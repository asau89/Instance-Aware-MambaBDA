import sys
# Set the base directory for the MambaCD package
sys.path.append('/home/granbell')

import argparse
import os
import time
import random
import numpy as np

from MambaCD.changedetection.configs.config import get_config
from MambaCD.changedetection.datasets.make_data_loader import make_data_loader, MultimodalDamageAssessmentDatset
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
import MambaCD.changedetection.utils_func.lovasz_loss as L

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn


# ============================================================
# Reproducibility Function
# ============================================================
def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Trainer Class
# ============================================================
class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        # Load training data
        self.train_data_loader = make_data_loader(args)

        # Evaluators for different tasks
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.evaluator_total = Evaluator(num_class=4)
        self.evaluator_event_noto = Evaluator(num_class=4)
        self.evaluator_event_marshall = Evaluator(num_class=4)

        # Build model
        self.deep_model = ChangeMambaMMBDA(
            output_building=2,
            output_damage=4,
            pretrained=args.pretrained_weight_path,  # ensure model supports this
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Resume from checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model_dict = self.deep_model.state_dict()
            checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(checkpoint)
            self.deep_model.load_state_dict(model_dict)

        # Optimizer & Scaler for AMP
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        self.scaler = GradScaler()  # <-- Added for mixed precision

    # ============================================================
    # Training Function
    # ============================================================
    def training(self):
        best_mIoU = 0.0
        best_round = {}
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        class_weights = torch.FloatTensor([1, 1, 1, 1]).cuda()

        progress_bar = tqdm(range(elem_num), desc="Training Progress")

        for itera_idx in progress_bar:
            itera, data = next(train_enumerator)
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            # Skip invalid labels
            if not (labels_clf != 255).any():
                continue

            self.optim.zero_grad(set_to_none=True)

            with autocast():
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
                lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)

                ce_loss_clf = F.cross_entropy(output_clf, labels_clf, weight=class_weights, ignore_index=255)
                lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)

                final_loss = ce_loss_loc + ce_loss_clf + (0.5 * lovasz_loss_loc + 0.75 * lovasz_loss_clf)

            self.scaler.scale(final_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            progress_bar.set_postfix({'Total Loss': f'{final_loss.item():.4f}'})

            # Validation checkpoint
            if (itera + 1) % 500 == 0:
                self.deep_model.eval()
                loc_f1_val, clf_f1_val, oa_val, miou_val, iou_each_val = self.validation()

                if miou_val > best_mIoU:
                    loc_f1_test, clf_f1_test, oa_test, miou_test, iou_each_test = self.test()
                    torch.save(self.deep_model.state_dict(),
                               os.path.join(self.model_save_path, f'best_model.pth'))
                    best_mIoU = miou_val
                    best_round = {
                        'best iter': itera + 1,
                        'loc f1 (val)': loc_f1_val * 100,
                        'clf f1 (val)': clf_f1_val * 100,
                        'OA (val)': oa_val * 100,
                        'mIoU (val)': miou_val * 100,
                        'sub class IoU (val)': iou_each_val * 100,
                        'loc f1 (test)': loc_f1_test * 100,
                        'clf f1 (test)': clf_f1_test * 100,
                        'OA (test)': oa_test * 100,
                        'mIoU (test)': miou_test * 100,
                        'sub class IoU (test)': iou_each_test * 100
                    }
                    print('\n✅ New best round:')
                    for k, v in best_round.items():
                        print(f'{k}: {v}')
                self.deep_model.train()

        print('\n🎯 Best Round Summary:')
        for k, v in best_round.items():
            print(f'{k}: {v}')

    # ============================================================
    # Validation Function
    # ============================================================
    def validation(self):
        print('\n--------- Starting Validation ---------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_total.reset()

        dataset = MultimodalDamageAssessmentDatset(
            self.args.val_dataset_path,
            self.args.val_data_name_list,
            256,
            None,
            'test'
        )
        val_loader = DataLoader(dataset, batch_size=4, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validating"):
                pre, post, lbl_loc, lbl_clf, _ = data
                pre, post = pre.cuda(), post.cuda()
                lbl_loc, lbl_clf = lbl_loc.cuda().long(), lbl_clf.cuda().long()

                with autocast():
                    out_loc, out_clf = self.deep_model(pre, post)

                out_loc = np.argmax(out_loc.cpu().numpy(), axis=1)
                out_clf = np.argmax(out_clf.cpu().numpy(), axis=1)
                lbl_loc = lbl_loc.cpu().numpy()
                lbl_clf = lbl_clf.cpu().numpy()

                self.evaluator_loc.add_batch(lbl_loc, out_loc)
                self.evaluator_clf.add_batch(lbl_clf[lbl_loc > 0], out_clf[lbl_loc > 0])
                self.evaluator_total.add_batch(lbl_clf, out_clf)

        loc_f1 = self.evaluator_loc.Pixel_F1_score()
        damage_f1 = self.evaluator_clf.Damage_F1_score()
        harmonic_f1 = len(damage_f1) / np.sum(1.0 / damage_f1)
        oa = self.evaluator_total.Pixel_Accuracy()
        iou_each = self.evaluator_total.Intersection_over_Union()
        miou = self.evaluator_total.Mean_Intersection_over_Union()

        print(f'OA={oa*100:.2f}, mIoU={miou*100:.2f}, IoU Each={iou_each*100}')
        return loc_f1, harmonic_f1, oa, miou, iou_each

    # ============================================================
    # Test Function
    # ============================================================
    def test(self):
        print('\n--------- Starting Testing ---------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_total.reset()

        dataset = MultimodalDamageAssessmentDatset(
            self.args.test_dataset_path,
            self.args.test_data_name_list,
            256,
            None,
            'test',
            suffix='.tif'
        )
        test_loader = DataLoader(dataset, batch_size=4, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for data in tqdm(test_loader, desc="Testing"):
                pre, post, lbl_loc, lbl_clf, _ = data
                pre, post = pre.cuda(), post.cuda()
                lbl_loc, lbl_clf = lbl_loc.cuda().long(), lbl_clf.cuda().long()

                with autocast():
                    out_loc, out_clf = self.deep_model(pre, post)

                out_loc = np.argmax(out_loc.cpu().numpy(), axis=1)
                out_clf = np.argmax(out_clf.cpu().numpy(), axis=1)
                lbl_loc = lbl_loc.cpu().numpy()
                lbl_clf = lbl_clf.cpu().numpy()

                self.evaluator_loc.add_batch(lbl_loc, out_loc)
                self.evaluator_clf.add_batch(lbl_clf[lbl_loc > 0], out_clf[lbl_loc > 0])
                self.evaluator_total.add_batch(lbl_clf, out_clf)

        loc_f1 = self.evaluator_loc.Pixel_F1_score()
        damage_f1 = self.evaluator_clf.Damage_F1_score()
        harmonic_f1 = len(damage_f1) / np.sum(1.0 / damage_f1)
        oa = self.evaluator_total.Pixel_Accuracy()
        iou_each = self.evaluator_total.Intersection_over_Union()
        miou = self.evaluator_total.Mean_Intersection_over_Union()

        print(f'OA={oa*100:.2f}, mIoU={miou*100:.2f}, IoU Each={iou_each*100}')
        return loc_f1, harmonic_f1, oa, miou, iou_each


# ============================================================
# Main Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Training on xBD dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument('--opts', default=None, nargs='+', help="Modify config options by adding 'KEY VALUE' pairs.")
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='xBD')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--val_dataset_path', type=str)
    parser.add_argument('--train_data_list_path', type=str)
    parser.add_argument('--test_data_list_path', type=str)
    parser.add_argument('--val_data_list_path', type=str)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MMMambaBDA')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    set_seed(args.seed)

    # Load data lists
    with open(args.train_data_list_path, "r") as f:
        args.train_data_name_list = [line.strip() for line in f]

    with open(args.test_data_list_path, "r") as f:
        args.test_data_name_list = [line.strip() for line in f]

    with open(args.val_data_list_path, "r") as f:
        args.val_data_name_list = [line.strip() for line in f]

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
