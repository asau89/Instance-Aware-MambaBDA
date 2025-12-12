# train_MambaBDA_bright.py

import sys
# Add your project root (adjust if your structure differs)
sys.path.append('/home/granbell')

import argparse
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import sobel

from MambaCD.changedetection.datasets.make_data_loader import make_data_loader, MultimodalDamageAssessmentDatset
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.ChangeMambaMMBDA_ForceDirected import ChangeMambaMMBDA
import MambaCD.changedetection.utils_func.lovasz_loss as L
# Import the new, spatially aware loss function
from MambaCD.changedetection.utils_func.instance_consistency_loss_forcedirected import InstanceConsistencyLoss
from torch.cuda.amp import autocast, GradScaler


# ============================================================
# ✅ 1. GLOBAL SEED FUNCTION
# ============================================================
def set_seed(seed: int = 42):
    """
    Ensures full reproducibility across:
    - Python random
    - NumPy
    - PyTorch (CPU & GPU)
    - cuDNN backend
    """
    print(f"Setting all random seeds to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Enforce deterministic operations (may slightly reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: for deterministic dataloader workers
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


# ============================================================
# ✅ 2. TRAINER CLASS
# ============================================================
class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        # Set up data loader with seeded workers for deterministic shuffling
        worker_init_fn = set_seed(args.seed)
        self.train_data_loader = make_data_loader(args, worker_init_fn=worker_init_fn)

        # Evaluators for localization, classification, and total metrics
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.evaluator_total = Evaluator(num_class=4)

    
        # Instance Consistency Loss Initialization
        self.inst_consistency_loss = InstanceConsistencyLoss()
        # Initialize model
        self.deep_model = ChangeMambaMMBDA(
            output_building=2, output_damage=4,
            pretrained=args.pretrained_weight_path,
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
        ).cuda()

        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Resume checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.deep_model.load_state_dict(state_dict, strict=False)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        self.scaler = GradScaler()

        # Lists for plotting losses
        self.iterations = []
        self.bda_losses, self.inst_losses, self.final_losses = [], [], []


    # ============================================================
    # ✅ 3. TRAINING LOOP
    # ============================================================
    def training(self):
        best_mIoU = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        class_weights = torch.FloatTensor([1, 1, 1, 1]).cuda()

        progress_bar = tqdm(range(elem_num), desc="Training Progress")
        for itera_idx in progress_bar:
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               continue

           # ==============================
            # === Forward + Loss Section ===
            # ==============================
            self.optim.zero_grad(set_to_none=True)

            # ---------------- AMP Forward Pass ----------------
            with autocast(enabled=True):
                pred_loc, pred_clf, features_loc, features_clf, features_loc_hr, features_clf_hr = self.deep_model(
                    pre_change_imgs, post_change_imgs
                )

                # --- Standard BDA Losses ---
                ce_loss_loc = F.cross_entropy(pred_loc, labels_loc, ignore_index=255)
                lovasz_loss_loc = L.lovasz_softmax(F.softmax(pred_loc, dim=1), labels_loc, ignore=255)

                ce_loss_clf = F.cross_entropy(pred_clf, labels_clf, weight=class_weights, ignore_index=255)
                lovasz_loss_clf = L.lovasz_softmax(F.softmax(pred_clf, dim=1), labels_clf, ignore=255)

                bda_loss = (ce_loss_loc + 0.5 * lovasz_loss_loc) + (ce_loss_clf + 0.75 * lovasz_loss_clf)

            # ---------------- FP32 Instance Loss ----------------
            # Compute instance losses outside autocast to prevent underflow
            feature_map_size_hr = features_loc_hr.shape[-2:]
            labels_loc_down_hr = F.interpolate(labels_loc.unsqueeze(1).float(), size=feature_map_size_hr, mode='nearest').squeeze(1).long()
            labels_clf_down_hr = F.interpolate(labels_clf.unsqueeze(1).float(), size=feature_map_size_hr, mode='nearest').squeeze(1).long()

            loss_inst_loc = self.inst_consistency_loss(features_loc_hr.float(), labels_loc_down_hr)
            loss_inst_clf = self.inst_consistency_loss(features_clf_hr.float(), labels_clf_down_hr)

            inst_loss = self.args.alpha * loss_inst_loc + self.args.beta * loss_inst_clf

            # --- Final Combined Loss ---
            final_loss = bda_loss + inst_loss

            # ====================================
            # === Backward + Optimization Stage ===
            # ====================================

            # Scale the loss before backward pass (AMP)
            self.scaler.scale(final_loss).backward()

            # Unscale gradients before clipping
            self.scaler.unscale_(self.optim)

            # Gradient clipping (safety for large updates)
            torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), 1.0)

            # Step optimizer
            self.scaler.step(self.optim)
            self.scaler.update()

            # Logging
            self.iterations.append(itera + 1)
            self.bda_losses.append(bda_loss.item())
            self.inst_losses.append(inst_loss.item())
            self.final_losses.append(final_loss.item())

            progress_bar.set_postfix({
                'BDA': f'{bda_loss.item():.4f}',
                'INST': f'{inst_loss.item():.4f}',
                'TOTAL': f'{final_loss.item():.4f}'
            })

            # Periodic validation
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

    def validation(self):
        print('---------starting validation-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_total.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.val_dataset_path, self.args.val_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=4, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            progress_bar = tqdm(val_data_loader, desc="Validating")
            for itera, data in enumerate(progress_bar):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                with autocast():
                    output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
                self.evaluator_total.add_batch(labels_clf, output_clf)

        print("---------Validation loop finished-----------")
        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / (damage_f1_score + 1e-8))
        final_OA = self.evaluator_total.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return loc_f1_score, harmonic_mean_f1, final_OA, mIoU, IoU_of_each_class

    def test(self):
        print('---------starting testing-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_total.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test', suffix='.tif')
        val_data_loader = DataLoader(dataset, batch_size=4, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            progress_bar = tqdm(val_data_loader, desc="Testing")
            for data in progress_bar:
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                with autocast():
                    output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
                self.evaluator_total.add_batch(labels_clf, output_clf)

        print("---------Testing loop finished-----------")
        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / (damage_f1_score + 1e-8))
        final_OA = self.evaluator_total.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return loc_f1_score, harmonic_mean_f1, final_OA, mIoU, IoU_of_each_class

    def plot_losses(self):
        # Ensure model_save_path exists for saving plots
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Plot 1: Main losses (BDA, Instance, Total)
        plt.figure(figsize=(12, 6))
        plt.plot(self.iterations, self.final_losses, label='Total Loss', color='red', alpha=0.8)
        plt.plot(self.iterations, self.bda_losses, label='BDA Loss', color='blue', alpha=0.8)
        plt.plot(self.iterations, self.inst_losses, label=f'Instance Consistency Loss (Weighted, alpha={self.args.alpha}, beta={self.args.beta})', color='green', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title('Training Losses Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path_main = os.path.join(self.model_save_path, 'training_losses_main.png')
        plt.savefig(plot_path_main)
        print(f"Main loss plot saved to {plot_path_main}")
        # plt.show() # Uncomment to display plot during execution

        # Plot 2: Unweighted Instance Consistency Losses
        plt.figure(figsize=(12, 6))
        plt.plot(self.iterations, self.inst_loc_losses_unweighted, label='Instance Localization Loss (Unweighted)', color='purple', alpha=0.8)
        plt.plot(self.iterations, self.inst_clf_losses_unweighted, label='Instance Classification Loss (Unweighted)', color='orange', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title('Unweighted Instance Consistency Losses Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path_inst = os.path.join(self.model_save_path, 'training_losses_instance_unweighted.png')
        plt.savefig(plot_path_inst)
        print(f"Unweighted instance loss plot saved to {plot_path_inst}")
        # plt.show() # Uncomment to display plot during execution

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
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for the localization component of the force-directed loss.")
    parser.add_argument('--beta', type=float, default=0.5, help="Weight for the classification component of the force-directed loss.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # Seed all RNGs
    set_seed(args.seed)

    # Load dataset name lists
    with open(args.train_data_list_path) as f: args.train_data_name_list = [x.strip() for x in f]
    with open(args.val_data_list_path) as f: args.val_data_name_list = [x.strip() for x in f]
    with open(args.test_data_list_path) as f: args.test_data_name_list = [x.strip() for x in f]

    trainer = Trainer(args)
    trainer.training()
    trainer.plot_losses()

if __name__ == "__main__":
    main()