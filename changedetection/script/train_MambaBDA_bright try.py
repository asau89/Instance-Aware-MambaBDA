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
from MambaCD.changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
import MambaCD.changedetection.utils_func.lovasz_loss as L
# Import the new, spatially aware loss function
from MambaCD.changedetection.utils_func.force_directed_loss import SpatiallyAwareForceLoss
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

        # Use the new Spatially Aware Force-Directed Loss
        self.force_directed_loss = SpatiallyAwareForceLoss(
            repulsion_margin=args.repulsion_margin,
            weight_attraction=args.weight_attraction,
            weight_repulsion=args.weight_repulsion,
            spatial_sigma=args.spatial_sigma,
        )

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
        self.inst_loc_losses_unweighted, self.inst_clf_losses_unweighted = [], []

    def get_boundary_weights(self, masks: torch.Tensor, boundary_weight=10.0) -> torch.Tensor:
        """
        Calculates weights for the cross-entropy loss that emphasize object boundaries.
        Args:
            masks (torch.Tensor): The ground truth masks, shape (B, H, W).
            boundary_weight (float): The weight to apply to boundary pixels.
        Returns:
            torch.Tensor: A weight map of shape (B, H, W) where boundary pixels have higher values.
        """
        with torch.no_grad():
            masks_np = masks.cpu().numpy()
            weights = torch.zeros_like(masks, dtype=torch.float32)

            for i in range(masks_np.shape[0]):
                # Use a simple binary distinction (background vs. any building) for boundary detection.
                binary_mask = (masks_np[i] > 0).astype(np.uint8)
                
                # Use Sobel filter to find edges/boundaries
                edge = sobel(binary_mask) > 0
                
                # Create a weight map where non-boundary pixels have weight 1
                # and boundary pixels have a higher weight.
                weight_map = np.ones_like(masks_np[i], dtype=np.float32)
                weight_map[edge] = boundary_weight
                weights[i] = torch.from_numpy(weight_map)
                
        return weights.to(masks.device)

    # ============================================================
    # ✅ 3. TRAINING LOOP
    # ============================================================
    def training(self):
        best_mIoU, best_round = 0.0, {}
        torch.cuda.empty_cache()
        progress_bar = tqdm(self.train_data_loader, desc="Training Progress")

        class_weights = torch.FloatTensor([1, 1, 1, 1]).cuda()

        for itera, data in enumerate(progress_bar):
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data
            pre_change_imgs, post_change_imgs = pre_change_imgs.cuda(), post_change_imgs.cuda()
            labels_loc, labels_clf = labels_loc.cuda().long(), labels_clf.cuda().long()

            if not (labels_clf != 255).any():
                continue

            self.optim.zero_grad()

            with autocast():
                pred_loc, pred_clf, feat_loc, feat_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                # --- Boundary-Aware Cross-Entropy Loss ---
                boundary_weights_loc = self.get_boundary_weights(labels_loc)
                boundary_weights_clf = self.get_boundary_weights(labels_clf)
                
                ce_loc_unreduced = F.cross_entropy(pred_loc, labels_loc, ignore_index=255, reduction='none')
                ce_loc = (ce_loc_unreduced * boundary_weights_loc).mean()
                
                lovasz_loc = L.lovasz_softmax(F.softmax(pred_loc, dim=1), labels_loc, ignore=255)
                
                ce_clf_unreduced = F.cross_entropy(pred_clf, labels_clf, weight=class_weights, ignore_index=255, reduction='none')
                ce_clf = (ce_clf_unreduced * boundary_weights_clf).mean()

                lovasz_clf = L.lovasz_softmax(F.softmax(pred_clf, dim=1), labels_clf, ignore=255)

                bda_loss = (ce_loc + 0.5 * lovasz_loc) + (ce_clf + 0.75 * lovasz_clf)

                size = feat_loc.shape[-2:]
                labels_loc_down = F.interpolate(labels_loc.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).long()
                labels_clf_down = F.interpolate(labels_clf.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).long()

                inst_loc = self.force_directed_loss(feat_loc, labels_loc_down)
                inst_clf = self.force_directed_loss(feat_clf, labels_clf_down)
                inst_loss = self.args.alpha * inst_loc + self.args.beta * inst_clf
                    
            final_loss = bda_loss + inst_loss

            self.scaler.scale(final_loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()

            # Logging
            self.iterations.append(itera + 1)
            self.bda_losses.append(bda_loss.item())
            self.inst_losses.append(inst_loss.item())
            self.final_losses.append(final_loss.item())
            self.inst_loc_losses_unweighted.append(inst_loc.item())
            self.inst_clf_losses_unweighted.append(inst_clf.item())

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
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        plt.figure(figsize=(12, 6))
        plt.plot(self.iterations, self.final_losses, label='Total Loss', color='red', alpha=0.8)
        plt.plot(self.iterations, self.bda_losses, label='BDA Loss (w/ Boundary Weight)', color='blue', alpha=0.8)
        plt.plot(self.iterations, self.inst_losses, label=f'Spatially Aware Force Loss (Weighted)', color='green', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title('Training Losses Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path_main = os.path.join(self.model_save_path, 'training_losses_main.png')
        plt.savefig(plot_path_main)
        print(f"Main loss plot saved to {plot_path_main}")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.iterations, self.inst_loc_losses_unweighted, label='Spatially Aware Loc Loss (Unweighted)', color='purple', alpha=0.8)
        plt.plot(self.iterations, self.inst_clf_losses_unweighted, label='Spatially Aware Clf Loss (Unweighted)', color='orange', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title('Unweighted Spatially Aware Losses Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path_inst = os.path.join(self.model_save_path, 'training_losses_instance_unweighted.png')
        plt.savefig(plot_path_inst)
        print(f"Unweighted instance loss plot saved to {plot_path_inst}")

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
    
    # --- Arguments for Spatially Aware Force-Directed Loss ---
    parser.add_argument('--repulsion_margin', type=float, default=2.0, help='Margin for inter-instance repulsion loss in feature space.')
    parser.add_argument('--weight_attraction', type=float, default=1.0, help='Weight for the attraction component of the instance loss.')
    parser.add_argument('--weight_repulsion', type=float, default=1.0, help='Weight for the repulsion component of the instance loss.')
    parser.add_argument('--spatial_sigma', type=float, default=50.0, help='Controls the falloff of the spatial weight for repulsion.')
    
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