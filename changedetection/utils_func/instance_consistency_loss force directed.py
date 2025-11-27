import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label as connected_components


class InstanceConsistencyLoss(nn.Module):
    """
    Instance Consistency Loss (Compactness + Separation)
    ----------------------------------------------------
    Encourages features of the same instance to cluster (compactness)
    while pushing apart different instance centroids (separation).

    Args:
        ignore_index (int): Label to ignore in ground truth.
        lambda_sep (float): Weight for separation term.
        min_pixels (int): Minimum pixels to consider a valid instance.
        normalize_feats (bool): Normalize features before computing distances.
    """

    def __init__(self, ignore_index=255, lambda_sep=2, min_pixels=4, normalize_feats=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_sep = lambda_sep
        self.min_pixels = min_pixels
        self.normalize_feats = normalize_feats

    def forward(self, features: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (B, C, H, W): Feature maps.
            gt_masks (B, H, W): Ground truth masks.

        Returns:
            torch.Tensor: Scalar instance consistency loss.
        """
        B, C, H, W = features.shape
        total_loss = 0.0

        for b in range(B):
            feat = features[b]
            mask = gt_masks[b]

            if self.normalize_feats:
                feat = F.normalize(feat, p=2, dim=0)

            # Get per-class connected components
            unique_classes = torch.unique(mask)
            unique_classes = [c.item() for c in unique_classes if c.item() not in [0, self.ignore_index]]

            centroids = []
            compact_losses = []

            for cls in unique_classes:
                binary_mask = (mask == cls).cpu().numpy().astype(np.uint8)
                instance_map, num_instances = connected_components(binary_mask)

                for inst_id in range(1, num_instances + 1):
                    inst_mask_np = (instance_map == inst_id)
                    num_pixels = inst_mask_np.sum()
                    if num_pixels < self.min_pixels:
                        continue

                    inst_mask = torch.from_numpy(inst_mask_np).to(feat.device)
                    instance_features = feat[:, inst_mask]
                    centroid = instance_features.mean(dim=1, keepdim=True)
                    centroids.append(centroid)

                    compact_loss = torch.mean((instance_features - centroid) ** 2)
                    compact_losses.append(compact_loss)

            if len(compact_losses) == 0:
                continue

            # Compute intra-instance compactness
            L_compact = torch.stack(compact_losses).mean()

            # Compute inter-instance separation
            if len(centroids) > 1:
                centroids = torch.cat(centroids, dim=1)
                # pairwise squared distances
                dist_matrix = torch.cdist(centroids.t(), centroids.t(), p=2) ** 2
                # exclude self-distances
                mask_diag = torch.eye(dist_matrix.size(0), device=dist_matrix.device).bool()
                dist_matrix = dist_matrix.masked_fill(mask_diag, float('inf'))
                L_sep = torch.exp(-dist_matrix).mean()
            else:
                L_sep = torch.tensor(0.0, device=feat.device)

            total_loss += L_compact + self.lambda_sep * L_sep

        return total_loss / max(1, B)
