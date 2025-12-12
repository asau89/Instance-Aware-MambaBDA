import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import label

class InstanceConsistencyLoss(nn.Module):
    """
    Computes the instance consistency loss, inspired by the DiSep methodology.
    This loss encourages feature vectors of pixels belonging to the same ground truth
    instance to be closer to their instance's centroid, thus promoting feature consistency.
    
    This implementation applies Connected Component Labeling on a per-class basis.
    
    This version strictly adheres to the provided mathematical equations from the image,
    including:
    - Averaging over all instances found by CCL (no minimum pixel count filter).
    - Averaging over the entire batch size for the final loss, even if some images
      have no contributing instances.
    """
    def __init__(self, ignore_index=255):
        super(InstanceConsistencyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, features: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Calculates the instance consistency loss.
        
        Args:
            features (torch.Tensor): The feature maps from the decoder, with shape (B, C, H, W).
            gt_masks (torch.Tensor): The ground truth masks, with shape (B, H, W).
            
        Returns:
            torch.Tensor: The calculated instance consistency loss (a scalar).
        """
        batch_size = features.shape[0]
        total_batch_loss = 0.0 # This will accumulate L_inst^image for all images in the batch

        for i in range(batch_size):
            mask_i = gt_masks[i]
            features_i = features[i]
            
            # Find all unique class labels present in the mask, excluding background and ignore_index
            unique_classes = torch.unique(mask_i)
            # Filter unique_classes as per logic for 'C_M' - classes in image M
            unique_classes = [c for c in unique_classes if c != 0 and c != self.ignore_index]

            # Even if no unique classes, this image's L_inst^image will be 0,
            # but it is still implicitly counted in the final batch_size denominator.
            
            image_instance_variance_sum = 0.0 # Sum of V_k for the current image
            total_instances_in_image_count = 0 # Denominator for L_inst^image, sum_{c in C_M} N_{c,i}

            for class_val in unique_classes:
                # Create a binary mask for the current class
                class_mask_np = (mask_i == class_val).cpu().numpy().astype(np.uint8)
                
                # Apply CCL to find instances *of this specific class*
                instance_map, num_instances = label(class_mask_np)
                
                if num_instances == 0:
                    continue # No instances for this class, move to next class

                # Iterate through all instances found for the current class
                for inst_id in range(1, num_instances + 1):
                    # Create a boolean mask for the current instance
                    instance_mask_np = (instance_map == inst_id)
                    
                    num_pixels_in_instance = instance_mask_np.sum()

                    # According to the formula, all instances from CCL contribute.
                    # However, we must ensure num_pixels_in_instance is not zero to prevent division by zero.
                    if num_pixels_in_instance == 0: 
                        continue # Skip truly empty masks, though CCL should prevent this for inst_id > 0

                    # Count this instance for the image-level denominator, as per formula sum_{c in C_M} N_{c,i}
                    total_instances_in_image_count += 1 

                    instance_mask_torch = torch.from_numpy(instance_mask_np).to(features.device)
                    
                    # Extract features for this instance
                    instance_features = features_i[:, instance_mask_torch]
                    
                    # Calculate the centroid (mean feature vector)
                    centroid = torch.mean(instance_features, dim=1, keepdim=True)
                    
                    # Calculate the Intra-Instance Variance (V_k)
                    # This is the average squared L2 distance from each pixel's feature to the centroid
                    V_k = torch.sum(torch.pow(instance_features - centroid, 2)) / num_pixels_in_instance
                    
                    image_instance_variance_sum += V_k
            
            # Calculate Final Instance Loss for one Image (L_inst^image)
            # As per formula: (sum of V_k) / (total number of instances in image)
            if total_instances_in_image_count > 0:
                L_inst_image = image_instance_variance_sum / total_instances_in_image_count
                total_batch_loss += L_inst_image
            else:
                # If an image has no valid instances, its L_inst^image is 0,
                # contributing 0 to total_batch_loss. It's still included in the final batch_size average.
                pass 

        # Final Equation for L_inst averaged over a batch of size B (1/B * sum of L_inst^image)
        if batch_size > 0:
            return total_batch_loss / batch_size
        else:
            return torch.tensor(0.0, device=features.device)