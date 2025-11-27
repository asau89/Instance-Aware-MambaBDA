import sys
import os
import numpy as np
import gradio as gr
from PIL import Image
import glob

# --- Configuration ---
MOCK_MODE = False # Set to False when you have the model and environment set up

if not MOCK_MODE:
    import torch
    import torchvision.transforms as transforms
    # Add project root to path to ensure MambaCD imports work
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # Import model and config
    from changedetection.configs.config import get_config
    from changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
    from changedetection.utils_func.metrics import Evaluator

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'changedetection', 'checkpoints')
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'changedetection', 'configs', 'vssm1', 'vssm_tiny_224_0229flex.yaml') 

def get_available_checkpoints():
    """Get list of available checkpoint models."""
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    
    checkpoint_folders = [d for d in os.listdir(CHECKPOINT_DIR) 
                         if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
    
    available_checkpoints = []
    for folder in checkpoint_folders:
        model_path = os.path.join(CHECKPOINT_DIR, folder, 'best_model.pth')
        if os.path.exists(model_path):
            available_checkpoints.append(folder)
    
    return sorted(available_checkpoints)

# Color mappings (using the same as infer_MambaBDA_BRIGHT.py)
ORI_LABEL_VALUE_DICT = {
    'background': (68, 1, 84),      # Dark purple
    'no_damage': (59, 82, 139),     # Blue/Purple
    'minor_damage': (33, 145, 140), # Teal/Green
    'major_damage': (253, 231, 37)  # Yellow
}

TARGET_LABEL_VALUE_DICT = {
    'background': 0,
    'no_damage': 1,
    'minor_damage': 2,
    'major_damage': 3,
}


def map_labels_to_colors(labels, ori_label_value_dict, target_label_value_dict):
    """Map label indices to RGB colors based on provided dictionaries."""
    target_to_ori = {v: k for k, v in target_label_value_dict.items()}
    H, W = labels.shape
    color_mapped_labels = np.zeros((H, W, 3), dtype=np.uint8)
    for target_label, ori_label in target_to_ori.items():
        mask = labels == target_label
        color_mapped_labels[mask] = ori_label_value_dict[ori_label]
    return color_mapped_labels

def load_model(checkpoint_name=None):
    """Load model with provided checkpoint name."""
    if MOCK_MODE:
        print("Running in MOCK MODE. Model will not be loaded.")
        return "MOCK_MODEL", None

    if checkpoint_name is None:
        print("No checkpoint selected.")
        return None, None

    model_weights_path = os.path.join(CHECKPOINT_DIR, checkpoint_name, 'best_model.pth')
    
    if not os.path.exists(model_weights_path):
        print(f"Model weights not found at {model_weights_path}")
        return None, None

    if not os.path.exists(CONFIG_FILE):
        print(f"Config file not found at {CONFIG_FILE}")
        return None, None

    # Real model loading
    try:
        class Args:
            cfg = CONFIG_FILE
            opts = None
            batch_size = 1
        args = Args()
        config = get_config(args)
        
        model = ChangeMambaMMBDA(
            output_building=2, output_damage=4, pretrained=None,
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
            use_checkpoint=False,
        )

        if os.path.exists(model_weights_path):
            print(f"Loading weights from {model_weights_path}")
            checkpoint = torch.load(model_weights_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"WARNING: Weights not found at {model_weights_path}")
            return None, None

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Initialize evaluators for metrics mode
        evaluator_loc = Evaluator(num_class=2)
        evaluator_clf = Evaluator(num_class=4)
        evaluator_total = Evaluator(num_class=4)
        
        return model, (evaluator_loc, evaluator_clf, evaluator_total)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Global model instance (will be loaded when user uploads files)
MODEL = None
EVALUATORS = None



if not MOCK_MODE:
    # Use the same normalization as the training/inference script
    # Mean and std values are for 0-255 range images
    MEAN = [123.675, 116.28, 103.53]
    STD = [58.395, 57.12, 57.375]
    
    def normalize_img(img):
        """Normalize image using the same method as training."""
        img_array = np.asarray(img, dtype=np.float32)
        mean_np = np.array(MEAN, dtype=np.float32)
        std_np = np.array(STD, dtype=np.float32)
        normalized_img = (img_array - mean_np) / std_np
        return normalized_img
    
    def preprocess_image(pil_image, is_post_disaster=False):
        """
        Preprocess PIL image for model input.
        Matches the preprocessing in make_data_loader.py exactly.
        NO RESIZING - images are processed at their original resolution!
        
        Args:
            pil_image: PIL Image
            is_post_disaster: If True, treats as SAR image (grayscale stacked to 3 channels)
        """
        # Convert to numpy array (no resizing!)
        img_np = np.array(pil_image, dtype=np.float32)
        
        # Handle pre-disaster (RGB) vs post-disaster (SAR/grayscale) differently
        if is_post_disaster:
            # Post-disaster: SAR image
            # If it's RGB, convert to grayscale first
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                # Take just one channel or convert to grayscale
                img_np = img_np[:, :, 0]  # Take first channel
            # Stack grayscale to 3 channels (like post_img = np.stack((post_img,) * 3, axis=-1))
            if len(img_np.shape) == 2:
                img_np = np.stack((img_np,) * 3, axis=-1)
        else:
            # Pre-disaster: RGB optical image
            # Take only first 3 channels (like pre_img = self.loader(pre_path)[:, :, 0:3])
            if len(img_np.shape) == 3 and img_np.shape[2] > 3:
                img_np = img_np[:, :, 0:3]
            elif len(img_np.shape) == 2:
                # If grayscale, convert to RGB
                img_np = np.stack((img_np,) * 3, axis=-1)
        
        # Normalize (same as __transforms with aug=False)
        img_normalized = normalize_img(img_np)
        
        # Transpose to (C, H, W)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_transposed).float()
        
        return img_tensor

def load_model_from_selection(checkpoint_name):
    """Handle model loading from checkpoint selection."""
    global MODEL, EVALUATORS
    
    if checkpoint_name is None or checkpoint_name == "":
        return "❌ Please select a checkpoint model"
    
    if MOCK_MODE:
        MODEL = "MOCK_MODEL"
        EVALUATORS = None
        return "✅ Model loaded successfully (MOCK MODE)!"
    
    try:
        print(f"Loading checkpoint: {checkpoint_name}")
        
        # Load the model
        loaded_model, loaded_evaluators = load_model(checkpoint_name)
        
        if loaded_model is None:
            return "❌ Failed to load model. Check console for errors."
        
        MODEL = loaded_model
        EVALUATORS = loaded_evaluators
        
        return f"✅ Model '{checkpoint_name}' loaded successfully!"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error loading model: {str(e)}"

def compute_metrics_loc_only(labels_loc, output_loc):
    """Compute only localization metrics"""
    if EVALUATORS is None:
        return None
    
    evaluator_loc, _, _ = EVALUATORS
    evaluator_loc.reset()
    evaluator_loc.add_batch(labels_loc, output_loc)
    
    loc_f1_score = evaluator_loc.Pixel_F1_score()
    loc_precision = evaluator_loc.Pixel_Precision_Rate()
    loc_recall = evaluator_loc.Pixel_Recall_Rate()
    loc_accuracy = evaluator_loc.Pixel_Accuracy()
    
    metrics_text = "===== LOCALIZATION METRICS =====\n"
    metrics_text += f"F1 Score: {loc_f1_score:.6f}\n"
    metrics_text += f"Precision: {loc_precision:.6f}\n"
    metrics_text += f"Recall: {loc_recall:.6f}\n"
    metrics_text += f"Accuracy: {100 * loc_accuracy:.4f}%\n"
    metrics_text += "================================"
    
    return metrics_text

def compute_metrics_clf_only(labels_loc, output_loc, labels_clf, output_clf):
    """Compute only classification metrics"""
    if EVALUATORS is None:
        return None
    
    _, evaluator_clf, evaluator_total = EVALUATORS
    evaluator_clf.reset()
    evaluator_total.reset()
    
    # Only evaluate damage classes where building exists
    output_clf_eval = output_clf[labels_loc > 0]
    labels_clf_eval = labels_clf[labels_loc > 0]
    if len(output_clf_eval) > 0:
        evaluator_clf.add_batch(labels_clf_eval, output_clf_eval)
    
    evaluator_total.add_batch(labels_clf, output_clf)
    
    damage_f1_score = evaluator_clf.Damage_F1_score()
    if isinstance(damage_f1_score, np.ndarray) and len(damage_f1_score) > 0:
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / (damage_f1_score + 1e-8))
    else:
        harmonic_mean_f1 = 0.0
    
    final_OA = evaluator_total.Pixel_Accuracy()
    IoU_of_each_class = evaluator_total.Intersection_over_Union()
    mIoU = evaluator_total.Mean_Intersection_over_Union()
    
    if not isinstance(IoU_of_each_class, np.ndarray):
        IoU_of_each_class = np.array([IoU_of_each_class])
    
    reverse_target_label_value_dict = {v: k for k, v in TARGET_LABEL_VALUE_DICT.items()}
    
    metrics_text = "===== CLASSIFICATION METRICS =====\n"
    metrics_text += f"Classification F1 (Harmonic Mean): {harmonic_mean_f1:.6f}\n"
    metrics_text += f"Overall Accuracy (OA): {100 * final_OA:.4f}%\n"
    metrics_text += f"Mean IoU (mIoU): {100 * mIoU:.4f}%\n\n"
    metrics_text += "Per-Class IoU:\n"
    for i, iou in enumerate(IoU_of_each_class):
        class_name = reverse_target_label_value_dict.get(i, f'class_{i}')
        metrics_text += f"  {class_name}: {100 * iou:.4f}%\n"
    
    if isinstance(damage_f1_score, np.ndarray) and len(damage_f1_score) > 0:
        class_names_for_f1 = [reverse_target_label_value_dict[i] for i in range(1, len(damage_f1_score) + 1)]
        metrics_text += "\nPer-Class F1:\n"
        for name, score in zip(class_names_for_f1, damage_f1_score):
            metrics_text += f"  {name}: {score:.6f}\n"
    metrics_text += "==================================="
    
    return metrics_text

def compute_metrics(labels_loc, output_loc, labels_clf, output_clf):
    """Compute evaluation metrics similar to infer_MambaBDA_BRIGHT.py"""
    if EVALUATORS is None:
        return None
    
    evaluator_loc, evaluator_clf, evaluator_total = EVALUATORS
    
    # Reset evaluators for this batch
    evaluator_loc.reset()
    evaluator_clf.reset()
    evaluator_total.reset()
    
    # Add batch for localization evaluation
    evaluator_loc.add_batch(labels_loc, output_loc)
    
    # Only evaluate damage classes where building exists
    output_clf_eval = output_clf[labels_loc > 0]
    labels_clf_eval = labels_clf[labels_loc > 0]
    if len(output_clf_eval) > 0:
        evaluator_clf.add_batch(labels_clf_eval, output_clf_eval)
    
    # Total evaluator includes all pixels (for OA & mIoU)
    evaluator_total.add_batch(labels_clf, output_clf)
    
    # Compute metrics
    loc_f1_score = evaluator_loc.Pixel_F1_score()
    damage_f1_score = evaluator_clf.Damage_F1_score()
    
    # Handle case where damage_f1_score might be empty or scalar
    if isinstance(damage_f1_score, np.ndarray) and len(damage_f1_score) > 0:
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / (damage_f1_score + 1e-8))
    else:
        harmonic_mean_f1 = 0.0
    
    oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1
    
    final_OA = evaluator_total.Pixel_Accuracy()
    IoU_of_each_class = evaluator_total.Intersection_over_Union()
    mIoU = evaluator_total.Mean_Intersection_over_Union()
    
    # Ensure IoU_of_each_class is an array
    if not isinstance(IoU_of_each_class, np.ndarray):
        IoU_of_each_class = np.array([IoU_of_each_class])
    
    # Per-class F1 report
    reverse_target_label_value_dict = {v: k for k, v in TARGET_LABEL_VALUE_DICT.items()}
    
    # Handle damage_f1_score as array or scalar
    if isinstance(damage_f1_score, np.ndarray) and len(damage_f1_score) > 0:
        class_names_for_f1 = [reverse_target_label_value_dict[i] for i in range(1, len(damage_f1_score) + 1)]
    else:
        class_names_for_f1 = []
        damage_f1_score = []
    
    metrics_text = "===== METRICS =====\n"
    metrics_text += f"Localization F1: {loc_f1_score:.6f}\n"
    metrics_text += f"Classification F1 (Harmonic Mean): {harmonic_mean_f1:.6f}\n"
    metrics_text += f"Overall Accuracy (OA): {100 * final_OA:.4f}%\n"
    metrics_text += f"Mean IoU (mIoU): {100 * mIoU:.4f}%\n"
    metrics_text += f"OA-F1 (weighted): {oaf1:.6f}\n\n"
    metrics_text += "Per-Class IoU:\n"
    for i, iou in enumerate(IoU_of_each_class):
        class_name = reverse_target_label_value_dict.get(i, f'class_{i}')
        metrics_text += f"  {class_name}: {100 * iou:.4f}%\n"
    
    if len(class_names_for_f1) > 0:
        metrics_text += "\nPer-Class F1:\n"
        for name, score in zip(class_names_for_f1, damage_f1_score):
            metrics_text += f"  {name}: {score:.6f}\n"
    metrics_text += "==================="
    
    return metrics_text

def run_inference(pre_image, post_image, ground_truth_loc=None, ground_truth_clf=None, calculate_metrics=False, output_mode="Both"):
    """
    Run inference on pre and post disaster images.
    
    Args:
        pre_image: Pre-disaster image (PIL Image)
        post_image: Post-disaster image (PIL Image, optional for Localization Only)
        ground_truth_loc: Ground truth localization mask (PIL Image, optional)
        ground_truth_clf: Ground truth classification mask (PIL Image, optional)
        calculate_metrics: Whether to calculate metrics (requires ground truth)
        output_mode: Output mode - "Localization Only", "Classification Only", or "Both"
    
    Returns:
        loc_vis: Localization visualization (PIL Image or None)
        clf_vis: Classification visualization (PIL Image or None)
        metrics_text: Metrics text (str or None)
    """
    # Validate inputs based on output mode
    if pre_image is None:
        return None, None, "❌ Please upload pre-disaster image"
    
    if output_mode != "Localization Only" and post_image is None:
        return None, None, "❌ Please upload post-disaster image for classification"

    if MOCK_MODE:
        # Generate dummy outputs based on mode
        w, h = pre_image.size
        loc_vis = None
        clf_vis = None
        
        if output_mode in ["Localization Only", "Both"]:
            # Dummy Localization: Random binary mask
            loc_vis = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
            loc_vis = Image.fromarray(loc_vis)
        
        if output_mode in ["Classification Only", "Both"]:
            # Dummy Classification: Random colors
            clf_vis = np.zeros((h, w, 3), dtype=np.uint8)
            colors = list(ORI_LABEL_VALUE_DICT.values())
            for i in range(h):
                for j in range(w):
                    clf_vis[i, j] = colors[np.random.randint(0, len(colors))]
            clf_vis = Image.fromarray(clf_vis)
        
        metrics_text = "Metrics not available in MOCK MODE" if calculate_metrics else None
        return loc_vis, clf_vis, metrics_text

    # Real Inference
    if MODEL is None:
        return None, None, "❌ Please load the model first by selecting a checkpoint"
    
    # Preprocess images using the same method as training
    # Pre-disaster: RGB optical image
    # Post-disaster: SAR image (treated as grayscale stacked to 3 channels)
    pre_tensor = preprocess_image(pre_image, is_post_disaster=False).unsqueeze(0)
    
    # For Localization Only mode, create a dummy post tensor (zeros) since model requires both inputs
    if output_mode == "Localization Only" and post_image is None:
        # Create a black/zero image with same dimensions as pre_image
        post_tensor = torch.zeros_like(pre_tensor)
    else:
        post_tensor = preprocess_image(post_image, is_post_disaster=True).unsqueeze(0)
    
    if torch.cuda.is_available():
        pre_tensor = pre_tensor.cuda()
        post_tensor = post_tensor.cuda()

    with torch.no_grad():
        output_loc, output_clf = MODEL(pre_tensor, post_tensor)
        output_loc_np = np.argmax(output_loc.data.cpu().numpy(), axis=1).squeeze()
        output_clf_np = np.argmax(output_clf.data.cpu().numpy(), axis=1).squeeze()

    # Generate visualizations based on output mode
    loc_vis = None
    clf_vis = None
    
    if output_mode in ["Localization Only", "Both"]:
        loc_vis_np = np.zeros_like(output_loc_np, dtype=np.uint8)
        loc_vis_np[output_loc_np > 0] = 255
        loc_vis = Image.fromarray(loc_vis_np)
    
    if output_mode in ["Classification Only", "Both"]:
        clf_vis_np = map_labels_to_colors(output_clf_np, ORI_LABEL_VALUE_DICT, TARGET_LABEL_VALUE_DICT)
        clf_vis_np[output_loc_np == 0] = ORI_LABEL_VALUE_DICT['background']
        clf_vis = Image.fromarray(clf_vis_np)

    # Calculate metrics if requested and ground truth is provided
    metrics_text = None
    if calculate_metrics and ground_truth_loc is not None and ground_truth_clf is not None:
        # Convert ground truth images to numpy arrays
        # Important: Don't convert to grayscale - preserve class labels (0, 1, 2, 3)
        gt_clf = np.array(ground_truth_clf)
        
        # If the image is RGB/RGBA, extract the label from first channel
        if len(gt_clf.shape) == 3:
            gt_clf = gt_clf[:, :, 0]
        
        # Resize to match output size if necessary
        if gt_clf.shape != output_clf_np.shape:
            from PIL import Image as PILImage
            gt_clf = np.array(PILImage.fromarray(gt_clf).resize(
                (output_clf_np.shape[1], output_clf_np.shape[0]), 
                PILImage.NEAREST
            ))
        
        # Derive localization ground truth from classification (same as original script)
        # loc_label[loc_label == 2] = 1; loc_label[loc_label == 3] = 1
        gt_loc_binary = gt_clf.copy()
        gt_loc_binary[gt_loc_binary == 2] = 1
        gt_loc_binary[gt_loc_binary == 3] = 1
        gt_loc_binary = (gt_loc_binary > 0).astype(np.int64)
        
        # Only compute metrics for the selected output mode
        if output_mode == "Localization Only":
            metrics_text = compute_metrics_loc_only(gt_loc_binary, output_loc_np)
        elif output_mode == "Classification Only":
            metrics_text = compute_metrics_clf_only(gt_loc_binary, output_loc_np, gt_clf, output_clf_np)
        else:  # Both
            metrics_text = compute_metrics(gt_loc_binary, output_loc_np, gt_clf, output_clf_np)
    elif calculate_metrics:
        metrics_text = "⚠️ Ground truth required for metrics calculation"

    return loc_vis, clf_vis, metrics_text


# --- Gradio UI ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

body {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
}

.gradio-container {
    background: transparent !important;
}

h1 {
    font-weight: 800 !important;
    font-size: 3rem !important;
    text-align: center;
    background: linear-gradient(to right, #2dd4bf, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.glass-panel {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 1rem;
    padding: 1.5rem;
    overflow: visible !important;
}

/* Scrollable listbox style for model selection */
.model-listbox {
    max-height: 300px !important;
    overflow-y: auto !important;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 0.5rem;
    padding: 0.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.primary-btn {
    background: linear-gradient(to right, #2dd4bf, #3b82f6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: transform 0.2s;
}

.primary-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 20px rgba(45, 212, 191, 0.2);
}
"""

# CSS removed from Blocks arg, injected via HTML
with gr.Blocks(title="Instance-Aware MambaBDA") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    with gr.Column(elem_classes="gradio-container"):
        gr.Markdown("# Instance-Aware MambaBDA")
        gr.Markdown("Next-Gen Building Damage Assessment with State Space Models", elem_classes="subtitle")
        
        if MOCK_MODE:
             gr.Markdown("⚠️ **DEMO MODE**: Running with simulated outputs.", elem_id="warning")
        
        # Model Loading Section
        with gr.Row():
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 0. Select Model")
                gr.Markdown("Choose a pretrained checkpoint model to use for inference.")
                
                available_checkpoints = get_available_checkpoints()
                if len(available_checkpoints) == 0:
                    gr.Markdown("⚠️ **No checkpoints found!** Please add model checkpoints to `changedetection/checkpoints/`")
                    checkpoint_selector = gr.Radio(
                        label="Model Checkpoint", 
                        choices=[], 
                        value=None,
                        interactive=True,
                        elem_classes=["model-listbox"]
                    )
                else:
                    checkpoint_selector = gr.Radio(
                        label="Model Checkpoint",
                        choices=available_checkpoints,
                        value=available_checkpoints[0] if available_checkpoints else None,
                        interactive=True,
                        elem_classes=["model-listbox"]
                    )
                
                btn_load_model = gr.Button("📥 Load Model", variant="secondary")
                model_status = gr.Textbox(label="Status", interactive=False, value="⏳ Select and load a checkpoint...")

        with gr.Row():
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 1. Upload Imagery")
                input_pre = gr.Image(
                    label="Pre-Disaster (Optical)", 
                    type="pil", 
                    elem_id="input-pre"
                )
                
                with gr.Group() as post_image_group:
                    input_post = gr.Image(
                        label="Post-Disaster (SAR)", 
                        type="pil", 
                        elem_id="input-post"
                    )
                
                gr.Markdown("### 2. Mode Selection")
                output_mode_radio = gr.Radio(
                    choices=["Both", "Localization Only", "Classification Only"],
                    value="Both",
                    label="Output Mode"
                )
                
                mode_radio = gr.Radio(
                    choices=["Without Metrics", "With Metrics"], 
                    value="Without Metrics",
                    label="Inference Mode"
                )
                
                # Ground truth inputs (visible only in metrics mode)
                with gr.Group(visible=False) as gt_group:
                    gr.Markdown("### 3. Upload Ground Truth (for Metrics)")
                    gr.Markdown("_Upload a single label image for both localization and classification_")
                    input_gt_label = gr.Image(label="Ground Truth Label", type="pil")
                
                btn_run = gr.Button("🚀 Analyze Damage", variant="primary", elem_classes="primary-btn")
            
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### Results")
                output_loc = gr.Image(
                    label="Building Localization", 
                    type="pil", 
                    interactive=False
                )
                output_clf = gr.Image(
                    label="Damage Classification", 
                    type="pil", 
                    interactive=False
                )
                
                # Metrics/Status output
                output_metrics = gr.Textbox(
                    label="Status / Metrics",
                    lines=15,
                    interactive=False,
                    placeholder="Results and metrics will appear here..."
                )
    
    # Load model button click
    btn_load_model.click(
        fn=load_model_from_selection,
        inputs=[checkpoint_selector],
        outputs=[model_status]
    )
    
    # Toggle post-disaster image based on output mode
    def toggle_post_image(output_mode):
        if output_mode == "Localization Only":
            return gr.update(visible=False)
        else:
            return gr.update(visible=True)
    
    output_mode_radio.change(
        fn=toggle_post_image,
        inputs=[output_mode_radio],
        outputs=[post_image_group]
    )

    # Toggle ground truth inputs based on mode selection
    def toggle_gt_inputs(mode):
        if mode == "With Metrics":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
    
    mode_radio.change(
        fn=toggle_gt_inputs,
        inputs=[mode_radio],
        outputs=[gt_group]
    )
    
    # Inference function wrapper
    def inference_wrapper(pre_img, post_img, mode, output_mode, gt_label):
        try:
            # Validate inputs based on output mode
            if pre_img is None:
                return None, None, "❌ Please upload pre-disaster image"
            
            if output_mode != "Localization Only" and post_img is None:
                return None, None, "❌ Please upload post-disaster image for classification/both modes"
            
            calculate_metrics = (mode == "With Metrics")
            loc_result, clf_result, metrics = run_inference(
                pre_img, post_img, gt_label, gt_label, calculate_metrics, output_mode
            )
            return loc_result, clf_result, metrics
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during inference:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return None, None, error_msg
    
    btn_run.click(
        fn=inference_wrapper,
        inputs=[input_pre, input_post, mode_radio, output_mode_radio, input_gt_label],
        outputs=[output_loc, output_clf, output_metrics]
    )

if __name__ == "__main__":
    demo.launch(share=True)

