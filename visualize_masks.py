import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def visualize_ground_truth_custom_color(tif_filepath, output_folder="labels"):
    """
    Reads and visualizes a .tif ground truth mask file using a specific,
    pre-defined color scheme and saves the output to a specified folder.
    
    Args:
        tif_filepath (str): The full path to the .tif file.
        output_folder (str): The name of the folder where the visualizations will be saved.
    """
    # --- 1. Define the custom color map and labels using the provided dictionaries ---
    # These dictionaries reflect the classes and their RGB values/IDs from your other script.
    # RGB values are 0-255
    ori_label_value_dict = {
        'background': (68, 1, 84),      # Approximated dark purple from the image's background/viridis start
        'no_damage': (59, 82, 139),     # Approximated darker blue/purple (Class ID 1)
        'minor_damage': (33, 145, 140), # Approximated teal/green (Class ID 2)
        'major_damage': (253, 231, 37)  # Approximated bright yellow (Class ID 3)
    }

    target_label_value_dict = {
        'background': 0,
        'no_damage': 1,
        'minor_damage': 2,
        'major_damage': 3,
    }

    # Construct COLOR_MAP (ID -> RGB) and CLASS_NAMES (ID -> Name)
    COLOR_MAP = {}
    CLASS_NAMES = {}
    for class_name, class_id in target_label_value_dict.items():
        COLOR_MAP[class_id] = ori_label_value_dict[class_name]
        CLASS_NAMES[class_id] = class_name

    # --- 2. Check and load the .tif file ---
    if not os.path.exists(tif_filepath):
        print(f"Error: The file '{tif_filepath}' was not found.")
        return
    try:
        mask = tifffile.imread(tif_filepath)
        print(f"Successfully loaded file: {tif_filepath}")
        print(f"Mask shape: {mask.shape}, Data type: {mask.dtype}")
    except Exception as e:
        print(f"Error reading the file '{tif_filepath}': {e}")
        return

    # --- 3. Convert single-channel mask to a 3-channel RGB image ---
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in COLOR_MAP.items():
        pixels_to_color = (mask == class_id)
        rgb_image[pixels_to_color] = color

    print("Converted mask to RGB image using custom color scheme.")

    # --- 4. Visualize the result with a custom legend ---
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb_image)
    ax.set_title(f"Ground Truth Visualization\n{os.path.basename(tif_filepath)}", fontsize=16)
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")

    legend_patches = []
    unique_values_in_mask = np.unique(mask)
    
    for class_id in sorted(unique_values_in_mask):
        if class_id in CLASS_NAMES:
            label = f"{CLASS_NAMES[class_id]}\n({', '.join(map(str, COLOR_MAP[class_id]))})"
            color = np.array(COLOR_MAP[class_id]) / 255.0
            patch = mpatches.Patch(color=color, label=label)
            legend_patches.append(patch)

    ax.legend(handles=legend_patches, 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              borderaxespad=0., 
              fontsize='large')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # --- 5. Save the image to the specified output folder ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: '{output_folder}'")

    output_filename = os.path.splitext(os.path.basename(tif_filepath))[0] + "_visualization.png"
    output_path = os.path.join(output_folder, output_filename)
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to: {output_path}")
    plt.close(fig)

# --- HOW TO USE ---

# 1. SPECIFY THE DIRECTORY CONTAINING YOUR TIF MASKS
TARGET_DIRECTORY = os.path.join("dataset", "dataset", "test", "target")
OUTPUT_FOLDER_NAME = "labels" 

# 2. ITERATE THROUGH ALL .tif FILES IN THE DIRECTORY AND RUN THE VISUALIZATION FUNCTION
if not os.path.isdir(TARGET_DIRECTORY):
    print(f"Error: The target directory '{TARGET_DIRECTORY}' was not found.")
else:
    print(f"Scanning directory: {TARGET_DIRECTORY} for .tif files...")
    tif_files_found = False
    for filename in os.listdir(TARGET_DIRECTORY):
        if filename.endswith(".tif"):
            tif_filepath = os.path.join(TARGET_DIRECTORY, filename)
            visualize_ground_truth_custom_color(tif_filepath, output_folder=OUTPUT_FOLDER_NAME)
            tif_files_found = True
    
    if not tif_files_found:
        print(f"No .tif files found in '{TARGET_DIRECTORY}'.")