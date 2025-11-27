import sys
sys.path.append('/home/granbell')
import numpy as np
from MambaCD.changedetection.datasets.make_data_loader import MultimodalDamageAssessmentDatset
from tqdm import tqdm

DATASET_PATH = '/home/granbell/MambaCD/dataset/dataset/train'
DATA_LIST_PATH = '/home/granbell/MambaCD/dataset/train_set.txt'

print("--- Starting Data Loader FULL SCAN Debug Script ---")

try:
    with open(DATA_LIST_PATH, "r") as f:
        train_data_name_list = [name.strip() for name in f]
    print(f"Successfully loaded {len(train_data_name_list)} file names from {DATA_LIST_PATH}")
except FileNotFoundError:
    print(f"ERROR: Could not find the data list file at {DATA_LIST_PATH}")
    sys.exit(1)

print("Initializing MultimodalDamageAssessmentDatset for a full scan...")
dataset = MultimodalDamageAssessmentDatset(
    dataset_path=DATASET_PATH,
    data_list=train_data_name_list,
    crop_size=256,
    type='train'
)
print("Dataset initialized.")

# =============================================================================
# THIS WILL SCAN THE ENTIRE DATASET - IT MAY TAKE A FEW MINUTES
# =============================================================================
all_unique_labels = set()
problematic_files = {}

# Use tqdm to show progress
for i in tqdm(range(len(dataset)), desc="Scanning all dataset samples"):
    try:
        # We only need the classification label for this test
        _, _, _, clf_label, data_idx = dataset[i]
        
        unique_in_sample = np.unique(clf_label)
        
        # Add all unique labels found in this sample to our master set
        all_unique_labels.update(unique_in_sample)
        
        # Check for any label that is not in the expected set [0, 1, 2, 3, 255]
        unexpected_labels = [l for l in unique_in_sample if l not in [0, 1, 2, 3, 255]]
        if unexpected_labels:
            problematic_files[data_idx] = unexpected_labels

    except Exception as e:
        print(f"\n[!!! ERROR !!!] An error occurred while loading sample index {i} ({dataset.data_list[i]}): {e}")
        continue

print("\n--- Scan Complete ---")
print(f"All unique label values found across the entire dataset: {sorted(list(all_unique_labels))}")

if problematic_files:
    print("\n[!!! FAILED !!!] Found unexpected label values in the following files:")
    for filename, labels in problematic_files.items():
        print(f"  - File: {filename}, Unexpected Labels: {labels}")
    print("\nThis is the cause of the IndexError. The labels in these files must be cleaned or remapped.")
elif np.max(list(all_unique_labels)) > 3 and 255 not in all_unique_labels:
     print("\n[!!! FAILED !!!] A label value greater than 3 was found, and it is NOT the ignore_index (255).")
     print("This confirms the remapping in the data loader is not working or there's an issue with the source data.")
else:
    print("\n[--- SUCCESS ---] All classification labels are within the expected range [0, 1, 2, 3] and the ignore index [255].")
    print("If the error still occurs, the problem is likely a stubborn Python cache issue.")