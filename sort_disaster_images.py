import os
import shutil

# --- CONFIGURATION ---
SOURCE_IMAGES_DIR = '.' 
DESTINATION_DIR = 'dataset_sorted'
PRE_DISASTER_PATTERN = "{base_name}_pre_disaster.tif"
POST_DISASTER_PATTERN = "{base_name}_post_disaster.tif"
MASK_PATTERN = "{base_name}_building_damage.tif"
DEST_MASK_SUBFOLDER = 'label'
TRAIN_LIST_FILE = 'train_set.txt'
TEST_LIST_FILE = 'test_set.txt'
VAL_LIST_FILE = 'val_set.txt'
# --- END OF CONFIGURATION ---


def read_file_list(filepath):
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: File list not found at '{filepath}'. Skipping this set.")
        return []

def process_and_move_files(base_names, set_name):
    if not base_names:
        return

    print(f"\n--- Processing {set_name.upper()} set ---")
    
    dest_a = os.path.join(DESTINATION_DIR, set_name, 'A')
    dest_b = os.path.join(DESTINATION_DIR, set_name, 'B')
    dest_label = os.path.join(DESTINATION_DIR, set_name, DEST_MASK_SUBFOLDER)
    
    os.makedirs(dest_a, exist_ok=True)
    os.makedirs(dest_b, exist_ok=True)
    os.makedirs(dest_label, exist_ok=True)

    moved_count = 0
    not_found_count = 0
    for base_name in base_names:
        # Construct the filenames
        pre_filename = PRE_DISASTER_PATTERN.format(base_name=base_name)
        post_filename = POST_DISASTER_PATTERN.format(base_name=base_name)
        mask_filename = MASK_PATTERN.format(base_name=base_name)
        
        src_pre_path = os.path.join(SOURCE_IMAGES_DIR, pre_filename)
        src_post_path = os.path.join(SOURCE_IMAGES_DIR, post_filename)
        src_mask_path = os.path.join(SOURCE_IMAGES_DIR, mask_filename)
        
        # <<< --- NEW DEBUG BLOCK --- >>>
        print(f"\n--- Checking for base name: '{base_name}'")
        print(f"  > Looking for PRE file at: '{src_pre_path}'")
        print(f"  > Looking for POST file at: '{src_post_path}'")
        print(f"  > Looking for MASK file at: '{src_mask_path}'")
        # <<< --- END OF DEBUG BLOCK --- >>>

        if os.path.exists(src_pre_path) and os.path.exists(src_post_path) and os.path.exists(src_mask_path):
            print("  [SUCCESS] Found all three files. Moving them.")
            shutil.move(src_pre_path, os.path.join(dest_a, pre_filename))
            shutil.move(src_post_path, os.path.join(dest_b, post_filename))
            shutil.move(src_mask_path, os.path.join(dest_label, mask_filename))
            moved_count += 1
        else:
            print("  [!] Warning: Could not find the full pre/post/mask triplet.")
            not_found_count += 1
            
    print(f"\n--- Moved {moved_count} image triplets to the '{set_name}' folders. ---")
    if not_found_count > 0:
        print(f"--- Could not find {not_found_count} image triplets. ---")

if __name__ == "__main__":
    print("Starting dataset sorting process in DEBUG mode...")
    train_base_names = read_file_list(TRAIN_LIST_FILE)
    test_base_names = read_file_list(TEST_LIST_FILE)
    val_base_names = read_file_list(VAL_LIST_FILE)
    process_and_move_files(train_base_names, 'train')
    process_and_move_files(test_base_names, 'test')
    process_and_move_files(val_base_names, 'val')
    print("\nSorting complete!")