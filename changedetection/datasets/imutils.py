import random
import numpy as np
from PIL import Image
# from scipy import misc # This import was commented out, keeping it that way
import torch
import torchvision

from PIL import ImageEnhance


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """
    Normalize image by subtracting mean and dividing by std.
    Ensures image is np.float32 to prevent implicit float64 conversion during calculations.
    """
    # CRITICAL CHANGE: Ensure the input image array is float32 for consistent calculations.
    # This prevents NumPy from implicitly converting to float64 during arithmetic operations,
    # which can interfere with PyTorch's Automatic Mixed Precision (AMP).
    img_array = np.asarray(img, dtype=np.float32)
    
    # Ensure mean and std are also float32 NumPy arrays
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)

    # Perform normalization directly on the float32 array
    normalized_img = (img_array - mean_np) / std_np
    
    return normalized_img

def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label

def random_fliplr_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_1, label_2


def random_fliplr_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.fliplr(label_cd)
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label_cd, label_1, label_2

def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label

def random_flipud_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_1, label_2


def random_flipud_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.flipud(label_cd)
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)

        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label_cd, label_1, label_2


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1 # Generates 1, 2, or 3 (for 90, 180, 270 degrees)

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_rot_bda(pre_img, post_img, label_1, label_2):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()

    return pre_img, post_img, label_1, label_2


def random_rot_mcd(pre_img, post_img, label_cd, label_1, label_2):
    k = random.randrange(3) + 1
    
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    label_cd = np.rot90(label_cd, k).copy()

    return pre_img, post_img, label_cd, label_1, label_2


# NOTE: random_crop and random_bi_image_crop seem to be unused in your MultimodalDamageAssessmentDatset,
# but I'm including them with potential dtype fixes for completeness.
def random_crop(img, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w, _ = img.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    # Use img.dtype for pixel data, assuming it's already float32 from normalize_img or uint8
    pad_image = np.zeros((H, W, 3), dtype=img.dtype) # Use original img dtype
    
    # Fill padding with mean values (these should be float32)
    mean_np = np.array(mean_rgb, dtype=np.float32)
    pad_image[:, :, 0] = mean_np[0]
    pad_image[:, :, 1] = mean_np[1]
    pad_image[:, :, 2] = mean_np[2]

    H_pad = int(np.random.randint(0, H - h + 1))
    W_pad = int(np.random.randint(0, W - w + 1))

    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = img # Assign original img

    def get_random_cropbox(cat_max_ratio=0.75):
        # NOTE: This cropping logic uses pad_image[..., 0] as a 'temp_label'.
        # If pad_image is image data (float32), unique values for unique_classes 
        # for segmentation labels. It should probably use an actual label.
        # This function might be intended for a different use case or needs a label input.
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_image[H_start:H_end, W_start:W_end, 0] # This is likely image channel data, not label
            index, cnt = np.unique(temp_label, return_counts=True)
            # The condition 'len(cnt > 1)' can be problematic if cnt is empty or has one element.
            # Should be 'len(cnt) > 1'
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()

    img = pad_image[H_start:H_end, W_start:W_end, :]

    return img


def random_bi_image_crop(pre_img, object, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    # NOTE: This function's `object` argument seems to be a label.
    # It doesn't perform padding, assumes object and pre_img are large enough.
    h, w = object.shape # Assumes object is 2D (label)

    H = max(crop_size, h)
    W = max(crop_size, w)

    H_start = random.randrange(0, H - crop_size + 1, 1)
    H_end = H_start + crop_size
    W_start = random.randrange(0, W - crop_size + 1, 1)
    W_end = W_start + crop_size

    pre_img = pre_img[H_start:H_end, W_start:W_end, :]
    object = object[H_start:H_end, W_start:W_end] # crop label (object)
    
    return pre_img, object


def random_crop_new(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    # Use pre_img.dtype as reference for image padding, assuming it's already float32
    pad_pre_image = np.zeros((H, W, 3), dtype=pre_img.dtype)
    pad_post_image = np.zeros((H, W, 3), dtype=post_img.dtype)
    
    # CRITICAL CHANGE for labels: Use uint8 for labels, filled with ignore_index
    pad_label = np.full((H, W), ignore_index, dtype=np.uint8) # Use np.full for clearer intent

    # Fill padding with mean values (these should be float32)
    mean_np = np.array(mean_rgb, dtype=np.float32)
    pad_pre_image[:, :, 0] = mean_np[0]
    pad_pre_image[:, :, 1] = mean_np[1]
    pad_pre_image[:, :, 2] = mean_np[2]

    pad_post_image[:, :, 0] = mean_np[0]
    pad_post_image[:, :, 1] = mean_np[1]
    pad_post_image[:, :, 2] = mean_np[2]

    H_pad = int(np.random.randint(0, H - h + 1))
    W_pad = int(np.random.randint(0, W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            # Fixed len(cnt > 1) -> len(cnt) > 1
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
   
    return pre_img, post_img, label


def random_crop_bda(pre_img, post_img, loc_label, clf_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    # Use pre_img.dtype as reference for image padding, assuming it's already float32
    pad_pre_image = np.zeros((H, W, 3), dtype=pre_img.dtype)
    pad_post_image = np.zeros((H, W, 3), dtype=post_img.dtype)
    
    # CRITICAL CHANGE for labels: Use uint8 for labels
    pad_loc_label = np.full((H, W), ignore_index, dtype=np.uint8)
    pad_clf_label = np.full((H, W), ignore_index, dtype=np.uint8)

    # Fill padding with mean values (these should be float32)
    mean_np = np.array(mean_rgb, dtype=np.float32)
    pad_pre_image[:, :, 0] = mean_np[0]
    pad_pre_image[:, :, 1] = mean_np[1]
    pad_pre_image[:, :, 2] = mean_np[2]

    pad_post_image[:, :, 0] = mean_np[0]
    pad_post_image[:, :, 1] = mean_np[1]
    pad_post_image[:, :, 2] = mean_np[2]

    H_pad = int(np.random.randint(0, H - h + 1))
    W_pad = int(np.random.randint(0, W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_loc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = loc_label
    pad_clf_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = clf_label

    def get_random_cropbox(cat_max_ratio=0.75):
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_loc_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            # Fixed len(cnt > 1) -> len(cnt) > 1
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    loc_label = pad_loc_label[H_start:H_end, W_start:W_end]
    clf_label = pad_clf_label[H_start:H_end, W_start:W_end]

    return pre_img, post_img, loc_label, clf_label


def random_crop_mcd(pre_img, post_img, label_cd, label_1, label_2, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label_1.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    # Use pre_img.dtype as reference for image padding, assuming it's already float32
    pad_pre_image = np.zeros((H, W, 3), dtype=pre_img.dtype)
    pad_post_image = np.zeros((H, W, 3), dtype=post_img.dtype)
    
    # CRITICAL CHANGE for labels: Use uint8 for labels
    pad_label_cd = np.full((H, W), ignore_index, dtype=np.uint8)
    pad_label_1 = np.full((H, W), ignore_index, dtype=np.uint8)
    pad_label_2 = np.full((H, W), ignore_index, dtype=np.uint8)

    # Fill padding with mean values (these should be float32)
    mean_np = np.array(mean_rgb, dtype=np.float32)
    pad_pre_image[:, :, 0] = mean_np[0]
    pad_pre_image[:, :, 1] = mean_np[1]
    pad_pre_image[:, :, 2] = mean_np[2]

    pad_post_image[:, :, 0] = mean_np[0]
    pad_post_image[:, :, 1] = mean_np[1]
    pad_post_image[:, :, 2] = mean_np[2]

    H_pad = int(np.random.randint(0, H - h + 1))
    W_pad = int(np.random.randint(0, W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img

    pad_label_cd[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_cd
    pad_label_1[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_1
    pad_label_2[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label_2

    def get_random_cropbox(cat_max_ratio=0.75):
        for i in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label_1[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            # Fixed len(cnt > 1) -> len(cnt) > 1
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label_cd = pad_label_cd[H_start:H_end, W_start:W_end]
    label_1 = pad_label_1[H_start:H_end, W_start:W_end]
    label_2 = pad_label_2[H_start:H_end, W_start:W_end]

    return pre_img, post_img, label_cd, label_1, label_2

# Placeholder for img_loader, assuming it's part of this imutils or provided externally
# If img_loader is intended to be part of imutils, you need to define its actual loading logic here.
# For example, using OpenCV:
# import cv2
# def img_loader(path):
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise FileNotFoundError(f"Image not found at {path}")
#     return img