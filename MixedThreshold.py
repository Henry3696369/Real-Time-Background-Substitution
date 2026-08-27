import numpy as np
import cv2
from scipy import ndimage

def mixed_mask(confidence_mask, low_t, high_t):
    _, w = confidence_mask.shape
    start_x =  w // 4       # 1/4 width
    end_x = (3 * w) // 4    # 3/4 width

    # Median Blur to remove high_value noise
    blurred_mask = cv2.medianBlur((confidence_mask * 255).astype(np.uint8), 5)
    blurred_mask  = blurred_mask / 255.0

    # Double shresholding
    candidates = (blurred_mask > low_t).astype(np.uint8)
    core_mask = (blurred_mask > high_t).astype(np.uint8)
    if not np.any(core_mask):
        return np.zeros_like(confidence_mask, dtype=np.float32)

    # Locate the hole!
    kernel = np.ones((3, 3), np.uint8)
    dialted_core = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, kernel)
    dialted_core[-1, start_x:end_x] = 1
    full_core = ndimage.binary_fill_holes(dialted_core).astype(np.uint8)
    inner_core = cv2.erode(full_core, np.ones((15, 15), np.uint8))
    holes = ((inner_core == 1) & (core_mask==0)).astype(np.uint8)

    # Deal with connectivity
    combined_candidates = cv2.bitwise_or(candidates, full_core)
    _, labels = cv2.connectedComponents(combined_candidates)
    labels_in_core = np.unique(labels[core_mask > 0])
    keep_area = np.isin(labels, labels_in_core) & (labels > 0)

    # Make floating-number alpha matte
    final_mask = np.where(keep_area, confidence_mask, 0.0).astype(np.float32)
    final_mask[ (holes == 1) & keep_area] = 1
    
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

    return final_mask
