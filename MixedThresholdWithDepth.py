import numpy as np
import cv2
from scipy import ndimage


def mixed_mask(depth_mask, confidence_mask, low_t, high_t):
    _, w = confidence_mask.shape
    start_x =  w // 4       # 1/4 width
    end_x = (3 * w) // 4    # 3/4 width

    # Median Blur to remove high_value noise
    blurred_mask = cv2.medianBlur((confidence_mask * 255).astype(np.uint8), 3)
    blurred_mask  = blurred_mask / 255.0
    candidates = (blurred_mask > low_t).astype(np.uint8)
    core_mask = (blurred_mask > high_t).astype(np.uint8)
    if not np.any(core_mask):
        return np.zeros_like(confidence_mask, dtype=np.float32)
    
    # Get the median depth of the core region
    ref_depth = np.median(depth_mask[(core_mask > 0) & (depth_mask > 0)])  
    # Get the depth_diff for every pixel
    depth_diff = np.abs(depth_mask - ref_depth)
    sigma = 0.2 
    
    # Locate the hole!
    kernel = np.ones((3, 3), np.uint8)
    dialted_core = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, kernel)
    dialted_core[-1, start_x:end_x] = 1
    full_core = ndimage.binary_fill_holes(dialted_core).astype(np.uint8)
    inner_core = cv2.erode(full_core, np.ones((15, 15), np.uint8))
    holes = ((inner_core == 1) & (core_mask==0)).astype(np.uint8)
    combined_candidates = cv2.bitwise_or(candidates, full_core) 
    valid_holes = (holes == 1) & ((depth_mask == 0) | (np.abs(depth_diff) < 0.3))


    
    # Deal with connectivity
    _, labels = cv2.connectedComponents(combined_candidates)
    labels_in_core = np.unique(labels[core_mask > 0])
    keep_area_topo = np.isin(labels, labels_in_core) & (labels > 0)

    temp_mask = np.where(keep_area_topo, confidence_mask, 0.0).astype(np.float32)
    
    # Build a new confidence map based on depth_diff using Gaussian probability function
    weight = np.zeros_like(confidence_mask)
    weight[keep_area_topo] = np.exp(-(depth_diff[keep_area_topo]**2) / (2 * sigma**2))
    
    # We get a new map based on integrated probability and connectivity
    mixed_confidence_mask = temp_mask * weight

    # We drop the low confidence pixel
    strong_area = mixed_confidence_mask > high_t
    keep_area = mixed_confidence_mask > low_t

    # We keep the original probability value for our selected region, and 
    # replace strong confidence region with 1
    final_mask = np.where(keep_area, confidence_mask, 0.0).astype(np.float32)
    final_mask = np.where(strong_area, 1, final_mask)
    final_mask[(valid_holes == 1) & keep_area_topo] = 1
    final_mask[(core_mask == 1)] = 1

    # Gaussian blur to make it smoother
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

    return final_mask