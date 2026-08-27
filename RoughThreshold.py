import numpy as np

def rough_mask(confidence_mask, threshold):

    binary_mask = confidence_mask > threshold

    stacked_mask = np.stack((binary_mask,) * 3, axis=-1)
    
    return stacked_mask