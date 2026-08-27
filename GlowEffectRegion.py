import cv2
def glow(region_mask, output_image):
    # I create 2 layers to make the effect better
    core_glow = cv2.GaussianBlur(region_mask, (15, 15), 0)
    outer_glow = cv2.GaussianBlur(region_mask, (75, 75), 0) 
    # Color grading
    output_image[:, :, 0] = cv2.addWeighted(output_image[:, :, 0], 1.0, core_glow, 0.1, 0)
    output_image[:, :, 1] = cv2.addWeighted(output_image[:, :, 1], 1.0, core_glow, 0.9, 0)
    output_image[:, :, 1] = cv2.addWeighted(output_image[:, :, 1], 1.0, outer_glow, 0.3, 0)

    output_image[:, :, 2] = cv2.addWeighted(output_image[:, :, 2], 1.0, core_glow, 1.2, 0)
    output_image[:, :, 2] = cv2.addWeighted(output_image[:, :, 2], 1.0, outer_glow, 0.5, 0)
    return output_image