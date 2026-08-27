import numpy as np
import cv2
            


def depth(frame, session, input_name):
    img_small = cv2.resize(frame, (256, 256))
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.transpose(2, 0, 1).astype(np.float32)
    img_input = img_input[np.newaxis, :] / 255.0
    outputs = session.run(None, {input_name: img_input})
    prediction = outputs[0].squeeze()
    p_min, p_max = prediction.min(), prediction.max()
    depth_norm_small = (prediction - p_min) / (p_max - p_min + 1e-6)
    depth_norm = cv2.resize(depth_norm_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth_norm