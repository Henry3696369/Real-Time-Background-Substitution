import cv2
import onnxruntime as ort
import mediapipe as mp
import numpy as np
import RoughThreshold
import MixedThreshold
import GlowEffectRegion
import time
import MixedThresholdWithDepth
import DepthMap

MODEL_PATH = "selfie_segmenter.tflite"


FACE_LANDMARKER_PATH = "face_landmarker.task"

# Initialize Mediapipe
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

segmenter_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_category_mask=False,
    output_confidence_masks=True
)



FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


providers = [('DmlExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']

try:
    session = ort.InferenceSession("midas_small.onnx", providers=providers)
    print("Gpu accelerating")
except Exception as e:
    print(f"Fail: {e}")
print(session.get_providers())

input_name = session.get_inputs()[0].name



# Open camera
# Open camera on Windows
cap = cv2.VideoCapture(0)
bg_image = cv2.imread('van.jpg') 

# Generate a blue background, if no bg_image;
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"The resolution of the camera is: {w}x{h}")
background = cv2.resize(bg_image, (w, h))
b_f = background.astype(np.float32)
b_f_bak = b_f



print("Opening Camera now...")
# Fps Variables
prev_frame_time = 0
new_frame_time = 0
fps_coeff = 0.05
smooth_fps = 0

# Mode Control Variables
eye_mode = 0
current_mode = 1
disappear_count = 0
art_mode = 0

# Mask Variables
final_mask = None
prev_gray = None
last_stable_mask = None
map_y, map_x = np.indices((h, w), dtype=np.float32) # For optical flow mask


with ImageSegmenter.create_from_options(segmenter_options) as segmenter, \
     FaceLandmarker.create_from_options(landmarker_options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("There are some errors.")
            break
        # Get Fps
        new_frame_time = time.time()
        time_diff = new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time

        if time_diff > 0:
            instant_fps = 1 / time_diff
        
            smooth_fps = (1 - fps_coeff) * smooth_fps + fps_coeff * instant_fps
        fps_text = f"FPS: {int(smooth_fps)}"

        # Mirroring the image
        frame = cv2.flip(frame, 1)

        #h, w = frame.shape[:2] # h is the height of frame, w is the width
        
        # Convert BGR to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # time labeling
        timestamp_ms = int(time.time() * 1000)

        # AI segmentation
        
        segmentation_result = segmenter.segment_for_video(mp_image, timestamp_ms)
        
        # get confindece map! (key point)
        confidence_mask = segmentation_result.confidence_masks[0].numpy_view().squeeze()    



        if current_mode == 0:
        # rough mask choose the piexels, the value of which are > 0.15
            final_mask = RoughThreshold.rough_mask(confidence_mask, 0.15)
            output_image = np.where(final_mask, frame, background)


        elif current_mode == 2:
            final_mask = MixedThreshold.mixed_mask(confidence_mask, 0.15, 0.9)
            final_mask = np.stack((final_mask,) * 3, axis=-1)
            f_f = frame.astype(np.float32)
            output_image = b_f + final_mask * (f_f - b_f)
            output_image = np.clip(output_image, 0, 255).astype(np.uint8)


        elif current_mode == 3:
            depth_norm = DepthMap.depth(frame, session, input_name)
            final_mask = MixedThresholdWithDepth.mixed_mask(depth_norm, confidence_mask, 0.15, 0.9)

            final_mask = np.stack((final_mask,) * 3, axis=-1)
            f_f = frame.astype(np.float32)
            output_image = b_f + final_mask * (f_f - b_f)
            output_image = np.clip(output_image, 0, 255).astype(np.uint8)
            # output_image = final_mask
            

        elif current_mode == 4:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_norm = DepthMap.depth(frame, session, input_name)
            # current_mixed_mask = MixedThreshold.mixed_mask(confidence_mask, 0.2, 0.8)
            current_mixed_mask = MixedThresholdWithDepth.mixed_mask(depth_norm, confidence_mask, 0.15, 0.9)
            if prev_gray is not None and last_stable_mask is not None:

                # Pry down twice
                small_curr = cv2.pyrDown(cv2.pyrDown(curr_gray))
                small_prev = cv2.pyrDown(cv2.pyrDown(prev_gray))
                
                # Farneback-Algrithom for flow
                flow_small = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None, 0.5, 3, 11, 3, 5, 1.2, 0)
                flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) * 4.0
 
                # Create the predicated mask based on the previous mask and the flow.
                remap_x = map_x + flow[..., 0]
                remap_y = map_y + flow[..., 1]
                predicated_mask = cv2.remap(last_stable_mask, remap_x, remap_y, cv2.INTER_LINEAR)
                
                if art_mode == 0:
                    alpha = 0.95
                    integrated_mask = cv2.addWeighted(current_mixed_mask, alpha, predicated_mask, 1 - alpha, 0)
                    final_mask = np.stack((integrated_mask,) * 3, axis=-1)
                elif art_mode == 1:
                    alpha = 0.3
                    integrated_mask = cv2.addWeighted(current_mixed_mask, alpha, predicated_mask, 1 - alpha, 0)
                    final_mask = np.stack((integrated_mask,) * 3, axis=-1)
                else:
                    
                    alpha_r, alpha_g, alpha_b = 0.05, 0.3, 0.8
                    mask_r = current_mixed_mask * alpha_r + predicated_mask * (1 - alpha_r)
                    mask_r = np.power(mask_r, 1.5)
                    mask_g = current_mixed_mask * alpha_g + predicated_mask * (1 - alpha_g)
                    mask_b = current_mixed_mask * alpha_b + predicated_mask * (1 - alpha_b)
                    mask_b = np.power(mask_b, 0.8)
                    final_mask = cv2.merge([mask_b, mask_g, mask_r])
                    final_mask = np.clip(final_mask * 1.2, 0, 1.0)
                    integrated_mask = mask_g
                    bg_remap_x = (map_x - flow[..., 0] * 1.8).astype(np.float32)
                    bg_remap_y = (map_y - flow[..., 1] * 1.8).astype(np.float32)
                    distorted_bg = cv2.remap(b_f, bg_remap_x, bg_remap_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
                    b_f = 0.5 * b_f + 0.5 * distorted_bg
            else:
                # For the first frame
                integrated_mask = current_mixed_mask
                final_mask = np.stack((integrated_mask,) * 3, axis=-1)

            prev_gray = curr_gray
            last_stable_mask = integrated_mask
            f_f = frame.astype(np.float32)
            output_image = b_f + final_mask * (f_f - b_f)
            output_image = np.clip(output_image, 0, 255).astype(np.uint8)



        elif current_mode == 1:
            output_image = frame.copy()

        elif current_mode == 11:
            output_image = (confidence_mask * 255).astype(np.uint8)

        elif current_mode == 12:
            depth_norm = DepthMap.depth(frame, session, input_name)
            output_image = depth_norm
        
        

        if eye_mode == 1:
            landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if landmarker_result.face_landmarks:
                disappear_count = 0
                landmarks = landmarker_result.face_landmarks[0]
                eyes_mask = np.zeros((h, w), dtype=np.uint8)
                laser_mask = np.zeros((h, w), dtype=np.uint8)
                # Locate the eyes' position
                for eye_indices in [(LEFT_EYE), (RIGHT_EYE)]:
                    eye_points = []
                    for idx in eye_indices:
                        pt = landmarks[idx]
                        px, py = int(pt.x * w), int(pt.y * h)
                        eye_points.append([px, py])
                    
                    customer_eye = np.array(eye_points, dtype=np.int32)
                    
                    cv2.fillPoly(eyes_mask, [customer_eye], 255)

                output_image = GlowEffectRegion.glow(eyes_mask, output_image)
            else :
                disappear_count +=1
            if disappear_count >30:
                output_image = cv2.GaussianBlur(output_image, (91, 91), 0) 
        cv2.putText(output_image, fps_text, (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Heng Computer Vision Project - UD CISC442', output_image)

        key = cv2.waitKey(5) & 0xFF
        if key == 27: # Exit by ESC
            break
        elif key == ord('0'): 
            current_mode = 0
            print("Switched to Rough Segmentation")
        elif key == ord('2'): 
            current_mode = 2
            print("Switched to Mixed Segmentation")

        elif key == ord('3'): 
            current_mode = 3
            print("Switched to Mixed Segmentation with depth map")

        elif key == ord('4'): 
            current_mode = 4
            print("Switched to Mixed Segmentation with optical flow")

        elif key == ord('8'): 
            art_mode = (art_mode + 1) % 3
            print("Art Mode Switch")
        elif key == ord('r'): 
            b_f = b_f_bak
            print("Reset Background image")
        elif key == ord('9'): 
            eye_mode = 1 - eye_mode
            if eye_mode == 0:
                print("Eye mode on")
            else:
                print("Eye mode off")       
        elif key == ord('1'): 
            current_mode = 1
            print("Switched to Origin Camera")
        elif key == ord('-'): 
            current_mode = 11
            print("Switched to Depth map")
        elif key == ord('='): 
            current_mode = 12
            print("Switched to Depth map")

cap.release()
cv2.destroyAllWindows()

