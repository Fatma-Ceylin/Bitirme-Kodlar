# STEP 1: Import the necessary modules.
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)
    return annotated_image

# STEP 5: Process the detection result (LIVE_STREAM callback).
annotated_image = None
def step5_callback(detection_result, output_image, timestamp_ms):
    global annotated_image
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), detection_result)

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=2,
    min_pose_detection_confidence=0.3,  # İkinci kişiyi daha kolay yakalaması için (varsayılan 0.5)
    min_pose_presence_confidence=0.3,   # Kişi ekrandayken kaybolmaması için (varsayılan 0.5)
    min_tracking_confidence=0.3,        # Kişiden kişiye atlamayı azaltmak için (varsayılan 0.5)
    output_segmentation_masks=False,
    result_callback=step5_callback)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_ts = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # STEP 4: Detect pose landmarks from the input image.
    ts = max(int(time.time() * 1000), last_ts + 1)
    last_ts = ts
    detector.detect_async(image, ts)

    gosterilecek = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR) if annotated_image is not None else frame
    cv2.imshow('Pose', gosterilecek)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()