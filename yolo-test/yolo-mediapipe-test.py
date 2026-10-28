import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from ultralytics import YOLO

# 1. İskelet çizim fonksiyonu (Kırpılan görüntü üzerine çizer)
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

# 2. MediaPipe PoseLandmarker kurulumu (IMAGE modunda ve tek kişi odaklı)
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,  # Her kırpılan kutuyu bağımsız işlemek için IMAGE modu
    num_poses=1,                            # Her kutuda zaten 1 kişi olacağı için 1 yapıyoruz
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# 3. YOLO modelini yükle ve webcam akışını başlat
model = YOLO("yolo26n.pt")
source = 0
results = model(source, stream=True)  # generator of Results objects

# 4. Akıştaki her bir kareyi (frame) sırayla işle
for result in results:
    # Orijinal kamera görüntüsünü al
    frame = result.orig_img.copy()

    # YOLO'nun bulduğu tüm kutuları (bounding box) dön
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Eğer tespit edilen nesne İNSAN ise (COCO veri setinde 0 = person)
        if cls_id == 0:
            # Kişiyi ana görüntüden kırp (crop)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Kırpılan görüntüyü RGB'ye çevir ve MediaPipe formatına hazırla
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_crop))

            # MediaPipe ile kırpılan görüntüde iskelet tespiti yap
            detection_result = detector.detect(mp_image)

            # İskelet noktalarını kırpılan görüntüye çiz ve ana görüntüdeki yerine geri yerleştir
            annotated_crop = draw_landmarks_on_image(rgb_crop, detection_result)
            frame[y1:y2, x1:x2] = cv2.cvtColor(annotated_crop, cv2.COLOR_RGB2BGR)

        # İsteğe bağlı: Tespit edilen nesnenin (insan veya başka nesne) kutusunu çiz
        label = model.names[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Sonucu ekranda göster
    cv2.imshow("YOLO + MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
detector.close()