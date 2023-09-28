import os
from ultralytics import YOLO
import cv2

# Inisialisasi model YOLO dan model path
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "last.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please check the path."
    )

model = YOLO(model_path)  # load a custom model

# Baca gambar
image_path = "dataset/images/train/DSC_2011.jpg"  # Ganti dengan path gambar Anda
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Image file '{image_path}' not found. Please check the path."
    )

# Deteksi objek pada gambar
results = model(image)[0]

# Ambang batas deteksi
threshold = 0.5
cle
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(
            image,
            results.names[int(class_id)].upper(),
            (int(x1), int(y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

# Tampilkan gambar dengan objek yang terdeteksi
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
