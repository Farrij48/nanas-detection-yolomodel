import os
from ultralytics import YOLO
import cv2

# Inisialisasi model YOLO dan model path
model_path = os.path.join(".", "runs", "detect", "train2", "weights", "last.pt")
model = YOLO(model_path)  # load a custom model

# Gunakan kamera laptop (0 mengacu pada kamera bawaan)
camera_source = 0

# Ambang batas deteksi
threshold = 0.5

cap = cv2.VideoCapture(camera_source)  # Gunakan kamera laptop

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(
                frame,
                results.names[int(class_id)].upper(),
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

    # Tampilkan frame yang telah diperbarui
    cv2.imshow("Live Object Detection", frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bebaskan sumber daya
cap.release()
cv2.destroyAllWindows()
