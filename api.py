import os
from ultralytics import YOLO
from flask import Flask, request, jsonify, send_file, Response
import cv2
import numpy as np

app = Flask(__name__)

# Inisialisasi model YOLO dan model path
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please check the path."
    )

model = YOLO(model_path)  # load a custom model

# Ambang batas deteksi
threshold = 0.5


@app.route("/detect", methods=["POST"])
def detect_objects():
    try:
        # Terima gambar dari permintaan POST
        image_file = request.files["image"]

        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Deteksi objek pada gambar
        results = model(image)[0]

        # Proses hasil deteksi
        detected_objects = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                detected_objects.append(
                    {
                        "class": results.names[int(class_id)].upper(),
                        "confidence": score,
                        "bounding_box": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                        },
                    }
                )

        return jsonify({"detected_objects": detected_objects}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Gunakan kamera laptop (0 mengacu pada kamera bawaan)
camera_source = 0


def generate_frames():
    cap = cv2.VideoCapture(camera_source)  # Gunakan kamera laptop

    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score > threshold:
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4
                    )
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

            # Mengubah frame menjadi bentuk yang bisa ditampilkan di web
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        except GeneratorExit:
            # Generator dihentikan, bebaskan sumber daya
            cap.release()
            cv2.destroyAllWindows()
            break


@app.route("/live_detection", methods=["GET"])
def live_detection():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
