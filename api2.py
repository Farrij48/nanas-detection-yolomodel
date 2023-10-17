import os
from ultralytics import YOLO
from flask import Flask, request, jsonify, send_file, Response
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Inisialisasi model YOLO dan model path
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please check the path."
    )

model = YOLO(model_path)  # load a custom model

# Ambang batas deteksi
threshold_live = 0.6


@app.route("/live_detect", methods=["POST"])
def detect_objects():
    try:
        # Ambil gambar dari permintaan POST
        image_file = request.files["image"]

        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        # Baca gambar menggunakan OpenCV
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Deteksi objek pada gambar
        results = model(image)[0]

        # Proses hasil deteksi
        detected_objects = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold_live:
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
