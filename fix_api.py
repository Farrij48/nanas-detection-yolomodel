import os
from ultralytics import YOLO
from flask import Flask, request, jsonify, send_file, Response
import cv2
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from collections import Counter

app = Flask(__name__)
CORS(app)

file = "features2.xlsx"
dataset = pd.read_excel(file)
glcm_properties = [
    "dissimilarity",
    "correlation",
    "homogeneity",
    "contrast",
    "ASM",
    "energy",
]

fitur = dataset.iloc[:, +1:-1].values
kelas = dataset.iloc[:, 30].values

scaler = StandardScaler()
scaler.fit(fitur)
fitur = scaler.transform(fitur)

classifier = KNeighborsClassifier(n_neighbors=13)
classifier.fit(fitur, kelas)

# Inisialisasi model YOLO dan model path
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Please check the path."
    )

model = YOLO(model_path)  # load a custom model

threshold_live = 0.5


def process_image(src):
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(mask.copy(), None, iterations=10)
    mask = cv2.erode(mask.copy(), None, iterations=10)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, mask]
    dst = cv2.merge(rgba, 4)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selected = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(selected)
    cropped = dst[y : y + h, x : x + w]
    mask = mask[y : y + h, x : x + w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
    clt = KMeans(n_clusters=3, n_init=10)
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    features = []
    features.append(dom_color[0])
    features.append(dom_color[1])
    features.append(dom_color[2])

    glcm = graycomatrix(
        gray,
        distances=[5],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    glcm_props = [
        propery for name in glcm_properties for propery in graycoprops(glcm, name)[0]
    ]
    for item in glcm_props:
        features.append(item)

    label_img = label(mask)
    props = regionprops(label_img)
    eccentricity = getattr(props[0], "eccentricity")
    area = getattr(props[0], "area")
    perimeter = getattr(props[0], "perimeter")
    metric = (4 * np.pi * area) / (perimeter * perimeter)
    features.append(metric)
    features.append(eccentricity)

    return features


@app.route("/detect_and_predict", methods=["POST"])
def detect_and_predict():
    try:
        image_file = request.files["image"]

        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Deteksi objek pada gambar menggunakan YOLO
        results = model(image)[0]

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

        # Jika tidak terdeteksi NANAS_MATANG ATAU NANAS_MENTAH, kirim pesan nanas tidak terdeteksi
        if not any(
            obj["class"] == "NANAS_MATANG" or obj["class"] == "NANAS_MENTAH"
            for obj in detected_objects
        ):
            return jsonify({"prediction": "tidak_diketahui"}), 200

        # Jika terdapat objek nanas, lakukan prediksi fitur
        features = process_image(image)
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        prediction = classifier.predict(features)
        return jsonify({"prediction": prediction[0]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
