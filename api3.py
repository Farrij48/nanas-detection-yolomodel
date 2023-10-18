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
import base64

app = Flask(__name__)

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


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_file = request.files["image"]

        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        features = process_image(image)
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        prediction = classifier.predict(features)
        return jsonify({"prediction": prediction[0]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
