import cv2
import numpy as np
import xlsxwriter
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, regionprops_table
from sklearn.cluster import KMeans

workbook = xlsxwriter.Workbook("features.xlsx")
worksheet = workbook.add_worksheet()

jenis = [
    "nanas_matang",
    "nanas_mentah",
]
jum_per_data = 25

hsv_properties = ["hue", "saturation", "value"]
glcm_properties = [
    "dissimilarity",
    "correlation",
    "homogeneity",
    "contrast",
    "ASM",
    "energy",
]
angles = ["0", "45", "90", "135"]
shape_properties = ["metric", "eccentricity"]

worksheet.write(0, 0, "File")
kolom = 1

# ---------------------------
# Writing excel header
for i in hsv_properties:
    worksheet.write(0, kolom, i)
    kolom += 1
for i in glcm_properties:
    for j in angles:
        worksheet.write(0, kolom, i + " " + j)
        kolom += 1
for i in shape_properties:
    worksheet.write(0, kolom, i)
    kolom += 1
worksheet.write(0, kolom, "Class")
kolom += 1
baris = 1

# Looping for each dataset
for i in jenis:
    for j in range(1, 26):
        kolom = 0
        file_name = "dataset/" + i + str(j) + ".bmp"
        print(file_name)
        worksheet.write(baris, kolom, file_name)
        kolom += 1
        # Preprocessing
        src = cv2.imread(file_name, 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask.copy(), None, iterations=10)
        mask = cv2.erode(mask.copy(), None, iterations=10)
        b, g, r = cv2.split(src)
        rgba = [b, g, r, mask]
        dst = cv2.merge(rgba, 4)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        selected = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(selected)
        cropped = dst[y : y + h, x : x + w]
        mask = mask[y : y + h, x : x + w]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # HSV
        hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
        clt = KMeans(n_clusters=3)
        labels = clt.fit_predict(image)
        label_counts = Counter(labels)
        dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        worksheet.write(baris, kolom, dom_color[0])
        kolom += 1
        worksheet.write(baris, kolom, dom_color[1])
        kolom += 1
        worksheet.write(baris, kolom, dom_color[2])
        kolom += 1
        dom_color_hsv = np.full(cropped.shape, dom_color, dtype="uint8")
        dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
        output_image = np.hstack((cropped, dom_color_bgr))

        cv2.imshow("Image Dominant Color", output_image)
        cv2.waitKey(0)
        # GLCM
        glcm = graycomatrix(
            gray,
            distances=[5],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        glcm_props = [
            propery
            for name in glcm_properties
            for propery in graycoprops(glcm, name)[0]
        ]
        for item in glcm_props:
            worksheet.write(baris, kolom, item)
            kolom += 1

        # Shape
        label_img = label(mask)
        props = regionprops(label_img)
        eccentricity = getattr(props[0], "eccentricity")
        area = getattr(props[0], "area")
        perimeter = getattr(props[0], "perimeter")
        metric = (4 * np.pi * area) / (perimeter * perimeter)
        worksheet.write(baris, kolom, metric)
        kolom += 1
        worksheet.write(baris, kolom, eccentricity)
        kolom += 1

        worksheet.write(baris, kolom, i)
        kolom += 1
        baris += 1
workbook.close()
# --------------------------
