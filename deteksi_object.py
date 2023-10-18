import cv2
import numpy as np

# Membaca citra input
A = cv2.imread("testing/nanasmatang1.jpeg")
cv2.imshow("Input Image", A)
cv2.waitKey(0)

# Mengubah citra RGB menjadi Grayscale
B = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", B)
cv2.waitKey(0)

# Melakukan operasi deteksi tepi menggunakan operator Roberts
Ix = cv2.filter2D(B, cv2.CV_64F, np.array([[0, 1], [-1, 0]]))
Iy = cv2.filter2D(B, cv2.CV_64F, np.array([[1, 0], [0, -1]]))
J = np.sqrt(Ix**2 + Iy**2)
cv2.imshow("Edge Detection", J.astype(np.uint8))
cv2.waitKey(0)

# Melakukan operasi biner menggunakan threshold
_, D = cv2.threshold(J, 20, 255, cv2.THRESH_BINARY)
cv2.imshow("Binarized Image", D)
cv2.waitKey(0)

# Operasi morphological close untuk mengisi lubang
kernel = np.ones((30, 30), np.uint8)
E = cv2.morphologyEx(D, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Hole Filling", E)
cv2.waitKey(0)

# Operasi area opening
F = cv2.morphologyEx(
    E, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
)
cv2.imshow("Area Opening", F)
cv2.waitKey(0)

# Menghitung ciri bentuk
contours, _ = cv2.findContours(F, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
metric = []
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / perimeter**2
    metric.append(circularity)

# Memberi label objek berdasarkan metrik bentuk
for i, contour in enumerate(contours):
    if metric[i] > 1:
        label = "Tomat"
    elif 1 >= metric[i] > 0.9:
        label = "Mangga"
    elif 0.9 >= metric[i] > 0.8:
        label = "Jambu"
    else:
        label = "Pisang"

    x, y, w, h = cv2.boundingRect(contour)
    A = cv2.rectangle(A, (x, y), (x + w, y + h), (255, 0, 0), 3)
    A = cv2.putText(A, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imshow("Objects with Labels", A)
cv2.waitKey(0)
cv2.destroyAllWindows()
