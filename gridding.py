import cv2
import numpy as np

img = "gridding practice.png"
img = cv2.imread(img)
lap = cv2.Laplacian(img, cv2.IPL_DEPTH_32F, ksize=3)
imgray = cv2.cvtColor(lap, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
size = img.shape
m = np.zeros(size, dtype=np.uint8)
for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) >= 1:
        color = (255, 255, 255)
        cv2.drawContours(m, cnt, -1, color, -1)
cv2.imwrite(str(img) + "contours.jpg", m)
