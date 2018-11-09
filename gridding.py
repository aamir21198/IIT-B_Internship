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
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float
microarray = io.imread('./big-grid.jpg')

"""
Scale between zero and one
"""

microarray = img_as_float(microarray)
from skimage import color
red = microarray[..., 0]
green = microarray[..., 1]
red_rgb = np.zeros_like(microarray)
red_rgb[..., 0] = red
green_rgb = np.zeros_like(microarray)
green_rgb[..., 1] = green
from skimage import filters as filters
mask = (green > 0.1)
z = red.copy()
z /= green
z[~mask] = 0
print(z.min(), z.max())

"""
Locating the grid
"""

both = (green + red)
from skimage import feature

sum_down_columns = both.sum(axis=0)
sum_across_rows = both.sum(axis=1)

dips_columns = feature.peak_local_max(sum_down_columns.max() - sum_down_columns)
dips_columns = dips_columns.ravel()

M = len(dips_columns)
column_distance = np.mean(np.diff(dips_columns))

dips_rows = feature.peak_local_max(sum_across_rows.max() - sum_across_rows)
dips_rows = dips_rows.ravel()

N = len(dips_rows)
row_distance = np.mean(np.diff(dips_rows))

print('Columns are a mean distance of %.2f apart' % column_distance)
print('Rows are a mean distance of %.2f apart' % row_distance)


P, Q = 500, 500

plt.figure(figsize=(15, 10))
plt.imshow(microarray[:P, :Q])

for i in dips_rows[dips_rows < P]:
    plt.plot([0, Q], [i, i], 'm')

for j in dips_columns[dips_columns < Q]:
    plt.plot([j, j], [0, P], 'm')

plt.axis('image')

plt.show()
