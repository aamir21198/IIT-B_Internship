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