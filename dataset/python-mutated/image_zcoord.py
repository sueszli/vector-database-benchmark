"""
==================================
Modifying the coordinate formatter
==================================

Modify the coordinate formatter to report the image "z" value of the nearest
pixel given x and y.  This functionality is built in by default; this example
just showcases how to customize the `~.axes.Axes.format_coord` function.
"""
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(19680801)
X = 10 * np.random.rand(5, 3)
(fig, ax) = plt.subplots()
ax.imshow(X)

def format_coord(x, y):
    if False:
        for i in range(10):
            print('nop')
    col = round(x)
    row = round(y)
    (nrows, ncols) = X.shape
    if 0 <= col < ncols and 0 <= row < nrows:
        z = X[row, col]
        return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}'
    else:
        return f'x={x:1.4f}, y={y:1.4f}'
ax.format_coord = format_coord
plt.show()