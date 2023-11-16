"""
==========================================
Blend transparency with color in 2D images
==========================================

Blend transparency with color to highlight parts of data with imshow.

A common use for `matplotlib.pyplot.imshow` is to plot a 2D statistical
map. The function makes it easy to visualize a 2D matrix as an image and add
transparency to the output. For example, one can plot a statistic (such as a
t-statistic) and color the transparency of each pixel according to its p-value.
This example demonstrates how you can achieve this effect.

First we will generate some data, in this case, we'll create two 2D "blobs"
in a 2D grid. One blob will be positive, and the other negative.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def normal_pdf(x, mean, var):
    if False:
        while True:
            i = 10
    return np.exp(-(x - mean) ** 2 / (2 * var))
(xmin, xmax, ymin, ymax) = (0, 100, 0, 100)
n_bins = 100
xx = np.linspace(xmin, xmax, n_bins)
yy = np.linspace(ymin, ymax, n_bins)
means_high = [20, 50]
means_low = [50, 60]
var = [150, 200]
gauss_x_high = normal_pdf(xx, means_high[0], var[0])
gauss_y_high = normal_pdf(yy, means_high[1], var[0])
gauss_x_low = normal_pdf(xx, means_low[0], var[1])
gauss_y_low = normal_pdf(yy, means_low[1], var[1])
weights = np.outer(gauss_y_high, gauss_x_high) - np.outer(gauss_y_low, gauss_x_low)
greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)
vmax = np.abs(weights).max()
imshow_kwargs = {'vmax': vmax, 'vmin': -vmax, 'cmap': 'RdYlBu', 'extent': (xmin, xmax, ymin, ymax)}
(fig, ax) = plt.subplots()
ax.imshow(greys)
ax.imshow(weights, **imshow_kwargs)
ax.set_axis_off()
alphas = np.ones(weights.shape)
alphas[:, 30:] = np.linspace(1, 0, 70)
(fig, ax) = plt.subplots()
ax.imshow(greys)
ax.imshow(weights, alpha=alphas, **imshow_kwargs)
ax.set_axis_off()
alphas = Normalize(0, 0.3, clip=True)(np.abs(weights))
alphas = np.clip(alphas, 0.4, 1)
(fig, ax) = plt.subplots()
ax.imshow(greys)
ax.imshow(weights, alpha=alphas, **imshow_kwargs)
ax.contour(weights[::-1], levels=[-0.1, 0.1], colors='k', linestyles='-')
ax.set_axis_off()
plt.show()
ax.contour(weights[::-1], levels=[-0.0001, 0.0001], colors='k', linestyles='-')
ax.set_axis_off()
plt.show()