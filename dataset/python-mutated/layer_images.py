"""
============
Layer Images
============

Layer images above one another using alpha blending
"""
import matplotlib.pyplot as plt
import numpy as np

def func3(x, y):
    if False:
        print('Hello World!')
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-(x ** 2 + y ** 2))
(dx, dy) = (0.05, 0.05)
x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
(X, Y) = np.meshgrid(x, y)
extent = (np.min(x), np.max(x), np.min(y), np.max(y))
fig = plt.figure(frameon=False)
Z1 = np.add.outer(range(8), range(8)) % 2
im1 = plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest', extent=extent)
Z2 = func3(X, Y)
im2 = plt.imshow(Z2, cmap=plt.cm.viridis, alpha=0.9, interpolation='bilinear', extent=extent)
plt.show()