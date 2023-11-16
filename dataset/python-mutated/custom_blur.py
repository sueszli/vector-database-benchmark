from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

def convolve2d(X, W):
    if False:
        i = 10
        return i + 15
    (n1, n2) = X.shape
    (m1, m2) = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i + m1, j:j + m2] += X[i, j] * W
    ret = Y[m1 // 2:-m1 // 2 + 1, m2 // 2:-m2 // 2 + 1]
    assert ret.shape == X.shape
    return ret
img = mpimg.imread('lena.png')
plt.imshow(img)
plt.show()
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()
W = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        dist = (i - 9.5) ** 2 + (j - 9.5) ** 2
        W[i, j] = np.exp(-dist / 50.0)
plt.imshow(W, cmap='gray')
plt.show()
out = convolve2d(bw, W)
plt.imshow(out, cmap='gray')
plt.show()
print(out.shape)
out = np.zeros(img.shape)
W /= W.sum()
for i in range(3):
    out[:, :, i] = convolve2d(img[:, :, i], W)
plt.imshow(out)
plt.show()