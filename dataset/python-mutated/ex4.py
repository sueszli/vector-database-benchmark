from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../../large_files/train.csv')
data = df.values
np.random.shuffle(data)
X = data[:, 1:]
Y = data[:, 0]

def rotate1(im):
    if False:
        i = 10
        return i + 15
    return np.rot90(im, 3)

def rotate2(im):
    if False:
        return 10
    (H, W) = im.shape
    im2 = np.zeros((W, H))
    for i in range(H):
        for j in range(W):
            im2[j, H - i - 1] = im[i, j]
    return im2
for i in range(X.shape[0]):
    im = X[i].reshape(28, 28)
    im = rotate2(im)
    plt.imshow(im, cmap='gray')
    plt.title('Label: %s' % Y[i])
    plt.show()
    ans = input('Continue? [Y/n]: ')
    if ans and ans[0].lower() == 'n':
        break