import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.misc import imresize

def disparity(im1, im2, max_disp, w):
    if False:
        i = 10
        return i + 15
    o = (w - 1) / 2
    L = np.pad(im1, o, 'constant')
    R = np.pad(im2, o, 'constant')
    min_ssd = np.ones(im1.shape) * np.Inf
    disp = np.zeros(im1.shape)
    for k in range(max_disp + 1):
        R[:, 1:] = R[:, 0:-1]
        R[:, :1] = 0
        ssd = np.ones(im1.shape) * np.Inf
        for y in np.arange(im1.shape[0]):
            for x in np.arange(k, im1.shape[1]):
                ssd[y, x] = np.sum((L[y:y + 2 * o + 1, x:x + 2 * o + 1] - R[y:y + 2 * o + 1, x:x + 2 * o + 1]) ** 2)
        disp[ssd < min_ssd] = k
        min_ssd = np.minimum(ssd, min_ssd)
    return disp
imgL = cv2.imread('data_road/image_2/uu_000001.png', 0)
imgR = cv2.imread('data_road_right/image_3/uu_000001.png', 0)
imgL = imresize(imgL, 0.3)
imgR = imresize(imgR, 0.3)
disp = disparity(imgL, imgR, 64, 11)
plt.figure()
plt.imshow(disp, cmap='Greys_r')
plt.show()