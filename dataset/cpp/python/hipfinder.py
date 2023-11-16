import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from scipy import signal

image_path = "../data/finish-line/bmps/220.bmp"
image = io.imread(image_path, as_gray=True)

filter1 = np.array([[1,0,-1],[2, 0, -2],[1,0,-1]])
filter2 = np.array([[1,2,1],[0, 0, 0],[-1,-2,-1]])
filter3 = np.array([[-2,-1,0],[-1, 0, 1],[0,1,2]])

res1 = signal.convolve2d(image, filter1)
res2 = signal.convolve2d(image, filter2)
res3 = signal.convolve2d(image, filter3)

output = np.zeros(res1.shape)
for x in range(output.shape[0]):
    for y in range(output.shape[1]):
        output[x,y] = max(res1[x,y], res2[x,y], res3[x,y])



plt.imshow(output, cmap="gray")
plt.show()



