# -*-coding:utf8-*-#
__author__ = 'play4fun'
"""
create time:15-10-24 下午5:42

函数 np.fft.fft2() 可以对信号   率  换  出结果是一个复杂的数组。
第一个参数是 入图像  求是灰 度格式。
第二个参数是可 的, 决定 出数组的大小。
 输出数组的大小和输入图像大小一样。如果输出结果比输入图像大 
  输入图像就需要在进行 FFT 前补0。如果输出结果比输入图像小的话   输入图像就会被切割。
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/messi5.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# 这里构建振幅图的公式没学过
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
