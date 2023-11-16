import cv2
import numpy as np

def acimage(image):
    if False:
        i = 10
        return i + 15
    image = np.array(image)
    w = image.shape[0]
    h = image.shape[1]
    ac_image = np.arange(w * h).reshape(w, h)
    for i in range(w):
        for ii in range(h):
            if ii == 0 and i == 0:
                ac_image[i][ii] = image[i][ii]
            elif i == 0 and ii != 0:
                ac_image[i][ii] = ac_image[i][ii - 1] + image[i][ii]
            elif i != 0 and ii == 0:
                ac_image[i][ii] = ac_image[i - 1][ii] + image[i][ii]
            else:
                ac_image[i][ii] = image[i][ii] + ac_image[i - 1][ii] + ac_image[i][ii - 1] - ac_image[i - 1][ii - 1]
    return ac_image

def ac_rec(image, x, y, w, h):
    if False:
        print('Hello World!')
    i = acimage(image)
    RectSum = i[x - 1][y - 1] + i[x + w - 1][y + h - 1] - i[x - 1][y + h - 1] - i[x + w - 1][y - 1]