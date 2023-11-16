import os
import numpy as np
import cv2

def save_images(path):
    if False:
        return 10
    for filename in os.listdir(path):
        if filename.endswith('.bmp'):
            name = filename[:-4]
            i = cv2.imread(path + filename)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            i = cv2.resize(i, (28, 28))
            cv2.imwrite('../data/finish-line/pngs/' + name + '.png', i)
save_images('../data/finish-line/bmps/train/')