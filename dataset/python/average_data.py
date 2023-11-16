import os
import numpy as np
import cv2


def load_images(path):
    pos = []
    neg = []
    for filename in os.listdir(path):
        if filename.endswith(".bmp"):
            i = cv2.imread(path+filename)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            i = cv2.resize(i, (28, 28))
            if "pos" in filename:
                pos.append(i)
            else:
                neg.append(i)
    pos_stack = np.stack(pos)
    neg_stack = np.stack(neg)

    return (pos_stack, neg_stack)

(pos_stack, neg_stack) = load_images("../data/finish-line/bmps/train/")
pos = np.mean(pos_stack, axis=0)
neg = np.mean(neg_stack, axis=0)
pos = cv2.resize(pos, (200,200))
neg = cv2.resize(neg, (200,200))
cv2.imwrite("pos.png", pos)
cv2.imwrite("neg.png", neg)
