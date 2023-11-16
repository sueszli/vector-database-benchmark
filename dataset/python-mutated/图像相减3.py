"""
图像相减3.py:

3张图片

"""
import cv2

def diff(img, img1):
    if False:
        i = 10
        return i + 15
    return cv2.absdiff(img, img1)

def diff_remove_bg(img0, img, img1):
    if False:
        print('Hello World!')
    d1 = diff(img0, img)
    d2 = diff(img, img1)
    return cv2.bitwise_and(d1, d2)
img1 = cv2.imread('subtract1.jpg', 0)
img2 = cv2.imread('subtract2.jpg', 0)
cv2.imshow('subtract1', img1)
cv2.imshow('subtract2', img2)
st = diff_remove_bg(img2, img1, img2)
cv2.imshow('after subtract', st)
cv2.waitKey(0)