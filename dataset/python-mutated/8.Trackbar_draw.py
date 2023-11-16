import cv2
import numpy as np

def nothing(x):
    if False:
        while True:
            i = 10
    pass
drawing = False
(ix, iy) = (-1, -1)
'\ncv2.getTrackbarPos() 函数的第一个参数是滑动条的名字 \n第二个参数 是滑动条被放置窗口的名字 \n第三个参数是滑动条的默认位置。\n第四个参数是滑动条的最大值 \n第五个函数是回调函数， 每次滑动条的滑动都会调用回调函 数。\n回调函数通常都会含有一个默认参数 就是滑动条的位置\n'

def draw_circle(event, x, y, flags, param):
    if False:
        while True:
            i = 10
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    color = (b, g, r)
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        (ix, iy) = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing is True:
            if mode is True:
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
                cv2.circle(img, (x, y), 3, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
img = np.zeros((512, 512, 3), np.uint8)
mode = False
cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.setMouseCallback('image', draw_circle)
while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1)
    if k == ord('m'):
        mode = not mode
    elif k == ord('q'):
        break