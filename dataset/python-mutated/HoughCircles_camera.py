"""
HoughCircles_camera.py:

用围棋-棋子来测试
"""
import cv2
import numpy as np
from skimage.measure import compare_mse as mse
import string, random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    if False:
        while True:
            i = 10
    return ''.join((random.choice(chars) for _ in range(size)))
cap = cv2.VideoCapture(0)
margin = 30

def draw_line_rectangle(frame, margin):
    if False:
        print('Hello World!')
    (rows, cols, ch) = frame.shape
    half = int(cols / 2)
    cv2.line(frame, (half, 0), (half, rows), (0, 0, 255), 2)
    up_left1 = (margin, margin)
    down_right1 = (cols - margin, rows - margin)
    cv2.rectangle(frame, up_left1, down_right1, (0, 255, 0), 3)
(ret, temp) = cap.read()
tm = 0
while cap.isOpened():
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(id_generator() + '.jpg', frame2)
    (ret, frame) = cap.read()
    m = mse(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    print('mse', m, '----\n')
    if abs(m - tm) < 2:
        continue
    else:
        temp = frame.copy()
        tm = m
    frame2 = frame[margin:frame.shape[0] - margin, margin:frame.shape[1] - margin]
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=30, minRadius=10, maxRadius=40)
    print(circles)
    cimg = frame2
    if circles is not None:
        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    draw_line_rectangle(frame, margin)
    cv2.imshow('houghlines', frame)
cap.release()
cv2.destroyAllWindows()