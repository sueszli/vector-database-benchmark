"""
画圆圈.py:随机覆盖，不同颜色，
"""
from time import sleep
import cv2
import numpy as np

def click_event(event, x, y, flags, param):
    if False:
        print('Hello World!')
    '\n    用左键点击屏幕，打印坐标\n    :param event:\n    :param x:\n    :param y:\n    :param flags:\n    :param param:\n    :return:\n    '
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y, flags, param)
cv2.namedWindow('Canvas', cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback('Canvas', click_event)
canvas = np.zeros((300, 300, 3), dtype='uint8')
while True:
    try:
        for i in range(0, 25):
            radius = np.random.randint(5, high=200)
            color = np.random.randint(0, high=256, size=(3,)).tolist()
            pt = np.random.randint(0, high=300, size=(2,))
            cv2.circle(canvas, tuple(pt), radius, color, -1)
        cv2.imshow('Canvas', canvas)
        key = cv2.waitKey(1000)
        if key == ord('q'):
            break
        else:
            continue
    except KeyboardInterrupt as e:
        print('KeyboardInterrupt', e)
    finally:
        cv2.imwrite('random-circles2.jpg', canvas)