"""
http://www.technicdynamic.com/2017/08/28/python-motion-detection-with-opencv-simple/

"""
import cv2
from datetime import datetime

def diffImg(t0, t1, t2):
    if False:
        print('Hello World!')
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)
threshold = 81500
cam = cv2.VideoCapture(0)
winName = 'Movement Indicator'
cv2.namedWindow(winName)
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
timeCheck = datetime.now().strftime('%Ss')
while True:
    cv2.imshow(winName, cam.read()[1])
    if cv2.countNonZero(diffImg(t_minus, t, t_plus)) > threshold and timeCheck != datetime.now().strftime('%Ss'):
        dimg = cam.read()[1]
    timeCheck = datetime.now().strftime('%Ss')
    t_minus = t
    t = t_plus
    t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    key = cv2.waitKey(10)
    if key == ord('q'):
        cv2.destroyWindow(winName)
        break