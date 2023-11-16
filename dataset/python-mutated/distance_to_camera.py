from imutils import paths
import numpy as np
import imutils
import cv2

def find_marker(image):
    if False:
        while True:
            i = 10
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    if False:
        i = 10
        return i + 15
    return knownWidth * focalLength / perWidth
KNOWN_DISTANCE = 12.0
KNOWN_WIDTH = 11.0
image = cv2.imread('paper.jpg')
marker = find_marker(image)
focalLength = marker[1][0] * KNOWN_DISTANCE / KNOWN_WIDTH
print('Focal Length:', focalLength)
image = cv2.imread('paper.jpg')
marker = find_marker(image)
inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
box = np.int0(box)
print(box)
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
cv2.circle(image, tuple(box[0]), 5, (255, 0, 0), thickness=2, lineType=8, shift=0)
cv2.circle(image, tuple(box[3]), 5, (0, 0, 255), thickness=2, lineType=8, shift=0)
cv2.putText(image, '%.2fft' % (inches / 12), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
cv2.imshow('image', image)
cv2.waitKey(0)