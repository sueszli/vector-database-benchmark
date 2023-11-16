import numpy as np
import cv2

def find_marker(image):
    if False:
        while True:
            i = 10
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    if False:
        i = 10
        return i + 15
    return knownWidth * focalLength / perWidth

def fmain(type):
    if False:
        i = 10
        return i + 15
    if type == 6:
        KNOWN_DISTANCE = 48.0
        KNOWN_WIDTH = 200
        KNOWN_HEIGHT = 16.27
        IMAGE_PATHS = ['1.jpg', '2.jpg', '3.jpg']
        image = cv2.imread(IMAGE_PATHS[0])
        marker = find_marker(image)
        focalLength = marker[1][0] * KNOWN_DISTANCE / KNOWN_WIDTH
        print('focalLength = ', focalLength)
        camera = cv2.VideoCapture(0)
        for imagePath in IMAGE_PATHS:
            image = cv2.imread(imagePath)
            marker = find_marker(image)
            inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
            box = cv2.boxPoints(marker)
            box = np.int0(box)
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
            cv2.putText(image, '%.2fcm' % (inches * 30.48 / 12), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            cv2.imshow('image', image)
            cv2.waitKey(0)
fmain(6)