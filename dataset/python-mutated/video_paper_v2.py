import numpy as np
import cv2

def find_marker(image):
    if False:
        for i in range(10):
            print('nop')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    if False:
        return 10
    return knownWidth * focalLength / perWidth

def mainf(type):
    if False:
        while True:
            i = 10
    KNOWN_DISTANCE = 24.0
    KNOWN_WIDTH = 11.69
    KNOWN_HEIGHT = 8.27
    IMAGE_PATHS = ['1.jpg', '2.jpg', '3.jpg']
    image = cv2.imread(IMAGE_PATHS[0])
    marker = find_marker(image)
    focalLength = marker[1][0] * KNOWN_DISTANCE / KNOWN_WIDTH
    print('focalLength = ', focalLength)
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        (grabbed, frame) = camera.read()
        marker = find_marker(frame)
        if marker == 0:
            print(marker)
            continue
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
        box = cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
        cv2.putText(frame, '%.2fcm' % (inches * 30.48 / 12), (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        cv2.imshow('capture', frame)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
        camera.release()
        cv2.destroyAllWindows()
mainf(1)