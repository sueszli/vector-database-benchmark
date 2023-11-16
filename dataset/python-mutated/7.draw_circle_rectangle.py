import cv2
import numpy as np
drawing = False
(ix, iy) = (-1, -1)

def draw_circle(event, x, y, flags, param):
    if False:
        i = 10
        return i + 15
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        (ix, iy) = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing is True:
            if mode is True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
img = np.zeros((512, 512, 3), np.uint8)
mode = False
cv2.namedWindow('image', 0)
cv2.setMouseCallback('image', draw_circle)
while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1)
    if k == ord('m'):
        mode = not mode
    elif k == ord('q'):
        break
cv2.destroyAllWindows()