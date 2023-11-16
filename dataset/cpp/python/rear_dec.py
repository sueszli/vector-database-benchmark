import cv2
import numpy as np

camera = cv2.VideoCapture ("video.avi")
camera.open("video.avi")
# camera = cv2.VideoCapture ("car.avi")
# camera.open("car.avi")

car_cascade = cv2.CascadeClassifier('cars.xml')
#car_cascade = cv2.CascadeClassifier('car_rear.xml')
while True:
    (grabbed,frame) = camera.read()

    grayvideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(grayvideo, 1.1, 1)
    # print(cars)
    # print(type(cars))
    # print(cars.shape)
    # 部分输出如下所示：
    # [[255  62  37  37]
    #  [144  25  35  35]
    #  [219  81  62  62]
    #  [246  52  54  54]]
    # < class 'numpy.ndarray'>
    # (4, 4)
    # ...

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("video",frame)

    if cv2.waitKey(1)== ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
