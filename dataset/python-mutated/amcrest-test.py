from amcrest import AmcrestCamera
import time
import datetime
import os
from PIL import Image
import io
from matplotlib import pyplot as plt
import cv2
import numpy as np
ip = '10.21.5.26'
port = 80
username = 'admin'
password = 'test_cam'
camera = AmcrestCamera(ip, port, username, password).camera
img_no = 1
ts = time.time()
time_string = '%d-%m-%Y_%Hh%Mm'
st = datetime.datetime.fromtimestamp(ts).strftime(time_string)
image_dir = './session_images/' + st
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
        print('[INFO] Directory created.')
    except:
        print('[ERROR] Directory could not be created.')
else:
    print('[INFO] Directory already present.')
sm = 0.1
md = 0.5
lg = 1

def moveLeft(sleep_time):
    if False:
        while True:
            i = 10
    camera.move_left(action='start', channel=0, vertical_speed=1)
    time.sleep(sleep_time)
    camera.move_left(action='stop', channel=0, vertical_speed=1)

def moveRight(sleep_time):
    if False:
        i = 10
        return i + 15
    camera.move_right(action='start', channel=0, vertical_speed=1)
    time.sleep(sleep_time)
    camera.move_right(action='stop', channel=0, vertical_speed=1)

def moveUp(sleep_time):
    if False:
        print('Hello World!')
    camera.move_up(action='start', channel=0, vertical_speed=1)
    time.sleep(sleep_time)
    camera.move_up(action='stop', channel=0, vertical_speed=1)

def moveDown(sleep_time):
    if False:
        i = 10
        return i + 15
    camera.move_down(action='start', channel=0, vertical_speed=1)
    time.sleep(sleep_time)
    camera.move_down(action='stop', channel=0, vertical_speed=1)

def takeSnap():
    if False:
        return 10
    timeLag(1.5)
    for i in range(1):
        global img_no
        global camera
        image_path = image_dir + '/image' + str(img_no) + '.png'
        img = camera.snapshot(0, image_path)
        print(img.status)
        if img.status != 200:
            camera = AmcrestCamera(ip, port, username, password).camera
            print('image data not found')
            print('trying again')
            return -1
        img_no += 1

def reset_x():
    if False:
        print('Hello World!')
    zoomOut(2)
    moveDown(2)

def zoomIn(sleep_time):
    if False:
        return 10
    camera.zoom_in(action='start', channel=0)
    time.sleep(sleep_time)
    camera.zoom_in(action='stop', channel=0)

def zoomOut(sleep_time):
    if False:
        i = 10
        return i + 15
    camera.zoom_out(action='start', channel=0)
    time.sleep(sleep_time)
    camera.zoom_out(action='stop', channel=0)

def timeLag(sleep_time):
    if False:
        return 10
    time.sleep(sleep_time)

def captureSession():
    if False:
        return 10
    reverse = 0
    takeSnap()
    for i in range(3):
        moveRight(lg)
        reverse += lg
        timeLag(0.2)
        takeSnap()
    moveLeft(reverse)

def snap():
    if False:
        return 10
    image = camera.snapshot(0).data
    return Image.open(io.BytesIO(image))

def tour():
    if False:
        return 10
    moveLeft(10)
    image = snap()
    img = plt.imshow(image)
    print('Completed moving left')
    for i in range(10):
        moveRight(md)
        timeLag(0.3)
        image = snap()
        print('snap')
        img.set_data(image)

def start():
    if False:
        return 10
    print('reset')
    moveRight(6)

def end():
    if False:
        for i in range(10):
            print('nop')
    print('end')
    moveLeft(5)
    moveDown(2)
    zoomOut(2)
start()
for x in range(9):
    takeSnap()
    zoomIn(0.3)
    up = 0
    while up < 2:
        if takeSnap() == -1:
            reset_x()
            up = 0
            continue
        print('[{}] {} snap'.format(x, up))
        moveUp(0.2 / (up + 1))
        zoomIn(0.2 + up / 10.0)
        up += 1
    takeSnap()
    print('snap!')
    zoomOut(2)
    moveDown(2)
    moveRight(0.5)
end()
'\n\nlisp=[]\nmoveLeft(20)\nmoveRight(3)\nprint("move left complete")\nfor x in xrange(20):\n\t#y=cv2.resize(np.array(snap()),(0,0),fx=0.5,fy=0.5)\n\t#cv2.imshow(\'disp\',y)\n\t#print(\'snap\')\n\t#cv2.waitKey(1)\n\t#moveRight(1)\n#cv2.destroyAllWindows()\n'