from amcrest import AmcrestCamera
import time
import datetime
import os
from PIL import Image
import io
from matplotlib import pyplot as plt
import cv2
import numpy as np

# camera configuarations
ip = '10.21.5.26'
port = 80
username = 'admin'
password = 'test_cam'
camera = AmcrestCamera(ip, port, username, password).camera
img_no = 1

# session details
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


# time configuarations
sm = 0.1
md = 0.5
lg = 1

# helper functions
def moveLeft(sleep_time):
	camera.move_left(action = "start", channel = 0, vertical_speed = 1)
	time.sleep(sleep_time)
	camera.move_left(action = "stop", channel = 0, vertical_speed = 1)


def moveRight(sleep_time):
	camera.move_right(action = "start", channel = 0, vertical_speed = 1)
	time.sleep(sleep_time)
	camera.move_right(action = "stop", channel = 0, vertical_speed = 1)

def moveUp(sleep_time):
	camera.move_up(action="start",channel=0, vertical_speed=1)
	time.sleep(sleep_time)
	camera.move_up(action="stop",channel=0,vertical_speed=1)

def moveDown(sleep_time):
        camera.move_down(action="start",channel=0, vertical_speed=1)
        time.sleep(sleep_time)
        camera.move_down(action="stop",channel=0,vertical_speed=1)

def takeSnap():
	timeLag(1.5)
	for i in range(1):
		global img_no
		global camera
		image_path = image_dir + '/image' + str(img_no) + '.png' 
		img = camera.snapshot(0, image_path)
		print(img.status)
		if img.status!=200:
			camera = AmcrestCamera(ip, port, username, password).camera
			print('image data not found')
			print('trying again')
			return -1
		img_no += 1

def reset_x():
	zoomOut(2)
	moveDown(2)

def zoomIn(sleep_time):
	camera.zoom_in(action = "start", channel = 0)
	time.sleep(sleep_time)
	camera.zoom_in(action = "stop", channel = 0)
	# camera.focus_far()

def zoomOut(sleep_time):
	camera.zoom_out(action = "start", channel = 0)
	time.sleep(sleep_time)
	camera.zoom_out(action = "stop", channel = 0)
	# camera.focus_near()

def timeLag(sleep_time):
	time.sleep(sleep_time)


def captureSession():
	reverse = 0
	takeSnap()
	for i in range(3):
		moveRight(lg); reverse += lg
		timeLag(0.2)
		takeSnap()
	moveLeft(reverse)

def snap():
	image = camera.snapshot(0).data
	return Image.open(io.BytesIO(image))
	

def tour():
	# Or, moveLeft at the beginning of program and keep track of position yourself
	moveLeft(10)
	image = snap()
	img = plt.imshow(image)
	#plt.show(block=True)
	print("Completed moving left")
	for i in range(10):
		moveRight(md)
		timeLag(0.3)
		image = snap()
		print("snap")
		img.set_data(image)
		#plt.imshow(image)
		#plt.draw()
		#plt.show()
		#plt.show()

def start():
	print('reset')
	#moveLeft(5)
	moveRight(6)
	#moveUp(1)

def end():
	print('end')
	moveLeft(5)
	moveDown(2)
	zoomOut(2)

#end()
start()
for x in range(9):
	#timeLag(1.5)
	takeSnap()
	zoomIn(0.3)
	up=0
	while up<2:
		if takeSnap()==-1:
			reset_x()
			up=0
			continue
		print('[{}] {} snap'.format(x,up))
		#zoomIn(0.2)
		#timeLag(1.5)
		moveUp(0.2/(up+1))
		zoomIn(0.2 + (up/10.0))
		up += 1
	takeSnap()
	print('snap!')
	#timeLag(1.5)
	#moveDown(2)
	zoomOut(2)
	moveDown(2)
	moveRight(0.5)	

end()
'''

lisp=[]
moveLeft(20)
moveRight(3)
print("move left complete")
for x in xrange(20):
	#y=cv2.resize(np.array(snap()),(0,0),fx=0.5,fy=0.5)
	#cv2.imshow('disp',y)
	#print('snap')
	#cv2.waitKey(1)
	#moveRight(1)
#cv2.destroyAllWindows()
'''	
#plt.ion()
#tour()
#captureSession()
