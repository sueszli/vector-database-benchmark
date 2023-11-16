'''
Run spsstereo on all images and get the disparity maps.
'''
import os
import sys
import subprocess

RIGHT_PATH = '/home/b1/Documents/data_Road_right/testing/image_3'
LEFT_PATH = '/home/b1/Documents/data_road/testing/image_2'
OUT_DIR = '/home/b1/Downloads/spsstereo/build/disparities/'

left_images = [LEFT_PATH + os.sep + f for f in os.listdir(LEFT_PATH) if os.path.isfile(os.path.join(LEFT_PATH, f))]

right_images = [RIGHT_PATH + os.sep + f for f in os.listdir(RIGHT_PATH) if os.path.isfile(os.path.join(RIGHT_PATH, f))]

left_images.sort()

right_images.sort()

for l,r in zip(left_images, right_images):
	print(subprocess.check_output("./spsstereo " + l + " " + r, shell=True))