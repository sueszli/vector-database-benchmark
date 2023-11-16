import cv2
import numpy as np
import os

points = []

ROOT_P = os.getcwd()
DATA_P = ROOT_P + '\\stereo_tests\\left\\data\\2020-03-12_outside_shot_5'
IMG_P = ROOT_P + '\\stereo_tests\\left\\img\\2020-03-12_outside_shot_5'

# def click_detect(event, x, y, flags, param):
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		points.append([[x,y]])

cv2.namedWindow('image')
# cv2.setMouseCallback('image', click_detect)

os.chdir(DATA_P)
points = np.load('points2.npy', allow_pickle=True)

if __name__ == '__main__':
	os.chdir(IMG_P)
	img_list = os.listdir()
	
	count = 0
	for i, file in enumerate(img_list):
		img = cv2.imread(file)
		if points[i,0,0]:
			cv2.drawMarker(img,(points[i,0,0],points[i,0,1]),(0, 0, 255),cv2.MARKER_CROSS,thickness=2,markerSize=10)
		cv2.imshow('image',img)
		key = cv2.waitKey(30) & 0xFF

		if key == ord('c'):
			continue

	# os.chdir(DATA_P)
	# np.save('points.npy', points)

	# print(points)
	cv2.destroyAllWindows()