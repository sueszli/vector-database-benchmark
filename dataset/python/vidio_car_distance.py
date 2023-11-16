#!usr/bin/python
# -*- coding: utf-8 -*-

#import the necessary packages
import numpy as np
import cv2
 
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    gray = cv2.GaussianBlur(gray, (5, 5), 0)        
    edged = cv2.Canny(gray, 35, 125)               
 
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 求最大面积 
    c = max(cnts, key = cv2.contourArea)
 
    # compute the bounding box of the of the paper region and return it
    # cv2.minAreaRect() c代表点集，返回rect[0]是最小外接矩形中心点坐标，
    # rect[1][0]是width，rect[1][1]是height，rect[2]是角度
    return cv2.minAreaRect(c)
 
def distance_to_camera(knownWidth, focalLength, perWidth):  
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth            

def fmain(type):
# initialize the known distance from the camera to the object, which
#type 6 means 越野车
  if type==6:
    KNOWN_DISTANCE = 48.0
    #KNOWN_DISTANCE = 610
    # initialize the known object width, which in this case, the car

    KNOWN_WIDTH = 200
    KNOWN_HEIGHT = 16.27
    #KNOWN_WIDTH = 297
    #KNOWN_HEIGHT = 210
    # initialize the list of images that we'll be using
    #IMAGE_PATHS = ["Picture1.jpg", "Picture2.jpg", "Picture3.jpg","picture4.jpg","Picture5.jpg","Picture6.jpg","Picture7.jpg"]

    IMAGE_PATHS = ["1.jpg", "2.jpg", "3.jpg"]


# from our camera, then find the paper marker in the image, and initialize
# the focal length
    image = cv2.imread(IMAGE_PATHS[0])
#image = cv2.imread('Picture1.jpg')
#cv2.imshow('first image',image)   
    marker = find_marker(image)
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
#focalLength = 811.82
    print('focalLength = ',focalLength)
    camera = cv2.VideoCapture(0)




# loop over the images

    for imagePath in IMAGE_PATHS:
    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera
        image = cv2.imread(imagePath)
        marker = find_marker(image)
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

    # draw a bounding box around the image and display it
        box = cv2.boxPoints(marker)
        box = np.int0(box)
        #% (inches *30.48/ 12)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, "%.2fcm" % (inches*30.48/12),
    	(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
    	2.0, (0, 255, 0), 3)
        cv2.imshow("image", image)
        cv2.waitKey(0)

    # while camera.isOpened():
    # # get a frame
    #     (grabbed, frame) = camera.read()
    #
    #     if not grabbed:
    #         break
    #
    #     marker = find_marker(frame)
    #     if marker == 0:
    #         print(marker)
    #         continue
    #     inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    #
    # # draw a bounding box around the image and display it
    #     box = np.int0(cv2.cv.BoxPoints(marker))
    #     cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    #     cv2.putText(frame, "%.2fcm" % (inches *30.48/ 12),
    #          (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
    #      2.0, (0, 255, 0), 3)
    #
    # # show a frame
    #     cv2.imshow("capture", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # camera.release()
    #
    # cv2.destroyAllWindows()

fmain(6)
