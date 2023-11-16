import glob
from random import random

import cv2 as cv
import numpy as np




# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)*0.02
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imagesLeft = glob.glob('left/*')
imagesLeft = sorted(imagesLeft)
imgpointsR = []
imagesRight = glob.glob('right/*')
imagesRight = sorted(imagesRight)
print("CUDA")
print(cv.cuda.getCudaEnabledDeviceCount())
print(len(imagesLeft))
for i in range(len(imagesLeft)):
    if (i % 10 == 0):
        print(i)
    # if random() < 0.9:
    #     print("Skipped")
    #     continue
    imgL = cv.imread(imagesLeft[i])
    imgR = cv.imread(imagesRight[i])
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, (7,9), None)
    retR, cornersR = cv.findChessboardCorners(grayR, (7,9), None)
    cv.destroyAllWindows()
    # If found, add object points, image points (after refining them)
    if retL == True and retR == True:
        objpoints.append(objp)
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        # Draw and display the corners
    else:
        print("Rejected!")
        print(imagesLeft[i])


print("All Points Found")
print(len(imgpointsL))
print("Calibrating")
        

# Camera Calibrations
if not len(imgpointsL) == len(imgpointsR):
    print("Error, not all points found")
    exit()
ret, mtxL, distL, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
if not ret:
    print("Error, Camera Left Calibration")
    exit()
ret, mtxR, distR, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpointsR, grayL.shape[::-1], None, None)
if not ret:
    print("Error, Camera Right Calibration")
    exit()

# Stereo Calibration
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1])
if not ret:
    print("Error, Stereo Calibration")
    exit()
print("Calibration Matrix L")
print(mtxL)
print("Distortion L")
print(distL)
print("Calibration Matrix R")
print(mtxR)
print("Distortion R")
print(distR)
print("Rotation")
print(R)
print("Translation")
print(T)

print("Essential")
print(E)
print("Fundamental")
print(F)

RTL = np.concatenate([np.eye(3), [[0],[0],[0]]], axis=-1)
PL = mtxL @ RTL
RTR = np.concatenate([R, T], axis=-1)
PR = mtxR @ RTR

print("Projection Matrices")
print(PL)
print(PR)

# leftImage = cv.cvtColor(leftImage, cv.COLOR_BGR2GRAY)
# rightImage = cv.cvtColor(rightImage, cv.COLOR_BGR2GRAY)

# ret, leftPoints = cv.findChessboardCorners(leftImage, (7,9), None)
# leftPoints = cv.undistortImagePoints(leftPoints, mtxL, distL)
# ret, rightPoints = cv.findChessboardCorners(rightImage, (7, 9), None)
# rightPoints = cv.undistortImagePoints(rightPoints, mtxR, distR)

# points4H = cv.triangulatePoints(PL, PR, leftPoints, rightPoints)

# points3D = points4H[0:3,:]/points4H[3,:]
# print (points3D)


# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c = img1.shape
#     img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
#     img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv.circle(img1,(int(pt1[0][0]), int(pt1[0][1])),5,color,-1)
#         img2 = cv.circle(img2,(int(pt2[0][0]), int(pt2[0][1])),5,color,-1)
#     return img1,img2

# lines1 = cv.computeCorrespondEpilines(rightPoints.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(leftImage,rightImage,lines1,leftPoints,rightPoints)
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv.computeCorrespondEpilines(leftPoints.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(rightImage,leftImage,lines2,rightPoints,leftPoints)

# cv.imshow("Left", img5)
# cv.waitKey(0)
# cv.imshow("Right", img3)
# cv.waitKey(0)

# R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtxL, distL, mtxR, distR, gray.shape, R, T)
# print("Stereo Rectified")