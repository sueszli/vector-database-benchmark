"""
calib3d.py:
在图像中绘制一些 2D 的线条来产生 3D 的效果

在棋盘的第一个角点绘制 3D 坐标  X Y Z
 X  为蓝色 Y  为绿色 Z  为红色。
 在视觉效果上来看 Z 轴应 是垂直于棋盘平面的。
"""
import cv2
import numpy as np
import glob
with np.load('B.npz') as X:
    (mtx, dist, _, _) = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

def draw(img, corners, imgpts):
    if False:
        i = 10
        return i + 15
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def draw_cube(img, corners, imgpts):
    if False:
        while True:
            i = 10
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    for (i, j) in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), 255, 3)
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
'\n很 常一样我们  加 图像。搜寻 7x6 的格子 如果发现 我们就把它 优化到亚像素级。然后使用函数:cv2.solvePnPRansac() 来 算旋 和变 换。但我们有了变换矩 之后 我们就可以利用它们将 些坐标 点映射到图 像平 中去。简单来  我们在图像平 上找到了与 3D 空 中的点 3,0,0  ,(0,3,0),(0,0,3) 相对应的点。然后我们就可以使用我们的函数 draw() 从图像 上的第一个 点开始绘制 接 些点的直线了。搞定   \n'
for fname in glob.glob('../data/left*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (ret, corners) = cv2.findChessboardCorners(gray, (7, 6), None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        (ret, rvecs, tvecs, inliers) = cv2.solvePnP(objp, corners2, mtx, dist)
        (imgpts, jac) = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 255
        if k == ord('s'):
            cv2.imwrite(fname[:6] + '.png', img)
cv2.destroyAllWindows()