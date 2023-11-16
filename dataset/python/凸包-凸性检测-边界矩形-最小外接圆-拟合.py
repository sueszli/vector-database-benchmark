# -*- coding: utf-8 -*-
# @Time    : 2017/7/12 下午8:28
# @Author  : play4fun
# @File    : 凸包-凸性检测-边界矩形-最小外接圆-拟合.py
# @Software: PyCharm

"""
凸包-凸性检测-边界矩形-最小外接圆-拟合.py:
"""
import cv2
import numpy as np

img=cv2.imread('../data/lightning.png',0)

image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt=contours[0]
'''
函数 cv2.convexHull() 可以用来检测一个曲线是否具有凸性缺  并能纠 正缺 。一般来  凸性曲线总是凸出来的 至少是平的。如果有地方凹 去 了就 叫做凸性缺 
例如下图中的手。红色曲线显示了手的凸包 凸性缺   双箭头标出来了。
'''
# convexHull(points, hull=None, clockwise=None, returnPoints=None)
hull = cv2.convexHull(points, hull, clockwise, returnPoints)

'''
points 我们 传入的 廓
• hull  输出  通常不需要  
• clockwise 方向标志。如果 置为 True  出的凸包是顺时针 方向的。 否则为逆时针 方向。
• returnPoints   值为 True。它会 回凸包上点的坐标。如果 置 为 False 就会 回与凸包点对应的 廓上的点。
'''
hull = cv2.convexHull(cnt)

# 凸性检测
# 函数 cv2.isContourConvex() 可以可以用来检测一个曲线是不是凸的。它只能 回 True 或 False。没什么大不了的。
k = cv2.isContourConvex(cnt)

# 边界矩形
'''
直边界矩形 一个直矩形 就是没有旋转的矩形 。它不会考虑对象是否旋转。 所以边界矩形的 积不是最小的。可以使用函数 cv2.boundingRect() 查 找得到。
 x y 为矩形左上角的坐标 w h 是矩形的宽和 。
'''
x, y, w, h = cv2.boundingRect(cnt)
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

'''
旋转矩形
这里，以最小面积绘制边界矩形，因此也考虑旋转。使用的功能是cv2.minAreaRect（）。它返回一个Box2D结构，其中包含以下条件 - （中心（x，y），（宽度，高度），旋转角度）。但是要绘制这个矩形，我们需要矩形的四个角。它是通过函数cv2.boxPoints（）
'''
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)

# 最小外接圆
# 函数 cv2.minEnclosingCircle() 可以帮我们找到一个对 的外切圆。它是所有能够包括对 的圆中 积最小的一个。
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
img = cv2.circle(img, center, radius, (0, 255, 0), 2)

# 椭圆拟合
# 使用的函数为 cv2.ellipse()  回值其实就是旋  界矩形的内切圆
ellipse = cv2.fitEllipse(cnt)
#((135.34278869628906, 134.22764587402344),(57.018402099609375, 166.91265869140625),136.8311767578125)
angle=ellipse[2]
im = cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# 直线拟合
# 我们可以根据一组点拟合出一条直线 同样我们也可以为图像中的白色点 拟合出一条直线。
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
