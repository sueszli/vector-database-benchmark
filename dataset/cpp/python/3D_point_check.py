import cv2
import numpy as np 
import os
import stereo_calibration_t as s_calib
import matplotlib.pyplot as plt

ROOT_P = 'D:\\documents\\local uni\\FYP\\code'
os.chdir(ROOT_P)

IMG_P = ROOT_P + '\\' + 'img' + '\\' + 'stereo_3D'
MAIN_P = ROOT_P + '\\' + 'testing'

if __name__ == "__main__":
	os.chdir(MAIN_P)
	s_cal = s_calib.StereoCal()
	s_cal.load_params('0.3071stereo_calib.npy')
	os.chdir(IMG_P)

	img_L = cv2.imread('l_0016.png', cv2.IMREAD_GRAYSCALE)
	img_R = cv2.imread('r_0016.png', cv2.IMREAD_GRAYSCALE)

	Lmap1, Lmap2 = cv2.initUndistortRectifyMap(s_cal.cameraMatrix1,s_cal.distCoeffs1,s_cal.R1,s_cal.P1,(1000,600),cv2.CV_32FC1)
	Rmap1, Rmap2 = cv2.initUndistortRectifyMap(s_cal.cameraMatrix2,s_cal.distCoeffs2,s_cal.R2,s_cal.P2,(1000,600),cv2.CV_32FC1)	
	img_L_r = cv2.remap(img_L, Lmap1, Lmap2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
	img_R_r = cv2.remap(img_R, Rmap1, Rmap2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
	
	img_o_1 = cv2.hconcat([img_L,img_R])
	img_o = cv2.hconcat([img_L_r,img_R_r])

	diff = 40
	for i in range(int(480/diff)):
		img_o[i*diff,:] = 255
		img_o_1[i*diff,:] = 255

	cv2.imshow("img",img_o)
	cv2.imwrite("img_original.png", img_o_1)
	cv2.imwrite("img_rectified.png", img_o)
	# cv2.waitKey(0)
	# quit()

	# pointsL = np.array([[[460.14,255.85]],[[518.33,254.16]],[[577.87,251.63]],
	# 					[[461.83,314.88]],[[520.86,312.35]],[[579.47,310.24]],
	# 					[[464.36,373.91]],[[523.39,371.80]],[[581.58,369.69]]], dtype=np.float32)

	# pointsR = np.array([[[35.30,257.67]],[[93.75,257.67]],[[152.56,257.29]],
	# 					[[35.30,317.25]],[[93.75,316.87]],[[152.57,316.11]],
	# 					[[35.30,375.69]],[[94.12,376.07]],[[152.57,376.07]]], dtype=np.float32)

	pointsL = np.array([[[460.14,255.85]],
						[[461.83,314.88]],
						[[464.36,373.91]]], dtype=np.float32)

	pointsR = np.array([[[35.30,257.67]],
						[[35.30,317.25]],
						[[35.30,375.69]]], dtype=np.float32)

	pointsLu = cv2.undistortPoints(pointsL, s_cal.cameraMatrix1, s_cal.distCoeffs1, R=s_cal.R1, P=s_cal.P1)
	pointsRu = cv2.undistortPoints(pointsR, s_cal.cameraMatrix2, s_cal.distCoeffs2, R=s_cal.R2, P=s_cal.P2)

	## -- Triangulate points in 3D space -- ##
	points4d = cv2.triangulatePoints(s_cal.P1,s_cal.P2,pointsLu,pointsRu)

	## -- Convert homogeneous coordinates to Euclidean space -- ##
	points3d = np.array([i/points4d[3] for i in points4d[:3]])

	print(points3d)

	## -- Plot points -- ##
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(-0.05, 0.05)
	ax.set_ylim(0.45, 0.55)
	ax.set_zlim(0, 0.1)

	x = 0
	y = 1
	z = 2

	for i,X in enumerate(points3d[x]):
		ax.scatter(xs=points3d[x][i]-(25E-2/2),ys=points3d[z][i],zs=points3d[y][i])
		if i>3:
			break
	plt.show()