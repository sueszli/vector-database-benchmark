import cv2
import numpy as np 	

if __name__ == "__main__":
	kernel = np.array([	[1, 1],
						[1, 1]], dtype=np.uint8)

	kernel1 = np.array([[0, 1],
						[1, 1]], dtype=np.uint8)

	kernel2 = np.array([[1, 0, 1],
						[0, 1, 0],
						[1, 0, 1]], dtype=np.uint8)

	# kernel = np.array([	[0,1,0],
	# 					[1,1,1],
	# 					[0,1,0]], dtype=np.uint8)

	img = cv2.imread("test.bmp", cv2.IMREAD_GRAYSCALE)
	img = ~img
	img_o = cv2.resize(img, (500, 500))


	img = cv2.dilate(img, kernel, iterations=2)
	img = cv2.erode(img, kernel, iterations=3)
	# img = cv2.dilate(img, kernel1, iterations=1)
	# img = cv2.erode(img, kernel, iterations=1)



	img = cv2.resize(img, (500, 500))
	cv2.imwrite("img_old.png", img_o)
	cv2.imwrite("img_new.png", img)
	img_out = cv2.hconcat((img_o,img))
	cv2.imshow("img", img_out)
	cv2.waitKey(0)