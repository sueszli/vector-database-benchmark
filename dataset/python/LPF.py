import cv2
import numpy as np 	

if __name__ == "__main__":
	kernel = np.ones((3,3), dtype=np.float32)
	img = np.zeros((6,6), dtype=np.float32)

	img[2:4,2:4] = 1

	img_o = cv2.filter2D(img, ddepth=-1, kernel=(1/4)*kernel)

	print(img_o)