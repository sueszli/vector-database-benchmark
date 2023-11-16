import cv2 as cv
import matplotlib.pyplot as plt

Lx,Ly = 253, 218
Rx,Ry = 49.5, 202.8

if __name__ == "__main__":
	img_L = cv.imread('right0.png', cv.IMREAD_GRAYSCALE)
	plt.imshow(img_L)
	plt.show()
	cv.waitKey(0)