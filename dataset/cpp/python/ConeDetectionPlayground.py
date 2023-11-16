
import cv2 as cv
import numpy as np

img_id = 1
right_image = cv.imread("right/"  + str(img_id) + ".png")
left_image = cv.imread("left/" + str(img_id) + ".png")

# cv.imshow("right", right_image)
# cv.imshow("left", left_image)
cv.waitKey(1000)
right_image = right_image[600:1000, 0:1400]
right_image = cv.cvtColor(right_image, cv.COLOR_RGB2HSV)

# Second attempt
cv.imshow("HSV", right_image)
cv.waitKey(0)

hue ,saturation ,value = cv.split(right_image)
cv.imshow("Saturation Image", saturation)
cv.waitKey(0)

retval, thresholded = cv.threshold(saturation, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("Thresholded", thresholded)
cv.waitKey(0)
# First attempt naive blob approach

# black_low = np.array([0, 0, 60])
# black_high = np.array([360, 100, 100])

# black_mask = cv.inRange(right_image, black_low, black_high)

# kernel = np.ones((3,3),np.uint8)
# black_mask = cv.morphologyEx(black_mask, cv.MORPH_CLOSE, kernel)

# kernel = np.ones((3,3),np.uint8)
# black_mask = cv.morphologyEx(black_mask, cv.MORPH_OPEN, kernel)

# # black_mask = cv.medianBlur(black_mask,7)

# black_mask = cv.bitwise_not(black_mask)

# params = cv.SimpleBlobDetector_Params()
# # params.filterByArea = True
# # params.minArea = 50
# # params.maxArea = 1500

# # params.filterByInertia = True
# # params.minInertiaRatio = 0
# # params.maxInertiaRatio = 1

# # params.thresholdStep = 0.1
# params.minConvexity = 0.9
# params.minDistBetweenBlobs = 10

# detector = cv.SimpleBlobDetector_create(params)

# keypoints = detector.detect(black_mask)

# black_mask = cv.drawKeypoints(black_mask, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv.imshow("output", black_mask)
# cv.waitKey(0)
