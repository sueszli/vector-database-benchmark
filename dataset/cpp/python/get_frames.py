import matplotlib.pyplot as plt
import cv2
BASE_PATH = '/home/deeplearning/Desktop/Attendance-System/Images/'
VIDEO = 'clip_data_3.MOV'
VIDEO_PATH = BASE_PATH + VIDEO
vidcap = cv2.VideoCapture(VIDEO_PATH)
success,image = vidcap.read()
count = 0
success = True
plt.imshow(image)
plt.show()
while success:
    if (count%20) == 1:
        cv2.imwrite("../clip_data2/frame%d.jpg" % (count/20), image)     # save frame as JPEG file
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
    if count>200:
        break
    count += 1
print(count)
