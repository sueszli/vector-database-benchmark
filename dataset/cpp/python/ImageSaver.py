#! /usr/bin/env python

import rospy

from sensor_msgs.msg import Image

import cv2 as cv
from cv_bridge import CvBridge


# Please add a path!
SAVE_IMAGE_PATH = "/media/mur/XavierSSD1/mur2022_full_system/CalibrationImages"

SAVE = 5000
SKIP = 0

class ImageSaver:
    def __init__(self):
        self.r_sub = rospy.Subscriber("/CameraRight/image_raw", Image, self.rightInput)
        self.l_sub = rospy.Subscriber("/CameraLeft/image_raw", Image, self.leftInput)

        self.r_saved = 0;
        self.l_saved = 0;
        self.r_to_skip = SKIP;
        self.l_to_skip = SKIP;

        self.cv_bridge = CvBridge()

    def rightInput(self, msg):
        if self.r_saved < SAVE and self.r_to_skip <= 0:
            img_path = SAVE_IMAGE_PATH + "/right/" + str(self.r_saved) + ".png"
            img = cv.cvtColor(self.cv_bridge.imgmsg_to_cv2(msg), cv.COLOR_BGR2RGB)
            cv.imwrite(img_path, img)
            self.r_saved += 1
            self.r_to_skip = SKIP
            print("Saved Right Image to:" + img_path)
        else:
            self.r_to_skip -= 1
    
    def leftInput(self, msg):
        if self.l_saved < SAVE and self.l_to_skip <= 0:
            img_path = SAVE_IMAGE_PATH + "/left/" + str(self.r_saved) + ".png"
            img = cv.cvtColor(self.cv_bridge.imgmsg_to_cv2(msg), cv.COLOR_BGR2RGB)
            cv.imwrite(img_path, img)
            self.l_saved += 1
            self.l_to_skip = SKIP
            print("Saved Left Image to:" + img_path)
        else:
            self.l_to_skip -= 1
    

if __name__ == '__main__':
    # Initialise the node
    global node_name
    node_name = "ImageSaver"
    rospy.init_node(node_name)
    image_saver = ImageSaver()
    # Spin as a single-threaded node
    rospy.spin()

    # Close any OpenCV windows
    cv.destroyAllWindows()
