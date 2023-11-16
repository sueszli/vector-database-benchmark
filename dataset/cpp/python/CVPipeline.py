#! /usr/bin/env python

import rospy

from sensor_msgs.msg import Image

import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

CLASS_FILE = "src/cone_detection_pipeline/src/models/yolo-tinyv4/cones.names"
MODEL_CONFIG = "src/cone_detection_pipeline/src/models/yolo-tinyv4/yolov4-tiny.cfg"
MODEL_WEIGHTS = "src/cone_detection_pipeline/src/models/yolo-tinyv4/yolov4-tiny-best.weights"

GPU = True

confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 832       #Width of network's input image
inpHeight = 832      #Height of network's input image

class CVPipeline:
    def __init__(self):
        self.r_sub = rospy.Subscriber("/CameraRight/image_raw", Image, self.rightInput)
        self.l_sub = rospy.Subscriber("/CameraLeft/image_raw", Image, self.leftInput)
        self.r_new = False
        self.l_new = False

        self.cv_bridge = CvBridge()

        self.r_image = None
        self.l_image = None

        self.net = cv.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)

        if(not GPU):
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            print('Using CPU device.')
        else:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
            print('Using GPU device.')
        self.classes = None
        with open(CLASS_FILE, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')


    def rightInput(self, msg):
        if not self.r_new:
            self.r_image = self.cv_bridge.imgmsg_to_cv2(msg)
            self.r_new = True
    
    def leftInput(self, msg):
        if not self.l_new:
            self.l_image = self.cv_bridge.imgmsg_to_cv2(msg)
            self.l_new = True

    def pipeline(self, event=None):
        if not (self.r_new and self.l_new):
            return
        
        r_blob = cv.dnn.blobFromImage(self.r_image, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # At end
        self.r_new = False
        self.l_new = False

    # From OpenCV Tutorial
    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        label = '%.2f' % conf
            
        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        
        return frame

# Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)



if __name__ == '__main__':
    # Initialise the node
    global node_name
    node_name = "CVPipeline"
    rospy.init_node(node_name)
    cv_pipeline = CVPipeline()

    rospy.Timer(rospy.Duration(1.0/10.0), cv_pipeline.pipeline)
    # Spin as a single-threaded node
    rospy.spin()

    # Close any OpenCV windows
    cv.destroyAllWindows()