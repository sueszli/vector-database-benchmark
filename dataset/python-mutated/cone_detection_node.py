from chardet import detect
import rospy
from sensor_msgs.msg import Image
from mur2022.msg import found_cone_msg
from random import randint
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge
LEFT_CAMERA_TOPIC = '/CameraLeft/image_raw'
RIGHT_CAMERA_TOPIC = '/CameraRight/image_raw'
DETECTED_CONE_TOPIC = '/stereo_cones'
CONE_DETECTION_FRAME = '/husky'
CLASS_FILE = '/media/mur/XavierSSD/mur2022_full_system/catkin_ws/src/mur2022/src/cones.names'
MODEL_CONFIG = '/media/mur/XavierSSD/mur2022_full_system/catkin_ws/src/mur2022/src/yolov4-tiny-cones.cfg'
MODEL_WEIGHTS = '/media/mur/XavierSSD/mur2022_full_system/catkin_ws/src/mur2022/src/yolov4-tiny-cones_best.weights'
try:
    GPU = cv.cuda.getCudaEnabledDeviceCount()
except:
    GPU = False
CONFIDENCE_THRESH = 0.75
NMS_THRESH = 0.75
INPUT_WIDTH = 832
INPUT_HEIGHT = 832

class Cone:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.x = 0
        self.y = 0
        self.color = None

    def __init__(self, x, y, color):
        if False:
            while True:
                i = 10
        self.x = x
        self.y = y
        self.color = color

class ConeDetector:

    def __init__(self):
        if False:
            return 10
        rospy.init_node('cone_detection_node')
        self.verbose = False
        self.cone_pub = rospy.Publisher(DETECTED_CONE_TOPIC, found_cone_msg, queue_size=20)
        self.r_sub = rospy.Subscriber(RIGHT_CAMERA_TOPIC, Image, self.rightInput)
        self.l_sub = rospy.Subscriber(LEFT_CAMERA_TOPIC, Image, self.leftInput)
        self.cone_pub = rospy.Publisher(DETECTED_CONE_TOPIC, found_cone_msg, queue_size=20)
        self.detected_cones = 0
        self.have_left = False
        self.have_right = False
        self.current_left = None
        self.current_right = None
        self.cv_bridge = CvBridge()
        self.net = cv.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
        self.classes = None
        with open(CLASS_FILE, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        print('Classes: ', self.classes)
        if not GPU:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            print('Using CPU device.')
        else:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
            print('Using GPU device.')

    def rightInput(self, msg):
        if False:
            print('Hello World!')
        if not self.have_right:
            img = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.current_right = img
            if self.verbose:
                print('Got right image')
            self.have_right = True

    def leftInput(self, msg):
        if False:
            i = 10
            return i + 15
        if not self.have_left:
            img = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.current_left = img
            if self.verbose:
                print('Got left image')
            self.have_left = True

    def publishCone(self, cone):
        if False:
            print('Hello World!')
        cone_msg = found_cone_msg()
        cone_msg.colour = cone.color
        cone_msg.point.header.seq = self.detected_cones
        cone_msg.point.header.frame_id = CONE_DETECTION_FRAME
        cone_msg.point.header.stamp = rospy.Time.now()
        cone_msg.point.point.x = cone.x
        cone_msg.point.point.y = cone.y
        cone_msg.point.point.z = 0
        self.cone_pub.publish(cone_msg)
        self.detected_cones += 1

    def detectCones(self):
        if False:
            print('Hello World!')
        if self.verbose:
            print('Checking if images saved...')
        if self.have_left and self.have_right:
            if self.verbose:
                print('Have them, running algorithm')
            found_cones = self.runAlgorithm(self.verbose)
            print('No. Cones Found: ', len(found_cones))
            for cone in found_cones:
                self.publishCone(cone)
            self.have_left = False
            self.have_right = False
        elif self.verbose:
            print('No Images, skipping')

    def runAlgorithm(self, verbose=True):
        if False:
            for i in range(10):
                print('nop')
        right_image = self.current_right
        left_image = self.current_left
        (h, w) = left_image.shape[:2]
        if verbose:
            cv.imshow('right', right_image)
            cv.waitKey(1000)
            cv.imshow('left', left_image)
            cv.waitKey(1000)
            cv.destroyAllWindows()
        blobR = cv.dnn.blobFromImage(right_image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
        self.net.setInput(blobR)
        outsR = self.net.forward(self.getOutputsNames())
        (boxesR, classR) = self.postprocess(right_image, outsR)
        if verbose:
            (t, _) = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(right_image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        blobL = cv.dnn.blobFromImage(left_image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
        self.net.setInput(blobL)
        outsL = self.net.forward(self.getOutputsNames())
        (boxesL, classL) = self.postprocess(left_image, outsL)
        if verbose:
            (t, _) = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(left_image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        if verbose:
            print('Found', len(boxesL), 'left boxes')
            print('Found', len(boxesR), 'right boxes')
        centerL = []
        classLnew = []
        for l in range(len(boxesL)):
            box = boxesL[l]
            x = box[0] + box[2] / 2
            y = box[1] + box[3] / 2
            if y > 600:
                try:
                    ind = centerL.index([x, y])
                except ValueError:
                    centerL.append([x, y])
                    classLnew.append(classL[l])
            elif verbose:
                print('Not Added!')
        centerL = np.array(centerL)
        classL = classLnew
        if verbose:
            print(centerL)
        centerR = []
        classRnew = []
        for r in range(len(boxesR)):
            box = boxesR[r]
            x = box[0] + box[2] / 2
            y = box[1] + box[3] / 2
            if y > 600:
                try:
                    ind = centerR.index([x, y])
                except ValueError:
                    centerR.append([x, y])
                    classRnew.append(classR[r])
            elif verbose:
                print('Not Added!')
        centerR = np.array(centerR)
        classR = classRnew
        if verbose:
            print(centerR)
        all_cost = np.ones((len(centerR), len(centerL))) * float('inf')
        for r in range(len(centerR)):
            for l in range(len(centerL)):
                if classR[r] == classL[l]:
                    (xL, yL) = self.world_from_L(centerL[l][0], centerL[l][1])
                    (xR, yR) = self.world_from_R(centerR[r][0], centerR[r][1])
                    cost = abs(xL - xR) + 2 * abs(yL - yR)
                    if abs(yL - yR) > 0.7:
                        cost = cost * 1000
                    all_cost[r][l] = cost
        if verbose:
            print(np.round(all_cost, 3))
        pairs = []
        for r in range(len(centerR)):
            for l in range(len(centerL)):
                if min(all_cost[r, :]) == all_cost[r, l] and min(all_cost[:, l]) == all_cost[r, l]:
                    pairs.append((r, l))
        if verbose:
            print(pairs)
        sortL = []
        sortR = []
        for (r, l) in pairs:
            sortR.append(r)
            sortL.append(l)
        centerL = centerL[sortL]
        centerR = centerR[sortR]
        classL = np.array(classL)
        classR = np.array(classR)
        classL = classL[sortL]
        classR = classR[sortR]
        if verbose:
            print(classR)
            for (pointL, pointR, cl, pair) in zip(centerL, centerR, classR, pairs):
                (xL, yL) = pointL
                (xR, yR) = pointR
                if cl == 0:
                    color = (100 + randint(0, 150), 0, 0)
                elif cl == 1:
                    color = (0, 50 + randint(0, 200), 0)
                elif cl == 2:
                    rn = 50 + randint(0, 200)
                    color = (0, rn, rn)
                else:
                    color = (0, 0, 0)
                right_image = cv.circle(right_image, (int(xR), int(yR)), 5, color, -1)
                cv.putText(right_image, str(pair[0]) + ',' + str(pair[1]), (int(xR), int(yR)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                left_image = cv.circle(left_image, (int(xL), int(yL)), 5, color, -1)
                cv.putText(left_image, str(pair[0]) + ',' + str(pair[1]), (int(xL), int(yL)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            left_image = cv.resize(left_image, (int(w / 2), int(h / 2)))
            cv.imshow('Left Image w/ matches', left_image)
            cv.waitKey(2000)
            right_image = cv.resize(right_image, (int(w / 2), int(h / 2)))
            cv.imshow('Right Image w/ matches', right_image)
            cv.waitKey(2000)
        cones = []
        for i in range(len(centerL)):
            (xL, yL) = self.world_from_L(centerL[i][0], centerL[i][1])
            (xR, yR) = self.world_from_R(centerR[i][0], centerR[i][1])
            x = (xL + xR) / 2
            y = (yL + yR) / 2
            if not self.valid_cone(x, y):
                continue
            if classL[i] == classR[i]:
                color = self.classes[classL[i]]
            else:
                color = 'unknown'
            cone = Cone(x, y, color)
            cones.append(cone)
            if verbose:
                print(pairs[i])
                print(xL, yL)
                print(xR, yR)
                print(x, y)
        return cones

    def world_from_L(self, u, v):
        if False:
            while True:
                i = 10
        u = (u - 1030) / 452.9
        v = (v - 725.9) / 76.41
        y = -0.06899 - 1.705 * u + 0.09506 * v + 0.06137 * u ** 2 + 0.7549 * u * v - 0.05601 * v ** 2 + 0.01561 * u ** 3 - 0.05147 * v * u ** 2 - 0.1594 * u * v ** 2 + 0.01651 * v ** 3
        x = 4.861 - 0.2278 * u - 2.396 * v - 0.05831 * u ** 2 + 0.1824 * u * v + 0.9738 * v ** 2 + 0.01231 * u ** 3 + 0.02113 * v * u ** 2 - 0.05309 * u * v ** 2 - 0.1779 * v ** 3
        return (x, y)

    def world_from_R(self, u, v):
        if False:
            i = 10
            return i + 15
        u = (u - 881.8) / 452.7
        v = (v - 764.5) / 77.06
        y = -0.04364 - 1.71 * u - 0.006886 * v + 0.06176 * u ** 2 + 0.7675 * u * v - 0.01835 * v ** 2 + 0.0171 * u ** 3 - 0.04484 * v * u ** 2 - 0.1662 * u * v ** 2 + 0.007698 * v ** 3
        x = 4.835 - 0.292 * u - 2.424 * v - 0.05401 * u ** 2 + 0.1824 * u * v + 1.043 * v ** 2 + 0.007452 * u ** 3 + 0.03166 * v * u ** 2 - 0.04845 * u * v ** 2 - 0.201 * v ** 3
        return (x, y)

    def valid_cone(self, x, y):
        if False:
            while True:
                i = 10
        if x < 0:
            print('x < 0')
            return False
        return True

    def drawPred(self, image, classId, conf, left, top, right, bottom):
        if False:
            print('Hello World!')
        cv.rectangle(image, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        if self.classes:
            assert classId < len(self.classes)
            label = '%s:%s' % (self.classes[classId], label)
        (labelSize, baseLine) = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(image, (left, int(top - round(1.5 * labelSize[1]))), (int(left + round(1.5 * labelSize[0])), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    def postprocess(self, image, outs):
        if False:
            while True:
                i = 10
        imageHeight = image.shape[0]
        imageWidth = image.shape[1]
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > CONFIDENCE_THRESH:
                    center_x = int(detection[0] * imageWidth)
                    center_y = int(detection[1] * imageHeight)
                    width = int(detection[2] * imageWidth)
                    height = int(detection[3] * imageHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESH, NMS_THRESH)
        oBoxes = []
        oClasses = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            oBoxes.append(box)
            oClasses.append(classIds[i])
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(image, classIds[i], confidences[i], left, top, left + width, top + height)
        return (oBoxes, oClasses)

    def getOutputsNames(self):
        if False:
            for i in range(10):
                print('nop')
        layersNames = self.net.getLayerNames()
        check = self.net.getUnconnectedOutLayers().tolist()
        return [layersNames[i[0] - 1] for i in check]

def mainLoop():
    if False:
        return 10
    cone_detector = ConeDetector()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        cone_detector.detectCones()
        rate.sleep()
if __name__ == '__main__':
    try:
        mainLoop()
    except rospy.ROSInterruptException:
        pass
    cv.destroyAllWindows()