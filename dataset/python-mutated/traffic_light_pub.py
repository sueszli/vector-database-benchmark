import rospy
from morai_msgs.msg import TrafficSignInfo
from morai_msgs.msg import TrafficSign
import torch
import cv2
import numpy as np
from PIL import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

class IMGParser:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.model = torch.hub.load('/home/kimsngi/catkin_ws/src/ssafy_2/scripts/yolov5/', 'custom', path='/home/kimsngi/catkin_ws/src/ssafy_2/scripts/yolov5/best5xl.pt', source='local', force_reload=True)
        self.image_sub = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.callback)
        self.traffic_sign_info_pub = rospy.Publisher('/traffic_sign_info', TrafficSignInfo, queue_size=10)
        self.img_rgb = None
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.img_rgb is not None:
                msg = []
                output = self.model(self.img_rgb)
                info = output.pandas().xyxy[0]
                for i in range(len(info)):
                    name = info.name[i]
                    precision = info.confidence[i]
                    sign = self.generate_msg_topic(name, precision)
                    msg.append(sign)
                self.traffic_sign_info_pub.publish(msg)
            rate.sleep()

    def callback(self, msg):
        if False:
            print('Hello World!')
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            print(e)

    def generate_msg_topic(self, string, precision):
        if False:
            print('Hello World!')
        type = int(string[4:6])
        value = int(string[6:])
        traffic_sign_msg = TrafficSign()
        traffic_sign_msg.traffic_light_type = type
        traffic_sign_msg.traffic_light_status = value
        traffic_sign_msg.detect_precision = int(precision * 100)
        return traffic_sign_msg

    def drawLabel(self, img, obj):
        if False:
            while True:
                i = 10
        shapes = np.zeros_like(img, np.uint8)
        copied = img.copy()
        out = cv2.cvtColor(copied, cv2.COLOR_BGR2RGB)
        for i in range(len(obj.pandas().xyxy[0])):
            x1 = int(obj.pandas().xyxy[0].xmin[i])
            y1 = int(obj.pandas().xyxy[0].ymin[i])
            x2 = int(obj.pandas().xyxy[0].xmax[i])
            y2 = int(obj.pandas().xyxy[0].ymax[i])
            conf = obj.pandas().xyxy[0].confidence[i]
            name = obj.pandas().xyxy[0].name[i]
            cv2.rectangle(shapes, (x1, y1), (x2, y2), (102, 255, 204), -1)
            alpha = 0.7
            mask = shapes.astype(bool)
            out[mask] = cv2.addWeighted(img, 0.5, shapes, 1 - alpha, 0)[mask]
            cv2.putText(out, str(name) + '  ' + str(conf), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (102, 255, 204))
        cv2.imshow('result', out)
        cv2.waitKey(1)
if __name__ == '__main__':
    rospy.init_node('traffic_sign_publisher', anonymous=True)
    image_parser = IMGParser()
    rospy.spin()