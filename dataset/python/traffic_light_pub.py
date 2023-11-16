#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
warnings.simplefilter("ignore", DeprecationWarning)


class IMGParser:
    def __init__(self):

        # 신호등 정보 학습된 모델 불러오기
        self.model = torch.hub.load('/home/kimsngi/catkin_ws/src/ssafy_2/scripts/yolov5/', 'custom',
                                    path='/home/kimsngi/catkin_ws/src/ssafy_2/scripts/yolov5/best5xl.pt', source='local', force_reload=True)

        # ROS Topic으로부터 데이터를 송, 수신하는 Subscriber, Publisher 생성
        self.image_sub = rospy.Subscriber(
            "/image_jpeg/compressed", CompressedImage, self.callback)
        self.traffic_sign_info_pub = rospy.Publisher(
            '/traffic_sign_info', TrafficSignInfo, queue_size=10)

        # 변수 초기화
        self.img_rgb = None
        rate = rospy.Rate(30)  # 30 hz

        while not rospy.is_shutdown():
            # 카메라로부터 이미지가 들어왔을 때 실행
            if self.img_rgb is not None:

                # 신호등 인식 메세지 stack
                msg = []
                # 이미지가 학습 모델을 통과한 결과값 output 에 저장
                output = self.model(self.img_rgb)

                # 이미지로부터 검출된 신호등 정보 추출
                info = output.pandas().xyxy[0]

                for i in range(len(info)):
                    # info 로부터 인식된 신호등 정보들의 classname추출
                    name = info.name[i]
                    # 인식 정확도 추출
                    precision = info.confidence[i]

                    # 인식된 정보들로부터 토픽 메세지 만들기
                    sign = self.generate_msg_topic(name, precision)

                    # 배열에 추가
                    msg.append(sign)

                # 완성된 메세지 publish
                self.traffic_sign_info_pub.publish(msg)

                # 화면에 출력하기 위한 라벨 그리기(선택사항)
                # self.drawLabel(self.img_rgb, output)

            rate.sleep()

    # 카메라로부터 받아온 이미지 전처리
    def callback(self, msg):
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        except CvBridgeError as e:
            print(e)

    # 인식된 신호등 정보로부터 데이터 정제
    def generate_msg_topic(self, string, precision):
        # filtering from ignOOOO form
        type = int(string[4:6])
        value = int(string[6:])

        traffic_sign_msg = TrafficSign()
        traffic_sign_msg.traffic_light_type = type
        traffic_sign_msg.traffic_light_status = value
        traffic_sign_msg.detect_precision = int(precision*100)

        return traffic_sign_msg

    # 이미지 imshow 시 인식된 결과 라벨링을 그려주는 함수(선택사항)
    def drawLabel(self, img, obj):
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
            out[mask] = cv2.addWeighted(img, 0.5, shapes, 1-alpha, 0)[mask]
            cv2.putText(out, str(name)+'  '+str(conf), (x1, y1),
                        cv2.FONT_HERSHEY_PLAIN, 1, (102, 255, 204))
        cv2.imshow('result', out)
        cv2.waitKey(1)


if __name__ == '__main__':

    rospy.init_node('traffic_sign_publisher', anonymous=True)

    image_parser = IMGParser()
    rospy.spin()
