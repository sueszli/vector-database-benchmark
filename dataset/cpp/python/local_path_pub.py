#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import rospy
import time
from math import sqrt, pow
import numpy as np
from std_msgs.msg import Float32, Int16, String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from lib.mgeo.class_defs import *


class local_path_pub:
    def __init__(self):
        rospy.init_node('local_path_pub', anonymous=True)

        # ROS Topic으로부터 데이터를 수신하는 Subscriber 생성
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        rospy.Subscriber('/global_path', Path, self.global_path_callback)
        rospy.Subscriber("/start_node", String, self.start_node_callback)
        rospy.Subscriber("/end_node", String, self.end_node_callback)

        # ROS Topic으로부터 데이터를 수신하는 Publisher 생성
        self.local_path_pub = rospy.Publisher(
            '/local_path', Path, queue_size=1)
        self.velocity_pub = rospy.Publisher(
            '/velocity1', Float32, queue_size=1)
        self.rec_global_path_pub = rospy.Publisher(
            '/rec_global_path', Int16, queue_size=1)

        # 변수 초기화
        self.is_odom = False
        self.is_path = False
        self.is_start = False
        self.is_end = False
        self.start_node = None
        self.end_node = None

        # Local Path 의 Size 결정
        self.local_path_size = 100          # 50 m
        self.max_velocity = 100.0 / 3.6     # 100 km/h
        self.friction = 0.8

        # 저장된 global path target_velocity 불러오기
        self.velocity_map = {}

        if os.path.isfile("velocity.json"):

            # 저장위치 /root/.ros/velocity.json
            with open("velocity.json", "r") as f:
                self.velocity_map = json.load(f)

        # 출발 노드, 도착 노드 설정
        sn, en = str(self.start_node), str(self.end_node)

        # global path에서 곡률, target_velocity 설정
        while True:

            if self.is_path is True and self.is_start is True and self.is_end is True:

                if self.velocity_map.get(sn) is None:
                    self.velocity_map[sn] = {}
                if self.velocity_map[sn].get(en) is None:
                    self.velocity_map[sn][en] = []

                if len(self.velocity_map[sn][en]) == 0:
                    velocity_list = self.find_target_velocity()
                    self.velocity_map[sn][en] = velocity_list

                    # 쓰기 실패 시 데이터 날아가므로 임시 트랜잭션
                    try:
                        with open("velocity2.json", "w") as f:
                            f.write(json.dumps(self.velocity_map))
                            os.rename("velocity2.json", "velocity.json")
                    except Exception as err:
                        print("쓰기 실패 : ", err)
                        os.remove("velocity2.json")
                else:
                    velocity_list = self.velocity_map[sn][en]

                break
            else:
                rospy.loginfo('Waiting global path data')

        rate = rospy.Rate(30)  # 30hz.

        while not rospy.is_shutdown():

            if self.is_odom is True and self.is_path is True and self.is_start is True and self.is_end:

                # 지역 경로 변수 초기화
                self.local_path_msg = Path()
                self.local_path_msg.header.frame_id = '/map'

                x = self.x
                y = self.y

                # Global Path 에서 차량 위치와 가장 가까운 포인트(Currenty Waypoint) 탐색
                min_dis = float('inf')
                current_waypoint = -1
                for i, waypoint in enumerate(self.global_path_msg.poses):

                    distance = sqrt(
                        pow(x-waypoint.pose.position.x, 2)+pow(y-waypoint.pose.position.y, 2))
                    if distance < min_dis:
                        min_dis = distance
                        current_waypoint = i

                # 가장 가까운 포인트(Currenty Waypoint) 위치부터 Local Path 생성 및 예외 처리
                if current_waypoint != -1:
                    for num in range(current_waypoint, len(self.global_path_msg.poses)):
                        if num - current_waypoint >= self.local_path_size:
                            break
                        tmp_pose = PoseStamped()
                        tmp_pose.pose.position.x = self.global_path_msg.poses[num].pose.position.x
                        tmp_pose.pose.position.y = self.global_path_msg.poses[num].pose.position.y
                        tmp_pose.pose.orientation.w = 1
                        self.local_path_msg.poses.append(tmp_pose)

                # Local Path 메세지 Publish
                self.local_path_pub.publish(self.local_path_msg)

                velocity_msg = Float32()
                velocity_msg = velocity_list[current_waypoint]

                self.velocity_pub.publish(velocity_msg)

            rate.sleep()

    # 콜백함수에서 처음 메시지가 들어오면 현재 위치를 저장
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.is_odom = True

    # 전역 경로 받아오기
    def global_path_callback(self, msg):
        self.is_path = True
        self.rec_global_path_pub.publish(1)
        self.global_path_msg = msg

    # 출발 노드 받아오기
    def start_node_callback(self, data):
        self.is_start = True
        self.start_node = data

    # 도착 노드 받아오기
    def end_node_callback(self, data):
        self.is_end = True
        self.end_node = data

    # target velocity 설정
    def find_target_velocity(self):

        velocity_list = []

        for i in range(0, len(self.global_path_msg.poses)):

            r = self.find_r(i)
            velocity = Float32()
            velocity = sqrt(r * 9.8 * self.friction)

            if velocity > self.max_velocity:
                velocity = self.max_velocity

            velocity_list.append(velocity)

        return velocity_list

    # 곡률반경 계산
    def find_r(self, now_waypoint):

        # 이전 경로 크기
        prev_size = 20

        # 최소 범위
        small_size = 25

        # 최대 범위
        big_size = 100

        big_X_array = []
        big_Y_array = []

        big_r = float('inf')
        small_r = float('inf')
        min_small_r = float('inf')

        en_b_idx = now_waypoint + big_size+1
        if en_b_idx >= len(self.global_path_msg.poses):
            big_size = len(self.global_path_msg.poses) - now_waypoint - 1

        if big_size > small_size:

            for big_idx in range(now_waypoint, now_waypoint + big_size+1, big_size//2):
                x = self.global_path_msg.poses[big_idx].pose.position.x
                y = self.global_path_msg.poses[big_idx].pose.position.y
                big_X_array.append([x, y, 1])
                big_Y_array.append([-(x**2)-(y**2)])

            if (np.linalg.det(big_X_array)):

                X_inverse = np.linalg.inv(big_X_array)

                A_array = X_inverse.dot(big_Y_array)
                a = A_array[0]*-0.5
                b = A_array[1]*-0.5
                c = A_array[2]
                big_r = sqrt(a*a + b*b - c)

        st_s_idx = now_waypoint - prev_size
        en_s_idx = now_waypoint + big_size

        if st_s_idx < 0:
            st_s_idx = 0
        if en_s_idx > len(self.global_path_msg.poses):
            en_s_idx = len(self.global_path_msg.poses)

        for path_idx in range(st_s_idx, en_s_idx - small_size, 3):

            small_X_array = []
            small_Y_array = []

            for small_idx in range(0, small_size+1, small_size//2):
                x = self.global_path_msg.poses[path_idx +
                                               small_idx].pose.position.x
                y = self.global_path_msg.poses[path_idx +
                                               small_idx].pose.position.y
                small_X_array.append([x, y, 1])
                small_Y_array.append([-(x**2)-(y**2)])

            if (np.linalg.det(small_X_array)):

                X_inverse = np.linalg.inv(small_X_array)

                A_array = X_inverse.dot(small_Y_array)
                a = A_array[0]*-0.5
                b = A_array[1]*-0.5
                c = A_array[2]

                small_r = sqrt(a*a + b*b - c)

            if min_small_r > small_r:
                min_small_r = small_r

        r = min(min_small_r, big_r)

        return r


if __name__ == '__main__':
    try:
        test_track = local_path_pub()
    except rospy.ROSInterruptException:
        pass
