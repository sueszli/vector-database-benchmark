#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32
from nav_msgs.msg import Path, Odometry
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList

import numpy as np
from tf.transformations import euler_from_quaternion

class AccCtrlPub:
    def __init__(self):
        rospy.init_node('acc_ctrl', anonymous=True)

        # ROS Topic으로부터 데이터를 수신하는 Subscriber 생성
        rospy.Subscriber('/local_path', Path, self.path_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.status_callback)
        rospy.Subscriber('/Object_topic', ObjectStatusList, self.object_info_callback)

        # ROS Topic으로 데이터를 송신하는 Publisher 생성
        self.target_vel_pub = rospy.Publisher('/velocity2', Float32, queue_size=1)

        # 초기 속도 설정
        self.target_vel = Float32()
        self.target_vel.data = 100.0 / 3.6  # 최대 속도

        # Callback 함수 실행 여부 확인을 위한 Flag 변수
        self.is_path = False
        self.is_odom = False
        self.is_status = False
        self.is_obj = False

        # 안전 거리 계산을 위한 시간/거리 계수 설정
        self.time_gain = 0.8
        self.dis_gain = 2.0

        # 30 Hz로 실행되는 ROS 루프
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.is_path and self.is_odom and self.is_status and self.is_obj:
                self.calc_acc_control(self.my_pos, self.heading, self.obj_data)

                self.target_vel_pub.publish(self.target_vel)

            rate.sleep()

    # ROS Topic으로부터 수신한 경로 데이터 처리
    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

    # ROS Topic으로부터 수신한 Odometry 데이터 처리
    def odom_callback(self, msg):
        self.is_odom = True
        odom_quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        _, _, self.heading = euler_from_quaternion(odom_quaternion)
        self.my_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    # ROS Topic으로부터 수신한 EgoVehicleStatus 데이터 처리
    def status_callback(self, msg):
        self.is_status = True
        self.status_data = msg

    # ROS Topic으로부터 수신한 ObjectStatusList 데이터 처리
    def object_info_callback(self, msg):
        self.is_obj = True
        self.obj_data = msg

    # 주어진 자동차 위치, 방향, 장애물 정보를 바탕으로 가감속 제어 계산
    def calc_acc_control(self, my_pos, heading, obj_data):
        self.target_vel.data = 100.0 / 3.6
        my_vel = [self.status_data.velocity.x, self.status_data.velocity.y]

        collision_dis = 0.2 * np.linalg.norm(my_vel) ** 2
        safety_dis = np.linalg.norm(my_vel) * self.time_gain + self.dis_gain

        acc_flag = False
        target_obj_dis = float('inf')
        target_obj_vel = float('inf')

        R = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)]
        ])

        for i in range(obj_data.num_of_npcs):
            obj = obj_data.npc_list[i]

            obj_pos = [obj.position.x, obj.position.y]
            obj_vel = [obj.velocity.x, obj.velocity.y]
            obj_vel = np.matmul(R, obj_vel)

            relative_pos = obj_pos - my_pos
            relative_vel = obj_vel - my_vel

            dis = np.linalg.norm(relative_pos)

            if dis < collision_dis:
                flag = self.calc_collision(relative_pos, relative_vel)
            if flag:
                break

            if dis < safety_dis:
                for pose in self.path.poses:
                    dis_path = np.linalg.norm([obj_pos[0] - pose.pose.position.x, obj_pos[1] - pose.pose.position.y])

                    if dis_path < 2.5:
                        if dis < target_obj_dis:
                            acc_flag = True
                            target_obj_dis = dis
                            target_obj_vel = np.linalg.norm(obj_vel)
                            break

        if acc_flag:
            self.target_vel.data = min(self.target_vel.data, target_obj_vel)

    # 장애물과의 충돌 여부를 계산하여 충돌이 예상될 경우 가감속 제어
    def calc_collision(self, relative_pos, relative_vel):
        closest_time = -np.dot(relative_pos, relative_vel) / (np.linalg.norm(relative_vel) ** 2)

        if closest_time < 0:
            return False

        closest_pos = relative_pos + relative_vel*closest_time

        closest_dis = np.linalg.norm(closest_pos)

        if closest_dis > 5:
            return False

        target_vel = self.status_data.velocity.x / 2
        self.target_vel.data = min(self.target_vel.data, target_vel)
        return True

if __name__ == '__main__':
    try:
        acc_ctrl = AccCtrlPub()
    except rospy.ROSInterruptException:
        pass
