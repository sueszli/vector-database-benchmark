#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import rospy
import os
from sensor_msgs.msg import Imu
from morai_msgs.msg import GPSMessage
from nav_msgs.msg import Odometry
from pyproj import Proj

class GPSIMUParser:
    def __init__(self):
        rospy.init_node('GPS_IMU_parser', anonymous=True)

        # ROS Topic으로부터 데이터를 수신하는 Subscriber 생성
        self.gps_sub = rospy.Subscriber("/gps", GPSMessage, self.navsat_callback)
        self.imu_sub = rospy.Subscriber("/imu", Imu, self.imu_callback)
        
        # Odom msg를 송신하는 Publisher 생성
        self.odom_pub = rospy.Publisher('/odom',Odometry, queue_size=1)
        
        # 초기화
        self.x, self.y = None, None
        self.is_imu=False
        self.is_gps=False

        # 좌표계 변환
        self.proj_UTM = Proj( proj='utm',zone=52,ellps='WGS84',preserve_units=False)

        # Odometry msg 변수 생성
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = '/odom'
        self.odom_msg.child_frame_id = '/base_link'

        rate = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():
            if self.is_imu==True and self.is_gps == True:

                # 맵 좌표계로 변환
                self.convertLL2UTM()

                # Odometry publish
                self.odom_pub.publish(self.odom_msg)

                os.system('clear')
                print(" ROS Odometry Msgs Pose ")
                print(self.odom_msg.pose.pose.position)
                print(" ROS Odometry Msgs Orientation ")
                print(self.odom_msg.pose.pose.orientation)

                rate.sleep()


    # GPS msg를 맵 좌표계로 변환하는 콜백 함수
    def navsat_callback(self, gps_msg):

        self.lat = gps_msg.latitude
        self.lon = gps_msg.longitude
        self.e_o = gps_msg.eastOffset
        self.n_o = gps_msg.northOffset
        self.is_gps=True

        xy_zone = self.proj_UTM(self.lon, self.lat)

        if self.lon == 0 and self.lat == 0:
            self.x = 0.0
            self.y = 0.0

        else:
            self.x = xy_zone[0] - self.e_o
            self.y = xy_zone[1] - self.n_o

        # 맵 좌표계 값으로 변환 된 좌표 데이터로 Odometry msg 갱신
        self.odom_msg.header.stamp = rospy.get_rostime()
        self.odom_msg.pose.pose.position.x = self.x
        self.odom_msg.pose.pose.position.y = self.y
        self.odom_msg.pose.pose.position.z = 0


    # imu msg를 통해 받은 차량의 자세 데이터를 Odometry msg에 추가하는 콜백 함수
    def imu_callback(self, data):

        if data.orientation.w == 0:
            self.odom_msg.pose.pose.orientation.x = 0.0
            self.odom_msg.pose.pose.orientation.y = 0.0
            self.odom_msg.pose.pose.orientation.z = 0.0
            self.odom_msg.pose.pose.orientation.w = 1.0
        else:
            self.odom_msg.pose.pose.orientation.x = data.orientation.x
            self.odom_msg.pose.pose.orientation.y = data.orientation.y
            self.odom_msg.pose.pose.orientation.z = data.orientation.z
            self.odom_msg.pose.pose.orientation.w = data.orientation.w
        self.is_imu=True

if __name__ == '__main__':
    try:
        GPS_IMU_parser = GPSIMUParser()
    except rospy.ROSInterruptException:
        pass
