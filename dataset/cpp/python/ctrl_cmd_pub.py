#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from math import cos, sin, atan2, sqrt, pow
import numpy as np
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry, Path
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus

class ctrl_cmd_pub:
    def __init__(self):
        # node 초기화
        rospy.init_node('ctrl_cmd', anonymous=True)
        rospy.loginfo('ctrl_cmd_pub open')

        # 구독, 발행 함수 선언
        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber('/velocity1', Float32, self.velocity_callback1)
        rospy.Subscriber('/velocity2', Float32, self.velocity_callback2)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        
        # 데이터 유무 플래그
        self.is_path = False
        self.is_velocity1 = False
        self.is_velocity2 = False
        self.is_odom = False
        self.is_status = False
        self.is_target_point = False
        
        # 발행 메시지 변수 선언
        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 1
        
        # 물리 변수 선언
        self.vehicle_length = 2.984
        self.lad = 15.0
        self.lad_min = 3
        self.lad_max = 20
        self.lad_gain = 0.6
        
        # p_gain, i_gain, d_gain 을 변수로 선언할 수 있음
        self.velocity_pid = pidControl(0.30, 0.00, 0.03)
        self.steering_pid = pidControl(1.30, 0.00, 0.00)

        self.target_steering = 0.0
        self.target_velocity1 = 100 / 3.6

        rospy.loginfo('ctrl_cmd_pub ready')

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.is_path is True and self.is_odom is True and self.is_status is True and self.is_velocity1 is True and self.is_velocity2 is True: # 메시지 수신 확인
                
                self.target_steering = self.find_target_steering()
                
                print( self.target_velocity1, self.target_velocity2)
                velocity_output = self.velocity_pid.output(min(self.target_velocity1, self.target_velocity2), self.status_msg.velocity.x)
                steering_output = self.steering_pid.output(self.target_steering, 0.0)

                if velocity_output > 0.0:
                    self.ctrl_cmd_msg.accel = velocity_output
                    self.ctrl_cmd_msg.brake = 0.0
                else:
                    self.ctrl_cmd_msg.accel = 0.0
                    self.ctrl_cmd_msg.brake = -velocity_output

                self.ctrl_cmd_msg.steering = steering_output
                
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                    
            rate.sleep()
        
    def path_callback(self, msg):
        self.path = msg
        
        self.is_path = True
    
    def velocity_callback1(self, msg):
        
        self.target_velocity1 = msg.data
        
        self.is_velocity1 = True
        
    def velocity_callback2(self, msg):
        
        self.target_velocity2 = msg.data / 3.6
        
        self.is_velocity2 = True
    
    def odom_callback(self, msg):
        self.current_position = msg.pose.pose.position
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        
        self.is_odom = True
    
    def status_callback(self, msg):
        self.status_msg = msg
        
        self.is_status = True
    
    def find_target_steering(self):

        self.lad = (self.status_msg.velocity.x)*self.lad_gain
        self.lad = max(self.lad, self.lad_min)
        self.lad = min(self.lad, self.lad_max)
        
        translation = [self.current_position.x, self.current_position.y]
        
        trans_matrix = np.array([
            [cos(self.vehicle_yaw)  ,-sin(self.vehicle_yaw) ,translation[0] ],
            [sin(self.vehicle_yaw)  ,cos(self.vehicle_yaw)  ,translation[1] ],
            [0                      ,0                      ,1              ]])
        
        det_trans_matrix = np.linalg.inv(trans_matrix)
        dis = 0
        
        for i, pose in enumerate(self.path.poses):
            path_point = pose.pose.position
            
            global_path_point = [path_point.x, path_point.y, 1]
            local_path_point = det_trans_matrix.dot(global_path_point)
            
            if local_path_point[0] > 0:
                dis = sqrt(pow(local_path_point[0], 2) + pow(local_path_point[1], 2))
                if dis >= self.lad:
                    break
                
        theta = atan2(local_path_point[1] + 0.024*dis,local_path_point[0])
        steering = atan2(2*self.vehicle_length*sin(theta), dis)
        
        return steering


class pidControl:
    def __init__(self, p_gain, i_gain, d_gain):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.prev_error = 0
        self.i_control = 0
        self.controlTime = 0.02
    
    def output(self, target_value, current_value):
        error = target_value - current_value
        
        p_control = self.p_gain * error
        self.i_control += self.i_gain * error * self.controlTime
        d_control = self.d_gain * (error - self.prev_error) / self.controlTime
        
        output = p_control + self.i_control + d_control
        self.prev_error = error
        
        return output

if __name__ == '__main__':
    try:
        ctrl_cmd = ctrl_cmd_pub()
    except rospy.ROSInterruptException:
        pass
