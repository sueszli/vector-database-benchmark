#!/usr/bin/env python
#-*- coding:utf-8 -*-

import time
import rospy
import cv2
import numpy as np
import os, rospkg
import json
import math
import random
import tf

from cv_bridge import CvBridgeError
from sklearn import linear_model

from nav_msgs.msg import Odometry,Path
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped,Point

class IMGParser:
    def __init__(self, pkg_name = 'ssafy_3'):

        # ROS Topic으로부터 카메라 이미지 데이터를 수신하는 Subscriber 생성
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)
        
        # ROS Topic으로부터 odometry 데이터를 수신하는 Subscriber
        rospy.Subscriber("odom", Odometry, self.odom_callback)

        # 차선 인식에 따른 fitting 정보 publisher 생성
        self.path_pub = rospy.Publisher('/lane_path', Path, queue_size=30)
        
        # 변수 초기화
        self.img_bgr = None
        self.img_lane = None
        self.edges = None 
        self.is_status = False

        # ROI 영역 설정
        self.crop_pts = np.array([[[0,480],[0,350],[280,200],[360,200],[640,350],[640,480]]])

        rospack = rospkg.RosPack()
        
        # 현재 path 정보 가져오기
        currentPath = rospack.get_path(pkg_name)
        
        #센서 정보 가져오기
        with open(os.path.join(currentPath, 'sensor/sensor_params.json'), 'r') as fp:
            sensor_params = json.load(fp)

        # 카메라 센서의 정보
        params_cam = sensor_params["params_cam"]

        bev_op = BEVTransform(params_cam=params_cam)
        # CURVEFit 설정
        curve_learner = CURVEFit(order=3, lane_width=3.5,y_margin=1, x_range=30, min_pts=50)
        
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            # 카메라, odommetry 정보가 들어올 경우
            if self.img_bgr is not None and self.is_status == True:
                # 관심영역 추출
                img_crop = self.mask_roi(self.img_bgr)

                # 원근변환(bird eye view)
                img_warp = bev_op.warp_bev_img(img_crop)

                # canny edge detetction 수행
                canny_output = self.canny_process(img_warp)

                # 원근 역변환
                img_f = bev_op.warp_inv_img(canny_output)

                # 차선 픽셀들만 전부 추출
                lane_pts = bev_op.recon_lane_pts(img_f)
                
                # 좌우차선 피팅
                x_pred, y_pred_l, y_pred_r = curve_learner.fit_curve(lane_pts)
                
                # 현재 차량 상태 갱신
                curve_learner.set_vehicle_status(self.status_msg)

                # 차선을 유지하기 위한 경로 메세지를 만든다.
                lane_path = curve_learner.write_path_msg(x_pred, y_pred_l, y_pred_r)

                # 피팅된 차선 데이터를 이미지로 표시해주기 위한 작업 
                xyl, xyr = bev_op.project_lane2img(x_pred, y_pred_l, y_pred_r)

                # 화면에 표시해주기 위해 그리는 동작
                img_lane_fit = self.draw_lane_img(img_warp, xyl[:, 0].astype(np.int32),
                                                            xyl[:, 1].astype(np.int32),
                                                            xyr[:, 0].astype(np.int32),
                                                            xyr[:, 1].astype(np.int32))

                # 위에서 만들어진 lane path 를 publish 한다.(/lane_path 토픽)
                self.path_pub.publish(lane_path)

                #cv2.imshow("birdview", img_lane_fit)
                #cv2.imshow("img_warp", img_warp)
                #cv2.imshow("origin_img", self.img_bgr)

                #cv2.waitKey(1)

                rate.sleep()

    def odom_callback(self,msg): 
        self.status_msg=msg    
        self.is_status = True

    def callback(self, msg):
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            self.img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
        except CvBridgeError as e:
            print(e)

    def canny_process(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.img_canny = cv2.Canny(img_gray, 50, 150)
        
        return self.img_canny

    def mask_roi(self, img):

        h = img.shape[0]
        w = img.shape[1]
        
        if len(img.shape)==3:

            # num of channel = 3

            c = img.shape[2]
            mask = np.zeros((h, w, c), dtype=np.uint8)

            mask_value = (255, 255, 255)

        else:
    
            # grayscale

            c = 1
            mask = np.zeros((h, w, c), dtype=np.uint8)

            mask_value = (255)

        cv2.fillPoly(mask, self.crop_pts, mask_value)

        mask = cv2.bitwise_and(mask, img)

        return mask



    def draw_lane_img(self, img, leftx, lefty, rightx, righty):

        point_np = cv2.cvtColor(np.copy(img), cv2.COLOR_GRAY2BGR)

        #Left Lane
        for ctr in zip(leftx, lefty):
            point_np = cv2.circle(point_np, ctr, 2, (255,0,0),-1)

        #Right Lane
        for ctr in zip(rightx, righty):
            point_np = cv2.circle(point_np, ctr, 2, (0,0,255),-1)

        return point_np

class BEVTransform:
    def __init__(self, params_cam, xb=10.0, zb=10.0):
        self.xb = xb
        self.zb = zb

        self.theta = np.deg2rad(params_cam["PITCH"])
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]
        self.x = params_cam["X"]
        
        self.alpha_r = np.deg2rad(params_cam["FOV"]/2)

        self.fc_y = params_cam["HEIGHT"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
        self.alpha_c = np.arctan2(params_cam["WIDTH"]/2, self.fc_y)

        self.fc_x = self.fc_y
            
        self.h = params_cam["Z"] + 0.34

        self.n = float(params_cam["WIDTH"])
        self.m = float(params_cam["HEIGHT"])

        self.RT_b2g = np.matmul(np.matmul(self.traslationMtx(xb, 0, zb),    self.rotationMtx(np.deg2rad(-90), 0, 0)),
                                                                            self.rotationMtx(0, 0, np.deg2rad(180)))

        self.proj_mtx = self.project2img_mtx(params_cam)

        self._build_tf(params_cam)


    def calc_Xv_Yu(self, U, V):
        Xv = self.h*(np.tan(self.theta)*(1-2*(V-1)/(self.m-1))*np.tan(self.alpha_r)-1)/\
            (-np.tan(self.theta)+(1-2*(V-1)/(self.m-1))*np.tan(self.alpha_r))

        Yu = (1-2*(U-1)/(self.n-1))*Xv*np.tan(self.alpha_c)

        return Xv, Yu


    def _build_tf(self, params_cam):
        v = np.array([params_cam["HEIGHT"]*0.5, params_cam["HEIGHT"]]).astype(np.float32)
        u = np.array([0, params_cam["WIDTH"]]).astype(np.float32)

        U, V = np.meshgrid(u, v)

        Xv, Yu = self.calc_Xv_Yu(U, V)

        xyz_g = np.concatenate([Xv.reshape([1,-1]) + params_cam["X"],
                                Yu.reshape([1,-1]),
                                np.zeros_like(Yu.reshape([1,-1])),
                                np.ones_like(Yu.reshape([1,-1]))], axis=0)
        
        xyz_bird = np.matmul(np.linalg.inv(self.RT_b2g), xyz_g)

        xyi = self.project_pts2img(xyz_bird)

        src_pts = np.concatenate([U.reshape([-1, 1]), V.reshape([-1, 1])], axis=1).astype(np.float32)
        dst_pts = xyi.astype(np.float32)

        self.perspective_tf = cv2.getPerspectiveTransform(src_pts, dst_pts)

        self.perspective_inv_tf = cv2.getPerspectiveTransform(dst_pts, src_pts)


    def warp_bev_img(self, img):
        img_warp = cv2.warpPerspective(img, self.perspective_tf, (self.width, self.height), flags=cv2.INTER_LINEAR)
        
        return img_warp

    
    def warp_inv_img(self, img_warp):    
        img_f = cv2.warpPerspective(img_warp, self.perspective_inv_tf, (self.width, self.height), flags=cv2.INTER_LINEAR)
        
        return img_f


    def recon_lane_pts(self, img):
        if cv2.countNonZero(img) != 0:
    
            UV_mark = cv2.findNonZero(img).reshape([-1,2])

            U, V = UV_mark[:, 0].reshape([-1,1]), UV_mark[:, 1].reshape([-1,1])
            
            Xv, Yu = self.calc_Xv_Yu(U, V)

            xyz_g = np.concatenate([Xv.reshape([1,-1]) + self.x,
                                Yu.reshape([1,-1]),
                                np.zeros_like(Yu.reshape([1,-1])),
                                np.ones_like(Yu.reshape([1,-1]))], axis=0)

            xyz_g = xyz_g[:, xyz_g[0,:]>=0]

        else:
            xyz_g = np.zeros((4, 10))

        return xyz_g


    def project_lane2img(self, x_pred, y_pred_l, y_pred_r):
        xyz_l_g = np.concatenate([x_pred.reshape([1,-1]),
                                  y_pred_l.reshape([1,-1]),
                                  np.zeros_like(y_pred_l.reshape([1,-1])),
                                  np.ones_like(y_pred_l.reshape([1,-1]))
                                  ], axis=0)

        xyz_r_g = np.concatenate([x_pred.reshape([1,-1]),
                                  y_pred_r.reshape([1,-1]),
                                  np.zeros_like(y_pred_r.reshape([1,-1])),
                                  np.ones_like(y_pred_r.reshape([1,-1]))
                                  ], axis=0)

        xyz_l_b = np.matmul(np.linalg.inv(self.RT_b2g), xyz_l_g)
        xyz_r_b = np.matmul(np.linalg.inv(self.RT_b2g), xyz_r_g)

        xyl = self.project_pts2img(xyz_l_b)
        xyr = self.project_pts2img(xyz_r_b)

        xyl = self.crop_pts(xyl)
        xyr = self.crop_pts(xyr)
        
        return xyl, xyr
        

    def project_pts2img(self, xyz_bird):
        xc, yc, zc = xyz_bird[0,:].reshape([1,-1]), xyz_bird[1,:].reshape([1,-1]), xyz_bird[2,:].reshape([1,-1])

        xn, yn = xc/(zc+0.0001), yc/(zc+0.0001)

        xyi = np.matmul(self.proj_mtx, np.concatenate([xn, yn, np.ones_like(xn)], axis=0))

        xyi = xyi[0:2,:].T
        
        return xyi

    def crop_pts(self, xyi):
        xyi = xyi[np.logical_and(xyi[:, 0]>=0, xyi[:, 0]<self.width), :]
        xyi = xyi[np.logical_and(xyi[:, 1]>=0, xyi[:, 1]<self.height), :]

        return xyi


    def traslationMtx(self,x, y, z):     
        M = np.array([[1,         0,              0,               x],
                    [0,         1,              0,               y],
                    [0,         0,              1,               z],
                    [0,         0,              0,               1],
                    ])
        
        return M

    def project2img_mtx(self,params_cam):    
        # focal lengths
        fc_x = params_cam["HEIGHT"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
        fc_y = params_cam["HEIGHT"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))

        #the center of image
        cx = params_cam["WIDTH"]/2
        cy = params_cam["HEIGHT"]/2
        
        #transformation matrix from 3D to 2D
        R_f = np.array([[fc_x,  0,      cx],
                        [0,     fc_y,   cy]])

        return R_f

    def rotationMtx(self,yaw, pitch, roll):    
        R_x = np.array([[1,         0,              0,                0],
                        [0,         math.cos(roll), -math.sin(roll) , 0],
                        [0,         math.sin(roll), math.cos(roll)  , 0],
                        [0,         0,              0,               1],
                        ])
                        
        R_y = np.array([[math.cos(pitch),    0,      math.sin(pitch) , 0],
                        [0,                  1,      0               , 0],
                        [-math.sin(pitch),   0,      math.cos(pitch) , 0],
                        [0,         0,              0,               1],
                        ])
        
        R_z = np.array([[math.cos(yaw),    -math.sin(yaw),    0,    0],
                        [math.sin(yaw),    math.cos(yaw),     0,    0],
                        [0,                0,                 1,    0],
                        [0,         0,              0,               1],
                        ])
                        
        R = np.matmul(R_x, np.matmul(R_y, R_z))
    
        return R


class CURVEFit:    
    def __init__(self, order=3, alpha=10, lane_width=4, y_margin=0.5, x_range=5, dx=0.5, min_pts=50):

        self.order = order
        self.lane_width = lane_width
        self.y_margin = y_margin
        self.x_range = x_range
        self.dx = dx
        self.min_pts = min_pts

        self.lane_path = Path()
        
        self.ransac_left = linear_model.RANSACRegressor(base_estimator=linear_model.Lasso(alpha=alpha),
                                                        max_trials=5,
                                                        loss='absolute_loss',
                                                        min_samples=self.min_pts,
                                                        residual_threshold=self.y_margin)

        self.ransac_right = linear_model.RANSACRegressor(base_estimator=linear_model.Lasso(alpha=alpha),
                                                        max_trials=5,
                                                        loss='absolute_loss',
                                                        min_samples=self.min_pts,
                                                        residual_threshold=self.y_margin)
        
        
        self._init_model()

    def _init_model(self):        
        X = np.stack([np.arange(0, 2, 0.02)**i for i in reversed(range(1, self.order+1))]).T
        y_l = 0.5*self.lane_width*np.ones_like(np.arange(0, 2, 0.02))
        y_r = -0.5*self.lane_width*np.ones_like(np.arange(0, 2, 0.02))

        self.ransac_left.fit(X, y_l)
        self.ransac_right.fit(X, y_r)

 
    def preprocess_pts(self, lane_pts):
        idx_list = []

        # Random Sampling
        for d in np.arange(0, self.x_range, self.dx):

            idx_full_list = np.where(np.logical_and(lane_pts[0, :]>=d, lane_pts[0, :]<d+self.dx))[0].tolist()
            
            idx_list += random.sample(idx_full_list, np.minimum(self.min_pts, len(idx_full_list)))

        lane_pts = lane_pts[:, idx_list]
        x_g = np.copy(lane_pts[0, :])
        y_g = np.copy(lane_pts[1, :])
        
        # 이전 Frame의 Fitting 정보를 활용하여 현재 Line에 대한 Point를 분류
        X_g = np.stack([x_g**i for i in reversed(range(1, self.order+1))]).T
                
        y_ransac_collect_r = self.ransac_right.predict(X_g)

        y_right = y_g[np.logical_and(y_g>=y_ransac_collect_r-self.y_margin, y_g<y_ransac_collect_r+self.y_margin)]
        x_right = x_g[np.logical_and(y_g>=y_ransac_collect_r-self.y_margin, y_g<y_ransac_collect_r+self.y_margin)]

        y_ransac_collect_l = self.ransac_left.predict(X_g)

        y_left = y_g[np.logical_and(y_g>=y_ransac_collect_l-self.y_margin, y_g<y_ransac_collect_l+self.y_margin)]
        x_left = x_g[np.logical_and(y_g>=y_ransac_collect_l-self.y_margin, y_g<y_ransac_collect_l+self.y_margin)]
        
        #1. Sampling 된 Point들의 X좌표를 활용하여 Prediction Input 생성
        #2. RANSAC을 활용하여 Prediction 수행
        #3. 결과로 나온 Y좌표가 실제 값의 Y좌표와 마진보다 작게 차이나면 Left, Right Lane Point 집합에 추가
        
        return x_left, y_left, x_right, y_right
    
    
    def fit_curve(self, lane_pts):
        # 기존 Curve를 바탕으로 Point 분류
        x_left, y_left, x_right, y_right = self.preprocess_pts(lane_pts)
        
        if len(y_left)==0 or len(y_right)==0:

            self._init_model()

            x_left, y_left, x_right, y_right = self.preprocess_pts(lane_pts)
        
        X_left = np.stack([x_left**i for i in reversed(range(1, self.order+1))]).T
        X_right = np.stack([x_right**i for i in reversed(range(1, self.order+1))]).T

        if y_left.shape[0]>=self.ransac_left.min_samples:
            self.ransac_left.fit(X_left, y_left)
        
        if y_right.shape[0]>=self.ransac_right.min_samples:
            self.ransac_right.fit(X_right, y_right)
            
        x_pred = np.arange(0, self.x_range, self.dx).astype(np.float32)
        X_pred = np.stack([x_pred**i for i in reversed(range(1, self.order+1))]).T
        
        y_pred_l = self.ransac_left.predict(X_pred)
        y_pred_r = self.ransac_right.predict(X_pred)


        if y_left.shape[0]>=self.ransac_left.min_samples and y_right.shape[0]>=self.ransac_right.min_samples:

            self.update_lane_width(y_pred_l, y_pred_r)

        if y_left.shape[0]<self.ransac_left.min_samples:
            
            y_pred_l = y_pred_r + self.lane_width

        if y_right.shape[0]<self.ransac_right.min_samples:

            y_pred_r = y_pred_l - self.lane_width
    

        if len(y_pred_l) == len(y_pred_r):

            if np.mean(y_pred_l + y_pred_r):

                if y_pred_r[x_pred==3.0]>0:
                    
                    y_pred_r = y_pred_l - self.lane_width

                elif y_pred_l[x_pred==3.0]<0:
                    
                    y_pred_l = y_pred_r + self.lane_width

            else:

                pass
        
        else:

            pass

        return x_pred, y_pred_l, y_pred_r

    def update_lane_width(self, y_pred_l, y_pred_r):
        self.lane_width = np.clip(np.max(y_pred_l-y_pred_r), 3.5, 5)
    
    def write_path_msg(self, x_pred, y_pred_l, y_pred_r,frame_id='/map'):
        self.lane_path = Path()

        trans_matrix = np.array([
                                [math.cos(self.vehicle_yaw), -math.sin(self.vehicle_yaw),self.vehicle_pos_x],
                                [math.sin(self.vehicle_yaw),  math.cos(self.vehicle_yaw),self.vehicle_pos_y],
                                [0,0,1]])

        self.lane_path.header.frame_id=frame_id

        for i in range(len(x_pred)) :

            local_result=np.array([[x_pred[i]],[(0.5)*(y_pred_l[i] + y_pred_r[i])],[1]])
            global_result=trans_matrix.dot(local_result)

            tmp_pose=PoseStamped()
            tmp_pose.pose.position.x = global_result[0][0]
            tmp_pose.pose.position.y = global_result[1][0]
            tmp_pose.pose.position.z = 0
            tmp_pose.pose.orientation.x = 0
            tmp_pose.pose.orientation.y = 0
            tmp_pose.pose.orientation.z = 0
            tmp_pose.pose.orientation.w = 1
            self.lane_path.poses.append(tmp_pose)

        return self.lane_path

    def set_vehicle_status(self, vehicle_status):

        odom_quaternion=(vehicle_status.pose.pose.orientation.x,vehicle_status.pose.pose.orientation.y,vehicle_status.pose.pose.orientation.z,vehicle_status.pose.pose.orientation.w)

        _,_,vehicle_yaw=tf.transformations.euler_from_quaternion(odom_quaternion)
        self.vehicle_yaw = vehicle_yaw
        self.vehicle_pos_x = vehicle_status.pose.pose.position.x
        self.vehicle_pos_y = vehicle_status.pose.pose.position.y


if __name__ == '__main__':

    rospy.init_node('lane_fitting', anonymous=True)

    image_parser = IMGParser()

    rospy.spin() 
