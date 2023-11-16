from threading import Thread
import pyrealsense2 as rs
import numpy as np
import math
import time
import cv2

class T265:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.rawImg1 = None
        self.rawImg2 = None
        self.Img1 = None
        self.Img2 = None
        self.psiRate = None
        self.vx = None
        self.vy = None
        self.vz = None
        self.ax = None
        self.ay = None
        self.az = None
        self.t1 = None
        self.dt = None
        self.isReceivingFrame = False
        self.isRunFrame = True
        self.frameThread = None
        self.frameStartTime = None
        self.frameCount = 0
        self.map1A = None
        self.map1B = None
        self.map2A = None
        self.map2B = None
        self.pipe = None
        self.start()

    def start(self):
        if False:
            print('Hello World!')
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.fisheye, 1)
        cfg.enable_stream(rs.stream.fisheye, 2)
        cfg.enable_stream(rs.stream.pose)
        self.pipe.start(cfg)
        self.createMaps()
        self.startFrameThread()

    def startFrameThread(self):
        if False:
            print('Hello World!')
        if self.frameThread == None:
            self.frameThread = Thread(target=self.acquireFrame)
            self.frameThread.start()
            print('T265 (Thread 0) start.')
            while self.isReceivingFrame != True:
                time.sleep(0.1)
            self.frameCount = 0
            self.frameStartTime = time.time()

    def acquireFrame(self):
        if False:
            i = 10
            return i + 15
        while self.isRunFrame:
            frames = self.pipe.wait_for_frames()
            f1 = frames.get_fisheye_frame(1)
            f2 = frames.get_fisheye_frame(2)
            pose = frames.get_pose_frame()
            if not f1 or not f2 or (not pose):
                continue
            if self.dt is None:
                self.dt = 1.0 / 30.0
                self.t1 = pose.timestamp
            else:
                t2 = pose.timestamp
                self.dt = (t2 - self.t1) / 1000.0
                self.t1 = t2
            self.rawImg1 = np.asanyarray(f1.get_data())
            self.rawImg2 = np.asanyarray(f2.get_data())
            self.psiRate = math.degrees(pose.get_pose_data().angular_velocity.y)
            self.vx = pose.get_pose_data().velocity.x
            self.vy = pose.get_pose_data().velocity.y
            self.vz = pose.get_pose_data().velocity.z
            self.ax = pose.get_pose_data().acceleration.x
            self.ay = pose.get_pose_data().acceleration.y
            self.az = pose.get_pose_data().acceleration.z
            self.Img1 = cv2.remap(self.rawImg1, self.map1A, self.map1B, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            self.Img2 = cv2.remap(self.rawImg2, self.map2A, self.map2B, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            self.frameCount += 1
            self.isReceivingFrame = True

    def cameraMatrix(self, intrinsics):
        if False:
            return 10
        return np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])

    def fisheyeDistortion(self, intrinsics):
        if False:
            return 10
        return np.array(intrinsics.coeffs[:4])

    def createMaps(self):
        if False:
            i = 10
            return i + 15
        profiles = self.pipe.get_active_profile()
        streams = {'f1': profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(), 'f2': profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
        intrinsics = {'f1': streams['f1'].get_intrinsics(), 'f2': streams['f2'].get_intrinsics()}
        K1 = self.cameraMatrix(intrinsics['f1'])
        D1 = self.fisheyeDistortion(intrinsics['f1'])
        K2 = self.cameraMatrix(intrinsics['f2'])
        D2 = self.fisheyeDistortion(intrinsics['f2'])
        (self.map1A, self.map1B) = cv2.fisheye.initUndistortRectifyMap(K1, D1, np.eye(3), K1, (848, 800), cv2.CV_16SC2)
        (self.map2A, self.map2B) = cv2.fisheye.initUndistortRectifyMap(K2, D2, np.eye(3), K2, (848, 800), cv2.CV_16SC2)

    def close(self):
        if False:
            return 10
        self.isRunFrame = False
        self.frameThread.join()
        print('\nThread 0 closed.')
        print('  Frame rate (T265): ', round(self.frameCount / (time.time() - self.frameStartTime), 1))
        self.pipe.stop()

def main():
    if False:
        while True:
            i = 10
    cam = T265()
    cv2.imwrite('Cam1.png', cam.Img1)
    cv2.imwrite('Cam2.png', cam.Img2)
    cam.close()
if __name__ == '__main__':
    main()