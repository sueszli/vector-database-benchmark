import cv2
from glob import glob
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import consts as c
import funcs as func
root_p = os.getcwd()

class StereoCal(object):

    def __init__(self, rms=None, cameraMatrix1=None, distCoeffs1=None, cameraMatrix2=None, distCoeffs2=None, R=None, T=None, E=None, F=None, R1=None, R2=None, P1=None, P2=None, Q=None, validPixROI1=None, validPixROI2=None):
        if False:
            for i in range(10):
                print('nop')
        self.rms = rms
        self.cameraMatrix1 = cameraMatrix1
        self.distCoeffs1 = distCoeffs1
        self.cameraMatrix2 = cameraMatrix2
        self.distCoeffs2 = distCoeffs2
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.validPixROI1 = validPixROI1
        self.validPixROI2 = validPixROI2

    def save_params(self, filename):
        if False:
            i = 10
            return i + 15
        with open(filename, 'wb') as f:
            np.savez(f, rms=self.rms, cameraMatrix1=self.cameraMatrix1, distCoeffs1=self.distCoeffs1, cameraMatrix2=self.cameraMatrix2, distCoeffs2=self.distCoeffs2, R=self.R, T=self.T, E=self.E, F=self.F, R1=self.R1, R2=self.R2, P1=self.P1, P2=self.P2, Q=self.Q, validPixROI1=self.validPixROI1, validPixROI2=self.validPixROI2)
            print(f'{filename} saved successfully')

    def load_params(self, filename):
        if False:
            for i in range(10):
                print('nop')
        try:
            with open(filename, 'rb') as f:
                myfile = np.load(f)
                print(f'{filename} loaded successfully')
                self.rms = myfile['rms']
                self.cameraMatrix1 = myfile['cameraMatrix1']
                self.distCoeffs1 = myfile['distCoeffs1']
                self.cameraMatrix2 = myfile['cameraMatrix2']
                self.distCoeffs2 = myfile['distCoeffs2']
                self.R = myfile['R']
                self.T = myfile['T']
                self.E = myfile['E']
                self.F = myfile['F']
                self.R1 = myfile['R1']
                self.R2 = myfile['R2']
                self.P1 = myfile['P1']
                self.P2 = myfile['P2']
                self.Q = myfile['Q']
                self.validPixROI1 = myfile['validPixROI1']
                self.validPixROI2 = myfile['validPixROI2']
                return True
        except OSError:
            print(f'{filename} does not exist')
        return None

    def get_params(self, camera_name):
        if False:
            print('Hello World!')
        if camera_name == c.LEFT_CLIENT:
            return (self.cameraMatrix1, self.distCoeffs1, self.R1, self.P1, self.validPixROI1)
        elif camera_name == c.RIGHT_CLIENT:
            return (self.cameraMatrix2, self.distCoeffs2, self.R2, self.P2, self.validPixROI2)
        else:
            return None

class CamCal(object):

    def __init__(self, rms=None, camera_matrix=None, dist_coefs=None, rvecs=None, tvecs=None):
        if False:
            i = 10
            return i + 15
        self.rms = rms
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.rvecs = rvecs
        self.tvecs = tvecs

    def save_params(self, filename):
        if False:
            while True:
                i = 10
        with open(filename, 'wb') as f:
            np.savez(f, rms=self.rms, camera_matrix=self.camera_matrix, dist_coefs=self.dist_coefs, rvecs=self.rvecs, tvecs=self.tvecs)
            print(f'{filename} saved successfully')

    def load_params(self, filename):
        if False:
            i = 10
            return i + 15
        try:
            os.chdir(func.make_path(root_p, c.DATA_DIR, c.CALIB_DIR, c.ACTIVE_CALIB_DIR))
            with open(filename, 'rb') as f:
                myfile = np.load(f)
                print(f'{filename} loaded successfully')
                self.rms = myfile['rms']
                self.camera_matrix = myfile['camera_matrix']
                self.dist_coefs = myfile['dist_coefs']
                self.rvecs = myfile['rvecs']
                self.tvecs = myfile['tvecs']
                return True
        except OSError:
            raise ValueError(f'{filename} does not exist')

def load_calibs():
    if False:
        while True:
            i = 10
    os.chdir(func.make_path(root_p, c.DATA_DIR, c.CALIB_DIR, c.ACTIVE_CALIB_DIR))
    names = os.listdir()
    left_cal = CamCal()
    right_cal = CamCal()
    for name in names:
        if c.LEFT_CALIB_F in name:
            left_cal.load_params(name)
        elif c.RIGHT_CALIB_F in name:
            right_cal.load_params(name)
    return (left_cal, right_cal)

def save_calib(calib, camera_name):
    if False:
        i = 10
        return i + 15
    os.chdir(c.DATA_P)
    calib.save_params(f'{calib.rms:0.4f}{camera_name}')
    os.chdir(c.ROOT_P)

def load_stereo_calib():
    if False:
        i = 10
        return i + 15
    os.chdir(c.STEREO_CALIB_P)
    names = os.listdir()
    s_cal = StereoCal()
    if c.ACTIVE_STEREO_F in names:
        s_cal.load_params(c.ACTIVE_STEREO_F)
    os.chdir(c.ROOT_P)
    return s_cal

def save_stereo_calib(stereo_calib):
    if False:
        for i in range(10):
            print('nop')
    os.chdir(func.make_path(root_p, c.DATA_DIR, c.STEREO_CALIB_DIR))
    stereo_calib.save_params(f'{stereo_calib.rms:0.4f}{c.STEREO_CALIB_F}')

def process_image(img_data, pattern_points):
    if False:
        i = 10
        return i + 15
    (n_frame, img) = img_data
    (found, corners) = cv2.findChessboardCorners(img, c.PATTERN_SIZE)
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    if not found:
        print(n_frame)
        print('chessboard not found')
        return None
    return (n_frame, corners.reshape(-1, 2), pattern_points)

def generate_pattern_points():
    if False:
        while True:
            i = 10
    pattern_points = np.zeros((np.prod(c.PATTERN_SIZE), 3), np.float32)
    pattern_points[:, :2] = np.indices(c.PATTERN_SIZE).T.reshape(-1, 2)
    pattern_points *= c.SQUARE_SIZE
    return pattern_points

def find_chessboards(img_data):
    if False:
        print('Hello World!')
    pattern_points = generate_pattern_points()
    chessboards = [process_image(img, pattern_points) for img in img_data]
    chessboards = [x for x in chessboards if x is not None]
    return chessboards

def validate_chessboards(left_chessboards, right_chessboards):
    if False:
        for i in range(10):
            print('nop')
    if len(left_chessboards) > 0 and len(right_chessboards) > 0:
        max_len = min(len(left_chessboards), len(right_chessboards))
        left_chessboards = left_chessboards[:max_len]
        right_chessboards = right_chessboards[:max_len]
        new_left_chessboards = []
        new_right_chessboards = []
        for (i, left_chessboard) in enumerate(left_chessboards):
            if left_chessboards[i][0] == right_chessboards[i][0]:
                new_left_chessboards.append(left_chessboards[i])
                new_right_chessboards.append(right_chessboards[i])
            else:
                for (j, right_chessboard) in enumerate(right_chessboards):
                    if left_chessboards[i][0] == right_chessboards[j][0]:
                        new_left_chessboards.append(left_chessboards[i])
                        new_right_chessboards.append(right_chessboards[j])
        return (new_left_chessboards, new_right_chessboards)
    return False

def calibrate_stereo(left_chessboards, right_chessboards, left_cam, right_cam, size):
    if False:
        for i in range(10):
            print('nop')
    object_points = []
    left_image_points = []
    right_image_points = []
    for (n_frame, image_points, obj_points) in left_chessboards:
        left_image_points.append(image_points)
        object_points.append(obj_points)
    for (n_frame, image_points, obj_points) in right_chessboards:
        right_image_points.append(image_points)
    term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-06)
    (RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F) = (None, None, None, None, None, None, None, None, None)
    if len(left_chessboards) > 8:
        (RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F) = cv2.stereoCalibrate(object_points, left_image_points, right_image_points, left_cam.camera_matrix, left_cam.dist_coefs, right_cam.camera_matrix, right_cam.dist_coefs, size, criteria=term_crit, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    else:
        print('there is not enough chessboard views for calibration, please repeat')
    return (RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F)

def calibrate_mono_local(camera_name, img_directory):
    if False:
        print('Hello World!')
    os.chdir(c.IMG_P + '//' + img_directory)
    img_list = glob('*****.png')
    pattern_points = generate_pattern_points()
    chessboards = [process_image((None, cv2.imread(img, cv2.IMREAD_GRAYSCALE)), pattern_points) for img in img_list]
    chessboards = [x for x in chessboards if x is not None]
    obj_points = []
    img_points = []
    for (frames, corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)
    term = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-06)
    (rms, camera_matrix, dist_coefs, rvecs, tvecs) = cv2.calibrateCamera(obj_points, img_points, c.RESOLUTION, distCoeffs=None, cameraMatrix=None, criteria=term)
    cal = CamCal(rms, camera_matrix, dist_coefs, rvecs, tvecs)
    save_calib(cal, camera_name)
    print(cal.rms)
    print(camera_matrix)

def calibrate_stereo_local():
    if False:
        return 10
    os.chdir(func.make_path(root_p, c.IMG_DIR, c.CALIB_IMG_S_DIR))
    img_list = os.listdir()
    left_img_data = []
    right_img_data = []
    size = 0
    for img_name in img_list:
        if 'l' in img_name or 'left' in img_name:
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            frame_n = img_name[-8:-4]
            left_img_data.append((frame_n, img))
        elif 'r' in img_name or 'right' in img_name:
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            frame_n = img_name[-8:-4]
            right_img_data.append((frame_n, img))
    print('left')
    left_chessboards = find_chessboards(left_img_data)
    print('right')
    right_chessboards = find_chessboards(right_img_data)
    (left_chessboards, right_chessboards) = validate_chessboards(left_chessboards, right_chessboards)
    (left_cal, right_cal) = load_calibs()
    (w, h) = c.RESOLUTION
    (RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F) = calibrate_stereo(left_chessboards, right_chessboards, left_cal, right_cal, (h, w))
    (R1, R2, P1, P2, Q, validPixROI1, validPixROI2) = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (h, w), R, T)
    s_cal = StereoCal(RMS, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, R1, R2, P1, P2, Q, validPixROI1, validPixROI2)
    save_stereo_calib(s_cal)
    (Lx, Ly) = (378.5, 304.5)
    (Rx, Ry) = (259.6, 298.6)
    LPointsd = np.array([[Lx, Ly]], dtype=np.float32).reshape(-1, 1, 2)
    RPointsd = np.array([[Rx, Ry]], dtype=np.float32).reshape(-1, 1, 2)
    LPointsu = cv2.undistortPoints(LPointsd, cameraMatrix1, distCoeffs1, R=R1, P=P1)
    RPointsu = cv2.undistortPoints(RPointsd, cameraMatrix2, distCoeffs2, R=R2, P=P2)
    points4d = cv2.triangulatePoints(P1, P2, LPointsu, RPointsu)
    points3d = np.array([i / points4d[3] for i in points4d[:3]])
    print(points3d)
    return True
if __name__ == '__main__':
    calibrate_stereo_local()