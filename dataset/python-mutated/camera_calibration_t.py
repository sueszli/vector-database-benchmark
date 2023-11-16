import cv2 as cv
from glob import glob
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import consts_t as c

class CamCal(object):

    def __init__(self, rms, camera_matrix, dist_coefs, rvecs, tvecs):
        if False:
            while True:
                i = 10
        self.rms = rms
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.rvecs = rvecs
        self.tvecs = tvecs

def save_params(filename, rms, camera_matrix, dist_coefs, rvecs, tvecs):
    if False:
        while True:
            i = 10
    with open(filename, 'wb') as f:
        np.savez(f, rms=rms, camera_matrix=camera_matrix, dist_coefs=dist_coefs, rvecs=rvecs, tvecs=tvecs)

def load_params(filename):
    if False:
        return 10
    with open(filename, 'rb') as f:
        myfile = np.load(f)
        cam_cal = CamCal(myfile['rms'], myfile['camera_matrix'], myfile['dist_coefs'], myfile['rvecs'], myfile['tvecs'])
        return cam_cal

def process_image(img_data, pattern_points):
    if False:
        i = 10
        return i + 15
    (n_frame, img) = img_data
    (found, corners) = cv.findChessboardCorners(img, c.pattern_size)
    if found:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    if not found:
        print('chessboard not found')
        return None
    return (n_frame, corners.reshape(-1, 2), pattern_points)

def generate_pattern_points():
    if False:
        for i in range(10):
            print('nop')
    pattern_points = np.zeros((np.prod(c.pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(c.pattern_size).T.reshape(-1, 2)
    pattern_points *= c.square_size
    return pattern_points

def find_chessboards(img_data):
    if False:
        for i in range(10):
            print('nop')
    pattern_points = generate_pattern_points()
    chessboards = [process_image(img, pattern_points) for img in img_data]
    chessboards = [x for x in chessboards if x is not None]
    return chessboards
if __name__ == '__main__':
    root = os.getcwd()
    img_dir = 'calib_R'
    os.chdir(root)
    os.chdir(img_dir)
    img_list = glob('***.png')
    img_list.sort()
    pattern_points = np.zeros((np.prod(c.pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(c.pattern_size).T.reshape(-1, 2)
    pattern_points *= c.square_size
    obj_points = []
    img_points = []
    (h, w) = cv.imread(img_list[0], cv.IMREAD_GRAYSCALE).shape[:2]
    chessboards = [process_image(f, True) for f in img_list]
    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)
    (rms, camera_matrix, dist_coefs, rvecs, tvecs) = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
    img_g = cv.imread(img_list[0], cv.IMREAD_GRAYSCALE)
    img_ud = cv.undistort(img_g, camera_matrix, dist_coefs, None, camera_matrix)
    os.chdir(root)
    save_params(f'{img_dir}.npy', rms, camera_matrix, dist_coefs, rvecs, tvecs)
    my_cam = load_params(f'{img_dir}.npy')
    print(my_cam.camera_matrix)
    (fovx, fovy, focal_length, principal_point, aspect_ratio) = cv.calibrationMatrixValues(camera_matrix, (w, h), c.sensor_size[0], c.c.sensor_size[1])
    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img_g, cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.title('Undistorted')
    plt.imshow(img_ud, cmap='gray')
    plt.axis('off')
    plt.show()