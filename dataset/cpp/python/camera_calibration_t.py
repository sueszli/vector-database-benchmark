import cv2 as cv
from glob import glob
import numpy as np

import pickle
import os

import matplotlib.pyplot as plt

import consts_t as c

class CamCal(object):
    def __init__(self, rms, camera_matrix, dist_coefs, rvecs, tvecs):
        self.rms = rms
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.rvecs = rvecs
        self.tvecs = tvecs

def save_params(filename, rms, camera_matrix, dist_coefs, rvecs, tvecs):
    with open(filename, 'wb') as f:
        np.savez(f, rms=rms, camera_matrix=camera_matrix, dist_coefs=dist_coefs, rvecs=rvecs, tvecs=tvecs)

def load_params(filename):
    with open(filename, 'rb') as f:
        myfile = np.load(f)
        cam_cal = CamCal(myfile['rms'],myfile['camera_matrix'],myfile['dist_coefs'],myfile['rvecs'],myfile['tvecs'])
        return cam_cal

def process_image(img_data, pattern_points):

    n_frame, img = img_data

    # check to ensure that the image matches the width/height of the initial image
    # assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ..." % (img.shape[1], img.shape[0]))

    found, corners = cv.findChessboardCorners(img, c.pattern_size)
    if found:
        # term defines when to stop refinement of subpixel coords
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if not found:
        print('chessboard not found')
        return None

    # print message if file contains chessboard
    # print('%s... OK' % n_frame)
    return (n_frame, corners.reshape(-1, 2), pattern_points)

def generate_pattern_points():
    # -- Setup point lists -- #
    pattern_points = np.zeros((np.prod(c.pattern_size), 3), np.float32)  # x,y,z for all points in image
    pattern_points[:, :2] = np.indices(c.pattern_size).T.reshape(-1, 2)  # p.u for all point positions
    pattern_points *= c.square_size  # scale by square size for point coords
    return pattern_points


def find_chessboards(img_data):
    # frames = []
    # obj_points = []
    # img_points = []

    pattern_points = generate_pattern_points()

    # find the chessboard points in all images
    chessboards = [process_image(img, pattern_points) for img in img_data]
    chessboards = [x for x in chessboards if x is not None]

    # split the chessboards into image points and object points
    # for(n_frame, corners, pattern_points) in chessboards:
    #     frames.append(n_frame)
    #     img_points.append(corners)
    #     obj_points.append(pattern_points)

    return chessboards

if __name__ == "__main__":

    # sensor_size = (3.68, 2.76)
    # square_size = 23.4E-3
    # pattern_size = (9, 6)  # number of points (where the black and white intersects)
    # draw_corners_flg = False

    root = os.getcwd()
    img_dir = "calib_R"

    os.chdir(root)
    os.chdir(img_dir)

    # -- Read in all calibration images -- #
    img_list = glob('***.png')
    img_list.sort()

    # -- Setup point lists -- #
    pattern_points = np.zeros((np.prod(c.pattern_size), 3), np.float32)  # x,y,z for all points in image
    pattern_points[:, :2] = np.indices(c.pattern_size).T.reshape(-1, 2)  # p.u for all point positions
    pattern_points *= c.square_size  # scale by square size for point coords

    obj_points = []
    img_points = []
    h, w = cv.imread(img_list[0], cv.IMREAD_GRAYSCALE).shape[:2]  # get height and width of image

    # find the chessboard points in all images
    chessboards = [process_image(f, True) for f in img_list]
    chessboards = [x for x in chessboards if x is not None]

    # split the chessboards into image points and object points
    for(corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # -- Calculate camera distortion parameters -- #
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w,h), None, None)

    # -- Undistort an image -- #
    img_g = cv.imread(img_list[0], cv.IMREAD_GRAYSCALE)
    img_ud = cv.undistort(img_g, camera_matrix, dist_coefs, None, camera_matrix)

    os.chdir(root)
    save_params(f"{img_dir}.npy", rms, camera_matrix, dist_coefs, rvecs, tvecs)
    # with open('calib_L.npy', 'wb') as f:
    #     np.savez(f, rms=rms, camera_matrix=camera_matrix, dist_coefs=dist_coefs, rvecs=rvecs, tvecs=tvecs)
    my_cam = load_params(f"{img_dir}.npy")
    print(my_cam.camera_matrix)
    # print('\n\n')
    # with open('calib_L.npy', 'rb') as f:
    #     myfile = np.load(f)
    #     a,b,c,d,e = myfile['rms'],myfile['camera_matrix'],myfile['dist_coefs'],myfile['rvecs'],myfile['tvecs']


    fovx, fovy, focal_length, principal_point, aspect_ratio = cv.calibrationMatrixValues(camera_matrix, (w,h), c.sensor_size[0], c.
        c.sensor_size[1])
    # print(f"fovx: {fovx}\nfovy: {fovy}\nfocal length: {focal_length}\nprincipal point: {principal_point}\naspect ratio: {aspect_ratio}")    

    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img_g, cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.title('Undistorted') 
    plt.imshow(img_ud, cmap='gray')
    plt.axis('off')
    plt.show()
