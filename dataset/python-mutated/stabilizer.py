"""
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
"""
import cv2
import numpy as np
import logging as log
from collections import deque
from .helper import logger_handler, check_CV_version, retrieve_best_interpolation, logcurr_vidgear_ver
logger = log.getLogger('Stabilizer')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

class Stabilizer:
    """
    This is an auxiliary class that enables Video Stabilization for vidgear with minimalistic latency, and at the expense
    of little to no additional computational requirements.

    The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses
    these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies
    heavily on **Threaded Queue mode** for error-free & ultra-fast frame handling.
    """

    def __init__(self, smoothing_radius=25, border_type='black', border_size=0, crop_n_zoom=False, logging=False):
        if False:
            return 10
        '\n        This constructor method initializes the object state and attributes of the Stabilizer class.\n\n        Parameters:\n            smoothing_radius (int): alter averaging window size.\n            border_type (str): changes the extended border type.\n            border_size (int): enables and set the value for extended border size to reduce the black borders.\n            crop_n_zoom (bool): enables cropping and zooming of frames(to original size) to reduce the black borders.\n            logging (bool): enables/disables logging.\n        '
        logcurr_vidgear_ver(logging=logging)
        self.__frame_queue = deque(maxlen=smoothing_radius)
        self.__frame_queue_indexes = deque(maxlen=smoothing_radius)
        self.__logging = False
        if logging:
            self.__logging = logging
        self.__clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.__smoothing_radius = smoothing_radius
        self.__smoothed_path = None
        self.__path = None
        self.__transforms = []
        self.__frame_transforms_smoothed = None
        self.__previous_gray = None
        self.__previous_keypoints = None
        (self.__frame_height, self.frame_width) = (0, 0)
        self.__crop_n_zoom = 0
        if crop_n_zoom and border_size:
            self.__crop_n_zoom = border_size
            self.__border_size = 0
            self.__frame_size = None
            if logging:
                logger.debug('Setting Cropping margin {} pixels'.format(border_size))
        else:
            self.__border_size = border_size
            if self.__logging and border_size:
                logger.debug('Setting Border size {} pixels'.format(border_size))
        border_modes = {'black': cv2.BORDER_CONSTANT, 'reflect': cv2.BORDER_REFLECT, 'reflect_101': cv2.BORDER_REFLECT_101, 'replicate': cv2.BORDER_REPLICATE, 'wrap': cv2.BORDER_WRAP}
        if border_type in ['black', 'reflect', 'reflect_101', 'replicate', 'wrap']:
            if not crop_n_zoom:
                self.__border_mode = border_modes[border_type]
                if self.__logging and border_type != 'black':
                    logger.debug('Setting Border type: {}'.format(border_type))
            else:
                if self.__logging and border_type != 'black':
                    logger.debug('Setting border type is disabled if cropping is enabled!')
                self.__border_mode = border_modes['black']
        else:
            if logging:
                logger.debug('Invalid input border type!')
            self.__border_mode = border_modes['black']
        self.__cv2_version = check_CV_version()
        self.__interpolation = retrieve_best_interpolation(['INTER_LINEAR_EXACT', 'INTER_LINEAR', 'INTER_AREA'])
        self.__box_filter = np.ones(smoothing_radius) / smoothing_radius

    def stabilize(self, frame):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method takes an unstabilized video frame, and returns a stabilized one.\n\n        Parameters:\n            frame (numpy.ndarray): inputs unstabilized video frames.\n        '
        if frame is None:
            return
        if self.__crop_n_zoom and self.__frame_size == None:
            self.__frame_size = frame.shape[:2]
        if not self.__frame_queue:
            previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous_gray = self.__clahe.apply(previous_gray)
            self.__previous_keypoints = cv2.goodFeaturesToTrack(previous_gray, maxCorners=200, qualityLevel=0.05, minDistance=30.0, blockSize=3, mask=None, useHarrisDetector=False, k=0.04)
            (self.__frame_height, self.frame_width) = frame.shape[:2]
            self.__frame_queue.append(frame)
            self.__frame_queue_indexes.append(0)
            self.__previous_gray = previous_gray[:]
        elif self.__frame_queue_indexes[-1] < self.__smoothing_radius - 1:
            self.__frame_queue.append(frame)
            self.__frame_queue_indexes.append(self.__frame_queue_indexes[-1] + 1)
            self.__generate_transformations()
        else:
            self.__frame_queue.append(frame)
            self.__frame_queue_indexes.append(self.__frame_queue_indexes[-1] + 1)
            self.__generate_transformations()
            for i in range(3):
                self.__smoothed_path[:, i] = self.__box_filter_convolve(self.__path[:, i], window_size=self.__smoothing_radius)
            deviation = self.__smoothed_path - self.__path
            self.__frame_transforms_smoothed = self.frame_transform + deviation
            return self.__apply_transformations()

    def __generate_transformations(self):
        if False:
            return 10
        '\n        An internal method that generate previous-to-current transformations [dx,dy,da].\n        '
        frame_gray = cv2.cvtColor(self.__frame_queue[-1], cv2.COLOR_BGR2GRAY)
        frame_gray = self.__clahe.apply(frame_gray)
        transformation = None
        try:
            (curr_kps, status, error) = cv2.calcOpticalFlowPyrLK(self.__previous_gray, frame_gray, self.__previous_keypoints, None)
            valid_curr_kps = curr_kps[status == 1]
            valid_previous_keypoints = self.__previous_keypoints[status == 1]
            if self.__cv2_version == 3:
                transformation = cv2.estimateRigidTransform(valid_previous_keypoints, valid_curr_kps, False)
            else:
                transformation = cv2.estimateAffinePartial2D(valid_previous_keypoints, valid_curr_kps)[0]
        except cv2.error as e:
            logger.warning('Video-Frame is too dark to generate any transformations!')
            transformation = None
        if not transformation is None:
            dx = transformation[0, 2]
            dy = transformation[1, 2]
            da = np.arctan2(transformation[1, 0], transformation[0, 0])
        else:
            dx = dy = da = 0
        self.__transforms.append([dx, dy, da])
        self.frame_transform = np.array(self.__transforms, dtype='float32')
        self.__path = np.cumsum(self.frame_transform, axis=0)
        self.__smoothed_path = np.copy(self.__path)
        self.__previous_keypoints = cv2.goodFeaturesToTrack(frame_gray, maxCorners=200, qualityLevel=0.05, minDistance=30.0, blockSize=3, mask=None, useHarrisDetector=False, k=0.04)
        self.__previous_gray = frame_gray[:]

    def __box_filter_convolve(self, path, window_size):
        if False:
            while True:
                i = 10
        '\n        An internal method that applies *normalized linear box filter* to path w.r.t averaging window\n\n        Parameters:\n\n        * path (numpy.ndarray): a cumulative sum of transformations\n        * window_size (int): averaging window size\n        '
        path_padded = np.pad(path, (window_size, window_size), 'median')
        path_smoothed = np.convolve(path_padded, self.__box_filter, mode='same')
        path_smoothed = path_smoothed[window_size:-window_size]
        assert path.shape == path_smoothed.shape
        return path_smoothed

    def __apply_transformations(self):
        if False:
            return 10
        '\n        An internal method that applies affine transformation to the given frame\n        from previously calculated transformations\n        '
        queue_frame = self.__frame_queue.popleft()
        queue_frame_index = self.__frame_queue_indexes.popleft()
        bordered_frame = cv2.copyMakeBorder(queue_frame, top=self.__border_size, bottom=self.__border_size, left=self.__border_size, right=self.__border_size, borderType=self.__border_mode, value=[0, 0, 0])
        alpha_bordered_frame = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2BGRA)
        alpha_bordered_frame[:, :, 3] = 0
        alpha_bordered_frame[self.__border_size:self.__border_size + self.__frame_height, self.__border_size:self.__border_size + self.frame_width, 3] = 255
        dx = self.__frame_transforms_smoothed[queue_frame_index, 0]
        dy = self.__frame_transforms_smoothed[queue_frame_index, 1]
        da = self.__frame_transforms_smoothed[queue_frame_index, 2]
        queue_frame_transform = np.zeros((2, 3), np.float32)
        queue_frame_transform[0, 0] = np.cos(da)
        queue_frame_transform[0, 1] = -np.sin(da)
        queue_frame_transform[1, 0] = np.sin(da)
        queue_frame_transform[1, 1] = np.cos(da)
        queue_frame_transform[0, 2] = dx
        queue_frame_transform[1, 2] = dy
        frame_wrapped = cv2.warpAffine(alpha_bordered_frame, queue_frame_transform, alpha_bordered_frame.shape[:2][::-1], borderMode=self.__border_mode)
        frame_stabilized = frame_wrapped[:, :, :3]
        if self.__crop_n_zoom:
            frame_cropped = frame_stabilized[self.__crop_n_zoom:-self.__crop_n_zoom, self.__crop_n_zoom:-self.__crop_n_zoom]
            frame_stabilized = cv2.resize(frame_cropped, self.__frame_size[::-1], interpolation=self.__interpolation)
        return frame_stabilized

    def clean(self):
        if False:
            i = 10
            return i + 15
        '\n        Cleans Stabilizer resources\n        '
        if self.__frame_queue:
            self.__frame_queue.clear()
            self.__frame_queue_indexes.clear()