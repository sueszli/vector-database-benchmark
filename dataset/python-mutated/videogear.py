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
import logging as log
from .helper import logger_handler, logcurr_vidgear_ver
from .camgear import CamGear
logger = log.getLogger('VideoGear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

class VideoGear:
    """
    VideoGear API provides a special internal wrapper around VidGear's exclusive Video Stabilizer class.
    VideoGear also acts as a Common Video-Capture API that provides internal access for both CamGear and PiGear APIs and their parameters with an exclusive enablePiCamera boolean flag.

    VideoGear is ideal when you need to switch to different video sources without changing your code much. Also, it enables easy stabilization for various video-streams (real-time or not)
    with minimum effort and writing way fewer lines of code.
    """

    def __init__(self, enablePiCamera=False, stabilize=False, camera_num=0, resolution=(640, 480), framerate=30, source=0, stream_mode=False, backend=0, time_delay=0, colorspace=None, logging=False, **options):
        if False:
            return 10
        "\n        This constructor method initializes the object state and attributes of the VideoGear class.\n\n        Parameters:\n            enablePiCamera (bool): provide access to PiGear(if True) or CamGear(if False) APIs respectively.\n            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.\n            camera_num (int): selects the camera module index which will be used as Rpi source.\n            resolution (tuple): sets the resolution (i.e. `(width,height)`) of the Rpi source.\n            framerate (int/float): sets the framerate of the Rpi source.\n            source (based on input): defines the source for the input stream.\n            stream_mode (bool): controls the exclusive YouTube Mode.\n            backend (int): selects the backend for OpenCV's VideoCapture class.\n            colorspace (str): selects the colorspace of the input stream.\n            logging (bool): enables/disables logging.\n            time_delay (int): time delay (in sec) before start reading the frames.\n            options (dict): provides ability to alter Tweak Parameters of CamGear, PiGear & Stabilizer.\n        "
        logcurr_vidgear_ver(logging=logging)
        self.__stablization_mode = stabilize
        self.__logging = False
        if logging:
            self.__logging = logging
        options = {str(k).strip(): v for (k, v) in options.items()}
        if self.__stablization_mode:
            from .stabilizer import Stabilizer
            s_radius = options.pop('SMOOTHING_RADIUS', 25)
            if not isinstance(s_radius, int):
                s_radius = 25
            border_size = options.pop('BORDER_SIZE', 0)
            if not isinstance(border_size, int):
                border_size = 0
            border_type = options.pop('BORDER_TYPE', 'black')
            if not isinstance(border_type, str):
                border_type = 'black'
            crop_n_zoom = options.pop('CROP_N_ZOOM', False)
            if not isinstance(crop_n_zoom, bool):
                crop_n_zoom = False
            self.__stabilizer_obj = Stabilizer(smoothing_radius=s_radius, border_type=border_type, border_size=border_size, crop_n_zoom=crop_n_zoom, logging=logging)
            self.__logging and logger.debug('Enabling Stablization Mode for the current video source!')
        if enablePiCamera:
            from .pigear import PiGear
            self.stream = PiGear(camera_num=camera_num, resolution=resolution, framerate=framerate, colorspace=colorspace, logging=logging, time_delay=time_delay, **options)
        else:
            self.stream = CamGear(source=source, stream_mode=stream_mode, backend=backend, colorspace=colorspace, logging=logging, time_delay=time_delay, **options)
        self.framerate = self.stream.framerate

    def start(self):
        if False:
            while True:
                i = 10
        '\n        Launches the internal *Threaded Frames Extractor* daemon of API in use.\n\n        **Returns:** A reference to the selected class object.\n        '
        self.stream.start()
        return self

    def read(self):
        if False:
            return 10
        "\n        Extracts frames synchronously from selected API's monitored deque, while maintaining a fixed-length frame\n        buffer in the memory, and blocks the thread if the deque is full.\n\n        **Returns:** A n-dimensional numpy array.\n        "
        while self.__stablization_mode:
            frame = self.stream.read()
            if frame is None:
                break
            frame_stab = self.__stabilizer_obj.stabilize(frame)
            if not frame_stab is None:
                return frame_stab
        return self.stream.read()

    def stop(self):
        if False:
            while True:
                i = 10
        '\n        Safely terminates the thread, and release the respective VideoStream resources.\n        '
        self.stream.stop()
        self.__logging and logger.debug('Terminating VideoGear.')
        if self.__stablization_mode:
            self.__stabilizer_obj.clean()