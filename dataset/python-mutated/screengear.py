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
import json
import platform
import numpy as np
import logging as log
from threading import Thread, Event
from collections import OrderedDict
from .helper import import_dependency_safe, capPropId, logger_handler, logcurr_vidgear_ver
mss = import_dependency_safe('from mss import mss', error='silent')
if not mss is None:
    from mss.exception import ScreenShotError
pysct = import_dependency_safe('pyscreenshot', error='silent')
dxcam = import_dependency_safe('dxcam', error='silent')
logger = log.getLogger('ScreenGear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

class ScreenGear:
    """
    ScreenGear is designed exclusively for targeting rapid Screencasting Capabilities, which means it can
    grab frames from your monitor in real-time, either by defining an area on the computer screen or full-screen,
    at the expense of inconsiderable latency. ScreenGear also seamlessly support frame capturing from multiple
    monitors as well as supports multiple backends.

    ScreenGear API implements a multi-threaded wrapper around dxcam, pyscreenshot, python-mss python library,
    and also flexibly supports its internal parameter.
    """

    def __init__(self, monitor=None, backend=None, colorspace=None, logging=False, **options):
        if False:
            i = 10
            return i + 15
        '\n        This constructor method initializes the object state and attributes of the ScreenGear class.\n\n        Parameters:\n            monitor (int): enables `mss` backend and sets the index of the monitor screen.\n            backend (str): select suitable backend for extracting frames.\n            colorspace (str): selects the colorspace of the input stream.\n            logging (bool): enables/disables logging.\n            options (dict): provides the flexibility to easily alter backend library parameters. Such as, manually set the dimensions of capture screen area etc.\n        '
        logcurr_vidgear_ver(logging=logging)
        self.__logging = logging if isinstance(logging, bool) else False
        self.__monitor_instance = None
        self.__backend = None
        assert monitor is None or (monitor and isinstance(monitor, (int, tuple))), '[ScreenGear:ERROR] :: Invalid `monitor` value detected!'
        if backend and monitor is None:
            self.__backend = backend.lower().strip()
        else:
            self.__backend = 'dxcam' if platform.system() == 'Windows' and dxcam else None
        screen_dims = {}
        screen_dims = {k.strip(): v for (k, v) in options.items() if k.strip() in ['top', 'left', 'width', 'height']}
        if screen_dims and len(screen_dims) == 4:
            key_order = ('top', 'left', 'width', 'height') if self.__backend != 'dxcam' else ('left', 'top', 'width', 'height')
            screen_dims = OrderedDict(((k, screen_dims[k]) for k in key_order))
            logging and logger.debug('Setting Capture-Area dimensions: {}'.format(json.dumps(screen_dims)))
        else:
            screen_dims.clear()
        if self.__backend == 'dxcam':
            self.__target_fps = options.pop('dxcam_target_fps', 0)
            if self.__target_fps and isinstance(self.__target_fps, (int, float)):
                self.__target_fps = int(self.__target_fps)
                logging and logger.debug('Setting Target FPS: {}'.format(self.__target_fps))
            else:
                self.__target_fps = 0
            assert platform.system() == 'Windows', '`dxcam` backend is only available for Windows Machines.'
            assert monitor is None or isinstance(monitor, int) or (isinstance(monitor, tuple) and len(monitor) == 2 and all((isinstance(x, int) for x in monitor))), 'For dxcam` backend, monitor` tuple value must be format `int` or `(int, int)` only.'
            import_dependency_safe('dxcam' if dxcam is None else '')
            if monitor is None:
                self.__capture_object = dxcam.create(region=tuple(screen_dims.values()) if screen_dims else None)
            else:
                self.__capture_object = dxcam.create(device_idx=monitor[0], output_idx=monitor[1], region=tuple(screen_dims.values()) if screen_dims else None) if isinstance(monitor, tuple) else dxcam.create(device_idx=monitor, region=tuple(screen_dims.values()) if screen_dims else None)
        elif monitor is None:
            import_dependency_safe('pyscreenshot' if pysct is None else '')
            self.__backend = 'pil' if self.__backend is None else self.__backend
            assert self.__backend in pysct.backends(), 'Unsupported backend {} provided!'.format(backend)
            self.__capture_object = pysct
        else:
            assert monitor and isinstance(monitor, int), '[ScreenGear:ERROR] :: Invalid `monitor` value must be integer with mss backend.'
            import_dependency_safe('from mss import mss' if mss is None else '', pkg_name='mss')
            self.__capture_object = mss()
            self.__backend and logger.warning('Backends are disabled for Monitor Indexing(monitor>=0)!')
            self.__monitor_instance = self.__capture_object.monitors[monitor]
        self.__backend and logging and logger.debug('Setting Backend: {}'.format(self.__backend.upper()))
        if colorspace:
            self.color_space = capPropId(colorspace.strip())
            logging and (not self.color_space is None) and logger.debug('Enabling `{}` colorspace for this video stream!'.format(colorspace.strip()))
        else:
            self.color_space = None
        self.__mss_capture_instance = None
        try:
            if self.__backend == 'dxcam':
                self.frame = self.__capture_object.grab()
            elif self.__monitor_instance is None:
                if screen_dims:
                    self.__mss_capture_instance = tuple(screen_dims.values())
                self.frame = np.asanyarray(self.__capture_object.grab(bbox=self.__mss_capture_instance, childprocess=False, backend=self.__backend))
            else:
                if screen_dims:
                    self.__mss_capture_instance = {'top': self.__monitor_instance['top'] + screen_dims['top'], 'left': self.__monitor_instance['left'] + screen_dims['left'], 'width': screen_dims['width'], 'height': screen_dims['height'], 'mon': monitor}
                else:
                    self.__mss_capture_instance = self.__monitor_instance
                self.frame = np.asanyarray(self.__capture_object.grab(self.__mss_capture_instance))
            self.frame = self.frame[:, :, ::-1] if self.__backend == 'dxcam' or not pysct is None else self.frame
            if not self.frame is None and (not self.color_space is None):
                self.frame = cv2.cvtColor(self.frame, self.color_space)
        except Exception as e:
            if isinstance(e, ScreenShotError):
                logging and logger.exception(self.__capture_object.get_error_details())
                raise ValueError('[ScreenGear:ERROR] :: ScreenShotError caught, Wrong dimensions passed to python-mss, Kindly Refer Docs!')
            else:
                raise SystemError('[ScreenGear:ERROR] :: Unable to grab any instance on this system, Are you running headless?')
        self.__thread = None
        self.__terminate = Event()

    def start(self):
        if False:
            return 10
        '\n        Launches the internal *Threaded Frames Extractor* daemon\n\n        **Returns:** A reference to the ScreenGear class object.\n        '
        self.__thread = Thread(target=self.__update, name='ScreenGear', args=())
        self.__thread.daemon = True
        self.__thread.start()
        if self.__backend == 'dxcam':
            self.__capture_object.start(target_fps=self.__target_fps, video_mode=True)
            self.__logging and self.__target_fps and logger.debug('Targeting FPS: {}'.format(self.__target_fps))
        return self

    def __update(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A **Threaded Frames Extractor**, that keep iterating frames from `mss` API to a internal monitored deque,\n        until the thread is terminated, or frames runs out.\n        '
        frame = None
        while not self.__terminate.is_set():
            try:
                if self.__backend == 'dxcam':
                    frame = self.__capture_object.get_latest_frame()
                elif self.__monitor_instance:
                    frame = np.asanyarray(self.__capture_object.grab(self.__mss_capture_instance))
                else:
                    frame = np.asanyarray(self.__capture_object.grab(bbox=self.__mss_capture_instance, childprocess=False, backend=self.__backend))
                assert not (frame is None or np.shape(frame) == ()), '[ScreenGear:ERROR] :: Failed to retrieve valid frame!'
                frame = frame[:, :, ::-1] if self.__backend == 'dxcam' or not pysct is None else frame
            except Exception as e:
                if isinstance(e, ScreenShotError):
                    raise RuntimeError(self.__capture_object.get_error_details())
                else:
                    logger.exception(str(e))
                self.__terminate.set()
                continue
            if not self.color_space is None:
                color_frame = None
                try:
                    if isinstance(self.color_space, int):
                        color_frame = cv2.cvtColor(frame, self.color_space)
                    else:
                        self.__logging and logger.warning('Global color_space parameter value `{}` is not a valid!'.format(self.color_space))
                        self.color_space = None
                except Exception as e:
                    self.color_space = None
                    if self.__logging:
                        logger.exception(str(e))
                        logger.warning('Input colorspace is not a valid colorspace!')
                if not color_frame is None:
                    self.frame = color_frame
                else:
                    self.frame = frame
            else:
                self.frame = frame
        self.__terminate.set()
        if self.__monitor_instance:
            self.__capture_object.close()
        if self.__backend == 'dxcam':
            self.__capture_object.stop()
            del self.__capture_object

    def read(self):
        if False:
            print('Hello World!')
        '\n        Extracts frames synchronously from monitored deque, while maintaining a fixed-length frame buffer in the memory,\n        and blocks the thread if the deque is full.\n\n        **Returns:** A n-dimensional numpy array.\n        '
        return self.frame

    def stop(self):
        if False:
            print('Hello World!')
        '\n        Safely terminates the thread, and release the resources.\n        '
        self.__logging and logger.debug('Terminating ScreenGear Processes.')
        self.__terminate.set()
        if self.__thread is not None:
            self.__thread.join()