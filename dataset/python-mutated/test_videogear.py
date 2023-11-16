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
import os
import sys
import pytest
import platform
import logging as log
import tempfile
from vidgear.gears import VideoGear
from vidgear.gears.helper import logger_handler
logger = log.getLogger('Test_videogear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)
_windows = True if os.name == 'nt' else False

def return_testvideo_path():
    if False:
        return 10
    '\n    returns Test video path\n    '
    path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
    return os.path.abspath(path)

@pytest.mark.skipif(platform.system() != 'Linux', reason='Not Implemented')
def test_PiGear_import():
    if False:
        i = 10
        return i + 15
    '\n    Testing VideoGear Import -> assign to fail when PiGear class is imported\n    '
    try:
        del sys.modules['picamera']
        del sys.modules['picamera.array']
    except KeyError:
        pass
    try:
        stream = VideoGear(enablePiCamera=True, logging=True).start()
        stream.stop()
    except Exception as e:
        if isinstance(e, ImportError):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
test_data = [('https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/example4_train_input.mp4', {'SMOOTHING_RADIUS': 5, 'BORDER_SIZE': 10, 'BORDER_TYPE': 'replicate', 'CROP_N_ZOOM': True}), ('https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/example_empty_train_input.mp4', {'SMOOTHING_RADIUS': 5, 'BORDER_SIZE': 15, 'BORDER_TYPE': 'reflect'}), ('https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/example4_train_input.mp4', {'SMOOTHING_RADIUS': '5', 'BORDER_SIZE': '15', 'BORDER_TYPE': ['reflect'], 'CROP_N_ZOOM': 'yes'}), (return_testvideo_path(), {'BORDER_TYPE': 'im_wrong'})]

@pytest.mark.parametrize('source, options', test_data)
def test_video_stablization(source, options):
    if False:
        for i in range(10):
            print('nop')
    "\n    Testing VideoGear's Video Stablization playback capabilities\n    "
    try:
        stab_stream = VideoGear(source=source, stabilize=True, logging=True, **options).start()
        framerate = stab_stream.framerate
        while True:
            frame = stab_stream.read()
            if frame is None:
                break
        stab_stream.stop()
        logger.debug('Input Framerate: {}'.format(framerate))
        assert framerate > 0
    except Exception as e:
        pytest.fail(str(e))