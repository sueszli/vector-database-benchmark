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
import time
import pytest
import logging as log
import platform
from vidgear.gears.helper import logger_handler
logger = log.getLogger('Test_pigear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

@pytest.mark.skipif(platform.system() != 'Linux', reason='Not Implemented')
def test_pigear_playback():
    if False:
        while True:
            i = 10
    "\n    Tests PiGear's playback capabilities\n    "
    try:
        from vidgear.gears import PiGear
        stream = PiGear(logging=True, colorspace='COLOR_BGR2GRAY').start()
        i = 0
        while i < 10:
            frame = stream.read()
            if frame is None:
                break
            i += 1
        stream.stop()
    except Exception as e:
        if isinstance(e, ImportError):
            logger.exception(e)
        else:
            pytest.fail(str(e))
test_data = [('invalid', None, '', 0, {}, None, AssertionError), (-1, 'invalid', '', 0.1, {}, None, AssertionError), (1, None, 'invalid', 0.1, {}, None, AssertionError), (0, (640, 480), 60, 0, {'HWFAILURE_TIMEOUT': 15.0}, None, ValueError), (0, (640, 480), 60, 'invalid', {'HWFAILURE_TIMEOUT': 'invalid'}, 'COLOR_BGR2INVALID', None), (0, (640, 480), 60, 1, {'create_bug': True}, 'None', RuntimeError), (0, (640, 480), 60, 0, {'create_bug': 'fail'}, 'None', RuntimeError), (-1, (640, 480), 60, 0, {'create_bug': ['fail']}, 'None', None), (0, (640, 480), 60, 0, {'HWFAILURE_TIMEOUT': 1.5, 'create_bug': 5}, 'COLOR_BGR2GRAY', SystemError)]

@pytest.mark.skipif(platform.system() != 'Linux', reason='Not Implemented')
@pytest.mark.parametrize('camera_num, resolution, framerate, time_delay, options, colorspace, exception_type', test_data)
def test_pigear_parameters(camera_num, resolution, framerate, time_delay, options, colorspace, exception_type):
    if False:
        for i in range(10):
            print('nop')
    "\n    Tests PiGear's options and colorspace.\n    "
    stream = None
    try:
        from vidgear.gears import PiGear
        stream = PiGear(camera_num=camera_num, resolution=resolution, framerate=framerate, logging=True, time_delay=time_delay, **options).start()
        i = 0
        while i < 20:
            frame = stream.read()
            if frame is None:
                break
            time.sleep(0.1)
            if i == 10:
                if colorspace == 'COLOR_BGR2INVALID':
                    stream.color_space = 1546755
                else:
                    stream.color_space = 'red'
            i += 1
    except Exception as e:
        if not exception_type is None and isinstance(e, exception_type):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if not stream is None:
            stream.stop()