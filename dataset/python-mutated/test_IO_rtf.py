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
import numpy as np
import pytest
from vidgear.gears import StreamGear

@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('size', [(480, 640, 5), [(480, 640, 1), (480, 640, 3)]])
def test_failedchannels(size):
    if False:
        while True:
            i = 10
    '\n    IO Test - made to fail with invalid channel lengths\n    '
    np.random.seed(0)
    if len(size) > 1:
        random_data_1 = np.random.random(size=size[0]) * 255
        input_data_ch1 = random_data_1.astype(np.uint8)
        random_data_2 = np.random.random(size=size[1]) * 255
        input_data_ch3 = random_data_2.astype(np.uint8)
        streamer = StreamGear('output.mpd', logging=True)
        streamer.stream(input_data_ch1)
        streamer.stream(input_data_ch3)
        streamer.terminate()
    else:
        random_data = np.random.random(size=size) * 255
        input_data = random_data.astype(np.uint8)
        streamer = StreamGear('output.mpd', logging=True)
        streamer.stream(input_data)
        streamer.terminate()

@pytest.mark.xfail(raises=ValueError)
def test_fail_framedimension():
    if False:
        return 10
    '\n    IO Test - made to fail with multiple frame dimension\n    '
    np.random.seed(0)
    random_data1 = np.random.random(size=(480, 640, 3)) * 255
    input_data1 = random_data1.astype(np.uint8)
    np.random.seed(0)
    random_data2 = np.random.random(size=(580, 640, 3)) * 255
    input_data2 = random_data2.astype(np.uint8)
    streamer = StreamGear(output='output.mpd')
    streamer.stream(None)
    streamer.stream(input_data1)
    streamer.stream(input_data2)
    streamer.terminate()

@pytest.mark.xfail(raises=RuntimeError)
def test_method_call_rtf():
    if False:
        i = 10
        return i + 15
    '\n    Method calling Test - Made to fail by calling method in the wrong context.\n    '
    stream_params = {'-video_source': 1234}
    streamer = StreamGear(output='output.mpd', logging=True, **stream_params)
    streamer.transcode_source()
    streamer.terminate()

@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('format', ['dash', 'hls'])
def test_invalid_params_rtf(format):
    if False:
        for i in range(10):
            print('nop')
    '\n    Invalid parameter Failure Test - Made to fail by calling invalid parameters\n    '
    np.random.seed(0)
    random_data = np.random.random(size=(480, 640, 3)) * 255
    input_data = random_data.astype(np.uint8)
    stream_params = {'-vcodec': 'unknown'}
    streamer = StreamGear(output='output{}'.format('.mpd' if format == 'dash' else '.m3u8'), format=format, logging=True, **stream_params)
    streamer.stream(input_data)
    streamer.stream(input_data)
    streamer.terminate()