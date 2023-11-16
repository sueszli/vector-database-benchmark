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
import cv2
import time
import queue
import numpy as np
import pytest
import logging as log
import platform
import tempfile
from vidgear.gears import CamGear
from vidgear.gears.helper import logger_handler
logger = log.getLogger('Test_camgear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)
_windows = True if os.name == 'nt' else False

def return_youtubevideo_params(url):
    if False:
        return 10
    '\n    returns Youtube Video parameters(FPS, dimensions) directly using Youtube-dl\n    '
    import yt_dlp
    ydl = yt_dlp.YoutubeDL({'outtmpl': '%(id)s%(ext)s', 'noplaylist': True, 'quiet': True, 'format': 'bestvideo'})
    with ydl:
        result = ydl.extract_info(url, download=False)
    return (int(result['width']), int(result['height']), float(result['fps']))

def return_testvideo_path():
    if False:
        while True:
            i = 10
    '\n    returns Test Video path\n    '
    path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
    return os.path.abspath(path)

def return_total_frame_count():
    if False:
        i = 10
        return i + 15
    '\n    simply counts the total frames in a given video\n    '
    stream = cv2.VideoCapture(return_testvideo_path())
    num_cv = 0
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            logger.debug('Total frames: {}'.format(num_cv))
            break
        num_cv += 1
    stream.release()
    return num_cv
test_data = [(return_testvideo_path(), {'THREAD_TIMEOUT': 300, 'CAP_PROP_FRAME_WIDTH ': 320, 'CAP_PROP_FRAME_HEIGHT': 240}), (return_testvideo_path(), {'THREAD_TIMEOUT': 'wrong', 'im_wrong': True, 'THREADED_QUEUE_MODE': False}), ('im_not_a_source.mp4', {'THREADED_QUEUE_MODE': 'invalid'})]

@pytest.mark.parametrize('source, options', test_data)
def test_threaded_queue_mode(source, options):
    if False:
        return 10
    '\n    Test for the Thread Queue Mode in CamGear API\n    '
    try:
        if platform.system() == 'Linux':
            stream_camgear = CamGear(source=source, backend=cv2.CAP_FFMPEG, logging=True, **options).start()
        else:
            stream_camgear = CamGear(source=source, logging=True, **options).start()
        camgear_frames_num = 0
        while True:
            frame = stream_camgear.read()
            if frame is None:
                logger.debug('VidGear Total frames: {}'.format(camgear_frames_num))
                break
            time.sleep(0.2)
            camgear_frames_num += 1
        stream_camgear.stop()
        actual_frame_num = return_total_frame_count()
        if 'THREADED_QUEUE_MODE' in options and (not options['THREADED_QUEUE_MODE']):
            assert camgear_frames_num < actual_frame_num
        else:
            assert camgear_frames_num == actual_frame_num
    except Exception as e:
        if isinstance(e, RuntimeError) and source == 'im_not_a_source.mp4':
            pass
        else:
            pytest.fail(str(e))

@pytest.mark.parametrize('url, quality, parameters', [('https://www.youtube.com/playlist?list=PLXsatjadpxK5wpQVrWKSxu4_ItvpwfCby', '720p', 'invalid'), ('https://youtu.be/uCy5OuSQnyA', '73p', 'invalid'), ('https://youtu.be/viOkh9al0xM', '720p', 'invalid'), ('https://www.dailymotion.com/video/x2yrnum', 'invalid', {'nocheckcertificate': True}), ('im_not_a_url', '', {})])
def test_stream_mode(url, quality, parameters):
    if False:
        print('Hello World!')
    '\n    Testing Stream Mode Playback capabilities of CamGear\n    '
    try:
        height = 0
        width = 0
        fps = 0
        options = {'STREAM_RESOLUTION': quality, 'STREAM_PARAMS': parameters}
        stream = CamGear(source=url, stream_mode=True, logging=True, **options).start()
        while True:
            frame = stream.read()
            if frame is None:
                break
            if height == 0 or width == 0:
                fps = stream.framerate
                (height, width) = frame.shape[:2]
                break
        stream.stop()
        logger.debug('WIDTH: {} HEIGHT: {} FPS: {}'.format(width, height, fps))
    except Exception as e:
        if isinstance(e, (RuntimeError, ValueError, cv2.error)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))

def test_network_playback():
    if False:
        i = 10
        return i + 15
    '\n    Testing Direct Network Video Playback capabilities of VidGear(with rtsp streaming)\n    '
    Publictest_rtsp_urls = ['rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov', 'rtsp://freja.hiof.no:1935/rtplive/definst/hessdalen03.stream', 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa', 'rtmp://semerkandglb.mediatriple.net:1935/semerkandliveedge/semerkand2']
    index = 0
    while index < len(Publictest_rtsp_urls):
        try:
            output_stream = CamGear(source=Publictest_rtsp_urls[index], logging=True).start()
            i = 0
            Output_data = []
            while i < 10:
                frame = output_stream.read()
                if frame is None:
                    break
                Output_data.append(frame)
                i += 1
            output_stream.stop()
            logger.debug('Output data shape:', np.array(Output_data).shape)
            if Output_data[-1].shape[:2] > (50, 50):
                break
        except Exception as e:
            if isinstance(e, RuntimeError):
                logger.debug('`{}` URL is not working'.format(Publictest_rtsp_urls[index]))
                index += 1
                continue
            else:
                pytest.fail(str(e))
    if index == len(Publictest_rtsp_urls):
        pytest.xfail('Test failed to play any URL!')

@pytest.mark.parametrize('conversion', ['COLOR_BGR2GRAY', 'COLOR_BGR2INVALID', 'COLOR_BGR2BGRA'])
def test_colorspaces(conversion):
    if False:
        for i in range(10):
            print('nop')
    '\n    Testing different colorspace with CamGear API.\n    '
    try:
        options = {'THREAD_TIMEOUT': 300}
        stream = CamGear(source=return_testvideo_path(), colorspace=conversion, logging=True, **options).start()
        while True:
            frame = stream.read()
            if frame is None:
                break
            if conversion == 'COLOR_BGR2INVALID':
                stream.color_space = conversion
                conversion = 'COLOR_BGR2INVALID2'
            if conversion == 'COLOR_BGR2INVALID2':
                stream.color_space = 1546755546
                conversion = ''
        stream.stop()
    except Exception as e:
        if not isinstance(e, (AssertionError, queue.Empty)):
            pytest.fail(str(e))
        else:
            logger.exception(str(e))