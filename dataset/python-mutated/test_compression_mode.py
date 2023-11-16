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
================================================
"""
import os
import re
import cv2
import pytest
import logging as log
import platform
import tempfile
import subprocess
from six import string_types
from deffcode import FFdecoder
from vidgear.gears import WriteGear
from vidgear.gears.helper import check_output, logger_handler
logger = log.getLogger('Test_commpression_mode')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)
_windows = True if os.name == 'nt' else False

def return_static_ffmpeg():
    if False:
        print('Hello World!')
    '\n    returns system specific FFmpeg static path\n    '
    path = ''
    if platform.system() == 'Windows':
        path += os.path.join(tempfile.gettempdir(), 'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe')
    elif platform.system() == 'Darwin':
        path += os.path.join(tempfile.gettempdir(), 'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg')
    else:
        path += os.path.join(tempfile.gettempdir(), 'Downloads/FFmpeg_static/ffmpeg/ffmpeg')
    return os.path.abspath(path)

def return_testvideo_path(fmt='av'):
    if False:
        return 10
    '\n    returns Test video path\n    '
    supported_fmts = {'av': 'BigBuckBunny_4sec.mp4', 'vo': 'BigBuckBunny_4sec_VO.mp4', 'ao': 'BigBuckBunny_4sec_AO.mp4'}
    req_fmt = fmt if fmt in supported_fmts else 'av'
    path = '{}/Downloads/Test_videos/{}'.format(tempfile.gettempdir(), supported_fmts[req_fmt])
    return os.path.abspath(path)

def remove_file_safe(path):
    if False:
        print('Hello World!')
    '\n    Remove file safely\n    '
    try:
        if path and os.path.isfile(os.path.abspath(path)):
            os.remove(path)
    except Exception as e:
        logger.exception(e)

def getFrameRate(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns framerate of video(at path provided) using FFmpeg\n    '
    process = subprocess.Popen([return_static_ffmpeg(), '-i', path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdout, _) = process.communicate()
    output = stdout.decode()
    match_dict = re.search('\\s(?P<fps>[\\d\\.]+?)\\stbr', output).groupdict()
    return float(match_dict['fps'])

def test_download_ffmpeg():
    if False:
        while True:
            i = 10
    '\n    Auxilary test to simply delete old ffmpeg binaries.\n    '
    try:
        import glob, shutil
        found = glob.glob(os.path.join(tempfile.gettempdir(), 'ffmpeg-static*'))
        if found and os.path.isdir(found[0]):
            shutil.rmtree(found[0])
    except Exception as e:
        if not isinstance(e, PermissionError):
            pytest.fail(str(e))

@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize('c_ffmpeg', [return_static_ffmpeg(), 'wrong_path'])
def test_input_framerate(c_ffmpeg):
    if False:
        print('Hello World!')
    '\n    Testing "-input_framerate" parameter provided by WriteGear(in Compression Mode)\n    '
    stream = cv2.VideoCapture(return_testvideo_path())
    test_video_framerate = stream.get(cv2.CAP_PROP_FPS)
    output_params = {'-input_framerate': test_video_framerate} if c_ffmpeg != 'wrong_path' else {'-input_framerate': 'wrong_input'}
    writer = WriteGear(output='Output_tif.mp4', custom_ffmpeg=c_ffmpeg, logging=True, **output_params)
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        writer.write(frame)
    stream.release()
    writer.close()
    output_video_framerate = getFrameRate(os.path.abspath('Output_tif.mp4'))
    assert test_video_framerate == output_video_framerate
    remove_file_safe('Output_tif.mp4')

@pytest.mark.parametrize('pixfmts', ['bgr24', 'rgba', 'invalid', 'invalid2', 'yuv444p', 'bgr48be'])
def test_write(pixfmts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Testing `frame_format` with different pixel formats.\n    '
    source = return_testvideo_path(fmt='vo')
    try:
        if pixfmts != 'invalid2':
            decoder = FFdecoder(source, frame_format=pixfmts, custom_ffmpeg=return_static_ffmpeg()).formulate()
            output_params = {'-input_pixfmt': pixfmts}
        else:
            decoder = FFdecoder(source, custom_ffmpeg=return_static_ffmpeg())
            decoder.metadata = dict(output_frames_pixfmt='yuvj422p')
            decoder.formulate()
            output_params = {'-input_pixfmt': 'yuvj422p'}
        writer = WriteGear(output='Output_tw.mp4', custom_ffmpeg=return_static_ffmpeg(), **output_params)
        for frame in decoder.generateFrame():
            writer.write(frame)
        decoder.terminate()
        writer.close()
        (basepath, _) = os.path.split(return_static_ffmpeg())
        ffprobe_path = os.path.join(basepath, 'ffprobe.exe' if os.name == 'nt' else 'ffprobe')
        assert os.path.isfile(ffprobe_path), 'FFprobe not Found!'
        result = check_output([ffprobe_path, '-v', 'error', '-count_frames', '-i', os.path.abspath('Output_tw.mp4')])
        if result:
            if not isinstance(result, string_types):
                result = result.decode()
            assert not any((x in result for x in ['Error', 'Invalid', 'error', 'invalid'])), 'Test failed!'
    except Exception as e:
        pytest.fail(str(e))
    finally:
        remove_file_safe('Output_tw.mp4')

@pytest.mark.xfail(raises=AssertionError)
def test_output_dimensions():
    if False:
        while True:
            i = 10
    '\n    Testing "-output_dimensions" special parameter provided by WriteGear(in Compression Mode)\n    '
    dimensions = (640, 480)
    stream = cv2.VideoCapture(return_testvideo_path())
    output_params = {}
    if platform.system() == 'Windows':
        output_params = {'-output_dimensions': dimensions, '-ffmpeg_download_path': tempfile.gettempdir(), '-disable_ffmpeg_window': True}
    else:
        output_params = {'-output_dimensions': dimensions}
    writer = WriteGear(output='Output_tod.mp4', custom_ffmpeg=return_static_ffmpeg(), logging=True, **output_params)
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        writer.write(frame)
    stream.release()
    writer.close()
    output = cv2.VideoCapture(os.path.abspath('Output_tod.mp4'))
    output_dim = (output.get(cv2.CAP_PROP_FRAME_WIDTH), output.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert output_dim[0] == 640 and output_dim[1] == 480
    output.release()
    remove_file_safe('Output_tod.mp4')
test_data_class = [('', '', {}, False), ('Output1.mp4', '', {}, True), (os.path.join(tempfile.gettempdir(), 'temp_write'), '', {}, True), ('Output2.mp4', '', {'-vcodec': 'libx264', '-crf': 0, '-preset': 'fast', '-ffpreheaders': False, '-disable_ffmpeg_window': True}, True), ('Output3.mp4', return_static_ffmpeg(), {'-c:v': 'libx265', '-vcodec': 'libx264', '-crf': 0, '-preset': 'veryfast', '-ffpreheaders': ['-re'], '-disable_ffmpeg_window': 'Invalid'}, True)]

@pytest.mark.parametrize('f_name, c_ffmpeg, output_params, result', test_data_class)
def test_WriteGear_compression(f_name, c_ffmpeg, output_params, result):
    if False:
        i = 10
        return i + 15
    '\n    Testing WriteGear Compression-Mode(FFmpeg) with different parameters\n    '
    try:
        stream = cv2.VideoCapture(return_testvideo_path())
        writer = WriteGear(output=f_name, compression_mode=True, **output_params)
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            writer.write(frame)
        stream.release()
        writer.close()
        remove_file_safe(f_name)
    except Exception as e:
        if result:
            pytest.fail(str(e))

@pytest.mark.parametrize('ffmpeg_cmd, logging, output_params', [(['-y', '-i', return_testvideo_path(), '-vn', '-acodec', 'copy', 'input_audio.aac'], False, {'-i': None, '-disable_force_termination': True}), (None, True, {'-i': None, '-disable_force_termination': 'OK'}), (['wrong_input', 'invalid_flag', 'break_things'], False, {}), (['wrong_input', 'invalid_flag', 'break_things'], True, {'-ffmpeg_download_path': 53} if platform.system() == 'Windows' else {'-disable_force_termination': 'OK'}), ('wrong_input', True, {'-disable_force_termination': True}), (['invalid'], True, {})])
def test_WriteGear_customFFmpeg(ffmpeg_cmd, logging, output_params):
    if False:
        while True:
            i = 10
    '\n    Testing WriteGear Compression-Mode(FFmpeg) custom FFmpeg Pipeline by seperating audio from video\n    '
    writer = None
    try:
        writer = WriteGear(output='Output.mp4', compression_mode=True if ffmpeg_cmd != ['invalid'] else False, logging=logging, **output_params)
        writer.execute_ffmpeg_cmd(ffmpeg_cmd)
        writer.close()
        if ffmpeg_cmd and isinstance(ffmpeg_cmd, list) and ('-acodec' in ffmpeg_cmd):
            assert os.path.isfile('input_audio.aac')
    except Exception as e:
        if isinstance(e, AssertionError):
            pytest.fail(str(e))
        elif isinstance(e, (ValueError, RuntimeError)):
            pytest.xfail('Test Passed!')
        else:
            logger.exception(str(e))