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
import pytest
import asyncio
import platform
import logging as log
import requests
import tempfile
import json
import numpy as np
from starlette.routing import Route
from starlette.responses import PlainTextResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from async_asgi_testclient import TestClient
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription
from vidgear.gears import VideoGear
from aiortc.mediastreams import MediaStreamError
from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears.helper import logger_handler
logger = log.getLogger('Test_webgear_rtc')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

@pytest.fixture
def event_loop():
    if False:
        return 10
    'Create an instance of the default event loop for each test case.'
    loop = asyncio.SelectorEventLoop()
    yield loop
    loop.close()

def return_testvideo_path():
    if False:
        i = 10
        return i + 15
    '\n    returns Test Video path\n    '
    path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
    return os.path.abspath(path)

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """
    kind = 'video'

    def __init__(self, track):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        return frame

async def get_RTCPeer_payload():
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer('stun:stun.l.google.com:19302')]))

    @pc.on('track')
    async def on_track(track):
        logger.debug('Receiving %s' % track.kind)
        if track.kind == 'video':
            pc.addTrack(VideoTransformTrack(track))

        @track.on('ended')
        async def on_ended():
            logger.info('Track %s ended', track.kind)
    pc.addTransceiver('video', direction='recvonly')
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    new_offer = pc.localDescription
    payload = {'sdp': new_offer.sdp, 'type': new_offer.type}
    return (pc, json.dumps(payload, separators=(',', ':')))

def hello_webpage(request):
    if False:
        for i in range(10):
            print('nop')
    '\n    returns PlainTextResponse callback for hello world webpage\n    '
    return PlainTextResponse('Hello, world!')

class Custom_Stream_Class:
    """
    Custom Streaming using OpenCV
    """

    def __init__(self, source=0):
        if False:
            return 10
        self.stream = cv2.VideoCapture(source)
        self.running = True

    def read(self):
        if False:
            i = 10
            return i + 15
        if self.stream is None:
            return None
        if self.running:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                return frame
            else:
                self.running = False
        return None

    def stop(self):
        if False:
            print('Hello World!')
        self.running = False
        if not self.stream is None:
            self.stream.release()

class Custom_Grayscale_class:
    """
    Custom Grayscale class for producing `ndim==3` grayscale frames
    """

    def __init__(self):
        if False:
            return 10
        self.running = True
        self.counter = 0

    def read(self, size=(480, 640, 1)):
        if False:
            print('Hello World!')
        self.counter += 1
        if self.running:
            frame = np.random.randint(0, 255, size=size, dtype=np.uint8)
            if self.counter < 11:
                return frame
            else:
                self.running = False
        return None

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        self.running = False

class Invalid_Custom_Channel_Class:
    """
    Custom Invalid WebGear_RTC Server
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.running = True
        self.stream = Custom_Grayscale_class()

    def read(self):
        if False:
            return 10
        return self.stream.read(size=(480, 640, 5))

    def stop(self):
        if False:
            while True:
                i = 10
        self.running = False
        self.stream.stop()

class Invalid_Custom_Stream_Class:
    """
    Custom Invalid WebGear_RTC Server
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.running = True

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        self.running = False
test_data = [(None, False, None, 0), (return_testvideo_path(), True, None, 0), (return_testvideo_path(), False, 'COLOR_BGR2HSV', 1)]

@pytest.mark.skipif(platform.python_version_tuple()[:2] >= ('3', '11'), reason='Random Failures!')
@pytest.mark.asyncio
@pytest.mark.parametrize('source, stabilize, colorspace, time_delay', test_data)
async def test_webgear_rtc_class(source, stabilize, colorspace, time_delay):
    """
    Test for various WebGear_RTC API parameters
    """
    try:
        web = WebGear_RTC(source=source, stabilize=stabilize, colorspace=colorspace, time_delay=time_delay, logging=True)
        async with TestClient(web()) as client:
            response = await client.get('/')
            assert response.status_code == 200
            response_404 = await client.get('/test')
            assert response_404.status_code == 404
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post('/offer', data=data, headers={'Content-Type': 'application/json'})
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get('/offer', data=data, headers={'Content-Type': 'application/json'})
            assert response_rtc_offer.status_code == 200
            await offer_pc.close()
        web.shutdown()
    except Exception as e:
        if source and (not isinstance(e, MediaStreamError)):
            pytest.fail(str(e))
test_data = [{'frame_size_reduction': 47, 'overwrite_default_files': 'invalid_value', 'enable_infinite_frames': 'invalid_value', 'enable_live_broadcast': 'invalid_value', 'custom_data_location': True}, {'frame_size_reduction': 'invalid_value', 'enable_live_broadcast': False, 'custom_data_location': 'im_wrong'}, {'custom_data_location': tempfile.gettempdir(), 'enable_infinite_frames': False}, {'overwrite_default_files': True, 'enable_live_broadcast': True, 'frame_size_reduction': 99}]

@pytest.mark.skipif(platform.python_version_tuple()[:2] >= ('3', '11'), reason='Random Failures!')
@pytest.mark.asyncio
@pytest.mark.parametrize('options', test_data)
async def test_webgear_rtc_options(options):
    """
    Test for various WebGear_RTC API internal options
    """
    web = None
    try:
        web = WebGear_RTC(source=return_testvideo_path(), logging=True, **options)
        async with TestClient(web()) as client:
            response = await client.get('/')
            assert response.status_code == 200
            if not 'enable_live_broadcast' in options or options['enable_live_broadcast'] == False:
                (offer_pc, data) = await get_RTCPeer_payload()
                response_rtc_answer = await client.post('/offer', data=data, headers={'Content-Type': 'application/json'})
                params = response_rtc_answer.json()
                answer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
                await offer_pc.setRemoteDescription(answer)
                response_rtc_offer = await client.get('/offer', data=data, headers={'Content-Type': 'application/json'})
                assert response_rtc_offer.status_code == 200
                await offer_pc.close()
        web.shutdown()
    except Exception as e:
        if isinstance(e, (AssertionError, MediaStreamError)):
            logger.exception(str(e))
        elif isinstance(e, requests.exceptions.Timeout):
            logger.exceptions(str(e))
        else:
            pytest.fail(str(e))
test_data = [{'frame_size_reduction': 40}, {'enable_live_broadcast': True, 'frame_size_reduction': 40}]

@pytest.mark.skipif(platform.system() == 'Windows' or platform.python_version_tuple()[:2] >= ('3', '11'), reason='Random Failures!')
@pytest.mark.asyncio
@pytest.mark.parametrize('options', test_data)
async def test_webpage_reload(options):
    """
    Test for testing WebGear_RTC API against Webpage reload
    disruptions
    """
    web = WebGear_RTC(source=return_testvideo_path(), logging=True, **options)
    try:
        async with TestClient(web()) as client:
            response = await client.get('/')
            assert response.status_code == 200
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post('/offer', data=data, headers={'Content-Type': 'application/json'})
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get('/offer', data=data, headers={'Content-Type': 'application/json'})
            assert response_rtc_offer.status_code == 200
            response_rtc_reload = await client.post('/close_connection', data='0')
            await offer_pc.close()
            offer_pc = None
            data = None
            logger.debug(response_rtc_reload.text)
            assert response_rtc_reload.text == 'OK', 'Test Failed!'
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post('/offer', data=data, headers={'Content-Type': 'application/json'})
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get('/offer', data=data, headers={'Content-Type': 'application/json'})
            assert response_rtc_offer.status_code == 200
            await offer_pc.close()
    except Exception as e:
        if 'enable_live_broadcast' in options and isinstance(e, (AssertionError, MediaStreamError)):
            pytest.xfail('Test Passed')
        else:
            pytest.fail(str(e))
    finally:
        web.shutdown()
test_stream_classes = [(None, False), (Custom_Stream_Class(source=return_testvideo_path()), True), (VideoGear(source=return_testvideo_path(), colorspace='COLOR_BGR2BGRA', logging=True), True), (Custom_Grayscale_class(), False), (Invalid_Custom_Channel_Class(), False), (Invalid_Custom_Stream_Class(), False)]

@pytest.mark.skipif(platform.python_version_tuple()[:2] >= ('3', '11'), reason='Random Failures!')
@pytest.mark.asyncio
@pytest.mark.parametrize('stream_class, result', test_stream_classes)
async def test_webgear_rtc_custom_stream_class(stream_class, result):
    """
    Test for WebGear_RTC API's custom source
    """
    options = {'custom_stream': stream_class, 'frame_size_reduction': 0 if not result else 45}
    try:
        web = WebGear_RTC(logging=True, **options)
        async with TestClient(web()) as client:
            response = await client.get('/')
            assert response.status_code == 200
            response_404 = await client.get('/test')
            assert response_404.status_code == 404
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post('/offer', data=data, headers={'Content-Type': 'application/json'})
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get('/offer', data=data, headers={'Content-Type': 'application/json'})
            assert response_rtc_offer.status_code == 200
            await offer_pc.close()
        web.shutdown()
    except Exception as e:
        if result and (not isinstance(e, (ValueError, MediaStreamError))):
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))
test_data_class = [(None, False), ([Middleware(CORSMiddleware, allow_origins=['*'])], True), ([Route('/hello', endpoint=hello_webpage)], False)]

@pytest.mark.skipif(platform.python_version_tuple()[:2] >= ('3', '11'), reason='Random Failures!')
@pytest.mark.asyncio
@pytest.mark.parametrize('middleware, result', test_data_class)
async def test_webgear_rtc_custom_middleware(middleware, result):
    """
    Test for WebGear_RTC API's custom middleware
    """
    try:
        web = WebGear_RTC(source=return_testvideo_path(), logging=True)
        web.middleware = middleware
        async with TestClient(web()) as client:
            response = await client.get('/')
            assert response.status_code == 200
        web.shutdown()
    except Exception as e:
        if result and (not isinstance(e, MediaStreamError)):
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))

@pytest.mark.skipif(platform.python_version_tuple()[:2] >= ('3', '11'), reason='Random Failures!')
@pytest.mark.asyncio
async def test_webgear_rtc_routes():
    """
    Test for WebGear_RTC API's custom routes
    """
    try:
        options = {'frame_size_reduction': 40}
        web = WebGear_RTC(source=return_testvideo_path(), logging=True, **options)
        web.routes.append(Route('/hello', endpoint=hello_webpage))
        async with TestClient(web()) as client:
            response = await client.get('/')
            assert response.status_code == 200
            response_hello = await client.get('/hello')
            assert response_hello.status_code == 200
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post('/offer', data=data, headers={'Content-Type': 'application/json'})
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get('/offer', data=data, headers={'Content-Type': 'application/json'})
            assert response_rtc_offer.status_code == 200
            await offer_pc.close()
        web.shutdown()
    except Exception as e:
        if not isinstance(e, MediaStreamError):
            pytest.fail(str(e))

@pytest.mark.skipif(platform.python_version_tuple()[:2] >= ('3', '11'), reason='Random Failures!')
@pytest.mark.asyncio
async def test_webgear_rtc_routes_validity():
    """
    Test WebGear_RTC Routes
    """
    options = {'enable_infinite_frames': False, 'enable_live_broadcast': True}
    web = WebGear_RTC(source=return_testvideo_path(), logging=True)
    try:
        web.routes.clear()
        async with TestClient(web()) as client:
            pass
    except Exception as e:
        if isinstance(e, (RuntimeError, MediaStreamError)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        web.shutdown()