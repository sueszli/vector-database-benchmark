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
import time
import fractions
import asyncio
import logging as log
from os.path import expanduser
from .helper import reducer, generate_webdata, create_blank_frame
from ..helper import logger_handler, retrieve_best_interpolation, import_dependency_safe, logcurr_vidgear_ver
from ..videogear import VideoGear
starlette = import_dependency_safe('starlette', error='silent')
if not starlette is None:
    from starlette.routing import Mount, Route
    from starlette.templating import Jinja2Templates
    from starlette.staticfiles import StaticFiles
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse, PlainTextResponse
logger = log.getLogger('WebGear_RTC')
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
aiortc = import_dependency_safe('aiortc', error='silent')
if not aiortc is None:
    from aiortc.rtcrtpsender import RTCRtpSender
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaRelay
    from aiortc.mediastreams import MediaStreamError
    from av import VideoFrame

    class RTC_VideoServer(VideoStreamTrack):
        """
        Default Internal Video-Server for WebGear_RTC,
        a inherit-class to aiortc's VideoStreamTrack API.
        """

        def __init__(self, enablePiCamera=False, stabilize=False, source=None, camera_num=0, stream_mode=False, backend=0, colorspace=None, resolution=(640, 480), framerate=25, logging=False, time_delay=0, **options):
            if False:
                for i in range(10):
                    print('nop')
            "\n            This constructor method initializes the object state and attributes of the RTC_VideoServer class.\n\n            Parameters:\n                enablePiCamera (bool): provide access to PiGear(if True) or CamGear(if False) APIs respectively.\n                stabilize (bool): enable access to Stabilizer Class for stabilizing frames.\n                camera_num (int): selects the camera module index which will be used as Rpi source.\n                resolution (tuple): sets the resolution (i.e. `(width,height)`) of the Rpi source.\n                framerate (int/float): sets the framerate of the Rpi source.\n                source (based on input): defines the source for the input stream.\n                stream_mode (bool): controls the exclusive YouTube Mode.\n                backend (int): selects the backend for OpenCV's VideoCapture class.\n                colorspace (str): selects the colorspace of the input stream.\n                logging (bool): enables/disables logging.\n                time_delay (int): time delay (in sec) before start reading the frames.\n                options (dict): provides ability to alter Tweak Parameters of WebGear_RTC, CamGear, PiGear & Stabilizer.\n            "
            logcurr_vidgear_ver(logging=logging)
            super().__init__()
            self.__logging = logging
            self.__enable_inf = False
            self.is_launched = False
            self.is_running = False
            self.__stream = None
            self.__frame_size_reduction = 20
            self.__interpolation = retrieve_best_interpolation(['INTER_LINEAR_EXACT', 'INTER_LINEAR', 'INTER_AREA'])
            if options:
                if 'frame_size_reduction' in options:
                    value = options['frame_size_reduction']
                    if isinstance(value, (int, float)) and value >= 0 and (value <= 90):
                        self.__frame_size_reduction = value
                    else:
                        logger.warning('Skipped invalid `frame_size_reduction` value!')
                    del options['frame_size_reduction']
                if 'enable_infinite_frames' in options:
                    value = options['enable_infinite_frames']
                    if isinstance(value, bool):
                        self.__enable_inf = value
                    else:
                        logger.warning('Skipped invalid `enable_infinite_frames` value!')
                    del options['enable_infinite_frames']
                if 'custom_stream' in options:
                    value = options['custom_stream']
                    if (hasattr(value, 'read') and callable(value.read)) and (hasattr(value, 'stop') and callable(value.stop)):
                        self.__stream = value
                        logger.critical('Using custom stream for its Default Internal Video-Server.')
                    else:
                        raise ValueError('[WebGear_RTC:ERROR] :: Invalid `custom_stream` value. Check VidGear docs!')
                    del options['custom_stream']
            if self.__stream is None:
                self.__stream = VideoGear(enablePiCamera=enablePiCamera, stabilize=stabilize, source=source, camera_num=camera_num, stream_mode=stream_mode, backend=backend, colorspace=colorspace, resolution=resolution, framerate=framerate, logging=logging, time_delay=time_delay, **options)
            self.__logging and logger.debug('Setting params:: Size Reduction:{}%{}'.format(self.__frame_size_reduction, ' and emulating infinite frames' if self.__enable_inf else ''))
            self.blank_frame = None
            self.__reset_enabled = False

        def launch(self):
            if False:
                while True:
                    i = 10
            '\n            Launches VideoGear stream\n            '
            self.__logging and logger.debug('Launching Internal RTC Video-Server')
            self.is_launched = True
            self.is_running = True
            if hasattr(self.__stream, 'start') and callable(self.__stream.start):
                self.__stream.start()

        async def next_timestamp(self):
            """
            VideoStreamTrack internal method for generating accurate timestamp.
            """
            if self.readyState != 'live':
                self.stop()
            if hasattr(self, '_timestamp') and (not self.__reset_enabled):
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + self._timestamp / VIDEO_CLOCK_RATE - time.time()
                await asyncio.sleep(wait)
            else:
                self.__logging and logger.debug('{} timestamps'.format('Resetting' if self.__reset_enabled else 'Setting'))
                self._start = time.time()
                self._timestamp = 0
                if self.__reset_enabled:
                    self.__reset_enabled = False
                    self.is_running = True
            return (self._timestamp, VIDEO_TIME_BASE)

        async def recv(self):
            """
            A coroutine function that yields `av.frame.Frame`.
            """
            (pts, time_base) = await self.next_timestamp()
            f_stream = None
            if self.__stream is None:
                raise MediaStreamError
            else:
                f_stream = self.__stream.read()
            if f_stream is None:
                if self.blank_frame is None or not self.is_running:
                    raise MediaStreamError
                else:
                    f_stream = self.blank_frame[:]
                if not self.__enable_inf and (not self.__reset_enabled):
                    self.__logging and logger.debug('Video-Stream Ended.')
                    self.terminate()
            elif self.blank_frame is None:
                self.blank_frame = create_blank_frame(frame=f_stream, text='No Input' if self.__enable_inf else 'The End', logging=self.__logging)
            if self.__frame_size_reduction:
                f_stream = await reducer(f_stream, percentage=self.__frame_size_reduction, interpolation=self.__interpolation)
            if f_stream.ndim == 3 and f_stream.shape[-1] == 4:
                f_format = 'bgra'
            elif f_stream.ndim == 2 or (f_stream.ndim == 3 and f_stream.shape[-1] == 1):
                f_stream = f_stream[:, :, 0] if f_stream.ndim == 3 and f_stream.shape[-1] == 1 else f_stream
                f_format = 'gray'
            elif f_stream.ndim == 3:
                f_format = 'bgr24'
            else:
                raise ValueError("Input frame of shape: {}, Isn't supported!".format(f_stream.shape))
            frame = VideoFrame.from_ndarray(f_stream, format=f_format)
            frame.pts = pts
            frame.time_base = time_base
            return frame

        async def reset(self):
            """
            Resets timestamp clock
            """
            self.__reset_enabled = True
            self.is_running = False

        def terminate(self):
            if False:
                i = 10
                return i + 15
            '\n            Gracefully terminates VideoGear stream\n            '
            if not self.__stream is None:
                self.is_running = False
                self.__logging and logger.debug('Terminating Internal RTC Video-Server')
                self.__stream.stop()
                self.__stream = None

class WebGear_RTC:
    """
    WebGear_RTC is similar to WeGear API in many aspects but utilizes WebRTC technology under the hood instead of Motion JPEG, which
    makes it suitable for building powerful video-streaming solutions for all modern browsers as well as native clients available on
    all major platforms.

    WebGear_RTC is implemented with the help of aiortc library which is built on top of asynchronous I/O framework for Web Real-Time
    Communication (WebRTC) and Object Real-Time Communication (ORTC) and supports many features like SDP generation/parsing, Interactive
    Connectivity Establishment with half-trickle and mDNS support, DTLS key and certificate generation, DTLS handshake, etc.

    WebGear_RTC can handle multiple consumers seamlessly and provides native support for ICE (Interactive Connectivity Establishment)
    protocol, STUN (Session Traversal Utilities for NAT), and TURN (Traversal Using Relays around NAT) servers that help us to easily
    establish direct media connection with the remote peers for uninterrupted data flow. It also allows us to define our custom Server
    as a source to transform frames easily before sending them across the network(see this doc example).

    WebGear_RTC API works in conjunction with Starlette ASGI application and can also flexibly interact with Starlette's ecosystem of
    shared middleware, mountable applications, Response classes, Routing tables, Static Files, Templating engine(with Jinja2), etc.

    Additionally, WebGear_RTC API also provides internal wrapper around VideoGear, which itself provides internal access to both
    CamGear and PiGear APIs.
    """

    def __init__(self, enablePiCamera=False, stabilize=False, source=None, camera_num=0, stream_mode=False, backend=0, colorspace=None, resolution=(640, 480), framerate=25, logging=False, time_delay=0, **options):
        if False:
            i = 10
            return i + 15
        "\n        This constructor method initializes the object state and attributes of the WebGear_RTC class.\n\n        Parameters:\n            enablePiCamera (bool): provide access to PiGear(if True) or CamGear(if False) APIs respectively.\n            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.\n            camera_num (int): selects the camera module index which will be used as Rpi source.\n            resolution (tuple): sets the resolution (i.e. `(width,height)`) of the Rpi source.\n            framerate (int/float): sets the framerate of the Rpi source.\n            source (based on input): defines the source for the input stream.\n            stream_mode (bool): controls the exclusive YouTube Mode.\n            backend (int): selects the backend for OpenCV's VideoCapture class.\n            colorspace (str): selects the colorspace of the input stream.\n            logging (bool): enables/disables logging.\n            time_delay (int): time delay (in sec) before start reading the frames.\n            options (dict): provides ability to alter Tweak Parameters of WebGear_RTC, CamGear, PiGear & Stabilizer.\n        "
        import_dependency_safe('starlette' if starlette is None else '')
        import_dependency_safe('aiortc' if aiortc is None else '')
        self.__logging = logging
        custom_data_location = ''
        data_path = ''
        overwrite_default = False
        self.__relay = None
        options = {str(k).strip(): v for (k, v) in options.items()}
        if options:
            if 'custom_data_location' in options:
                value = options['custom_data_location']
                if isinstance(value, str):
                    assert os.access(value, os.W_OK), "[WebGear_RTC:ERROR] :: Permission Denied!, cannot write WebGear_RTC data-files to '{}' directory!".format(value)
                    assert os.path.isdir(os.path.abspath(value)), '[WebGear_RTC:ERROR] :: `custom_data_location` value must be the path to a directory and not to a file!'
                    custom_data_location = os.path.abspath(value)
                else:
                    logger.warning('Skipped invalid `custom_data_location` value!')
                del options['custom_data_location']
            if 'overwrite_default_files' in options:
                value = options['overwrite_default_files']
                if isinstance(value, bool):
                    overwrite_default = value
                else:
                    logger.warning('Skipped invalid `overwrite_default_files` value!')
                del options['overwrite_default_files']
            if 'enable_live_broadcast' in options:
                value = options['enable_live_broadcast']
                if isinstance(value, bool):
                    if value:
                        self.__relay = MediaRelay()
                        options['enable_infinite_frames'] = True
                        logger.critical('Enabled live broadcasting for Peer connection(s).')
                    else:
                        None
                else:
                    logger.warning('Skipped invalid `enable_live_broadcast` value!')
                del options['enable_live_broadcast']
        if custom_data_location:
            data_path = generate_webdata(custom_data_location, c_name='webgear_rtc', overwrite_default=overwrite_default, logging=logging)
        else:
            data_path = generate_webdata(os.path.join(expanduser('~'), '.vidgear'), c_name='webgear_rtc', overwrite_default=overwrite_default, logging=logging)
        self.__logging and logger.debug('`{}` is the default location for saving WebGear_RTC data-files.'.format(data_path))
        self.__templates = Jinja2Templates(directory='{}/templates'.format(data_path))
        self.__exception_handlers = {404: self.__not_found, 500: self.__server_error}
        self.routes = [Route('/', endpoint=self.__homepage), Route('/offer', self.__offer, methods=['GET', 'POST']), Mount('/static', app=StaticFiles(directory='{}/static'.format(data_path)), name='static')]
        self.middleware = []
        if 'custom_stream' in options or not source is None:
            self.__default_rtc_server = RTC_VideoServer(enablePiCamera=enablePiCamera, stabilize=stabilize, source=source, camera_num=camera_num, stream_mode=stream_mode, backend=backend, colorspace=colorspace, resolution=resolution, framerate=framerate, logging=logging, time_delay=time_delay, **options)
            self.routes.append(Route('/close_connection', self.__reset_connections, methods=['POST']))
        else:
            raise ValueError('[WebGear_RTC:ERROR] :: Source cannot be NoneType without Custom Stream(`custom_stream`) defined!')
        self.__rt_org_copy = self.routes[:]
        self.__pcs = set()

    def __call__(self):
        if False:
            while True:
                i = 10
        '\n        Implements a custom Callable method for WebGear_RTC application.\n        '
        assert not self.routes is None, 'Routing tables are NoneType!'
        if not isinstance(self.routes, list) or not all((x in self.routes for x in self.__rt_org_copy)):
            raise RuntimeError('[WebGear_RTC:ERROR] :: Routing tables are not valid!')
        assert not self.middleware is None, 'Middlewares are NoneType!'
        if self.middleware and (not isinstance(self.middleware, list) or not all((isinstance(x, Middleware) for x in self.middleware))):
            raise RuntimeError('[WebGear_RTC:ERROR] :: Middlewares are not valid!')
        self.__logging and logger.debug('Running Starlette application.')
        return Starlette(debug=True if self.__logging else False, routes=self.routes, middleware=self.middleware, exception_handlers=self.__exception_handlers, on_shutdown=[self.__on_shutdown])

    async def __offer(self, request):
        """
        Generates JSON Response with a WebRTC Peer Connection of Video Server.
        """
        params = await request.json()
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
        if not self.__default_rtc_server is None and (not self.__default_rtc_server.is_launched):
            self.__logging and logger.debug('Initiating Video Streaming.')
            self.__default_rtc_server.launch()
        pc = RTCPeerConnection()
        self.__pcs.add(pc)
        self.__logging and logger.info('Created WebRTC Peer Connection.')

        @pc.on('iceconnectionstatechange')
        async def on_iceconnectionstatechange():
            logger.debug('ICE connection state is %s' % pc.iceConnectionState)
            if pc.iceConnectionState == 'failed':
                logger.error('ICE connection state failed.')
                if self.__relay is None:
                    await pc.close()
                    self.__pcs.discard(pc)
        await pc.setRemoteDescription(offer)
        for t in pc.getTransceivers():
            capabilities = RTCRtpSender.getCapabilities('video')
            preferences = list(filter(lambda x: x.name == 'H265', capabilities.codecs))
            t.setCodecPreferences(preferences)
            if t.kind == 'video':
                pc.addTrack(self.__relay.subscribe(self.__default_rtc_server) if not self.__relay is None else self.__default_rtc_server)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return JSONResponse({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

    async def __homepage(self, request):
        """
        Return an HTML index page.
        """
        return self.__templates.TemplateResponse(request, 'index.html')

    async def __not_found(self, request, exc):
        """
        Return an HTML 404 page.
        """
        return self.__templates.TemplateResponse(request, '404.html', status_code=404)

    async def __server_error(self, request, exc):
        """
        Return an HTML 500 page.
        """
        return self.__templates.TemplateResponse(request, '500.html', status_code=500)

    async def __reset_connections(self, request):
        """
        Resets all connections and recreates VideoServer timestamps
        """
        parameter = await request.json()
        if self.__relay is None and (not self.__default_rtc_server is None) and self.__default_rtc_server.is_running:
            logger.critical('Resetting Server')
            if parameter != 0:
                coros = [pc.close() for pc in self.__pcs]
                await asyncio.gather(*coros)
                self.__pcs.clear()
            await self.__default_rtc_server.reset()
            return PlainTextResponse('OK')
        else:
            return PlainTextResponse('DISABLED')

    async def __on_shutdown(self):
        """
        Implements a Callable to be run on application shutdown
        """
        self.shutdown()
        coros = [pc.close() for pc in self.__pcs]
        await asyncio.gather(*coros)
        self.__pcs.clear()

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gracefully shutdown video-server\n        '
        if not self.__default_rtc_server is None:
            self.__logging and logger.debug('Closing Video Server.')
            self.__default_rtc_server.terminate()
            self.__default_rtc_server = None
        self.__default_rtc_server = None