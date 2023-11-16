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
import asyncio
import inspect
import numpy as np
import logging as log
from os.path import expanduser
from .helper import reducer, generate_webdata, create_blank_frame
from ..helper import logger_handler, retrieve_best_interpolation, import_dependency_safe, logcurr_vidgear_ver
from ..videogear import VideoGear
starlette = import_dependency_safe('starlette', error='silent')
if not starlette is None:
    from starlette.routing import Mount, Route
    from starlette.responses import StreamingResponse, JSONResponse
    from starlette.templating import Jinja2Templates
    from starlette.staticfiles import StaticFiles
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
simplejpeg = import_dependency_safe('simplejpeg', error='silent', min_version='1.6.1')
logger = log.getLogger('WebGear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

class WebGear:
    """
    WebGear is a powerful ASGI Video-Broadcaster API ideal for transmitting Motion-JPEG-frames from a single source to multiple recipients via the browser.

    WebGear API works on Starlette's ASGI application and provides a highly extensible and flexible async wrapper around its complete framework. WebGear can
    flexibly interact with Starlette's ecosystem of shared middleware, mountable applications, Response classes, Routing tables, Static Files, Templating
    engine(with Jinja2), etc.

    WebGear API uses an intraframe-only compression scheme under the hood where the sequence of video-frames are first encoded as JPEG-DIB (JPEG with Device-Independent Bit compression)
    and then streamed over HTTP using Starlette's Multipart Streaming Response and a Uvicorn ASGI Server. This method imposes lower processing and memory requirements, but the quality
    is not the best, since JPEG compression is not very efficient for motion video.

    In layman's terms, WebGear acts as a powerful Video Broadcaster that transmits live video-frames to any web-browser in the network. Additionally, WebGear API also provides internal
    wrapper around VideoGear, which itself provides internal access to both CamGear and PiGear APIs, thereby granting it exclusive power for transferring frames incoming from any source to the network.
    """

    def __init__(self, enablePiCamera=False, stabilize=False, source=None, camera_num=0, stream_mode=False, backend=0, colorspace=None, resolution=(640, 480), framerate=25, logging=False, time_delay=0, **options):
        if False:
            return 10
        "\n        This constructor method initializes the object state and attributes of the WebGear class.\n\n        Parameters:\n            enablePiCamera (bool): provide access to PiGear(if True) or CamGear(if False) APIs respectively.\n            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.\n            camera_num (int): selects the camera module index which will be used as Rpi source.\n            resolution (tuple): sets the resolution (i.e. `(width,height)`) of the Rpi source.\n            framerate (int/float): sets the framerate of the Rpi source.\n            source (based on input): defines the source for the input stream.\n            stream_mode (bool): controls the exclusive YouTube Mode.\n            backend (int): selects the backend for OpenCV's VideoCapture class.\n            colorspace (str): selects the colorspace of the input stream.\n            logging (bool): enables/disables logging.\n            time_delay (int): time delay (in sec) before start reading the frames.\n            options (dict): provides ability to alter Tweak Parameters of WebGear, CamGear, PiGear & Stabilizer.\n        "
        logcurr_vidgear_ver(logging=logging)
        import_dependency_safe('starlette' if starlette is None else '')
        import_dependency_safe('simplejpeg' if simplejpeg is None else '', min_version='1.6.1')
        self.__skip_generate_webdata = False
        self.__jpeg_compression_quality = 90
        self.__jpeg_compression_fastdct = True
        self.__jpeg_compression_fastupsample = False
        self.__jpeg_compression_colorspace = 'BGR'
        self.__logging = logging
        self.__frame_size_reduction = 25
        self.__interpolation = retrieve_best_interpolation(['INTER_LINEAR_EXACT', 'INTER_LINEAR', 'INTER_AREA'])
        custom_video_endpoint = ''
        custom_data_location = ''
        data_path = ''
        overwrite_default = False
        self.__enable_inf = False
        options = {str(k).strip(): v for (k, v) in options.items()}
        if options:
            if 'skip_generate_webdata' in options:
                value = options['skip_generate_webdata']
                if isinstance(value, bool):
                    self.__skip_generate_webdata = value
                else:
                    logger.warning('Skipped invalid `skip_generate_webdata` value!')
                del options['skip_generate_webdata']
            if 'jpeg_compression_colorspace' in options:
                value = options['jpeg_compression_colorspace']
                if isinstance(value, str) and value.strip().upper() in ['RGB', 'BGR', 'RGBX', 'BGRX', 'XBGR', 'XRGB', 'GRAY', 'RGBA', 'BGRA', 'ABGR', 'ARGB', 'CMYK']:
                    self.__jpeg_compression_colorspace = value.strip().upper()
                else:
                    logger.warning('Skipped invalid `jpeg_compression_colorspace` value!')
                del options['jpeg_compression_colorspace']
            if 'jpeg_compression_quality' in options:
                value = options['jpeg_compression_quality']
                if isinstance(value, (int, float)) and value >= 10 and (value <= 100):
                    self.__jpeg_compression_quality = int(value)
                else:
                    logger.warning('Skipped invalid `jpeg_compression_quality` value!')
                del options['jpeg_compression_quality']
            if 'jpeg_compression_fastdct' in options:
                value = options['jpeg_compression_fastdct']
                if isinstance(value, bool):
                    self.__jpeg_compression_fastdct = value
                else:
                    logger.warning('Skipped invalid `jpeg_compression_fastdct` value!')
                del options['jpeg_compression_fastdct']
            if 'jpeg_compression_fastupsample' in options:
                value = options['jpeg_compression_fastupsample']
                if isinstance(value, bool):
                    self.__jpeg_compression_fastupsample = value
                else:
                    logger.warning('Skipped invalid `jpeg_compression_fastupsample` value!')
                del options['jpeg_compression_fastupsample']
            if 'frame_size_reduction' in options:
                value = options['frame_size_reduction']
                if isinstance(value, (int, float)) and value >= 0 and (value <= 90):
                    self.__frame_size_reduction = value
                else:
                    logger.warning('Skipped invalid `frame_size_reduction` value!')
                del options['frame_size_reduction']
            if 'custom_video_endpoint' in options:
                value = options['custom_video_endpoint']
                if value and isinstance(value, str) and value.strip().isalnum():
                    custom_video_endpoint = value.strip()
                    logging and logger.critical('Using custom video endpoint path: `/{}`'.format(custom_video_endpoint))
                else:
                    logger.warning('Skipped invalid `custom_video_endpoint` value!')
                del options['custom_video_endpoint']
            if 'custom_data_location' in options:
                value = options['custom_data_location']
                if value and isinstance(value, str):
                    assert os.access(value, os.W_OK), "[WebGear:ERROR] :: Permission Denied!, cannot write WebGear data-files to '{}' directory!".format(value)
                    assert os.path.isdir(os.path.abspath(value)), '[WebGear:ERROR] :: `custom_data_location` value must be the path to a directory and not to a file!'
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
            if 'enable_infinite_frames' in options:
                value = options['enable_infinite_frames']
                if isinstance(value, bool):
                    self.__enable_inf = value
                else:
                    logger.warning('Skipped invalid `enable_infinite_frames` value!')
                del options['enable_infinite_frames']
        if not self.__skip_generate_webdata:
            if custom_data_location:
                data_path = generate_webdata(custom_data_location, c_name='webgear', overwrite_default=overwrite_default, logging=logging)
            else:
                data_path = generate_webdata(os.path.join(expanduser('~'), '.vidgear'), c_name='webgear', overwrite_default=overwrite_default, logging=logging)
            self.__logging and logger.debug('`{}` is the default location for saving WebGear data-files.'.format(data_path))
            self.__templates = Jinja2Templates(directory='{}/templates'.format(data_path))
            self.routes = [Route('/', endpoint=self.__homepage), Route('/{}'.format(custom_video_endpoint if custom_video_endpoint else 'video'), endpoint=self.__video), Mount('/static', app=StaticFiles(directory='{}/static'.format(data_path)), name='static')]
        else:
            self.__logging and logger.critical('WebGear Data-Files Auto-Generation WorkFlow has been manually disabled.')
            self.routes = [Route('/{}'.format(custom_video_endpoint if custom_video_endpoint else 'video'), endpoint=self.__video)]
            self.__logging and logger.warning('Only `/video` route is available for this instance.')
        self.__exception_handlers = {404: self.__not_found, 500: self.__server_error}
        self.middleware = []
        if source is None:
            self.config = {'generator': None}
            self.__stream = None
        else:
            self.__stream = VideoGear(enablePiCamera=enablePiCamera, stabilize=stabilize, source=source, camera_num=camera_num, stream_mode=stream_mode, backend=backend, colorspace=colorspace, resolution=resolution, framerate=framerate, logging=logging, time_delay=time_delay, **options)
            self.config = {'generator': self.__producer}
        if self.__logging:
            if source is None:
                logger.warning('Given source is of NoneType. Therefore, JPEG Frame-Compression is disabled!')
            else:
                logger.debug('Enabling JPEG Frame-Compression with Colorspace:`{}`, Quality:`{}`%, Fastdct:`{}`, and Fastupsample:`{}`.'.format(self.__jpeg_compression_colorspace, self.__jpeg_compression_quality, 'enabled' if self.__jpeg_compression_fastdct else 'disabled', 'enabled' if self.__jpeg_compression_fastupsample else 'disabled'))
        self.__rt_org_copy = self.routes[:]
        self.blank_frame = None
        self.__isrunning = True

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implements a custom Callable method for WebGear application.\n        '
        assert not self.routes is None, 'Routing tables are NoneType!'
        if not isinstance(self.routes, list) or not all((x in self.routes for x in self.__rt_org_copy)):
            raise RuntimeError('[WebGear:ERROR] :: Routing tables are not valid!')
        assert not self.middleware is None, 'Middlewares are NoneType!'
        if self.middleware and (not isinstance(self.middleware, list) or not all((isinstance(x, Middleware) for x in self.middleware))):
            raise RuntimeError('[WebGear:ERROR] :: Middlewares are not valid!')
        if isinstance(self.config, dict) and 'generator' in self.config:
            if self.config['generator'] is None or not inspect.isasyncgen(self.config['generator']()):
                raise ValueError('[WebGear:ERROR] :: Invalid configuration. Assigned generator must be a asynchronous generator function/method only!')
        else:
            raise RuntimeError('[WebGear:ERROR] :: Assigned configuration is invalid!')
        self.__logging and logger.debug('Initiating Video Streaming.')
        if not self.__stream is None:
            self.__stream.start()
        self.__logging and logger.debug('Running Starlette application.')
        return Starlette(debug=True if self.__logging else False, routes=self.routes, middleware=self.middleware, exception_handlers=self.__exception_handlers, on_shutdown=[self.shutdown])

    async def __producer(self):
        """
        WebGear's default asynchronous frame producer/generator.
        """
        while self.__isrunning:
            frame = self.__stream.read()
            if frame is None:
                frame = self.blank_frame if self.blank_frame is None else self.blank_frame[:]
                if not self.__enable_inf:
                    self.__isrunning = False
            elif self.blank_frame is None:
                self.blank_frame = create_blank_frame(frame=frame, text='No Input' if self.__enable_inf else 'The End', logging=self.__logging)
            if self.__frame_size_reduction:
                frame = await reducer(frame, percentage=self.__frame_size_reduction, interpolation=self.__interpolation)
            if self.__jpeg_compression_colorspace == 'GRAY':
                if frame.ndim == 2:
                    frame = np.expand_dims(frame, axis=2)
                encodedImage = simplejpeg.encode_jpeg(frame, quality=self.__jpeg_compression_quality, colorspace=self.__jpeg_compression_colorspace, fastdct=self.__jpeg_compression_fastdct)
            else:
                encodedImage = simplejpeg.encode_jpeg(frame, quality=self.__jpeg_compression_quality, colorspace=self.__jpeg_compression_colorspace, colorsubsampling='422', fastdct=self.__jpeg_compression_fastdct)
            yield (b'--frame\r\nContent-Type:image/jpeg\r\n\r\n' + encodedImage + b'\r\n')
            await asyncio.sleep(0)

    async def __video(self, scope):
        """
        Returns a async video streaming response.
        """
        assert scope['type'] in ['http', 'https']
        return StreamingResponse(self.config['generator'](), media_type='multipart/x-mixed-replace; boundary=frame')

    async def __homepage(self, request):
        """
        Returns an HTML index page.
        """
        return self.__templates.TemplateResponse(request, 'index.html') if not self.__skip_generate_webdata else JSONResponse({'detail': 'WebGear Data-Files Auto-Generation WorkFlow is disabled!'}, status_code=404)

    async def __not_found(self, request, exc):
        """
        Returns an HTML 404 page.
        """
        return self.__templates.TemplateResponse(request, '404.html', status_code=404) if not self.__skip_generate_webdata else JSONResponse({'detail': 'WebGear Data-Files Auto-Generation WorkFlow is disabled!'}, status_code=404)

    async def __server_error(self, request, exc):
        """
        Returns an HTML 500 page.
        """
        return self.__templates.TemplateResponse(request, '500.html', status_code=500) if not self.__skip_generate_webdata else JSONResponse({'detail': 'WebGear Data-Files Auto-Generation WorkFlow is disabled!'}, status_code=500)

    def shutdown(self):
        if False:
            i = 10
            return i + 15
        '\n        Implements a Callable to be run on application shutdown\n        '
        if not self.__stream is None:
            self.__logging and logger.debug('Closing Video Streaming.')
            self.__isrunning = False
            self.__stream.stop()
            self.__stream = None