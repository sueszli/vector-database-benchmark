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
import time
import queue
import logging as log
from threading import Thread, Event
from .helper import capPropId, logger_handler, check_CV_version, get_supported_resolution, check_gstreamer_support, import_dependency_safe, logcurr_vidgear_ver
logger = log.getLogger('CamGear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)
yt_dlp = import_dependency_safe('yt_dlp', error='silent')
if not yt_dlp is None:
    from yt_dlp import YoutubeDL

    class YT_backend:
        """
        CamGear's Internal YT-DLP Backend Class for extracting metadata from Streaming URLs.

        Parameters:
            source_url (string): defines the URL of source stream
            logging (bool): enables/disables logging.
            options (dict): provides ability to alter yt-dlp backend params.
        """

        def __init__(self, source_url, logging=False, **stream_params):
            if False:
                for i in range(10):
                    print('nop')
            self.__logging = logging
            self.is_livestream = False
            self.streams_metadata = {}
            self.streams = {}
            self.supported_resolutions = {'256x144': '144p', '426x240': '240p', '640x360': '360p', '854x480': '480p', '1280x720': '720p', '1920x1080': '1080p', '2560x1440': '1440p', '3840x2160': '2160p', '7680x4320': '4320p'}
            self.source_url = source_url
            self.ydl_opts = {'format': 'best*[vcodec!=none]', 'quiet': True, 'prefer_insecure': False, 'no_warnings': True if logging else False, 'dump_single_json': True, 'extract_flat': True, 'skip_download': True}
            stream_params.pop('format', None)
            stream_params.pop('dump_single_json', None)
            stream_params.pop('extract_flat', None)
            std_hdrs = stream_params.pop('std_headers', None)
            if not std_hdrs is None and isinstance(std_hdrs, dict):
                yt_dlp.utils.std_headers.update(std_hdrs)
            self.ydl_opts.update(stream_params)
            self.meta_data = self.__extract_meta()
            if not self.meta_data is None and (not 'entries' in self.meta_data) and (len(self.meta_data.get('formats', {})) > 0):
                self.is_livestream = self.meta_data.get('is_live', False)
                self.streams_metadata = self.meta_data.get('formats', {})
                self.streams = self.__extract_streams()
                if self.streams:
                    logger.info('[Backend] :: Streaming URL is fully supported. Available Streams are: [{}]'.format(', '.join(list(self.streams.keys()))))
                else:
                    raise ValueError("[Backend] :: Streaming URL isn't supported. No usable video streams found!")
            else:
                raise ValueError("[Backend] :: Streaming URL isn't valid{}".format(". Playlists aren't supported yet!" if not self.meta_data is None and 'entries' in self.meta_data else '!'))

        def __extract_meta(self):
            if False:
                i = 10
                return i + 15
            extracted_data = None
            with YoutubeDL(self.ydl_opts) as ydl:
                try:
                    extracted_data = ydl.extract_info(self.source_url, download=False)
                except yt_dlp.utils.DownloadError as e:
                    raise RuntimeError(' [Backend] : ' + str(e))
            return extracted_data

        def __extract_streams(self):
            if False:
                return 10
            streams = {}
            streams_copy = {}
            for stream in self.streams_metadata:
                stream_dim = stream.get('resolution', '')
                stream_url = stream.get('url', '')
                stream_protocol = stream.get('protocol', '')
                stream_with_video = False if stream.get('vcodec', 'none') == 'none' else True
                stream_with_audio = False if stream.get('acodec', 'none') == 'none' else True
                if stream_with_video and stream_dim and stream_url and (stream_protocol != 'http_dash_segments'):
                    if stream_dim in self.supported_resolutions:
                        stream_res = self.supported_resolutions[stream_dim]
                        if not stream_with_audio or stream_protocol in ['https', 'http'] or (not stream_res in streams):
                            streams[stream_res] = stream_url
                    if not stream_with_audio or stream_protocol in ['https', 'http'] or (not stream_dim in streams_copy):
                        streams_copy[stream_dim] = stream_url
            streams['best'] = streams_copy[list(streams_copy.keys())[-1]]
            streams['worst'] = streams_copy[list(streams_copy.keys())[0]]
            return streams

class CamGear:
    """
    CamGear supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia video file format (upto 4k tested),
    any network stream URL such as http(s), rtp, rtsp, rtmp, mms, etc. It also supports Gstreamer's RAW pipelines.

    CamGear API provides a flexible, high-level multi-threaded wrapper around OpenCV's VideoCapture API with direct access to almost all of its available parameters.
    It relies on Threaded Queue mode for threaded, error-free and synchronized frame handling.

    CamGear internally implements `yt_dlp` backend class for seamlessly pipelining live video-frames and metadata from various streaming services like YouTube, Dailymotion,
    Twitch, and [many more ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites)
    """

    def __init__(self, source=0, stream_mode=False, backend=0, colorspace=None, logging=False, time_delay=0, **options):
        if False:
            return 10
        "\n        This constructor method initializes the object state and attributes of the CamGear class.\n\n        Parameters:\n            source (based on input): defines the source for the input stream.\n            stream_mode (bool): controls the exclusive **Stream Mode** for handling streaming URLs.\n            backend (int): selects the backend for OpenCV's VideoCapture class.\n            colorspace (str): selects the colorspace of the input stream.\n            logging (bool): enables/disables logging.\n            time_delay (int): time delay (in sec) before start reading the frames.\n            options (dict): provides ability to alter Source Tweak Parameters.\n        "
        logcurr_vidgear_ver(logging=logging)
        self.__logging = False
        if logging:
            self.__logging = logging
        self.ytv_metadata = {}
        if stream_mode:
            gst_support = check_gstreamer_support(logging=logging)
            stream_resolution = get_supported_resolution(options.pop('STREAM_RESOLUTION', 'best'), logging=logging)
            if not yt_dlp is None:
                yt_stream_params = options.pop('STREAM_PARAMS', {})
                if isinstance(yt_stream_params, dict):
                    yt_stream_params = {str(k).strip(): v for (k, v) in yt_stream_params.items()}
                else:
                    yt_stream_params = {}
                try:
                    logger.info('Verifying Streaming URL using yt-dlp backend. Please wait...')
                    ytbackend = YT_backend(source_url=source, logging=logging, **yt_stream_params)
                    if ytbackend:
                        self.ytv_metadata = ytbackend.meta_data
                        if ytbackend.is_livestream:
                            logger.warning('Livestream URL detected. It is advised to use GStreamer backend(`cv2.CAP_GSTREAMER`) with it.')
                        if not stream_resolution in ytbackend.streams.keys():
                            logger.warning('Specified stream-resolution `{}` is not available. Reverting to `best`!'.format(stream_resolution))
                            stream_resolution = 'best'
                        elif self.__logging:
                            logger.debug('Using `{}` resolution for streaming.'.format(stream_resolution))
                        source = ytbackend.streams[stream_resolution]
                        self.__logging and logger.debug('YouTube source ID: `{}`, Title: `{}`, Quality: `{}`'.format(self.ytv_metadata['id'], self.ytv_metadata['title'], stream_resolution))
                except Exception as e:
                    raise ValueError('[CamGear:ERROR] :: Stream Mode is enabled but Input URL is invalid!')
            else:
                import_dependency_safe('yt_dlp')
        self.__youtube_mode = stream_mode
        self.__threaded_queue_mode = options.pop('THREADED_QUEUE_MODE', True)
        if not isinstance(self.__threaded_queue_mode, bool):
            self.__threaded_queue_mode = True
        self.__thread_timeout = options.pop('THREAD_TIMEOUT', None)
        if self.__thread_timeout and isinstance(self.__thread_timeout, (int, float)):
            self.__thread_timeout = float(self.__thread_timeout)
        else:
            self.__thread_timeout = None
        self.__queue = None
        if self.__threaded_queue_mode and isinstance(source, str):
            self.__queue = queue.Queue(maxsize=96)
            self.__logging and logger.debug('Enabling Threaded Queue Mode for the current video source!')
        else:
            self.__threaded_queue_mode = False
            self.__logging and logger.warning('Threaded Queue Mode is disabled for the current video source!')
        if self.__thread_timeout:
            logger.debug('Setting Video-Thread Timeout to {}s.'.format(self.__thread_timeout))
        self.stream = None
        if backend and isinstance(backend, int):
            if check_CV_version() == 3:
                self.stream = cv2.VideoCapture(source + backend)
            else:
                self.stream = cv2.VideoCapture(source, backend)
            logger.debug('Setting backend `{}` for this source.'.format(backend))
        else:
            self.stream = cv2.VideoCapture(source)
        self.color_space = None
        options = {str(k).strip(): v for (k, v) in options.items()}
        for (key, value) in options.items():
            property = capPropId(key)
            if not property is None:
                self.stream.set(property, value)
        if not colorspace is None:
            self.color_space = capPropId(colorspace.strip())
            if self.__logging and (not self.color_space is None):
                logger.debug('Enabling `{}` colorspace for this video stream!'.format(colorspace.strip()))
        self.framerate = 0.0
        _fps = self.stream.get(cv2.CAP_PROP_FPS)
        if _fps > 1.0:
            self.framerate = _fps
        if time_delay and isinstance(time_delay, (int, float)):
            time.sleep(time_delay)
        (grabbed, self.frame) = self.stream.read()
        if grabbed:
            if not self.color_space is None:
                self.frame = cv2.cvtColor(self.frame, self.color_space)
            if self.__threaded_queue_mode:
                self.__queue.put(self.frame)
        else:
            raise RuntimeError('[CamGear:ERROR] :: Source is invalid, CamGear failed to initialize stream on this source!')
        self.__thread = None
        self.__terminate = Event()
        self.__stream_read = Event()

    def start(self):
        if False:
            print('Hello World!')
        '\n        Launches the internal *Threaded Frames Extractor* daemon.\n\n        **Returns:** A reference to the CamGear class object.\n        '
        self.__thread = Thread(target=self.__update, name='CamGear', args=())
        self.__thread.daemon = True
        self.__thread.start()
        return self

    def __update(self):
        if False:
            i = 10
            return i + 15
        "\n        A **Threaded Frames Extractor**, that keep iterating frames from OpenCV's VideoCapture API to a internal monitored queue,\n        until the thread is terminated, or frames runs out.\n        "
        while not self.__terminate.is_set():
            self.__stream_read.clear()
            (grabbed, frame) = self.stream.read()
            self.__stream_read.set()
            if not grabbed:
                if self.__threaded_queue_mode:
                    if self.__queue.empty():
                        break
                    else:
                        continue
                else:
                    break
            if not self.color_space is None:
                color_frame = None
                try:
                    if isinstance(self.color_space, int):
                        color_frame = cv2.cvtColor(frame, self.color_space)
                    else:
                        raise ValueError('Global color_space parameter value `{}` is not a valid!'.format(self.color_space))
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
            if self.__threaded_queue_mode:
                self.__queue.put(self.frame)
        self.__threaded_queue_mode and self.__queue.put(None)
        self.__threaded_queue_mode = False
        self.__terminate.set()
        self.__stream_read.set()
        self.stream.release()

    def read(self):
        if False:
            while True:
                i = 10
        '\n        Extracts frames synchronously from monitored queue, while maintaining a fixed-length frame buffer in the memory,\n        and blocks the thread if the queue is full.\n\n        **Returns:** A n-dimensional numpy array.\n        '
        while self.__threaded_queue_mode and (not self.__terminate.is_set()):
            return self.__queue.get(timeout=self.__thread_timeout)
        return self.frame if not self.__terminate.is_set() and self.__stream_read.wait(timeout=self.__thread_timeout) else None

    def stop(self):
        if False:
            print('Hello World!')
        '\n        Safely terminates the thread, and release the VideoStream resources.\n        '
        self.__logging and logger.debug('Terminating processes.')
        self.__threaded_queue_mode = False
        self.__stream_read.set()
        self.__terminate.set()
        if self.__thread is not None:
            if not self.__queue is None:
                while not self.__queue.empty():
                    try:
                        self.__queue.get_nowait()
                    except queue.Empty:
                        continue
                    self.__queue.task_done()
            self.__thread.join()