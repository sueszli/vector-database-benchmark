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
import platform
import pathlib
import logging as log
import subprocess as sp
from .helper import capPropId, dict2Args, is_valid_url, logger_handler, check_WriteAccess, get_valid_ffmpeg_path, get_supported_pixfmts, get_supported_vencoders, check_gstreamer_support, logcurr_vidgear_ver
logger = log.getLogger('WriteGear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

class WriteGear:
    """
    WriteGear handles various powerful Video-Writer Tools that provide us the freedom to do almost anything imaginable with multimedia data.

    WriteGear API provides a complete, flexible, and robust wrapper around FFmpeg, a leading multimedia framework. WriteGear can process real-time frames into a lossless
    compressed video-file with any suitable specification (such as bitrate, codec, framerate, resolution, subtitles, etc.). It is powerful enough to perform complex tasks such as
    Live-Streaming (such as for Twitch) and Multiplexing Video-Audio with real-time frames in way fewer lines of code.

    Best of all, WriteGear grants users the complete freedom to play with any FFmpeg parameter with its exclusive Custom Commands function without relying on any
    third-party API.

    In addition to this, WriteGear also provides flexible access to OpenCV's VideoWriter API tools for video-frames encoding without compression.

    ??? tip "Modes of Operation"

        WriteGear primarily operates in following modes:

        * **Compression Mode**: In this mode, WriteGear utilizes powerful **FFmpeg** inbuilt encoders to encode lossless multimedia files.
                                This mode provides us the ability to exploit almost any parameter available within FFmpeg, effortlessly and flexibly,
                                and while doing that it robustly handles all errors/warnings quietly.

        * **Non-Compression Mode**: In this mode, WriteGear utilizes basic **OpenCV's inbuilt VideoWriter API** tools. This mode also supports all
                                    parameters manipulation available within VideoWriter API, but it lacks the ability to manipulate encoding parameters
                                    and other important features like video compression, audio encoding, etc.

    """

    def __init__(self, output='', compression_mode=True, custom_ffmpeg='', logging=False, **output_params):
        if False:
            while True:
                i = 10
        "\n        This constructor method initializes the object state and attributes of the WriteGear class.\n\n        Parameters:\n            output (str): sets the valid filename/path/URL for encoding.\n            compression_mode (bool): selects the WriteGear's Primary Mode of Operation.\n            custom_ffmpeg (str): assigns the location of custom path/directory for custom FFmpeg executables.\n            logging (bool): enables/disables logging.\n            output_params (dict): provides the flexibility to control supported internal parameters and FFmpeg properties.\n        "
        logcurr_vidgear_ver(logging=logging)
        assert not 'output_filename' in output_params, '[WriteGear:ERROR] :: The `output_filename` parameter has been renamed to `output`. Refer Docs for more info.'
        self.__compression = compression_mode if isinstance(compression_mode, bool) else False
        self.__os_windows = True if os.name == 'nt' else False
        self.__logging = logging if isinstance(logging, bool) else False
        self.__output_parameters = {}
        self.__inputheight = None
        self.__inputwidth = None
        self.__inputchannels = None
        self.__inputdtype = None
        self.__process = None
        self.__ffmpeg = ''
        self.__initiate_process = True
        self.__ffmpeg_window_disabler_patch = False
        self.__out_file = None
        gstpipeline_mode = False
        if not output:
            raise ValueError('[WriteGear:ERROR] :: Kindly provide a valid `output` value. Refer Docs for more info.')
        else:
            abs_path = os.path.abspath(output)
            if check_WriteAccess(os.path.dirname(abs_path), is_windows=self.__os_windows, logging=self.__logging):
                if os.path.isdir(abs_path):
                    abs_path = os.path.join(abs_path, 'VidGear-{}.mp4'.format(time.strftime('%Y%m%d-%H%M%S')))
                self.__out_file = abs_path
            else:
                logger.info("`{}` isn't a valid system path or directory. Skipped!".format(output))
        self.__output_parameters = {str(k).strip(): str(v).strip() if not isinstance(v, (list, tuple, int, float)) else v for (k, v) in output_params.items()}
        self.__logging and logger.debug('Output Parameters: `{}`'.format(self.__output_parameters))
        if self.__compression:
            self.__logging and logger.debug('Compression Mode is enabled therefore checking for valid FFmpeg executable.')
            __ffmpeg_download_path = self.__output_parameters.pop('-ffmpeg_download_path', '')
            if not isinstance(__ffmpeg_download_path, str):
                __ffmpeg_download_path = ''
            self.__output_dimensions = self.__output_parameters.pop('-output_dimensions', None)
            if not isinstance(self.__output_dimensions, (list, tuple)):
                self.__output_dimensions = None
            self.__inputframerate = self.__output_parameters.pop('-input_framerate', 0.0)
            if not isinstance(self.__inputframerate, (float, int)):
                self.__inputframerate = 0.0
            else:
                self.__inputframerate = float(self.__inputframerate)
            self.__inputpixfmt = self.__output_parameters.pop('-input_pixfmt', None)
            if not isinstance(self.__inputpixfmt, str):
                self.__inputpixfmt = None
            else:
                self.__inputpixfmt = self.__inputpixfmt.strip()
            self.__ffmpeg_preheaders = self.__output_parameters.pop('-ffpreheaders', [])
            if not isinstance(self.__ffmpeg_preheaders, list):
                self.__ffmpeg_preheaders = []
            disable_force_termination = self.__output_parameters.pop('-disable_force_termination', False if '-i' in self.__output_parameters else True)
            if isinstance(disable_force_termination, bool):
                self.__forced_termination = not disable_force_termination
            else:
                self.__forced_termination = True if '-i' in self.__output_parameters else False
            ffmpeg_window_disabler_patch = self.__output_parameters.pop('-disable_ffmpeg_window', False)
            if not self.__os_windows or logging:
                logger.warning('Optional `-disable_ffmpeg_window` flag is only available on Windows OS with `logging=False`. Discarding!')
            elif isinstance(ffmpeg_window_disabler_patch, bool):
                self.__ffmpeg_window_disabler_patch = ffmpeg_window_disabler_patch
            else:
                self.__ffmpeg_window_disabler_patch = False
            self.__ffmpeg = get_valid_ffmpeg_path(custom_ffmpeg, self.__os_windows, ffmpeg_download_path=__ffmpeg_download_path, logging=self.__logging)
            if self.__ffmpeg:
                self.__logging and logger.debug('Found valid FFmpeg executable: `{}`.'.format(self.__ffmpeg))
            else:
                logger.warning('Disabling Compression Mode since no valid FFmpeg executable found on this machine!')
                if self.__logging and (not self.__os_windows):
                    logger.debug('Kindly install a working FFmpeg module or provide a valid custom FFmpeg binary path. See docs for more info.')
                self.__compression = False
        elif '-gst_pipeline_mode' in self.__output_parameters:
            if isinstance(self.__output_parameters['-gst_pipeline_mode'], bool):
                gstpipeline_mode = self.__output_parameters['-gst_pipeline_mode'] and check_gstreamer_support(logging=logging)
                self.__logging and logger.debug('GStreamer Pipeline Mode successfully activated!')
            else:
                gstpipeline_mode = False
                self.__logging and logger.warning('GStreamer Pipeline Mode failed to activate!')
        if self.__compression and self.__ffmpeg:
            if self.__out_file is None:
                if platform.system() == 'Linux' and pathlib.Path(output).is_char_device():
                    self.__logging and logger.debug('Path:`{}` is a valid Linux Video Device path.'.format(output))
                    self.__out_file = output
                elif is_valid_url(self.__ffmpeg, url=output, logging=self.__logging):
                    self.__logging and logger.debug('URL:`{}` is valid and successfully configured for streaming.'.format(output))
                    self.__out_file = output
                else:
                    raise ValueError('[WriteGear:ERROR] :: output value:`{}` is not supported in Compression Mode.'.format(output))
            self.__forced_termination and logger.debug('Forced termination is enabled for this FFmpeg process.')
            self.__logging and logger.debug('Compression Mode with FFmpeg backend is configured properly.')
        else:
            if self.__out_file is None and (not gstpipeline_mode):
                raise ValueError('[WriteGear:ERROR] :: output value:`{}` is not supported in Non-Compression Mode.'.format(output))
            if gstpipeline_mode:
                self.__output_parameters['-backend'] = 'CAP_GSTREAMER'
                self.__out_file = output
            self.__logging and logger.debug('Non-Compression Mode is successfully configured in GStreamer Pipeline Mode.')
            logger.critical('Compression Mode is disabled, Activating OpenCV built-in Writer!')

    def write(self, frame, rgb_mode=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Pipelines `ndarray` frames to respective API _(**FFmpeg** in Compression Mode & **OpenCV's VideoWriter API** in Non-Compression Mode)_.\n\n        Parameters:\n            frame (ndarray): a valid numpy frame\n            rgb_mode (boolean): enable this flag to activate RGB mode _(i.e. specifies that incoming frames are of RGB format(instead of default BGR)_.\n\n        "
        if frame is None:
            return
        (height, width) = frame.shape[:2]
        channels = frame.shape[-1] if frame.ndim == 3 else 1
        dtype = frame.dtype
        if self.__initiate_process:
            self.__inputheight = height
            self.__inputwidth = width
            self.__inputchannels = channels
            self.__inputdtype = dtype
            self.__logging and logger.debug('InputFrame => Height:{} Width:{} Channels:{} Datatype:{}'.format(self.__inputheight, self.__inputwidth, self.__inputchannels, self.__inputdtype))
        if height != self.__inputheight or width != self.__inputwidth:
            raise ValueError('[WriteGear:ERROR] :: All video-frames must have same size!')
        if channels != self.__inputchannels:
            raise ValueError('[WriteGear:ERROR] :: All video-frames must have same number of channels!')
        if dtype != self.__inputdtype:
            raise ValueError('[WriteGear:ERROR] :: All video-frames must have same datatype!')
        if self.__compression:
            if self.__initiate_process:
                self.__PreprocessFFParams(channels, dtype=dtype, rgb=rgb_mode)
                assert self.__process is not None
            try:
                self.__process.stdin.write(frame.tobytes())
            except (OSError, IOError):
                logger.error('BrokenPipeError caught, Wrong values passed to FFmpeg Pipe. Kindly Refer Docs!')
                raise ValueError
        else:
            if self.__initiate_process:
                self.__start_CVProcess()
                assert self.__process is not None
                self.__logging and logger.info('RGBA and 16-bit grayscale video frames are not supported by OpenCV yet. Kindly switch on `compression_mode` to use them!')
            self.__process.write(frame)

    def __PreprocessFFParams(self, channels, dtype=None, rgb=False):
        if False:
            print('Hello World!')
        '\n        Internal method that pre-processes FFmpeg Parameters before beginning to pipeline frames.\n\n        Parameters:\n            channels (int): Number of channels in input frame.\n            dtype (str): Datatype of input frame.\n            rgb_mode (boolean): Whether to activate `RGB mode`?\n        '
        self.__initiate_process = False
        input_parameters = {}
        dimensions = ''
        if self.__output_dimensions is None:
            dimensions += '{}x{}'.format(self.__inputwidth, self.__inputheight)
        else:
            dimensions += '{}x{}'.format(self.__output_dimensions[0], self.__output_dimensions[1])
        input_parameters['-s'] = str(dimensions)
        if not self.__inputpixfmt is None and self.__inputpixfmt in get_supported_pixfmts(self.__ffmpeg):
            input_parameters['-pix_fmt'] = self.__inputpixfmt
        elif dtype.kind == 'u' and dtype.itemsize == 2:
            pix_fmt = None
            if channels == 1:
                pix_fmt = 'gray16'
            elif channels == 2:
                pix_fmt = 'ya16'
            elif channels == 3:
                pix_fmt = 'rgb48' if rgb else 'bgr48'
            elif channels == 4:
                pix_fmt = 'rgba64' if rgb else 'bgra64'
            else:
                raise ValueError('[WriteGear:ERROR] :: Frames with channels outside range 1-to-4 are not supported!')
            input_parameters['-pix_fmt'] = pix_fmt + ('be' if dtype.byteorder == '>' else 'le')
        elif channels == 1:
            input_parameters['-pix_fmt'] = 'gray'
        elif channels == 2:
            input_parameters['-pix_fmt'] = 'ya8'
        elif channels == 3:
            input_parameters['-pix_fmt'] = 'rgb24' if rgb else 'bgr24'
        elif channels == 4:
            input_parameters['-pix_fmt'] = 'rgba' if rgb else 'bgra'
        else:
            raise ValueError('[WriteGear:ERROR] :: Frames with channels outside range 1-to-4 are not supported!')
        if self.__inputframerate > 0.0:
            self.__logging and logger.debug('Setting Input framerate: {}'.format(self.__inputframerate))
            input_parameters['-framerate'] = str(self.__inputframerate)
        self.__start_FFProcess(input_params=input_parameters, output_params=self.__output_parameters)

    def __start_FFProcess(self, input_params, output_params):
        if False:
            while True:
                i = 10
        '\n        An Internal method that launches FFmpeg subprocess pipeline in Compression Mode\n        for pipelining frames to `stdin`.\n\n        Parameters:\n            input_params (dict): Input FFmpeg parameters\n            output_params (dict): Output FFmpeg parameters\n        '
        input_parameters = dict2Args(input_params)
        supported_vcodecs = get_supported_vencoders(self.__ffmpeg)
        default_vcodec = [vcodec for vcodec in ['libx264', 'libx265', 'libxvid', 'mpeg4'] if vcodec in supported_vcodecs][0] or 'unknown'
        if '-c:v' in output_params:
            output_params['-vcodec'] = output_params.pop('-c:v', default_vcodec)
        if not '-vcodec' in output_params:
            output_params['-vcodec'] = default_vcodec
        if default_vcodec != 'unknown' and (not output_params['-vcodec'] in supported_vcodecs):
            logger.critical('Provided FFmpeg does not support `{}` video-encoder. Switching to default supported `{}` encoder!'.format(output_params['-vcodec'], default_vcodec))
            output_params['-vcodec'] = default_vcodec
        if output_params['-vcodec'] in supported_vcodecs:
            if output_params['-vcodec'] in ['libx265', 'libx264']:
                if not '-crf' in output_params:
                    output_params['-crf'] = '18'
                if not '-preset' in output_params:
                    output_params['-preset'] = 'fast'
            if output_params['-vcodec'] in ['libxvid', 'mpeg4']:
                if not '-qscale:v' in output_params:
                    output_params['-qscale:v'] = '3'
        else:
            raise RuntimeError('[WriteGear:ERROR] :: Provided FFmpeg does not support any suitable/usable video-encoders for compression. Kindly disable compression mode or switch to another FFmpeg binaries(if available).')
        output_parameters = dict2Args(output_params)
        cmd = [self.__ffmpeg, '-y'] + self.__ffmpeg_preheaders + ['-f', 'rawvideo', '-vcodec', 'rawvideo'] + input_parameters + ['-i', '-'] + output_parameters + [self.__out_file]
        if self.__logging:
            logger.debug('Executing FFmpeg command: `{}`'.format(' '.join(cmd)))
            self.__process = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
        else:
            self.__process = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT, creationflags=sp.DETACHED_PROCESS if self.__ffmpeg_window_disabler_patch else 0)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        "\n        Handles entry with the `with` statement. See [PEP343 -- The 'with' statement'](https://peps.python.org/pep-0343/).\n\n        **Returns:** Returns a reference to the WriteGear Class\n        "
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        "\n        Handles exit with the `with` statement. See [PEP343 -- The 'with' statement'](https://peps.python.org/pep-0343/).\n        "
        self.close()

    def execute_ffmpeg_cmd(self, command=None):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Executes user-defined FFmpeg Terminal command, formatted as a python list(in Compression Mode only).\n\n        Parameters:\n            command (list): inputs list data-type command.\n\n        '
        if command is None or not command:
            logger.warning('Input command is empty, Nothing to execute!')
            return
        elif not isinstance(command, list):
            raise ValueError('[WriteGear:ERROR] :: Invalid input command datatype! Kindly read docs.')
        if not self.__compression:
            raise RuntimeError('[WriteGear:ERROR] :: Compression Mode is disabled, Kindly enable it to access this function.')
        cmd = [self.__ffmpeg] + command
        try:
            if self.__logging:
                logger.debug('Executing FFmpeg command: `{}`'.format(' '.join(cmd)))
                sp.run(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
            else:
                sp.run(cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        except (OSError, IOError):
            logger.error('BrokenPipeError caught, Wrong command passed to FFmpeg Pipe, Kindly Refer Docs!')
            raise ValueError

    def __start_CVProcess(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An Internal method that launches OpenCV VideoWriter process in Non-Compression\n        Mode with given settings.\n        '
        self.__initiate_process = False
        FPS = 0
        BACKEND = ''
        FOURCC = 0
        COLOR = True
        if '-fourcc' not in self.__output_parameters:
            FOURCC = cv2.VideoWriter_fourcc(*'MJPG')
        if '-fps' not in self.__output_parameters:
            FPS = 25
        HEIGHT = self.__inputheight
        WIDTH = self.__inputwidth
        try:
            for (key, value) in self.__output_parameters.items():
                if key == '-fourcc':
                    FOURCC = cv2.VideoWriter_fourcc(*value.upper())
                elif key == '-fps':
                    FPS = int(value)
                elif key == '-backend':
                    BACKEND = capPropId(value.upper())
                elif key == '-color':
                    COLOR = bool(value)
                else:
                    pass
        except Exception as e:
            self.__logging and logger.exception(str(e))
            raise ValueError('[WriteGear:ERROR] :: Wrong Values passed to OpenCV Writer, Kindly Refer Docs!')
        self.__logging and logger.debug('FILE_PATH: {}, FOURCC = {}, FPS = {}, WIDTH = {}, HEIGHT = {}, BACKEND = {}'.format(self.__out_file, FOURCC, FPS, WIDTH, HEIGHT, BACKEND))
        if BACKEND:
            self.__process = cv2.VideoWriter(self.__out_file, apiPreference=BACKEND, fourcc=FOURCC, fps=FPS, frameSize=(WIDTH, HEIGHT), isColor=COLOR)
        else:
            self.__process = cv2.VideoWriter(self.__out_file, fourcc=FOURCC, fps=FPS, frameSize=(WIDTH, HEIGHT), isColor=COLOR)
        assert self.__process.isOpened(), '[WriteGear:ERROR] :: Failed to intialize OpenCV Writer!'

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        Safely terminates various WriteGear process.\n        '
        if self.__logging:
            logger.debug('Terminating WriteGear Processes.')
        if self.__compression:
            if self.__process is None or not self.__process.poll() is None:
                return
            self.__process.stdin and self.__process.stdin.close()
            self.__process.stdout and self.__process.stdout.close()
            self.__forced_termination and self.__process.terminate()
            self.__process.wait()
        else:
            if self.__process is None:
                return
            self.__process.release()
        self.__process = None