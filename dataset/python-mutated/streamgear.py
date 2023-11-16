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
import math
import platform
import pathlib
import difflib
import logging as log
import subprocess as sp
from tqdm import tqdm
from fractions import Fraction
from collections import OrderedDict
from .helper import capPropId, dict2Args, delete_ext_safe, extract_time, is_valid_url, logger_handler, validate_audio, validate_video, check_WriteAccess, get_video_bitrate, get_valid_ffmpeg_path, logcurr_vidgear_ver
logger = log.getLogger('StreamGear')
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

class StreamGear:
    """
    StreamGear automates transcoding workflow for generating Ultra-Low Latency, High-Quality, Dynamic & Adaptive Streaming Formats (such as MPEG-DASH and HLS) in just few lines of python code.
    StreamGear provides a standalone, highly extensible, and flexible wrapper around FFmpeg multimedia framework for generating chunked-encoded media segments of the content.

    SteamGear easily transcodes source videos/audio files & real-time video-frames and breaks them into a sequence of multiple smaller chunks/segments of suitable length. These segments make it
    possible to stream videos at different quality levels (different bitrates or spatial resolutions) and can be switched in the middle of a video from one quality level to another – if bandwidth
    permits – on a per-segment basis. A user can serve these segments on a web server that makes it easier to download them through HTTP standard-compliant GET requests.

    SteamGear also creates a Manifest/Playlist file (such as MPD in-case of DASH and M3U8 in-case of HLS) besides segments that describe these segment information (timing, URL, media characteristics like video resolution and bit rates)
     and is provided to the client before the streaming session.

    SteamGear currently supports MPEG-DASH (Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1) and Apple HLS (HTTP live streaming).
    """

    def __init__(self, output='', format='dash', custom_ffmpeg='', logging=False, **stream_params):
        if False:
            while True:
                i = 10
        '\n        This constructor method initializes the object state and attributes of the StreamGear class.\n\n        Parameters:\n            output (str): sets the valid filename/path for storing the StreamGear assets.\n            format (str): select the adaptive HTTP streaming format(DASH and HLS).\n            custom_ffmpeg (str): assigns the location of custom path/directory for custom FFmpeg executables.\n            logging (bool): enables/disables logging.\n            stream_params (dict): provides the flexibility to control supported internal parameters and FFmpeg properties.\n        '
        logcurr_vidgear_ver(logging=logging)
        self.__os_windows = True if os.name == 'nt' else False
        self.__logging = logging if logging and isinstance(logging, bool) else False
        self.__params = {}
        self.__inputheight = None
        self.__inputwidth = None
        self.__inputchannels = None
        self.__sourceframerate = None
        self.__process = None
        self.__ffmpeg = ''
        self.__initiate_stream = True
        self.__params = {str(k).strip(): str(v).strip() if not isinstance(v, (dict, list, int, float)) else v for (k, v) in stream_params.items()}
        __ffmpeg_download_path = self.__params.pop('-ffmpeg_download_path', '')
        if not isinstance(__ffmpeg_download_path, str):
            __ffmpeg_download_path = ''
        self.__ffmpeg = get_valid_ffmpeg_path(str(custom_ffmpeg), self.__os_windows, ffmpeg_download_path=__ffmpeg_download_path, logging=self.__logging)
        if self.__ffmpeg:
            self.__logging and logger.debug('Found valid FFmpeg executables: `{}`.'.format(self.__ffmpeg))
        else:
            raise RuntimeError('[StreamGear:ERROR] :: Failed to find FFmpeg assets on this system. Kindly compile/install FFmpeg or provide a valid custom FFmpeg binary path!')
        audio = self.__params.pop('-audio', '')
        if audio and isinstance(audio, str):
            if os.path.isfile(audio):
                self.__audio = os.path.abspath(audio)
            elif is_valid_url(self.__ffmpeg, url=audio, logging=self.__logging):
                self.__audio = audio
            else:
                self.__audio = ''
        elif audio and isinstance(audio, list):
            self.__audio = audio
        else:
            self.__audio = ''
        if self.__audio and self.__logging:
            logger.debug('External audio source detected!')
        source = self.__params.pop('-video_source', '')
        if source and isinstance(source, str) and (len(source) > 1):
            if os.path.isfile(source):
                self.__video_source = os.path.abspath(source)
            elif is_valid_url(self.__ffmpeg, url=source, logging=self.__logging):
                self.__video_source = source
            else:
                self.__video_source = ''
            if self.__video_source:
                validation_results = validate_video(self.__ffmpeg, video_path=self.__video_source)
                assert not validation_results is None, '[StreamGear:ERROR] :: Given `{}` video_source is Invalid, Check Again!'.format(self.__video_source)
                self.__aspect_source = validation_results['resolution']
                self.__fps_source = validation_results['framerate']
                self.__logging and logger.debug('Given video_source is valid and has {}x{} resolution, and a framerate of {} fps.'.format(self.__aspect_source[0], self.__aspect_source[1], self.__fps_source))
            else:
                logger.warning('No valid video_source provided.')
        else:
            self.__video_source = ''
        self.__inputframerate = self.__params.pop('-input_framerate', 0.0)
        if isinstance(self.__inputframerate, (float, int)):
            self.__inputframerate = float(self.__inputframerate)
        else:
            self.__inputframerate = 0.0
        self.__clear_assets = self.__params.pop('-clear_prev_assets', False)
        if not isinstance(self.__clear_assets, bool):
            self.__clear_assets = False
        self.__livestreaming = self.__params.pop('-livestream', False)
        if not isinstance(self.__livestreaming, bool):
            self.__livestreaming = False
        supported_formats = ['dash', 'hls']
        if not format is None and format and isinstance(format, str):
            _format = format.strip().lower()
            if _format in supported_formats:
                self.__format = _format
                logger.info('StreamGear will generate files for {} HTTP streaming format.'.format(self.__format.upper()))
            elif difflib.get_close_matches(_format, supported_formats):
                raise ValueError('[StreamGear:ERROR] :: Incorrect format! Did you mean `{}`?'.format(difflib.get_close_matches(_format, supported_formats)[0]))
            else:
                raise ValueError('[StreamGear:ERROR] :: format value `{}` not valid/supported!'.format(format))
        else:
            raise ValueError('[StreamGear:ERROR] :: format value is Missing/Incorrect. Check vidgear docs!')
        if not output:
            raise ValueError('[StreamGear:ERROR] :: Kindly provide a valid `output` value. Refer Docs for more information.')
        else:
            abs_path = os.path.abspath(output)
            if check_WriteAccess(os.path.dirname(abs_path), is_windows=self.__os_windows, logging=self.__logging):
                valid_extension = 'mpd' if self.__format == 'dash' else 'm3u8'
                assets_exts = [('chunk-stream', '.m4s'), ('chunk-stream', '.ts'), '.{}'.format(valid_extension)]
                if self.__video_source:
                    assets_exts.append(('chunk-stream', os.path.splitext(self.__video_source)[1]))
                if os.path.isdir(abs_path):
                    if self.__clear_assets:
                        delete_ext_safe(abs_path, assets_exts, logging=self.__logging)
                    abs_path = os.path.join(abs_path, '{}-{}.{}'.format(self.__format, time.strftime('%Y%m%d-%H%M%S'), valid_extension))
                elif self.__clear_assets and os.path.isfile(abs_path):
                    delete_ext_safe(os.path.dirname(abs_path), assets_exts, logging=self.__logging)
                assert abs_path.endswith(valid_extension), 'Given `{}` path has invalid file-extension w.r.t selected format: `{}`!'.format(output, self.__format.upper())
                self.__logging and logger.debug('Path:`{}` is sucessfully configured for streaming.'.format(abs_path))
                self.__out_file = abs_path.replace('\\', '/')
            elif platform.system() == 'Linux' and pathlib.Path(output).is_char_device():
                self.__logging and logger.debug('Path:`{}` is a valid Linux Video Device path.'.format(output))
                self.__out_file = output
            elif is_valid_url(self.__ffmpeg, url=output, logging=self.__logging):
                self.__logging and logger.debug('URL:`{}` is valid and sucessfully configured for streaming.'.format(output))
                self.__out_file = output
            else:
                raise ValueError('[StreamGear:ERROR] :: Output value:`{}` is not valid/supported!'.format(output))
        logger.info('StreamGear has been successfully configured for {} Mode.'.format('Single-Source' if self.__video_source else 'Real-time Frames'))

    def stream(self, frame, rgb_mode=False):
        if False:
            return 10
        '\n        Pipelines `ndarray` frames to FFmpeg Pipeline for transcoding into multi-bitrate streamable assets.\n\n        Parameters:\n            frame (ndarray): a valid numpy frame\n            rgb_mode (boolean): enable this flag to activate RGB mode _(i.e. specifies that incoming frames are of RGB format instead of default BGR)_.\n\n        '
        if self.__video_source:
            raise RuntimeError('[StreamGear:ERROR] :: `stream()` function cannot be used when streaming from a `-video_source` input file. Kindly refer vidgear docs!')
        if frame is None:
            return
        (height, width) = frame.shape[:2]
        channels = frame.shape[-1] if frame.ndim == 3 else 1
        if self.__initiate_stream:
            self.__inputheight = height
            self.__inputwidth = width
            self.__inputchannels = channels
            self.__sourceframerate = 25.0 if not self.__inputframerate else self.__inputframerate
            self.__logging and logger.debug('InputFrame => Height:{} Width:{} Channels:{}'.format(self.__inputheight, self.__inputwidth, self.__inputchannels))
        if height != self.__inputheight or width != self.__inputwidth:
            raise ValueError('[StreamGear:ERROR] :: All frames must have same size!')
        if channels != self.__inputchannels:
            raise ValueError('[StreamGear:ERROR] :: All frames must have same number of channels!')
        if self.__initiate_stream:
            self.__PreProcess(channels=channels, rgb=rgb_mode)
            assert self.__process is not None
        try:
            self.__process.stdin.write(frame.tobytes())
        except (OSError, IOError):
            logger.error('BrokenPipeError caught, Wrong values passed to FFmpeg Pipe, Kindly Refer Docs!')
            raise ValueError

    def transcode_source(self):
        if False:
            i = 10
            return i + 15
        '\n        Transcodes entire Video Source _(with audio)_ into multi-bitrate streamable assets\n        '
        if not self.__video_source:
            raise RuntimeError('[StreamGear:ERROR] :: `transcode_source()` function cannot be used without a valid `-video_source` input. Kindly refer vidgear docs!')
        self.__inputheight = int(self.__aspect_source[1])
        self.__inputwidth = int(self.__aspect_source[0])
        self.__sourceframerate = float(self.__fps_source)
        self.__PreProcess()

    def __PreProcess(self, channels=0, rgb=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal method that pre-processes default FFmpeg parameters before beginning pipelining.\n\n        Parameters:\n            channels (int): Number of channels\n            rgb_mode (boolean): activates RGB mode _(if enabled)_.\n        '
        self.__initiate_stream = False
        input_parameters = OrderedDict()
        output_parameters = OrderedDict()
        default_codec = 'libx264rgb' if rgb else 'libx264'
        output_parameters['-vcodec'] = self.__params.pop('-vcodec', default_codec)
        output_parameters['-vf'] = self.__params.pop('-vf', 'format=yuv420p')
        aspect_ratio = Fraction(self.__inputwidth / self.__inputheight).limit_denominator(10)
        output_parameters['-aspect'] = ':'.join(str(aspect_ratio).split('/'))
        if output_parameters['-vcodec'] in ['libx264', 'libx264rgb', 'libx265', 'libvpx-vp9']:
            output_parameters['-crf'] = self.__params.pop('-crf', '20')
        if output_parameters['-vcodec'] in ['libx264', 'libx264rgb']:
            if not self.__video_source:
                output_parameters['-profile:v'] = self.__params.pop('-profile:v', 'high')
            output_parameters['-tune'] = self.__params.pop('-tune', 'zerolatency')
            output_parameters['-preset'] = self.__params.pop('-preset', 'veryfast')
        if output_parameters['-vcodec'] == 'libx265':
            output_parameters['-x265-params'] = self.__params.pop('-x265-params', 'lossless=1')
        if self.__audio:
            bitrate = validate_audio(self.__ffmpeg, source=self.__audio)
            if bitrate:
                logger.info('Detected External Audio Source is valid, and will be used for streams.')
                output_parameters['{}'.format('-core_asource' if isinstance(self.__audio, list) else '-i')] = self.__audio
                output_parameters['-acodec'] = self.__params.pop('-acodec', 'aac' if isinstance(self.__audio, list) else 'copy')
                output_parameters['a_bitrate'] = bitrate
                output_parameters['-core_audio'] = ['-map', '1:a:0'] if self.__format == 'dash' else []
            else:
                logger.warning('Audio source `{}` is not valid, Skipped!'.format(self.__audio))
        elif self.__video_source:
            bitrate = validate_audio(self.__ffmpeg, source=self.__video_source)
            if bitrate:
                logger.info('Source Audio will be used for streams.')
                output_parameters['-acodec'] = 'aac' if self.__format == 'hls' else 'copy'
                output_parameters['a_bitrate'] = bitrate
            else:
                logger.warning('No valid audio_source available. Disabling audio for streams!')
        else:
            logger.warning('No valid audio_source provided. Disabling audio for streams!')
        if '-acodec' in output_parameters and output_parameters['-acodec'] == 'aac':
            output_parameters['-movflags'] = '+faststart'
        if self.__sourceframerate > 0 and (not self.__video_source):
            self.__logging and logger.debug('Setting Input framerate: {}'.format(self.__sourceframerate))
            input_parameters['-framerate'] = str(self.__sourceframerate)
        if not self.__video_source:
            dimensions = '{}x{}'.format(self.__inputwidth, self.__inputheight)
            input_parameters['-video_size'] = str(dimensions)
            if channels == 1:
                input_parameters['-pix_fmt'] = 'gray'
            elif channels == 2:
                input_parameters['-pix_fmt'] = 'ya8'
            elif channels == 3:
                input_parameters['-pix_fmt'] = 'rgb24' if rgb else 'bgr24'
            elif channels == 4:
                input_parameters['-pix_fmt'] = 'rgba' if rgb else 'bgra'
            else:
                raise ValueError('[StreamGear:ERROR] :: Frames with channels outside range 1-to-4 are not supported!')
        process_params = self.__handle_streams(input_params=input_parameters, output_params=output_parameters)
        assert not process_params is None, '[StreamGear:ERROR] :: {} stream cannot be initiated!'.format(self.__format.upper())
        self.__Build_n_Execute(process_params[0], process_params[1])

    def __handle_streams(self, input_params, output_params):
        if False:
            print('Hello World!')
        '\n        An internal function that parses various streams and its parameters.\n\n        Parameters:\n            input_params (dict): Input FFmpeg parameters\n            output_params (dict): Output FFmpeg parameters\n        '
        bpp = self.__params.pop('-bpp', 0.1)
        if isinstance(bpp, (float, int)) and bpp > 0.0:
            bpp = float(bpp) if bpp > 0.001 else 0.1
        else:
            bpp = 0.1
        self.__logging and logger.debug('Setting bit-per-pixels: {} for this stream.'.format(bpp))
        gop = self.__params.pop('-gop', 0)
        if isinstance(gop, (int, float)) and gop > 0:
            gop = int(gop)
        else:
            gop = 2 * int(self.__sourceframerate)
        self.__logging and logger.debug('Setting GOP: {} for this stream.'.format(gop))
        if self.__format != 'hls':
            output_params['-map'] = 0
        else:
            output_params['-corev0'] = ['-map', '0:v']
            if '-acodec' in output_params:
                output_params['-corea0'] = ['-map', '{}:a'.format(1 if '-core_audio' in output_params else 0)]
        if '-s:v:0' in self.__params:
            del self.__params['-s:v:0']
        output_params['-s:v:0'] = '{}x{}'.format(self.__inputwidth, self.__inputheight)
        if '-b:v:0' in self.__params:
            del self.__params['-b:v:0']
        output_params['-b:v:0'] = str(get_video_bitrate(int(self.__inputwidth), int(self.__inputheight), self.__sourceframerate, bpp)) + 'k'
        if '-b:a:0' in self.__params:
            del self.__params['-b:a:0']
        a_bitrate = output_params.pop('a_bitrate', '')
        if '-acodec' in output_params and a_bitrate:
            output_params['-b:a:0'] = a_bitrate
        streams = self.__params.pop('-streams', {})
        output_params = self.__evaluate_streams(streams, output_params, bpp)
        if output_params['-vcodec'] in ['libx264', 'libx264rgb']:
            if not '-bf' in self.__params:
                output_params['-bf'] = 1
            if not '-sc_threshold' in self.__params:
                output_params['-sc_threshold'] = 0
            if not '-keyint_min' in self.__params:
                output_params['-keyint_min'] = gop
        if output_params['-vcodec'] in ['libx264', 'libx264rgb', 'libvpx-vp9']:
            if not '-g' in self.__params:
                output_params['-g'] = gop
        if output_params['-vcodec'] == 'libx265':
            output_params['-core_x265'] = ['-x265-params', 'keyint={}:min-keyint={}'.format(gop, gop)]
        processed_params = None
        if self.__format == 'dash':
            processed_params = self.__generate_dash_stream(input_params=input_params, output_params=output_params)
        else:
            processed_params = self.__generate_hls_stream(input_params=input_params, output_params=output_params)
        return processed_params

    def __evaluate_streams(self, streams, output_params, bpp):
        if False:
            print('Hello World!')
        '\n        Internal function that Extracts, Evaluates & Validates user-defined streams\n\n        Parameters:\n            streams (dict): Indivisual streams formatted as list of dict.\n            output_params (dict): Output FFmpeg parameters\n        '
        output_params['stream_count'] = 1
        if not streams:
            logger.warning('No `-streams` are provided!')
            return output_params
        if isinstance(streams, list) and all((isinstance(x, dict) for x in streams)):
            stream_count = 1
            source_aspect_ratio = self.__inputwidth / self.__inputheight
            self.__logging and logger.debug('Processing {} streams.'.format(len(streams)))
            for stream in streams:
                stream_copy = stream.copy()
                intermediate_dict = {}
                if self.__format != 'hls':
                    intermediate_dict['-core{}'.format(stream_count)] = ['-map', '0']
                else:
                    intermediate_dict['-corev{}'.format(stream_count)] = ['-map', '0:v']
                    if '-acodec' in output_params:
                        intermediate_dict['-corea{}'.format(stream_count)] = ['-map', '{}:a'.format(1 if '-core_audio' in output_params else 0)]
                resolution = stream.pop('-resolution', '')
                dimensions = resolution.lower().split('x') if resolution and isinstance(resolution, str) else []
                if len(dimensions) == 2 and dimensions[0].isnumeric() and dimensions[1].isnumeric():
                    expected_width = math.floor(int(dimensions[1]) * source_aspect_ratio)
                    if int(dimensions[0]) != expected_width:
                        logger.warning('Given stream resolution `{}` is not in accordance with the Source Aspect-Ratio. Stream Output may appear Distorted!'.format(resolution))
                    intermediate_dict['-s:v:{}'.format(stream_count)] = resolution
                else:
                    logger.error('Missing `-resolution` value, Stream `{}` Skipped!'.format(stream_copy))
                    continue
                video_bitrate = stream.pop('-video_bitrate', '')
                if video_bitrate and isinstance(video_bitrate, str) and video_bitrate.endswith(('k', 'M')):
                    intermediate_dict['-b:v:{}'.format(stream_count)] = video_bitrate
                else:
                    fps = stream.pop('-framerate', 0.0)
                    if dimensions and isinstance(fps, (float, int)) and (fps > 0):
                        intermediate_dict['-b:v:{}'.format(stream_count)] = '{}k'.format(get_video_bitrate(int(dimensions[0]), int(dimensions[1]), fps, bpp))
                    else:
                        logger.error('Unable to determine Video-Bitrate for the stream `{}`, Skipped!'.format(stream_copy))
                        continue
                audio_bitrate = stream.pop('-audio_bitrate', '')
                if '-acodec' in output_params:
                    if audio_bitrate and audio_bitrate.endswith(('k', 'M')):
                        intermediate_dict['-b:a:{}'.format(stream_count)] = audio_bitrate
                    elif dimensions:
                        aspect_width = int(dimensions[0])
                        intermediate_dict['-b:a:{}'.format(stream_count)] = '{}k'.format(128 if aspect_width > 800 else 96)
                output_params.update(intermediate_dict)
                intermediate_dict.clear()
                stream_copy.clear()
                stream_count += 1
            output_params['stream_count'] = stream_count
            self.__logging and logger.debug('All streams processed successfully!')
        else:
            logger.warning('Invalid type `-streams` skipped!')
        return output_params

    def __generate_hls_stream(self, input_params, output_params):
        if False:
            i = 10
            return i + 15
        '\n        An internal function that parses user-defined parameters and generates\n        suitable FFmpeg Terminal Command for transcoding input into HLS Stream.\n\n        Parameters:\n            input_params (dict): Input FFmpeg parameters\n            output_params (dict): Output FFmpeg parameters\n        '
        default_hls_segment_type = self.__params.pop('-hls_segment_type', 'mpegts')
        if isinstance(default_hls_segment_type, str) and default_hls_segment_type.strip() in ['fmp4', 'mpegts']:
            output_params['-hls_segment_type'] = default_hls_segment_type.strip()
        else:
            output_params['-hls_segment_type'] = 'mpegts'
        if self.__livestreaming:
            default_hls_list_size = self.__params.pop('-hls_list_size', 6)
            if isinstance(default_hls_list_size, int) and default_hls_list_size > 0:
                output_params['-hls_list_size'] = default_hls_list_size
            else:
                output_params['-hls_list_size'] = 6
            output_params['-hls_init_time'] = self.__params.pop('-hls_init_time', 4)
            output_params['-hls_time'] = self.__params.pop('-hls_time', 6)
            output_params['-hls_flags'] = self.__params.pop('-hls_flags', 'delete_segments+discont_start+split_by_time')
            output_params['-remove_at_exit'] = self.__params.pop('-remove_at_exit', 0)
        else:
            output_params['-hls_list_size'] = 0
            output_params['-hls_playlist_type'] = 'vod'
        output_params['-hls_base_url'] = self.__params.pop('-hls_base_url', '')
        output_params['-allowed_extensions'] = 'ALL'
        segment_template = '{}-stream%v-%03d.{}' if output_params['stream_count'] > 1 else '{}-stream-%03d.{}'
        output_params['-hls_segment_filename'] = segment_template.format(os.path.join(os.path.dirname(self.__out_file), 'chunk'), 'm4s' if output_params['-hls_segment_type'] == 'fmp4' else 'ts')
        output_params['-hls_allow_cache'] = 0
        output_params['-f'] = 'hls'
        return (input_params, output_params)

    def __generate_dash_stream(self, input_params, output_params):
        if False:
            return 10
        '\n        An internal function that parses user-defined parameters and generates\n        suitable FFmpeg Terminal Command for transcoding input into MPEG-dash Stream.\n\n        Parameters:\n            input_params (dict): Input FFmpeg parameters\n            output_params (dict): Output FFmpeg parameters\n        '
        if self.__livestreaming:
            output_params['-window_size'] = self.__params.pop('-window_size', 5)
            output_params['-extra_window_size'] = self.__params.pop('-extra_window_size', 5)
            output_params['-remove_at_exit'] = self.__params.pop('-remove_at_exit', 0)
            output_params['-seg_duration'] = self.__params.pop('-seg_duration', 20)
            output_params['-use_timeline'] = 0
        else:
            output_params['-seg_duration'] = self.__params.pop('-seg_duration', 5)
            output_params['-use_timeline'] = 1
        output_params['-use_template'] = 1
        output_params['-adaptation_sets'] = 'id=0,streams=v {}'.format('id=1,streams=a' if '-acodec' in output_params else '')
        output_params['-f'] = 'dash'
        return (input_params, output_params)

    def __Build_n_Execute(self, input_params, output_params):
        if False:
            return 10
        '\n        An Internal function that launches FFmpeg subprocess and pipelines commands.\n\n        Parameters:\n            input_params (dict): Input FFmpeg parameters\n            output_params (dict): Output FFmpeg parameters\n        '
        if '-core_asource' in output_params:
            output_params.move_to_end('-core_asource', last=False)
        if '-i' in output_params:
            output_params.move_to_end('-i', last=False)
        stream_count = output_params.pop('stream_count', 1)
        input_commands = dict2Args(input_params)
        output_commands = dict2Args(output_params)
        stream_commands = dict2Args(self.__params)
        hls_commands = []
        if self.__format == 'hls' and stream_count > 1:
            stream_map = ''
            for count in range(0, stream_count):
                stream_map += 'v:{}{} '.format(count, ',a:{}'.format(count) if '-acodec' in output_params else ',')
            hls_commands += ['-master_pl_name', os.path.basename(self.__out_file), '-var_stream_map', stream_map.strip(), os.path.join(os.path.dirname(self.__out_file), 'stream_%v.m3u8')]
        if self.__logging:
            logger.debug('User-Defined Output parameters: `{}`'.format(' '.join(output_commands) if output_commands else None))
            logger.debug('Additional parameters: `{}`'.format(' '.join(stream_commands) if stream_commands else None))
        ffmpeg_cmd = None
        hide_banner = [] if self.__logging else ['-hide_banner']
        if self.__video_source:
            ffmpeg_cmd = [self.__ffmpeg, '-y'] + (['-re'] if self.__livestreaming else []) + hide_banner + ['-i', self.__video_source] + input_commands + output_commands + stream_commands
        else:
            ffmpeg_cmd = [self.__ffmpeg, '-y'] + hide_banner + ['-f', 'rawvideo', '-vcodec', 'rawvideo'] + input_commands + ['-i', '-'] + output_commands + stream_commands
        ffmpeg_cmd.extend([self.__out_file] if not hls_commands else hls_commands)
        logger.critical('Transcoding streaming chunks. Please wait...')
        self.__process = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE, stdout=sp.DEVNULL if not self.__video_source and (not self.__logging) else sp.PIPE, stderr=None if self.__logging else sp.STDOUT)
        if self.__video_source:
            return_code = 0
            pbar = None
            sec_prev = 0
            if not self.__logging:
                while True:
                    data = self.__process.stdout.readline()
                    if data:
                        data = data.decode('utf-8')
                        if pbar is None:
                            if 'Duration:' in data:
                                sec_duration = extract_time(data)
                                pbar = tqdm(total=sec_duration, desc='Processing Frames', unit='frame')
                        elif 'time=' in data:
                            sec_current = extract_time(data)
                            if sec_current:
                                pbar.update(sec_current - sec_prev)
                                sec_prev = sec_current
                    elif self.__process.poll() is not None:
                        break
                return_code = self.__process.poll()
            else:
                self.__process.communicate()
                return_code = self.__process.returncode
            if pbar:
                pbar.close()
            if return_code:
                logger.error('StreamGear failed to initiate stream for this video source!')
                error = sp.CalledProcessError(return_code, ffmpeg_cmd)
                raise error
            else:
                logger.critical('Transcoding Ended. {} Streaming assets are successfully generated at specified path.'.format(self.__format.upper()))

    def __enter__(self):
        if False:
            print('Hello World!')
        "\n        Handles entry with the `with` statement. See [PEP343 -- The 'with' statement'](https://peps.python.org/pep-0343/).\n\n        **Returns:** Returns a reference to the StreamGear Class\n        "
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        "\n        Handles exit with the `with` statement. See [PEP343 -- The 'with' statement'](https://peps.python.org/pep-0343/).\n        "
        self.terminate()

    def terminate(self):
        if False:
            print('Hello World!')
        '\n        Safely terminates StreamGear.\n        '
        if self.__process is None or not self.__process.poll() is None:
            return
        if self.__process.stdin:
            self.__process.stdin.close()
        if isinstance(self.__audio, list):
            self.__process.terminate()
        self.__process.wait()
        self.__process = None
        logger.critical('Transcoding Ended. {} Streaming assets are successfully generated at specified path.'.format(self.__format.upper()))