"""Implements all the functions to read a video or a picture using ffmpeg."""
import os
import re
import subprocess as sp
import warnings
import numpy as np
from moviepy.config import FFMPEG_BINARY
from moviepy.tools import convert_to_seconds, cross_platform_popen_params

class FFMPEG_VideoReader:
    """Class for video byte-level reading with ffmpeg."""

    def __init__(self, filename, decode_file=True, print_infos=False, bufsize=None, pixel_format='rgb24', check_duration=True, target_resolution=None, resize_algo='bicubic', fps_source='fps'):
        if False:
            for i in range(10):
                print('nop')
        self.filename = filename
        self.proc = None
        infos = ffmpeg_parse_infos(filename, check_duration=check_duration, fps_source=fps_source, decode_file=decode_file, print_infos=print_infos)
        self.fps = infos['video_fps']
        self.size = infos['video_size']
        self.rotation = abs(infos.get('video_rotation', 0))
        if self.rotation in [90, 270]:
            self.size = [self.size[1], self.size[0]]
        if target_resolution:
            if None in target_resolution:
                ratio = 1
                for (idx, target) in enumerate(target_resolution):
                    if target:
                        ratio = target / self.size[idx]
                self.size = (int(self.size[0] * ratio), int(self.size[1] * ratio))
            else:
                self.size = target_resolution
        self.resize_algo = resize_algo
        self.duration = infos['video_duration']
        self.ffmpeg_duration = infos['duration']
        self.n_frames = infos['video_n_frames']
        self.bitrate = infos['video_bitrate']
        self.infos = infos
        self.pixel_format = pixel_format
        self.depth = 4 if pixel_format[-1] == 'a' else 3
        if bufsize is None:
            (w, h) = self.size
            bufsize = self.depth * w * h + 100
        self.bufsize = bufsize
        self.initialize()

    def initialize(self, start_time=0):
        if False:
            i = 10
            return i + 15
        '\n        Opens the file, creates the pipe.\n\n        Sets self.pos to the appropriate value (1 if start_time == 0 because\n        it pre-reads the first frame).\n        '
        self.close(delete_lastread=False)
        if start_time != 0:
            offset = min(1, start_time)
            i_arg = ['-ss', '%.06f' % (start_time - offset), '-i', self.filename, '-ss', '%.06f' % offset]
        else:
            i_arg = ['-i', self.filename]
        cmd = [FFMPEG_BINARY] + i_arg + ['-loglevel', 'error', '-f', 'image2pipe', '-vf', 'scale=%d:%d' % tuple(self.size), '-sws_flags', self.resize_algo, '-pix_fmt', self.pixel_format, '-vcodec', 'rawvideo', '-']
        popen_params = cross_platform_popen_params({'bufsize': self.bufsize, 'stdout': sp.PIPE, 'stderr': sp.PIPE, 'stdin': sp.DEVNULL})
        self.proc = sp.Popen(cmd, **popen_params)
        self.pos = self.get_frame_number(start_time)
        self.lastread = self.read_frame()

    def skip_frames(self, n=1):
        if False:
            while True:
                i = 10
        'Reads and throws away n frames'
        (w, h) = self.size
        for i in range(n):
            self.proc.stdout.read(self.depth * w * h)
        self.pos += n

    def read_frame(self):
        if False:
            print('Hello World!')
        '\n        Reads the next frame from the file.\n        Note that upon (re)initialization, the first frame will already have been read\n        and stored in ``self.lastread``.\n        '
        (w, h) = self.size
        nbytes = self.depth * w * h
        s = self.proc.stdout.read(nbytes)
        if len(s) != nbytes:
            warnings.warn('In file %s, %d bytes wanted but %d bytes read at frame index %d (out of a total %d frames), at time %.02f/%.02f sec. Using the last valid frame instead.' % (self.filename, nbytes, len(s), self.pos, self.n_frames, 1.0 * self.pos / self.fps, self.duration), UserWarning)
            if not hasattr(self, 'last_read'):
                raise IOError(f'MoviePy error: failed to read the first frame of video file {self.filename}. That might mean that the file is corrupted. That may also mean that you are using a deprecated version of FFMPEG. On Ubuntu/Debian for instance the version in the repos is deprecated. Please update to a recent version from the website.')
            result = self.last_read
        else:
            if hasattr(np, 'frombuffer'):
                result = np.frombuffer(s, dtype='uint8')
            else:
                result = np.fromstring(s, dtype='uint8')
            result.shape = (h, w, len(s) // (w * h))
            self.last_read = result
        self.pos += 1
        return result

    def get_frame(self, t):
        if False:
            for i in range(10):
                print('nop')
        'Read a file video frame at time t.\n\n        Note for coders: getting an arbitrary frame in the video with\n        ffmpeg can be painfully slow if some decoding has to be done.\n        This function tries to avoid fetching arbitrary frames\n        whenever possible, by moving between adjacent frames.\n        '
        pos = self.get_frame_number(t) + 1
        if not self.proc:
            print('Proc not detected')
            self.initialize(t)
            return self.last_read
        if pos == self.pos:
            return self.last_read
        elif pos < self.pos or pos > self.pos + 100:
            self.initialize(t)
            return self.lastread
        else:
            self.skip_frames(pos - self.pos - 1)
            result = self.read_frame()
            return result

    def get_frame_number(self, t):
        if False:
            for i in range(10):
                print('nop')
        'Helper method to return the frame number at time ``t``'
        return int(self.fps * t + 1e-05)

    def close(self, delete_lastread=True):
        if False:
            for i in range(10):
                print('nop')
        'Closes the reader terminating the process, if is still open.'
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.stdout.close()
                self.proc.stderr.close()
                self.proc.wait()
            self.proc = None
        if delete_lastread and hasattr(self, 'last_read'):
            del self.last_read

    def __del__(self):
        if False:
            return 10
        self.close()

def ffmpeg_read_image(filename, with_mask=True, pixel_format=None):
    if False:
        while True:
            i = 10
    "Read an image file (PNG, BMP, JPEG...).\n\n    Wraps FFMPEG_Videoreader to read just one image.\n    Returns an ImageClip.\n\n    This function is not meant to be used directly in MoviePy.\n    Use ImageClip instead to make clips out of image files.\n\n    Parameters\n    ----------\n\n    filename\n      Name of the image file. Can be of any format supported by ffmpeg.\n\n    with_mask\n      If the image has a transparency layer, ``with_mask=true`` will save\n      this layer as the mask of the returned ImageClip\n\n    pixel_format\n      Optional: Pixel format for the image to read. If is not specified\n      'rgb24' will be used as the default format unless ``with_mask`` is set\n      as ``True``, then 'rgba' will be used.\n\n    "
    if not pixel_format:
        pixel_format = 'rgba' if with_mask else 'rgb24'
    reader = FFMPEG_VideoReader(filename, pixel_format=pixel_format, check_duration=False)
    im = reader.last_read
    del reader
    return im

class FFmpegInfosParser:
    """Finite state ffmpeg `-i` command option file information parser.
    Is designed to parse the output fast, in one loop. Iterates line by
    line of the `ffmpeg -i <filename> [-f null -]` command output changing
    the internal state of the parser.

    Parameters
    ----------

    filename
      Name of the file parsed, only used to raise accurate error messages.

    infos
      Information returned by FFmpeg.

    fps_source
      Indicates what source data will be preferably used to retrieve fps data.

    check_duration
      Enable or disable the parsing of the duration of the file. Useful to
      skip the duration check, for example, for images.

    decode_file
      Indicates if the whole file has been decoded. The duration parsing strategy
      will differ depending on this argument.
    """

    def __init__(self, infos, filename, fps_source='fps', check_duration=True, decode_file=False):
        if False:
            return 10
        self.infos = infos
        self.filename = filename
        self.check_duration = check_duration
        self.fps_source = fps_source
        self.duration_tag_separator = 'time=' if decode_file else 'Duration: '
        self._reset_state()

    def _reset_state(self):
        if False:
            return 10
        'Reinitializes the state of the parser. Used internally at\n        initialization and at the end of the parsing process.\n        '
        self._inside_file_metadata = False
        self._inside_output = False
        self._default_stream_found = False
        self._current_input_file = {'streams': []}
        self._current_stream = None
        self._current_chapter = None
        self.result = {'video_found': False, 'audio_found': False, 'metadata': {}, 'inputs': []}
        self._last_metadata_field_added = None

    def parse(self):
        if False:
            return 10
        'Parses the information returned by FFmpeg in stderr executing their binary\n        for a file with ``-i`` option and returns a dictionary with all data needed\n        by MoviePy.\n        '
        input_chapters = []
        for line in self.infos.splitlines()[1:]:
            if self.duration_tag_separator == 'time=' and self.check_duration and ('time=' in line):
                self.result['duration'] = self.parse_duration(line)
            elif self._inside_output or line[0] != ' ':
                if self.duration_tag_separator == 'time=' and (not self._inside_output):
                    self._inside_output = True
            elif not self._inside_file_metadata and line.startswith('  Metadata:'):
                self._inside_file_metadata = True
            elif line.startswith('  Duration:'):
                self._inside_file_metadata = False
                if self.check_duration and self.duration_tag_separator == 'Duration: ':
                    self.result['duration'] = self.parse_duration(line)
                bitrate_match = re.search('bitrate: (\\d+) kb/s', line)
                self.result['bitrate'] = int(bitrate_match.group(1)) if bitrate_match else None
                start_match = re.search('start: (\\d+\\.?\\d+)', line)
                self.result['start'] = float(start_match.group(1)) if start_match else None
            elif self._inside_file_metadata:
                (field, value) = self.parse_metadata_field_value(line)
                if field == '':
                    field = self._last_metadata_field_added
                    value = self.result['metadata'][field] + '\n' + value
                else:
                    self._last_metadata_field_added = field
                self.result['metadata'][field] = value
            elif line.lstrip().startswith('Stream '):
                if self._current_stream:
                    self._current_input_file['streams'].append(self._current_stream)
                main_info_match = re.search('^Stream\\s#(\\d+):(\\d+)(?:\\[\\w+\\])?\\(?(\\w+)?\\)?:\\s(\\w+):', line.lstrip())
                (input_number, stream_number, language, stream_type) = main_info_match.groups()
                input_number = int(input_number)
                stream_number = int(stream_number)
                stream_type_lower = stream_type.lower()
                if language == 'und':
                    language = None
                self._current_stream = {'input_number': input_number, 'stream_number': stream_number, 'stream_type': stream_type_lower, 'language': language, 'default': not self._default_stream_found or line.endswith('(default)')}
                self._default_stream_found = True
                if self._current_stream['default']:
                    self.result[f'default_{stream_type_lower}_input_number'] = input_number
                    self.result[f'default_{stream_type_lower}_stream_number'] = stream_number
                if self._current_chapter:
                    input_chapters[input_number].append(self._current_chapter)
                    self._current_chapter = None
                if 'input_number' not in self._current_input_file:
                    self._current_input_file['input_number'] = input_number
                elif self._current_input_file['input_number'] != input_number:
                    if len(input_chapters) >= input_number + 1:
                        self._current_input_file['chapters'] = input_chapters[input_number]
                    self.result['inputs'].append(self._current_input_file)
                    self._current_input_file = {'input_number': input_number}
                try:
                    (global_data, stream_data) = self.parse_data_by_stream_type(stream_type, line)
                except NotImplementedError as exc:
                    warnings.warn(f'{str(exc)}\nffmpeg output:\n\n{self.infos}', UserWarning)
                else:
                    self.result.update(global_data)
                    self._current_stream.update(stream_data)
            elif line.startswith('    Metadata:'):
                continue
            elif self._current_stream:
                if 'metadata' not in self._current_stream:
                    self._current_stream['metadata'] = {}
                (field, value) = self.parse_metadata_field_value(line)
                if self._current_stream['stream_type'] == 'video':
                    (field, value) = self.video_metadata_type_casting(field, value)
                    if field == 'rotate':
                        self.result['video_rotation'] = value
                if field == '':
                    field = self._last_metadata_field_added
                    value = self._current_stream['metadata'][field] + '\n' + value
                else:
                    self._last_metadata_field_added = field
                self._current_stream['metadata'][field] = value
            elif line.startswith('    Chapter'):
                if self._current_chapter:
                    if len(input_chapters) < self._current_chapter['input_number'] + 1:
                        input_chapters.append([])
                    input_chapters[self._current_chapter['input_number']].append(self._current_chapter)
                chapter_data_match = re.search('^    Chapter #(\\d+):(\\d+): start (\\d+\\.?\\d+?), end (\\d+\\.?\\d+?)', line)
                (input_number, chapter_number, start, end) = chapter_data_match.groups()
                self._current_chapter = {'input_number': int(input_number), 'chapter_number': int(chapter_number), 'start': float(start), 'end': float(end)}
            elif self._current_chapter:
                if 'metadata' not in self._current_chapter:
                    self._current_chapter['metadata'] = {}
                (field, value) = self.parse_metadata_field_value(line)
                if field == '':
                    field = self._last_metadata_field_added
                    value = self._current_chapter['metadata'][field] + '\n' + value
                else:
                    self._last_metadata_field_added = field
                self._current_chapter['metadata'][field] = value
        if self._current_input_file:
            self._current_input_file['streams'].append(self._current_stream)
            if len(input_chapters) == self._current_input_file['input_number'] + 1:
                self._current_input_file['chapters'] = input_chapters[self._current_input_file['input_number']]
            self.result['inputs'].append(self._current_input_file)
        if self.result['video_found'] and self.check_duration:
            self.result['video_n_frames'] = int(self.result['duration'] * self.result['video_fps'])
            self.result['video_duration'] = self.result['duration']
        else:
            self.result['video_n_frames'] = 1
            self.result['video_duration'] = None
        if self.result['audio_found'] and (not self.result.get('audio_bitrate')):
            self.result['audio_bitrate'] = None
            for streams_input in self.result['inputs']:
                for stream in streams_input['streams']:
                    if stream['stream_type'] == 'audio' and stream.get('bitrate'):
                        self.result['audio_bitrate'] = stream['bitrate']
                        break
                if self.result['audio_bitrate'] is not None:
                    break
        result = self.result
        self._reset_state()
        return result

    def parse_data_by_stream_type(self, stream_type, line):
        if False:
            return 10
        'Parses data from "Stream ... {stream_type}" line.'
        try:
            return {'Audio': self.parse_audio_stream_data, 'Video': self.parse_video_stream_data, 'Data': lambda _line: ({}, {})}[stream_type](line)
        except KeyError:
            raise NotImplementedError(f'{stream_type} stream parsing is not supported by moviepy and will be ignored')

    def parse_audio_stream_data(self, line):
        if False:
            while True:
                i = 10
        'Parses data from "Stream ... Audio" line.'
        (global_data, stream_data) = ({'audio_found': True}, {})
        try:
            stream_data['fps'] = int(re.search(' (\\d+) Hz', line).group(1))
        except (AttributeError, ValueError):
            stream_data['fps'] = 'unknown'
        match_audio_bitrate = re.search('(\\d+) kb/s', line)
        stream_data['bitrate'] = int(match_audio_bitrate.group(1)) if match_audio_bitrate else None
        if self._current_stream['default']:
            global_data['audio_fps'] = stream_data['fps']
            global_data['audio_bitrate'] = stream_data['bitrate']
        return (global_data, stream_data)

    def parse_video_stream_data(self, line):
        if False:
            while True:
                i = 10
        'Parses data from "Stream ... Video" line.'
        (global_data, stream_data) = ({'video_found': True}, {})
        try:
            match_video_size = re.search(' (\\d+)x(\\d+)[,\\s]', line)
            if match_video_size:
                stream_data['size'] = [int(num) for num in match_video_size.groups()]
        except Exception:
            raise IOError("MoviePy error: failed to read video dimensions in file '%s'.\nHere are the file infos returned byffmpeg:\n\n%s" % (self.filename, self.infos))
        match_bitrate = re.search('(\\d+) kb/s', line)
        stream_data['bitrate'] = int(match_bitrate.group(1)) if match_bitrate else None
        if self.fps_source == 'fps':
            try:
                fps = self.parse_fps(line)
            except (AttributeError, ValueError):
                fps = self.parse_tbr(line)
        elif self.fps_source == 'tbr':
            try:
                fps = self.parse_tbr(line)
            except (AttributeError, ValueError):
                fps = self.parse_fps(line)
        else:
            raise ValueError("fps source '%s' not supported parsing the video '%s'" % (self.fps_source, self.filename))
        coef = 1000.0 / 1001.0
        for x in [23, 24, 25, 30, 50]:
            if fps != x and abs(fps - x * coef) < 0.01:
                fps = x * coef
        stream_data['fps'] = fps
        if self._current_stream['default'] or 'video_size' not in self.result:
            global_data['video_size'] = stream_data.get('size', None)
        if self._current_stream['default'] or 'video_bitrate' not in self.result:
            global_data['video_bitrate'] = stream_data.get('bitrate', None)
        if self._current_stream['default'] or 'video_fps' not in self.result:
            global_data['video_fps'] = stream_data['fps']
        return (global_data, stream_data)

    def parse_fps(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Parses number of FPS from a line of the ``ffmpeg -i`` command output.'
        return float(re.search(' (\\d+.?\\d*) fps', line).group(1))

    def parse_tbr(self, line):
        if False:
            return 10
        'Parses number of TBS from a line of the ``ffmpeg -i`` command output.'
        s_tbr = re.search(' (\\d+.?\\d*k?) tbr', line).group(1)
        if s_tbr[-1] == 'k':
            tbr = float(s_tbr[:-1]) * 1000
        else:
            tbr = float(s_tbr)
        return tbr

    def parse_duration(self, line):
        if False:
            print('Hello World!')
        'Parse the duration from the line that outputs the duration of\n        the container.\n        '
        try:
            time_raw_string = line.split(self.duration_tag_separator)[-1]
            match_duration = re.search('([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])', time_raw_string)
            return convert_to_seconds(match_duration.group(1))
        except Exception:
            raise IOError("MoviePy error: failed to read the duration of file '%s'.\nHere are the file infos returned by ffmpeg:\n\n%s" % (self.filename, self.infos))

    def parse_metadata_field_value(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Returns a tuple with a metadata field-value pair given a ffmpeg `-i`\n        command output line.\n        '
        (raw_field, raw_value) = line.split(':', 1)
        return (raw_field.strip(' '), raw_value.strip(' '))

    def video_metadata_type_casting(self, field, value):
        if False:
            return 10
        'Cast needed video metadata fields to other types than the default str.'
        if field == 'rotate':
            return (field, float(value))
        return (field, value)

def ffmpeg_parse_infos(filename, check_duration=True, fps_source='fps', decode_file=False, print_infos=False):
    if False:
        return 10
    'Get the information of a file using ffmpeg.\n\n    Returns a dictionary with next fields:\n\n    - ``"duration"``\n    - ``"metadata"``\n    - ``"inputs"``\n    - ``"video_found"``\n    - ``"video_fps"``\n    - ``"video_n_frames"``\n    - ``"video_duration"``\n    - ``"video_bitrate"``\n    - ``"video_metadata"``\n    - ``"audio_found"``\n    - ``"audio_fps"``\n    - ``"audio_bitrate"``\n    - ``"audio_metadata"``\n\n    Note that "video_duration" is slightly smaller than "duration" to avoid\n    fetching the incomplete frames at the end, which raises an error.\n\n    Parameters\n    ----------\n\n    filename\n      Name of the file parsed, only used to raise accurate error messages.\n\n    infos\n      Information returned by FFmpeg.\n\n    fps_source\n      Indicates what source data will be preferably used to retrieve fps data.\n\n    check_duration\n      Enable or disable the parsing of the duration of the file. Useful to\n      skip the duration check, for example, for images.\n\n    decode_file\n      Indicates if the whole file must be read to retrieve their duration.\n      This is needed for some files in order to get the correct duration (see\n      https://github.com/Zulko/moviepy/pull/1222).\n    '
    cmd = [FFMPEG_BINARY, '-hide_banner', '-i', filename]
    if decode_file:
        cmd.extend(['-f', 'null', '-'])
    popen_params = cross_platform_popen_params({'bufsize': 10 ** 5, 'stdout': sp.PIPE, 'stderr': sp.PIPE, 'stdin': sp.DEVNULL})
    proc = sp.Popen(cmd, **popen_params)
    (output, error) = proc.communicate()
    infos = error.decode('utf8', errors='ignore')
    proc.terminate()
    del proc
    if print_infos:
        print(infos)
    try:
        return FFmpegInfosParser(infos, filename, fps_source=fps_source, check_duration=check_duration, decode_file=decode_file).parse()
    except Exception as exc:
        if os.path.isdir(filename):
            raise IsADirectoryError(f"'{filename}' is a directory")
        elif not os.path.exists(filename):
            raise FileNotFoundError(f"'{filename}' not found")
        raise IOError(f'Error passing `ffmpeg -i` command output:\n\n{infos}') from exc