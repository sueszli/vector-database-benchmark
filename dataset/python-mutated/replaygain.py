import collections
import enum
import math
import optparse
import os
import queue
import signal
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from multiprocessing.pool import ThreadPool
from threading import Event, Thread
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union, cast
from confuse import ConfigView
from beets import ui
from beets.importer import ImportSession, ImportTask
from beets.library import Album, Item, Library
from beets.plugins import BeetsPlugin
from beets.util import command_output, cpu_count, displayable_path, py3_path, syspath

class ReplayGainError(Exception):
    """Raised when a local (to a track or an album) error occurs in one
    of the backends.
    """

class FatalReplayGainError(Exception):
    """Raised when a fatal error occurs in one of the backends."""

class FatalGstreamerPluginReplayGainError(FatalReplayGainError):
    """Raised when a fatal error occurs in the GStreamerBackend when
    loading the required plugins."""

def call(args: List[Any], log: Logger, **kwargs: Any):
    if False:
        for i in range(10):
            print('nop')
    'Execute the command and return its output or raise a\n    ReplayGainError on failure.\n    '
    try:
        return command_output(args, **kwargs)
    except subprocess.CalledProcessError as e:
        log.debug(e.output.decode('utf8', 'ignore'))
        raise ReplayGainError('{} exited with status {}'.format(args[0], e.returncode))
    except UnicodeEncodeError:
        raise ReplayGainError('argument encoding failed')

def db_to_lufs(db: float) -> float:
    if False:
        i = 10
        return i + 15
    'Convert db to LUFS.\n\n    According to https://wiki.hydrogenaud.io/index.php?title=\n      ReplayGain_2.0_specification#Reference_level\n    '
    return db - 107

def lufs_to_db(db: float) -> float:
    if False:
        while True:
            i = 10
    'Convert LUFS to db.\n\n    According to https://wiki.hydrogenaud.io/index.php?title=\n      ReplayGain_2.0_specification#Reference_level\n    '
    return db + 107

@dataclass
class Gain:
    gain: float
    peak: float

class PeakMethod(enum.Enum):
    true = 1
    sample = 2

class RgTask:
    """State and methods for a single replaygain calculation (rg version).

    Bundles the state (parameters and results) of a single replaygain
    calculation (either for one item, one disk, or one full album).

    This class provides methods to store the resulting gains and peaks as plain
    old rg tags.
    """

    def __init__(self, items: Sequence[Item], album: Optional[Album], target_level: float, peak_method: Optional[PeakMethod], backend_name: str, log: Logger):
        if False:
            print('Hello World!')
        self.items = items
        self.album = album
        self.target_level = target_level
        self.peak_method = peak_method
        self.backend_name = backend_name
        self._log = log
        self.album_gain: Optional[Gain] = None
        self.track_gains: Optional[List[Gain]] = None

    def _store_track_gain(self, item: Item, track_gain: Gain):
        if False:
            print('Hello World!')
        'Store track gain for a single item in the database.'
        item.rg_track_gain = track_gain.gain
        item.rg_track_peak = track_gain.peak
        item.store()
        self._log.debug('applied track gain {0} LU, peak {1} of FS', item.rg_track_gain, item.rg_track_peak)

    def _store_album_gain(self, item: Item, album_gain: Gain):
        if False:
            i = 10
            return i + 15
        'Store album gain for a single item in the database.\n\n        The caller needs to ensure that `self.album_gain is not None`.\n        '
        item.rg_album_gain = album_gain.gain
        item.rg_album_peak = album_gain.peak
        item.store()
        self._log.debug('applied album gain {0} LU, peak {1} of FS', item.rg_album_gain, item.rg_album_peak)

    def _store_track(self, write: bool):
        if False:
            print('Hello World!')
        'Store track gain for the first track of the task in the database.'
        item = self.items[0]
        if self.track_gains is None or len(self.track_gains) != 1:
            raise ReplayGainError('ReplayGain backend `{}` failed for track {}'.format(self.backend_name, item))
        self._store_track_gain(item, self.track_gains[0])
        if write:
            item.try_write()
        self._log.debug('done analyzing {0}', item)

    def _store_album(self, write: bool):
        if False:
            i = 10
            return i + 15
        'Store track/album gains for all tracks of the task in the database.'
        if self.album_gain is None or self.track_gains is None or len(self.track_gains) != len(self.items):
            raise ReplayGainError('ReplayGain backend `{}` failed for some tracks in album {}'.format(self.backend_name, self.album))
        for (item, track_gain) in zip(self.items, self.track_gains):
            self._store_track_gain(item, track_gain)
            self._store_album_gain(item, self.album_gain)
            if write:
                item.try_write()
            self._log.debug('done analyzing {0}', item)

    def store(self, write: bool):
        if False:
            i = 10
            return i + 15
        'Store computed gains for the items of this task in the database.'
        if self.album is not None:
            self._store_album(write)
        else:
            self._store_track(write)

class R128Task(RgTask):
    """State and methods for a single replaygain calculation (r128 version).

    Bundles the state (parameters and results) of a single replaygain
    calculation (either for one item, one disk, or one full album).

    This class provides methods to store the resulting gains and peaks as R128
    tags.
    """

    def __init__(self, items: Sequence[Item], album: Optional[Album], target_level: float, backend_name: str, log: Logger):
        if False:
            return 10
        super().__init__(items, album, target_level, None, backend_name, log)

    def _store_track_gain(self, item: Item, track_gain: Gain):
        if False:
            return 10
        item.r128_track_gain = track_gain.gain
        item.store()
        self._log.debug('applied r128 track gain {0} LU', item.r128_track_gain)

    def _store_album_gain(self, item: Item, album_gain: Gain):
        if False:
            return 10
        '\n\n        The caller needs to ensure that `self.album_gain is not None`.\n        '
        item.r128_album_gain = album_gain.gain
        item.store()
        self._log.debug('applied r128 album gain {0} LU', item.r128_album_gain)
AnyRgTask = TypeVar('AnyRgTask', bound=RgTask)

class Backend(ABC):
    """An abstract class representing engine for calculating RG values."""
    NAME = ''
    do_parallel = False

    def __init__(self, config: ConfigView, log: Logger):
        if False:
            i = 10
            return i + 15
        'Initialize the backend with the configuration view for the\n        plugin.\n        '
        self._log = log

    @abstractmethod
    def compute_track_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            print('Hello World!')
        'Computes the track gain for the tracks belonging to `task`, and sets\n        the `track_gains` attribute on the task. Returns `task`.\n        '
        raise NotImplementedError()

    @abstractmethod
    def compute_album_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            print('Hello World!')
        'Computes the album gain for the album belonging to `task`, and sets\n        the `album_gain` attribute on the task. Returns `task`.\n        '
        raise NotImplementedError()

class FfmpegBackend(Backend):
    """A replaygain backend using ffmpeg's ebur128 filter."""
    NAME = 'ffmpeg'
    do_parallel = True

    def __init__(self, config: ConfigView, log: Logger):
        if False:
            print('Hello World!')
        super().__init__(config, log)
        self._ffmpeg_path = 'ffmpeg'
        try:
            ffmpeg_version_out = call([self._ffmpeg_path, '-version'], log)
        except OSError:
            raise FatalReplayGainError(f'could not find ffmpeg at {self._ffmpeg_path}')
        incompatible_ffmpeg = True
        for line in ffmpeg_version_out.stdout.splitlines():
            if line.startswith(b'configuration:'):
                if b'--enable-libebur128' in line:
                    incompatible_ffmpeg = False
            if line.startswith(b'libavfilter'):
                version = line.split(b' ', 1)[1].split(b'/', 1)[0].split(b'.')
                version = tuple(map(int, version))
                if version >= (6, 67, 100):
                    incompatible_ffmpeg = False
        if incompatible_ffmpeg:
            raise FatalReplayGainError('Installed FFmpeg version does not support ReplayGain.calculation. Either libavfilter version 6.67.100 or above orthe --enable-libebur128 configuration option is required.')

    def compute_track_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            return 10
        'Computes the track gain for the tracks belonging to `task`, and sets\n        the `track_gains` attribute on the task. Returns `task`.\n        '
        task.track_gains = [self._analyse_item(item, task.target_level, task.peak_method, count_blocks=False)[0] for item in task.items]
        return task

    def compute_album_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            for i in range(10):
                print('nop')
        'Computes the album gain for the album belonging to `task`, and sets\n        the `album_gain` attribute on the task. Returns `task`.\n        '
        target_level_lufs = db_to_lufs(task.target_level)
        track_results: List[Tuple[Gain, int]] = [self._analyse_item(item, task.target_level, task.peak_method, count_blocks=True) for item in task.items]
        track_gains: List[Gain] = [tg for (tg, _nb) in track_results]
        album_peak = max((tg.peak for tg in track_gains))
        n_blocks = sum((nb for (_tg, nb) in track_results))

        def sum_of_track_powers(track_gain: Gain, track_n_blocks: int):
            if False:
                print('Hello World!')
            loudness = target_level_lufs - track_gain.gain
            power = 10 ** ((loudness + 0.691) / 10)
            return track_n_blocks * power
        if n_blocks > 0:
            sum_powers = sum((sum_of_track_powers(tg, nb) for (tg, nb) in track_results))
            album_gain = -0.691 + 10 * math.log10(sum_powers / n_blocks)
        else:
            album_gain = -70
        album_gain = target_level_lufs - album_gain
        self._log.debug('{}: gain {} LU, peak {}', task.album, album_gain, album_peak)
        task.album_gain = Gain(album_gain, album_peak)
        task.track_gains = track_gains
        return task

    def _construct_cmd(self, item: Item, peak_method: Optional[PeakMethod]) -> List[Union[str, bytes]]:
        if False:
            i = 10
            return i + 15
        'Construct the shell command to analyse items.'
        return [self._ffmpeg_path, '-nostats', '-hide_banner', '-i', item.path, '-map', 'a:0', '-filter', 'ebur128=peak={}'.format('none' if peak_method is None else peak_method.name), '-f', 'null', '-']

    def _analyse_item(self, item: Item, target_level: float, peak_method: Optional[PeakMethod], count_blocks: bool=True) -> Tuple[Gain, int]:
        if False:
            i = 10
            return i + 15
        'Analyse item. Return a pair of a Gain object and the number\n        of gating blocks above the threshold.\n\n        If `count_blocks` is False, the number of gating blocks returned\n        will be 0.\n        '
        target_level_lufs = db_to_lufs(target_level)
        self._log.debug(f'analyzing {item}')
        cmd = self._construct_cmd(item, peak_method)
        self._log.debug('executing {0}', ' '.join(map(displayable_path, cmd)))
        output = call(cmd, self._log).stderr.splitlines()
        if peak_method is None:
            peak = 0.0
        else:
            line_peak = self._find_line(output, f'  {peak_method.name.capitalize()} peak:'.encode(), start_line=len(output) - 1, step_size=-1)
            peak = self._parse_float(output[self._find_line(output, b'    Peak:', line_peak)])
            peak = 10 ** (peak / 20)
        line_integrated_loudness = self._find_line(output, b'  Integrated loudness:', start_line=len(output) - 1, step_size=-1)
        gain = self._parse_float(output[self._find_line(output, b'    I:', line_integrated_loudness)])
        gain = target_level_lufs - gain
        n_blocks = 0
        if count_blocks:
            gating_threshold = self._parse_float(output[self._find_line(output, b'    Threshold:', start_line=line_integrated_loudness)])
            for line in output:
                if not line.startswith(b'[Parsed_ebur128'):
                    continue
                if line.endswith(b'Summary:'):
                    continue
                line = line.split(b'M:', 1)
                if len(line) < 2:
                    continue
                if self._parse_float(b'M: ' + line[1]) >= gating_threshold:
                    n_blocks += 1
            self._log.debug('{}: {} blocks over {} LUFS'.format(item, n_blocks, gating_threshold))
        self._log.debug('{}: gain {} LU, peak {}'.format(item, gain, peak))
        return (Gain(gain, peak), n_blocks)

    def _find_line(self, output: Sequence[bytes], search: bytes, start_line: int=0, step_size: int=1) -> int:
        if False:
            while True:
                i = 10
        'Return index of line beginning with `search`.\n\n        Begins searching at index `start_line` in `output`.\n        '
        end_index = len(output) if step_size > 0 else -1
        for i in range(start_line, end_index, step_size):
            if output[i].startswith(search):
                return i
        raise ReplayGainError('ffmpeg output: missing {} after line {}'.format(repr(search), start_line))

    def _parse_float(self, line: bytes) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Extract a float from a key value pair in `line`.\n\n        This format is expected: /[^:]:[[:space:]]*value.*/, where `value` is\n        the float.\n        '
        parts = line.split(b':', 1)
        if len(parts) < 2:
            raise ReplayGainError(f'ffmpeg output: expected key value pair, found {line!r}')
        value = parts[1].lstrip()
        value = value.split(b' ', 1)[0]
        try:
            return float(value)
        except ValueError:
            raise ReplayGainError(f'ffmpeg output: expected float value, found {value!r}')

class CommandBackend(Backend):
    NAME = 'command'
    do_parallel = True

    def __init__(self, config: ConfigView, log: Logger):
        if False:
            i = 10
            return i + 15
        super().__init__(config, log)
        config.add({'command': '', 'noclip': True})
        self.command = cast(str, config['command'].as_str())
        if self.command:
            if not os.path.isfile(self.command):
                raise FatalReplayGainError('replaygain command does not exist: {}'.format(self.command))
        else:
            for cmd in ('mp3gain', 'aacgain'):
                try:
                    call([cmd, '-v'], self._log)
                    self.command = cmd
                except OSError:
                    pass
        if not self.command:
            raise FatalReplayGainError('no replaygain command found: install mp3gain or aacgain')
        self.noclip = config['noclip'].get(bool)

    def compute_track_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            i = 10
            return i + 15
        'Computes the track gain for the tracks belonging to `task`, and sets\n        the `track_gains` attribute on the task. Returns `task`.\n        '
        supported_items = list(filter(self.format_supported, task.items))
        output = self.compute_gain(supported_items, task.target_level, False)
        task.track_gains = output
        return task

    def compute_album_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            return 10
        'Computes the album gain for the album belonging to `task`, and sets\n        the `album_gain` attribute on the task. Returns `task`.\n        '
        supported_items = list(filter(self.format_supported, task.items))
        if len(supported_items) != len(task.items):
            self._log.debug('tracks are of unsupported format')
            task.album_gain = None
            task.track_gains = None
            return task
        output = self.compute_gain(supported_items, task.target_level, True)
        task.album_gain = output[-1]
        task.track_gains = output[:-1]
        return task

    def format_supported(self, item: Item) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks whether the given item is supported by the selected tool.'
        if 'mp3gain' in self.command and item.format != 'MP3':
            return False
        elif 'aacgain' in self.command and item.format not in ('MP3', 'AAC'):
            return False
        return True

    def compute_gain(self, items: Sequence[Item], target_level: float, is_album: bool) -> List[Gain]:
        if False:
            i = 10
            return i + 15
        'Computes the track or album gain of a list of items, returns\n        a list of TrackGain objects.\n\n        When computing album gain, the last TrackGain object returned is\n        the album gain\n        '
        if not items:
            self._log.debug('no supported tracks to analyze')
            return []
        'Compute ReplayGain values and return a list of results\n        dictionaries as given by `parse_tool_output`.\n        '
        cmd: List[Union[bytes, str]] = [self.command, '-o', '-s', 's']
        if self.noclip:
            cmd = cmd + ['-k']
        else:
            cmd = cmd + ['-c']
        cmd = cmd + ['-d', str(int(target_level - 89))]
        cmd = cmd + [syspath(i.path) for i in items]
        self._log.debug('analyzing {0} files', len(items))
        self._log.debug('executing {0}', ' '.join(map(displayable_path, cmd)))
        output = call(cmd, self._log).stdout
        self._log.debug('analysis finished')
        return self.parse_tool_output(output, len(items) + (1 if is_album else 0))

    def parse_tool_output(self, text: bytes, num_lines: int) -> List[Gain]:
        if False:
            i = 10
            return i + 15
        'Given the tab-delimited output from an invocation of mp3gain\n        or aacgain, parse the text and return a list of dictionaries\n        containing information about each analyzed file.\n        '
        out = []
        for line in text.split(b'\n')[1:num_lines + 1]:
            parts = line.split(b'\t')
            if len(parts) != 6 or parts[0] == b'File':
                self._log.debug('bad tool output: {0}', text)
                raise ReplayGainError('mp3gain failed')
            gain = float(parts[2])
            peak = float(parts[3]) / (1 << 15)
            out.append(Gain(gain, peak))
        return out

class GStreamerBackend(Backend):
    NAME = 'gstreamer'

    def __init__(self, config: ConfigView, log: Logger):
        if False:
            print('Hello World!')
        super().__init__(config, log)
        self._import_gst()
        self._src = self.Gst.ElementFactory.make('filesrc', 'src')
        self._decbin = self.Gst.ElementFactory.make('decodebin', 'decbin')
        self._conv = self.Gst.ElementFactory.make('audioconvert', 'conv')
        self._res = self.Gst.ElementFactory.make('audioresample', 'res')
        self._rg = self.Gst.ElementFactory.make('rganalysis', 'rg')
        if self._src is None or self._decbin is None or self._conv is None or (self._res is None) or (self._rg is None):
            raise FatalGstreamerPluginReplayGainError('Failed to load required GStreamer plugins')
        self._rg.set_property('forced', True)
        self._sink = self.Gst.ElementFactory.make('fakesink', 'sink')
        self._pipe = self.Gst.Pipeline()
        self._pipe.add(self._src)
        self._pipe.add(self._decbin)
        self._pipe.add(self._conv)
        self._pipe.add(self._res)
        self._pipe.add(self._rg)
        self._pipe.add(self._sink)
        self._src.link(self._decbin)
        self._conv.link(self._res)
        self._res.link(self._rg)
        self._rg.link(self._sink)
        self._bus = self._pipe.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect('message::eos', self._on_eos)
        self._bus.connect('message::error', self._on_error)
        self._bus.connect('message::tag', self._on_tag)
        self._decbin.connect('pad-added', self._on_pad_added)
        self._decbin.connect('pad-removed', self._on_pad_removed)
        self._main_loop = self.GLib.MainLoop()
        self._files: List[bytes] = []

    def _import_gst(self):
        if False:
            while True:
                i = 10
        'Import the necessary GObject-related modules and assign `Gst`\n        and `GObject` fields on this object.\n        '
        try:
            import gi
        except ImportError:
            raise FatalReplayGainError('Failed to load GStreamer: python-gi not found')
        try:
            gi.require_version('Gst', '1.0')
        except ValueError as e:
            raise FatalReplayGainError(f'Failed to load GStreamer 1.0: {e}')
        from gi.repository import GLib, GObject, Gst
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            GObject.threads_init()
        Gst.init([sys.argv[0]])
        self.GObject = GObject
        self.GLib = GLib
        self.Gst = Gst

    def compute(self, items: Sequence[Item], target_level: float, album: bool):
        if False:
            return 10
        if len(items) == 0:
            return
        self._error = None
        self._files = [i.path for i in items]
        self._file_tags: DefaultDict[bytes, Dict[str, float]] = collections.defaultdict(dict)
        self._rg.set_property('reference-level', target_level)
        if album:
            self._rg.set_property('num-tracks', len(self._files))
        if self._set_first_file():
            self._main_loop.run()
            if self._error is not None:
                raise self._error

    def compute_track_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            while True:
                i = 10
        'Computes the track gain for the tracks belonging to `task`, and sets\n        the `track_gains` attribute on the task. Returns `task`.\n        '
        self.compute(task.items, task.target_level, False)
        if len(self._file_tags) != len(task.items):
            raise ReplayGainError('Some tracks did not receive tags')
        ret = []
        for item in task.items:
            ret.append(Gain(self._file_tags[item.path]['TRACK_GAIN'], self._file_tags[item.path]['TRACK_PEAK']))
        task.track_gains = ret
        return task

    def compute_album_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            i = 10
            return i + 15
        'Computes the album gain for the album belonging to `task`, and sets\n        the `album_gain` attribute on the task. Returns `task`.\n        '
        items = list(task.items)
        self.compute(items, task.target_level, True)
        if len(self._file_tags) != len(items):
            raise ReplayGainError('Some items in album did not receive tags')
        track_gains = []
        for item in items:
            try:
                gain = self._file_tags[item.path]['TRACK_GAIN']
                peak = self._file_tags[item.path]['TRACK_PEAK']
            except KeyError:
                raise ReplayGainError('results missing for track')
            track_gains.append(Gain(gain, peak))
        last_tags = self._file_tags[items[-1].path]
        try:
            gain = last_tags['ALBUM_GAIN']
            peak = last_tags['ALBUM_PEAK']
        except KeyError:
            raise ReplayGainError('results missing for album')
        task.album_gain = Gain(gain, peak)
        task.track_gains = track_gains
        return task

    def close(self):
        if False:
            i = 10
            return i + 15
        self._bus.remove_signal_watch()

    def _on_eos(self, bus, message):
        if False:
            i = 10
            return i + 15
        if not self._set_next_file():
            self._pipe.set_state(self.Gst.State.NULL)
            self._main_loop.quit()

    def _on_error(self, bus, message):
        if False:
            return 10
        self._pipe.set_state(self.Gst.State.NULL)
        self._main_loop.quit()
        (err, debug) = message.parse_error()
        f = self._src.get_property('location')
        self._error = ReplayGainError(f'Error {err!r} - {debug!r} on file {f!r}')

    def _on_tag(self, bus, message):
        if False:
            for i in range(10):
                print('nop')
        tags = message.parse_tag()

        def handle_tag(taglist, tag, userdata):
            if False:
                i = 10
                return i + 15
            if tag == self.Gst.TAG_TRACK_GAIN:
                self._file_tags[self._file]['TRACK_GAIN'] = taglist.get_double(tag)[1]
            elif tag == self.Gst.TAG_TRACK_PEAK:
                self._file_tags[self._file]['TRACK_PEAK'] = taglist.get_double(tag)[1]
            elif tag == self.Gst.TAG_ALBUM_GAIN:
                self._file_tags[self._file]['ALBUM_GAIN'] = taglist.get_double(tag)[1]
            elif tag == self.Gst.TAG_ALBUM_PEAK:
                self._file_tags[self._file]['ALBUM_PEAK'] = taglist.get_double(tag)[1]
            elif tag == self.Gst.TAG_REFERENCE_LEVEL:
                self._file_tags[self._file]['REFERENCE_LEVEL'] = taglist.get_double(tag)[1]
        tags.foreach(handle_tag, None)

    def _set_first_file(self) -> bool:
        if False:
            while True:
                i = 10
        if len(self._files) == 0:
            return False
        self._file = self._files.pop(0)
        self._pipe.set_state(self.Gst.State.NULL)
        self._src.set_property('location', py3_path(syspath(self._file)))
        self._pipe.set_state(self.Gst.State.PLAYING)
        return True

    def _set_file(self) -> bool:
        if False:
            return 10
        'Initialize the filesrc element with the next file to be analyzed.'
        if len(self._files) == 0:
            return False
        self._file = self._files.pop(0)
        self._src.sync_state_with_parent()
        self._src.get_state(self.Gst.CLOCK_TIME_NONE)
        self._decbin.sync_state_with_parent()
        self._decbin.get_state(self.Gst.CLOCK_TIME_NONE)
        self._decbin.unlink(self._conv)
        self._decbin.set_state(self.Gst.State.READY)
        self._src.set_state(self.Gst.State.READY)
        self._src.set_property('location', py3_path(syspath(self._file)))
        self._decbin.link(self._conv)
        self._pipe.set_state(self.Gst.State.READY)
        return True

    def _set_next_file(self) -> bool:
        if False:
            print('Hello World!')
        'Set the next file to be analyzed while keeping the pipeline\n        in the PAUSED state so that the rganalysis element can correctly\n        handle album gain.\n        '
        self._pipe.set_state(self.Gst.State.PAUSED)
        self._pipe.get_state(self.Gst.CLOCK_TIME_NONE)
        ret = self._set_file()
        if ret:
            self._pipe.seek_simple(self.Gst.Format.TIME, self.Gst.SeekFlags.FLUSH, 0)
            self._pipe.set_state(self.Gst.State.PLAYING)
        return ret

    def _on_pad_added(self, decbin, pad):
        if False:
            i = 10
            return i + 15
        sink_pad = self._conv.get_compatible_pad(pad, None)
        assert sink_pad is not None
        pad.link(sink_pad)

    def _on_pad_removed(self, decbin, pad):
        if False:
            while True:
                i = 10
        peer = pad.get_peer()
        assert peer is None

class AudioToolsBackend(Backend):
    """ReplayGain backend that uses `Python Audio Tools
    <http://audiotools.sourceforge.net/>`_ and its capabilities to read more
    file formats and compute ReplayGain values using it replaygain module.
    """
    NAME = 'audiotools'

    def __init__(self, config: ConfigView, log: Logger):
        if False:
            print('Hello World!')
        super().__init__(config, log)
        self._import_audiotools()

    def _import_audiotools(self):
        if False:
            for i in range(10):
                print('nop')
        "Check whether it's possible to import the necessary modules.\n        There is no check on the file formats at runtime.\n\n        :raises :exc:`ReplayGainError`: if the modules cannot be imported\n        "
        try:
            import audiotools
            import audiotools.replaygain
        except ImportError:
            raise FatalReplayGainError('Failed to load audiotools: audiotools not found')
        self._mod_audiotools = audiotools
        self._mod_replaygain = audiotools.replaygain

    def open_audio_file(self, item: Item):
        if False:
            while True:
                i = 10
        'Open the file to read the PCM stream from the using\n        ``item.path``.\n\n        :return: the audiofile instance\n        :rtype: :class:`audiotools.AudioFile`\n        :raises :exc:`ReplayGainError`: if the file is not found or the\n        file format is not supported\n        '
        try:
            audiofile = self._mod_audiotools.open(py3_path(syspath(item.path)))
        except OSError:
            raise ReplayGainError(f'File {item.path} was not found')
        except self._mod_audiotools.UnsupportedFile:
            raise ReplayGainError(f'Unsupported file type {item.format}')
        return audiofile

    def init_replaygain(self, audiofile, item: Item):
        if False:
            while True:
                i = 10
        'Return an initialized :class:`audiotools.replaygain.ReplayGain`\n        instance, which requires the sample rate of the song(s) on which\n        the ReplayGain values will be computed. The item is passed in case\n        the sample rate is invalid to log the stored item sample rate.\n\n        :return: initialized replagain object\n        :rtype: :class:`audiotools.replaygain.ReplayGain`\n        :raises: :exc:`ReplayGainError` if the sample rate is invalid\n        '
        try:
            rg = self._mod_replaygain.ReplayGain(audiofile.sample_rate())
        except ValueError:
            raise ReplayGainError(f'Unsupported sample rate {item.samplerate}')
            return
        return rg

    def compute_track_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            i = 10
            return i + 15
        'Computes the track gain for the tracks belonging to `task`, and sets\n        the `track_gains` attribute on the task. Returns `task`.\n        '
        gains = [self._compute_track_gain(i, task.target_level) for i in task.items]
        task.track_gains = gains
        return task

    def _with_target_level(self, gain: float, target_level: float):
        if False:
            i = 10
            return i + 15
        'Return `gain` relative to `target_level`.\n\n        Assumes `gain` is relative to 89 db.\n        '
        return gain + (target_level - 89)

    def _title_gain(self, rg, audiofile, target_level: float):
        if False:
            print('Hello World!')
        'Get the gain result pair from PyAudioTools using the `ReplayGain`\n        instance `rg` for the given `audiofile`.\n\n        Wraps `rg.title_gain(audiofile.to_pcm())` and throws a\n        `ReplayGainError` when the library fails.\n        '
        try:
            (gain, peak) = rg.title_gain(audiofile.to_pcm())
        except ValueError as exc:
            self._log.debug('error in rg.title_gain() call: {}', exc)
            raise ReplayGainError('audiotools audio data error')
        return (self._with_target_level(gain, target_level), peak)

    def _compute_track_gain(self, item: Item, target_level: float):
        if False:
            print('Hello World!')
        'Compute ReplayGain value for the requested item.\n\n        :rtype: :class:`Gain`\n        '
        audiofile = self.open_audio_file(item)
        rg = self.init_replaygain(audiofile, item)
        (rg_track_gain, rg_track_peak) = self._title_gain(rg, audiofile, target_level)
        self._log.debug('ReplayGain for track {0} - {1}: {2:.2f}, {3:.2f}', item.artist, item.title, rg_track_gain, rg_track_peak)
        return Gain(gain=rg_track_gain, peak=rg_track_peak)

    def compute_album_gain(self, task: AnyRgTask) -> AnyRgTask:
        if False:
            while True:
                i = 10
        'Computes the album gain for the album belonging to `task`, and sets\n        the `album_gain` attribute on the task. Returns `task`.\n        '
        item = list(task.items)[0]
        audiofile = self.open_audio_file(item)
        rg = self.init_replaygain(audiofile, item)
        track_gains = []
        for item in task.items:
            audiofile = self.open_audio_file(item)
            (rg_track_gain, rg_track_peak) = self._title_gain(rg, audiofile, task.target_level)
            track_gains.append(Gain(gain=rg_track_gain, peak=rg_track_peak))
            self._log.debug('ReplayGain for track {0}: {1:.2f}, {2:.2f}', item, rg_track_gain, rg_track_peak)
        (rg_album_gain, rg_album_peak) = rg.album_gain()
        rg_album_gain = self._with_target_level(rg_album_gain, task.target_level)
        self._log.debug('ReplayGain for album {0}: {1:.2f}, {2:.2f}', task.items[0].album, rg_album_gain, rg_album_peak)
        task.album_gain = Gain(gain=rg_album_gain, peak=rg_album_peak)
        task.track_gains = track_gains
        return task

class ExceptionWatcher(Thread):
    """Monitors a queue for exceptions asynchronously.
    Once an exception occurs, raise it and execute a callback.
    """

    def __init__(self, queue: queue.Queue, callback: Callable[[], None]):
        if False:
            for i in range(10):
                print('nop')
        self._queue = queue
        self._callback = callback
        self._stopevent = Event()
        Thread.__init__(self)

    def run(self):
        if False:
            i = 10
            return i + 15
        while not self._stopevent.is_set():
            try:
                exc = self._queue.get_nowait()
                self._callback()
                raise exc
            except queue.Empty:
                pass

    def join(self, timeout: Optional[float]=None):
        if False:
            while True:
                i = 10
        self._stopevent.set()
        Thread.join(self, timeout)
BACKEND_CLASSES: List[Type[Backend]] = [CommandBackend, GStreamerBackend, AudioToolsBackend, FfmpegBackend]
BACKENDS: Dict[str, Type[Backend]] = {b.NAME: b for b in BACKEND_CLASSES}

class ReplayGainPlugin(BeetsPlugin):
    """Provides ReplayGain analysis."""

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.config.add({'overwrite': False, 'auto': True, 'backend': 'command', 'threads': cpu_count(), 'parallel_on_import': False, 'per_disc': False, 'peak': 'true', 'targetlevel': 89, 'r128': ['Opus'], 'r128_targetlevel': lufs_to_db(-23)})
        self.force_on_import = cast(bool, self.config['overwrite'].get(bool))
        self.backend_name = self.config['backend'].as_str()
        if self.backend_name not in BACKENDS:
            raise ui.UserError('Selected ReplayGain backend {} is not supported. Please select one of: {}'.format(self.backend_name, ', '.join(BACKENDS.keys())))
        peak_method = self.config['peak'].as_str()
        if peak_method not in PeakMethod.__members__:
            raise ui.UserError('Selected ReplayGain peak method {} is not supported. Please select one of: {}'.format(peak_method, ', '.join(PeakMethod.__members__)))
        self.peak_method = PeakMethod[peak_method]
        if self.config['auto']:
            self.register_listener('import_begin', self.import_begin)
            self.register_listener('import', self.import_end)
            self.import_stages = [self.imported]
        self.r128_whitelist = self.config['r128'].as_str_seq()
        try:
            self.backend_instance = BACKENDS[self.backend_name](self.config, self._log)
        except (ReplayGainError, FatalReplayGainError) as e:
            raise ui.UserError(f'replaygain initialization failed: {e}')
        self.pool = None

    def should_use_r128(self, item: Item) -> bool:
        if False:
            print('Hello World!')
        'Checks the plugin setting to decide whether the calculation\n        should be done using the EBU R128 standard and use R128_ tags instead.\n        '
        return item.format in self.r128_whitelist

    @staticmethod
    def has_r128_track_data(item: Item) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return item.r128_track_gain is not None

    @staticmethod
    def has_rg_track_data(item: Item) -> bool:
        if False:
            return 10
        return item.rg_track_gain is not None and item.rg_track_peak is not None

    def track_requires_gain(self, item: Item) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self.should_use_r128(item):
            if not self.has_r128_track_data(item):
                return True
        elif not self.has_rg_track_data(item):
            return True
        return False

    @staticmethod
    def has_r128_album_data(item: Item) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return item.r128_track_gain is not None and item.r128_album_gain is not None

    @staticmethod
    def has_rg_album_data(item: Item) -> bool:
        if False:
            return 10
        return item.rg_album_gain is not None and item.rg_album_peak is not None

    def album_requires_gain(self, album: Album) -> bool:
        if False:
            while True:
                i = 10
        for item in album.items():
            if self.should_use_r128(item):
                if not self.has_r128_album_data(item):
                    return True
            elif not self.has_rg_album_data(item):
                return True
        return False

    def create_task(self, items: Sequence[Item], use_r128: bool, album: Optional[Album]=None) -> RgTask:
        if False:
            while True:
                i = 10
        if use_r128:
            return R128Task(items, album, self.config['r128_targetlevel'].as_number(), self.backend_instance.NAME, self._log)
        else:
            return RgTask(items, album, self.config['targetlevel'].as_number(), self.peak_method, self.backend_instance.NAME, self._log)

    def handle_album(self, album: Album, write: bool, force: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "Compute album and track replay gain store it in all of the\n        album's items.\n\n        If ``write`` is truthy then ``item.write()`` is called for each\n        item. If replay gain information is already present in all\n        items, nothing is done.\n        "
        if not force and (not self.album_requires_gain(album)):
            self._log.info('Skipping album {0}', album)
            return
        items_iter = iter(album.items())
        use_r128 = self.should_use_r128(next(items_iter))
        if any((use_r128 != self.should_use_r128(i) for i in items_iter)):
            self._log.error('Cannot calculate gain for album {0} (incompatible formats)', album)
            return
        self._log.info('analyzing {0}', album)
        discs: Dict[int, List[Item]] = {}
        if self.config['per_disc'].get(bool):
            for item in album.items():
                if discs.get(item.disc) is None:
                    discs[item.disc] = []
                discs[item.disc].append(item)
        else:
            discs[1] = album.items()

        def store_cb(task: RgTask):
            if False:
                for i in range(10):
                    print('nop')
            task.store(write)
        for (discnumber, items) in discs.items():
            task = self.create_task(items, use_r128, album=album)
            try:
                self._apply(self.backend_instance.compute_album_gain, args=[task], kwds={}, callback=store_cb)
            except ReplayGainError as e:
                self._log.info('ReplayGain error: {0}', e)
            except FatalReplayGainError as e:
                raise ui.UserError(f'Fatal replay gain error: {e}')

    def handle_track(self, item: Item, write: bool, force: bool=False):
        if False:
            print('Hello World!')
        'Compute track replay gain and store it in the item.\n\n        If ``write`` is truthy then ``item.write()`` is called to write\n        the data to disk.  If replay gain information is already present\n        in the item, nothing is done.\n        '
        if not force and (not self.track_requires_gain(item)):
            self._log.info('Skipping track {0}', item)
            return
        use_r128 = self.should_use_r128(item)

        def store_cb(task: RgTask):
            if False:
                for i in range(10):
                    print('nop')
            task.store(write)
        task = self.create_task([item], use_r128)
        try:
            self._apply(self.backend_instance.compute_track_gain, args=[task], kwds={}, callback=store_cb)
        except ReplayGainError as e:
            self._log.info('ReplayGain error: {0}', e)
        except FatalReplayGainError as e:
            raise ui.UserError(f'Fatal replay gain error: {e}')

    def open_pool(self, threads: int):
        if False:
            for i in range(10):
                print('nop')
        'Open a `ThreadPool` instance in `self.pool`'
        if self.pool is None and self.backend_instance.do_parallel:
            self.pool = ThreadPool(threads)
            self.exc_queue: queue.Queue[Exception] = queue.Queue()
            signal.signal(signal.SIGINT, self._interrupt)
            self.exc_watcher = ExceptionWatcher(self.exc_queue, self.terminate_pool)
            self.exc_watcher.start()

    def _apply(self, func: Callable[..., AnyRgTask], args: List[Any], kwds: Dict[str, Any], callback: Callable[[AnyRgTask], Any]):
        if False:
            while True:
                i = 10
        if self.pool is not None:

            def handle_exc(exc):
                if False:
                    for i in range(10):
                        print('nop')
                'Handle exceptions in the async work.'
                if isinstance(exc, ReplayGainError):
                    self._log.info(exc.args[0])
                else:
                    self.exc_queue.put(exc)
            self.pool.apply_async(func, args, kwds, callback, error_callback=handle_exc)
        else:
            callback(func(*args, **kwds))

    def terminate_pool(self):
        if False:
            print('Hello World!')
        'Forcibly terminate the `ThreadPool` instance in `self.pool`\n\n        Sends SIGTERM to all processes.\n        '
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _interrupt(self, signal, frame):
        if False:
            i = 10
            return i + 15
        try:
            self._log.info('interrupted')
            self.terminate_pool()
            sys.exit(0)
        except SystemExit:
            pass

    def close_pool(self):
        if False:
            for i in range(10):
                print('nop')
        'Regularly close the `ThreadPool` instance in `self.pool`.'
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.exc_watcher.join()
            self.pool = None

    def import_begin(self, session: ImportSession):
        if False:
            print('Hello World!')
        'Handle `import_begin` event -> open pool'
        threads = cast(int, self.config['threads'].get(int))
        if self.config['parallel_on_import'] and self.config['auto'] and threads:
            self.open_pool(threads)

    def import_end(self, paths):
        if False:
            print('Hello World!')
        'Handle `import` event -> close pool'
        self.close_pool()

    def imported(self, session: ImportSession, task: ImportTask):
        if False:
            for i in range(10):
                print('nop')
        'Add replay gain info to items or albums of ``task``.'
        if self.config['auto']:
            if task.is_album:
                self.handle_album(task.album, False, self.force_on_import)
            else:
                assert hasattr(task, 'item')
                self.handle_track(task.item, False, self.force_on_import)

    def command_func(self, lib: Library, opts: optparse.Values, args: List[str]):
        if False:
            while True:
                i = 10
        try:
            write = ui.should_write(opts.write)
            force = opts.force
            if opts.threads != 0:
                threads = opts.threads or cast(int, self.config['threads'].get(int))
                self.open_pool(threads)
            if opts.album:
                albums = lib.albums(ui.decargs(args))
                self._log.info('Analyzing {} albums ~ {} backend...'.format(len(albums), self.backend_name))
                for album in albums:
                    self.handle_album(album, write, force)
            else:
                items = lib.items(ui.decargs(args))
                self._log.info('Analyzing {} tracks ~ {} backend...'.format(len(items), self.backend_name))
                for item in items:
                    self.handle_track(item, write, force)
            self.close_pool()
        except (SystemExit, KeyboardInterrupt):
            pass

    def commands(self) -> List[ui.Subcommand]:
        if False:
            print('Hello World!')
        'Return the "replaygain" ui subcommand.'
        cmd = ui.Subcommand('replaygain', help='analyze for ReplayGain')
        cmd.parser.add_album_option()
        cmd.parser.add_option('-t', '--threads', dest='threads', type=int, help='change the number of threads,             defaults to maximum available processors')
        cmd.parser.add_option('-f', '--force', dest='force', action='store_true', default=False, help='analyze all files, including those that already have ReplayGain metadata')
        cmd.parser.add_option('-w', '--write', default=None, action='store_true', help="write new metadata to files' tags")
        cmd.parser.add_option('-W', '--nowrite', dest='write', action='store_false', help="don't write metadata (opposite of -w)")
        cmd.func = self.command_func
        return [cmd]