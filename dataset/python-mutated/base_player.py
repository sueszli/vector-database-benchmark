import asyncio
import logging
import warnings
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from feeluown.utils.dispatch import Signal
from feeluown.player.playlist import PlaybackMode, Playlist
from .metadata import Metadata
__all__ = ('Playlist', 'AbstractPlayer', 'PlaybackMode', 'State')
logger = logging.getLogger(__name__)

class State(IntEnum):
    """
    Player states.
    """
    stopped = 0
    paused = 1
    playing = 2

class AbstractPlayer(metaclass=ABCMeta):
    """Player abstrace base class.

    Note that signals may be emitted from different thread. You should
    take care of race condition.
    """

    def __init__(self, _=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        :param _: keep this arg to keep backward compatibility\n        '
        self._position = 0
        self._volume = 100
        self._playlist = None
        self._state = State.stopped
        self._duration = None
        self._current_media = None
        self._current_metadata = Metadata()
        self.position_changed = Signal()
        self.seeked = Signal()
        self.state_changed = Signal()
        self.duration_changed = Signal()
        self.media_about_to_changed = Signal()
        self.media_changed = Signal()
        self.media_loaded = Signal()
        self.metadata_changed = Signal()
        self.media_finished = Signal()
        self.media_loading_failed = Signal()
        self.volume_changed = Signal()

    @property
    def state(self):
        if False:
            return 10
        'Player state\n\n        :rtype: State\n        '
        return self._state

    @state.setter
    def state(self, value):
        if False:
            print('Hello World!')
        'set player state, emit state changed signal\n\n        outer object should not set state directly,\n        use ``pause`` / ``resume`` / ``stop`` / ``play`` method instead.\n        '
        self._state = value
        self.state_changed.emit(value)

    @property
    def current_media(self):
        if False:
            while True:
                i = 10
        return self._current_media

    @property
    def current_metadata(self) -> Metadata:
        if False:
            for i in range(10):
                print('nop')
        "Metadata for the current media\n\n        Check `MetadataFields` for all possible fields. Note that some fields\n        can be missed if them are unknown. For example, a video's metadata\n        may have no genre info.\n        "
        return self._current_metadata

    @property
    def position(self):
        if False:
            print('Hello World!')
        'player position, the units is seconds'
        return self._position

    @position.setter
    def position(self, position):
        if False:
            i = 10
            return i + 15
        'set player position, the units is seconds'

    @property
    def volume(self):
        if False:
            return 10
        return self._volume

    @volume.setter
    def volume(self, value):
        if False:
            print('Hello World!')
        value = 0 if value < 0 else value
        value = 100 if value > 100 else value
        self._volume = value
        self.volume_changed.emit(value)

    @property
    def duration(self):
        if False:
            print('Hello World!')
        'player media duration, the units is seconds'
        return self._duration

    @duration.setter
    def duration(self, value):
        if False:
            for i in range(10):
                print('nop')
        value = value or 0
        if value != self._duration:
            self._duration = value
            self.duration_changed.emit(value)

    @abstractmethod
    def play(self, media, video=True, metadata=None):
        if False:
            print('Hello World!')
        'play media\n\n        :param media: a local file absolute path, or a http url that refers to a\n            media file\n        :param video: show video or not\n        :param metadata: metadata for the media\n        '

    @abstractmethod
    def set_play_range(self, start=None, end=None):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def resume(self):
        if False:
            i = 10
            return i + 15
        'play playback'

    @abstractmethod
    def pause(self):
        if False:
            while True:
                i = 10
        'pause player'

    @abstractmethod
    def toggle(self):
        if False:
            for i in range(10):
                print('nop')
        'toggle player state'

    @abstractmethod
    def stop(self):
        if False:
            i = 10
            return i + 15
        'stop player'

    @abstractmethod
    def shutdown(self):
        if False:
            print('Hello World!')
        'shutdown player, do some clean up here'

    @property
    def playlist(self):
        if False:
            print('Hello World!')
        "(DEPRECATED) player playlist\n\n        Player SHOULD not know the existence of playlist. However, in the\n        very beginning, the player depends on playlist and listen playlist's\n        signal. Other programs may depends on the playlist property and\n        we keep it for backward compatibility.\n\n        TODO: maybe add a DeprecationWarning in v3.8.\n\n        :return: :class:`.Playlist`\n        "
        return self._playlist

    def set_playlist(self, playlist):
        if False:
            i = 10
            return i + 15
        self._playlist = playlist

    @property
    def current_song(self):
        if False:
            while True:
                i = 10
        '(Deprecated) alias of playlist.current_song\n\n        Please use playlist.current_song instead.\n        '
        warnings.warn('use playlist.current_model instead', DeprecationWarning)
        return self._playlist.current_song

    def load_song(self, song) -> asyncio.Task:
        if False:
            return 10
        '加载歌曲\n\n        如果目标歌曲与当前歌曲不相同，则修改播放列表当前歌曲，\n        播放列表会发出 song_changed 信号，player 监听到信号后调用 play 方法，\n        到那时才会真正的播放新的歌曲。如果和当前播放歌曲相同，则忽略。\n\n        .. note::\n\n            调用方应该直接调用 playlist.current_song = song 来切换歌曲\n        '
        assert song is not None
        warnings.warn('use playlist.set_current_model instead, this will be removed in v3.8', DeprecationWarning)
        return self._playlist.set_current_song(song)

    def play_song(self, song):
        if False:
            return 10
        '加载并播放指定歌曲'
        warnings.warn('use playlist.set_current_model instead, this will be removed in v3.8', DeprecationWarning)
        return self._playlist.set_current_song(song)

    def play_songs(self, songs):
        if False:
            for i in range(10):
                print('nop')
        '(alpha) play list of songs'
        warnings.warn('use playlist.init_from instead, this will be removed in v3.8', DeprecationWarning)
        self.playlist.set_models(songs, next_=True)