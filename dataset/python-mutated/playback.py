from __future__ import annotations
import logging
import urllib.parse
from collections.abc import Iterable
from typing import TYPE_CHECKING
from pykka.messages import ProxyCall
from pykka.typing import proxy_method
from mopidy.audio import PlaybackState
from mopidy.core import listener
from mopidy.exceptions import CoreError
from mopidy.internal import deprecation, models, validation
from mopidy.types import DurationMs, UriScheme
if TYPE_CHECKING:
    from mopidy.audio.actor import AudioProxy
    from mopidy.backend import BackendProxy
    from mopidy.core.actor import Backends, Core
    from mopidy.models import TlTrack, Track
    from mopidy.types import Uri
logger = logging.getLogger(__name__)

class PlaybackController:

    def __init__(self, audio: AudioProxy | None, backends: Backends, core: Core) -> None:
        if False:
            while True:
                i = 10
        self.backends = backends
        self.core = core
        self._audio = audio
        self._stream_title: str | None = None
        self._state = PlaybackState.STOPPED
        self._current_tl_track: TlTrack | None = None
        self._pending_tl_track: TlTrack | None = None
        self._pending_position: DurationMs | None = None
        self._last_position: DurationMs | None = None
        self._previous: bool = False
        self._start_at_position: DurationMs | None = None
        self._start_paused: bool = False
        if self._audio:
            self._audio.set_about_to_finish_callback(self._on_about_to_finish_callback)

    def _get_backend(self, tl_track: TlTrack | None) -> BackendProxy | None:
        if False:
            i = 10
            return i + 15
        if tl_track is None:
            return None
        uri_scheme = UriScheme(urllib.parse.urlparse(tl_track.track.uri).scheme)
        return self.backends.with_playback.get(uri_scheme, None)

    def get_current_tl_track(self) -> TlTrack | None:
        if False:
            print('Hello World!')
        'Get the currently playing or selected track.\n\n        Returns a :class:`mopidy.models.TlTrack` or :class:`None`.\n        '
        return self._current_tl_track

    def _set_current_tl_track(self, value: TlTrack | None) -> None:
        if False:
            while True:
                i = 10
        "Set the currently playing or selected track.\n\n        *Internal:* This is only for use by Mopidy's test suite.\n        "
        self._current_tl_track = value

    def get_current_track(self) -> Track | None:
        if False:
            while True:
                i = 10
        'Get the currently playing or selected track.\n\n        Extracted from :meth:`get_current_tl_track` for convenience.\n\n        Returns a :class:`mopidy.models.Track` or :class:`None`.\n        '
        return getattr(self.get_current_tl_track(), 'track', None)

    def get_current_tlid(self) -> int | None:
        if False:
            i = 10
            return i + 15
        'Get the currently playing or selected TLID.\n\n        Extracted from :meth:`get_current_tl_track` for convenience.\n\n        Returns a :class:`int` or :class:`None`.\n\n        .. versionadded:: 1.1\n        '
        return getattr(self.get_current_tl_track(), 'tlid', None)

    def get_stream_title(self) -> str | None:
        if False:
            print('Hello World!')
        'Get the current stream title or :class:`None`.'
        return self._stream_title

    def get_state(self) -> PlaybackState:
        if False:
            while True:
                i = 10
        'Get The playback state.'
        return self._state

    def set_state(self, new_state: PlaybackState) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the playback state.\n\n        Must be :attr:`PLAYING`, :attr:`PAUSED`, or :attr:`STOPPED`.\n\n        Possible states and transitions:\n\n        .. digraph:: state_transitions\n\n            "STOPPED" -> "PLAYING" [ label="play" ]\n            "STOPPED" -> "PAUSED" [ label="pause" ]\n            "PLAYING" -> "STOPPED" [ label="stop" ]\n            "PLAYING" -> "PAUSED" [ label="pause" ]\n            "PLAYING" -> "PLAYING" [ label="play" ]\n            "PAUSED" -> "PLAYING" [ label="resume" ]\n            "PAUSED" -> "STOPPED" [ label="stop" ]\n        '
        validation.check_choice(new_state, validation.PLAYBACK_STATES)
        (old_state, self._state) = (self.get_state(), new_state)
        logger.debug('Changing state: %s -> %s', old_state, new_state)
        self._trigger_playback_state_changed(old_state, new_state)

    def get_time_position(self) -> DurationMs:
        if False:
            print('Hello World!')
        'Get time position in milliseconds.'
        if self._pending_position is not None:
            return self._pending_position
        backend = self._get_backend(self.get_current_tl_track())
        if not backend:
            return DurationMs(0)
        return backend.playback.get_time_position().get()

    def _on_end_of_stream(self) -> None:
        if False:
            print('Hello World!')
        self.set_state(PlaybackState.STOPPED)
        if self._current_tl_track:
            self._trigger_track_playback_ended(self.get_time_position())
        self._set_current_tl_track(None)

    def _on_stream_changed(self, _uri: Uri) -> None:
        if False:
            while True:
                i = 10
        if self._last_position is None:
            position = self.get_time_position()
        else:
            (position, self._last_position) = (self._last_position, None)
        if self._pending_position is None:
            self._trigger_track_playback_ended(position)
        self._stream_title = None
        if self._pending_tl_track:
            self._set_current_tl_track(self._pending_tl_track)
            self._pending_tl_track = None
            if self._pending_position is None:
                self.set_state(PlaybackState.PLAYING)
                self._trigger_track_playback_started()
                seek_ok = False
                if self._start_at_position:
                    seek_ok = self.seek(self._start_at_position)
                    self._start_at_position = None
                if not seek_ok and self._start_paused:
                    self.pause()
                    self._start_paused = False
            else:
                self._seek(self._pending_position)
                self.set_state(PlaybackState.PLAYING)
                self._trigger_track_playback_started()

    def _on_position_changed(self, _position: int) -> None:
        if False:
            i = 10
            return i + 15
        if self._pending_position is not None:
            self._trigger_seeked(self._pending_position)
            self._pending_position = None
            if self._start_paused:
                self._start_paused = False
                self.pause()

    def _on_about_to_finish_callback(self) -> None:
        if False:
            print('Hello World!')
        'Callback that performs a blocking actor call to the real callback.\n\n        This is passed to audio, which is allowed to call this code from the\n        audio thread. We pass execution into the core actor to ensure that\n        there is no unsafe access of state in core. This must block until\n        we get a response.\n        '
        self.core.actor_ref.ask(ProxyCall(attr_path=('playback', '_on_about_to_finish'), args=(), kwargs={}))

    def _on_about_to_finish(self) -> None:
        if False:
            print('Hello World!')
        if self._state == PlaybackState.STOPPED:
            return
        if self._current_tl_track is not None:
            if self._current_tl_track.track.length is not None:
                self._last_position = DurationMs(self._current_tl_track.track.length)
            else:
                self._last_position = None
        else:
            pass
        pending = self.core.tracklist.eot_track(self._current_tl_track)
        count = self.core.tracklist.get_length() * 2
        while pending:
            backend = self._get_backend(pending)
            if backend:
                try:
                    if backend.playback.change_track(pending.track).get():
                        self._pending_tl_track = pending
                        break
                except Exception:
                    logger.exception('%s backend caused an exception.', backend.actor_ref.actor_class.__name__)
            self.core.tracklist._mark_unplayable(pending)
            pending = self.core.tracklist.eot_track(pending)
            count -= 1
            if not count:
                logger.info('No playable track in the list.')
                break

    def _on_tracklist_change(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tell the playback controller that the current playlist has changed.\n\n        Used by :class:`mopidy.core.TracklistController`.\n        '
        tl_tracks = self.core.tracklist.get_tl_tracks()
        if not tl_tracks:
            self.stop()
            self._set_current_tl_track(None)
        elif self.get_current_tl_track() not in tl_tracks:
            self._set_current_tl_track(None)

    def next(self) -> None:
        if False:
            print('Hello World!')
        'Change to the next track.\n\n        The current playback state will be kept. If it was playing, playing\n        will continue. If it was paused, it will still be paused, etc.\n        '
        state = self.get_state()
        current = self._pending_tl_track or self._current_tl_track
        count = self.core.tracklist.get_length() * 2
        while current:
            pending = self.core.tracklist.next_track(current)
            if self._change(pending, state):
                break
            self.core.tracklist._mark_unplayable(pending)
            current = pending
            count -= 1
            if not count:
                logger.info('No playable track in the list.')
                break

    def pause(self) -> None:
        if False:
            while True:
                i = 10
        'Pause playback.'
        backend = self._get_backend(self.get_current_tl_track())
        if not backend or backend.playback.pause().get():
            self.set_state(PlaybackState.PAUSED)
            self._trigger_track_playback_paused()

    def play(self, tl_track: TlTrack | None=None, tlid: int | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Play the given track, or if the given tl_track and tlid is\n        :class:`None`, play the currently active track.\n\n        Note that the track **must** already be in the tracklist.\n\n        .. deprecated:: 3.0\n            The ``tl_track`` argument. Use ``tlid`` instead.\n\n        :param tl_track: track to play\n        :param tlid: TLID of the track to play\n        '
        if sum((o is not None for o in [tl_track, tlid])) > 1:
            raise ValueError('At most one of "tl_track" and "tlid" may be set')
        if tl_track is not None:
            validation.check_instance(tl_track, models.TlTrack)
        if tlid is not None:
            validation.check_integer(tlid, min=1)
        if tl_track:
            deprecation.warn('core.playback.play:tl_track_kwarg')
        if tl_track is None and tlid is not None:
            for tl_track in self.core.tracklist.get_tl_tracks():
                if tl_track.tlid == tlid:
                    break
            else:
                tl_track = None
        if tl_track is not None:
            if tl_track not in self.core.tracklist.get_tl_tracks():
                raise AssertionError
        elif tl_track is None and self.get_state() == PlaybackState.PAUSED:
            self.resume()
            return
        current = self._pending_tl_track or self._current_tl_track
        pending = tl_track or current or self.core.tracklist.next_track(None)
        count = self.core.tracklist.get_length() * 2
        while pending:
            if self._change(pending, PlaybackState.PLAYING):
                break
            self.core.tracklist._mark_unplayable(pending)
            current = pending
            pending = self.core.tracklist.next_track(current)
            count -= 1
            if not count:
                logger.info('No playable track in the list.')
                break

    def _change(self, pending_tl_track: TlTrack | None, state: PlaybackState) -> bool:
        if False:
            i = 10
            return i + 15
        self._pending_tl_track = pending_tl_track
        if not pending_tl_track:
            self.stop()
            self._on_end_of_stream()
            return True
        backend = self._get_backend(pending_tl_track)
        if not backend:
            return False
        self._last_position = self.get_time_position()
        backend.playback.prepare_change()
        try:
            if not backend.playback.change_track(pending_tl_track.track).get():
                return False
        except Exception:
            logger.exception('%s backend caused an exception.', backend.actor_ref.actor_class.__name__)
            return False
        if state == PlaybackState.PLAYING:
            try:
                return backend.playback.play().get()
            except TypeError:
                logger.error('%s needs to be updated to work with this version of Mopidy.', backend)
                return False
        elif state == PlaybackState.PAUSED:
            return backend.playback.pause().get()
        elif state == PlaybackState.STOPPED:
            self._current_tl_track = self._pending_tl_track
            self._pending_tl_track = None
            return True
        raise CoreError(f'Unknown playback state: {state}')

    def previous(self) -> None:
        if False:
            print('Hello World!')
        'Change to the previous track.\n\n        The current playback state will be kept. If it was playing, playing\n        will continue. If it was paused, it will still be paused, etc.\n        '
        self._previous = True
        state = self.get_state()
        current = self._pending_tl_track or self._current_tl_track
        count = self.core.tracklist.get_length() * 2
        while current:
            pending = self.core.tracklist.previous_track(current)
            if self._change(pending, state):
                break
            self.core.tracklist._mark_unplayable(pending)
            current = pending
            count -= 1
            if not count:
                logger.info('No playable track in the list.')
                break

    def resume(self) -> None:
        if False:
            while True:
                i = 10
        'If paused, resume playing the current track.'
        if self.get_state() != PlaybackState.PAUSED:
            return
        backend = self._get_backend(self.get_current_tl_track())
        if backend and backend.playback.resume().get():
            self.set_state(PlaybackState.PLAYING)
            self._trigger_track_playback_resumed()

    def seek(self, time_position: DurationMs) -> bool:
        if False:
            return 10
        'Seeks to time position given in milliseconds.\n\n        Returns :class:`True` if successful, else :class:`False`.\n\n        :param time_position: time position in milliseconds\n        '
        validation.check_integer(time_position)
        if time_position < 0:
            logger.debug('Client seeked to negative position. Seeking to zero.')
            time_position = DurationMs(0)
        if not self.core.tracklist.get_length():
            return False
        if self.get_state() == PlaybackState.STOPPED:
            self.play()
        tl_track = self._current_tl_track or self._pending_tl_track
        if tl_track is None or tl_track.track.length is None:
            return False
        if time_position < 0:
            time_position = DurationMs(0)
        elif time_position > tl_track.track.length:
            self.next()
            return True
        self._pending_position = time_position
        if self._current_tl_track and self._pending_tl_track:
            return self._change(self._current_tl_track, self.get_state())
        return self._seek(time_position)

    def _seek(self, time_position: DurationMs) -> bool:
        if False:
            print('Hello World!')
        backend = self._get_backend(self.get_current_tl_track())
        if not backend:
            return False
        return backend.playback.seek(time_position).get()

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        'Stop playing.'
        if self.get_state() != PlaybackState.STOPPED:
            self._last_position = self.get_time_position()
            backend = self._get_backend(self.get_current_tl_track())
            if not backend or backend.playback.stop().get():
                self.set_state(PlaybackState.STOPPED)

    def _trigger_track_playback_paused(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Triggering track playback paused event')
        if self.get_current_tl_track() is None:
            return
        listener.CoreListener.send('track_playback_paused', tl_track=self.get_current_tl_track(), time_position=self.get_time_position())

    def _trigger_track_playback_resumed(self) -> None:
        if False:
            i = 10
            return i + 15
        logger.debug('Triggering track playback resumed event')
        if self.get_current_tl_track() is None:
            return
        listener.CoreListener.send('track_playback_resumed', tl_track=self.get_current_tl_track(), time_position=self.get_time_position())

    def _trigger_track_playback_started(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.get_current_tl_track() is None:
            return
        logger.debug('Triggering track playback started event')
        tl_track = self.get_current_tl_track()
        if tl_track is None:
            return
        self.core.tracklist._mark_playing(tl_track)
        self.core.history._add_track(tl_track.track)
        listener.CoreListener.send('track_playback_started', tl_track=tl_track)

    def _trigger_track_playback_ended(self, time_position_before_stop: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        tl_track = self.get_current_tl_track()
        if tl_track is None:
            return
        logger.debug('Triggering track playback ended event')
        if not self._previous:
            self.core.tracklist._mark_played(self._current_tl_track)
        self._previous = False
        listener.CoreListener.send('track_playback_ended', tl_track=tl_track, time_position=time_position_before_stop)

    def _trigger_playback_state_changed(self, old_state: PlaybackState, new_state: PlaybackState) -> None:
        if False:
            while True:
                i = 10
        logger.debug('Triggering playback state change event')
        listener.CoreListener.send('playback_state_changed', old_state=old_state, new_state=new_state)

    def _trigger_seeked(self, time_position: int) -> None:
        if False:
            print('Hello World!')
        logger.debug('Triggering seeked event')
        listener.CoreListener.send('seeked', time_position=time_position)

    def _save_state(self) -> models.PlaybackState:
        if False:
            for i in range(10):
                print('nop')
        return models.PlaybackState(tlid=self.get_current_tlid(), time_position=self.get_time_position(), state=self.get_state())

    def _load_state(self, state: models.PlaybackState, coverage: Iterable[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if state and 'play-last' in coverage and (state.tlid is not None):
            if state.state == PlaybackState.PAUSED:
                self._start_paused = True
            if state.state in (PlaybackState.PLAYING, PlaybackState.PAUSED):
                self._start_at_position = DurationMs(state.time_position)
                self.play(tlid=state.tlid)

class PlaybackControllerProxy:
    get_current_tl_track = proxy_method(PlaybackController.get_current_tl_track)
    get_current_track = proxy_method(PlaybackController.get_current_track)
    get_current_tlid = proxy_method(PlaybackController.get_current_tlid)
    get_stream_title = proxy_method(PlaybackController.get_stream_title)
    get_state = proxy_method(PlaybackController.get_state)
    set_state = proxy_method(PlaybackController.set_state)
    get_time_position = proxy_method(PlaybackController.get_time_position)
    next = proxy_method(PlaybackController.next)
    pause = proxy_method(PlaybackController.pause)
    play = proxy_method(PlaybackController.play)
    previous = proxy_method(PlaybackController.previous)
    seek = proxy_method(PlaybackController.seek)
    stop = proxy_method(PlaybackController.stop)