from __future__ import annotations
import logging
import random
from collections.abc import Iterable
from typing import TYPE_CHECKING
from pykka.typing import proxy_method
from mopidy import exceptions
from mopidy.core import listener
from mopidy.internal import deprecation, validation
from mopidy.internal.models import TracklistState
from mopidy.models import TlTrack, Track
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from mopidy.core.actor import Core
    from mopidy.types import Query, TracklistField, Uri

class TracklistController:

    def __init__(self, core: Core) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.core = core
        self._next_tlid: int = 1
        self._tl_tracks: list[TlTrack] = []
        self._version: int = 0
        self._consume: bool = False
        self._random: bool = False
        self._shuffled: list[TlTrack] = []
        self._repeat: bool = False
        self._single: bool = False

    def get_tl_tracks(self) -> list[TlTrack]:
        if False:
            i = 10
            return i + 15
        'Get tracklist as list of :class:`mopidy.models.TlTrack`.'
        return self._tl_tracks[:]

    def get_tracks(self) -> list[Track]:
        if False:
            i = 10
            return i + 15
        'Get tracklist as list of :class:`mopidy.models.Track`.'
        return [tl_track.track for tl_track in self._tl_tracks]

    def get_length(self) -> int:
        if False:
            while True:
                i = 10
        'Get length of the tracklist.'
        return len(self._tl_tracks)

    def get_version(self) -> int:
        if False:
            return 10
        'Get the tracklist version.\n\n        Integer which is increased every time the tracklist is changed. Is not\n        reset before Mopidy is restarted.\n        '
        return self._version

    def _increase_version(self) -> None:
        if False:
            i = 10
            return i + 15
        self._version += 1
        self.core.playback._on_tracklist_change()
        self._trigger_tracklist_changed()

    def get_consume(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Get consume mode.\n\n        :class:`True`\n            Tracks are removed from the tracklist when they have been played.\n        :class:`False`\n            Tracks are not removed from the tracklist.\n        '
        return self._consume

    def set_consume(self, value: bool) -> None:
        if False:
            print('Hello World!')
        'Set consume mode.\n\n        :class:`True`\n            Tracks are removed from the tracklist when they have been played.\n        :class:`False`\n            Tracks are not removed from the tracklist.\n        '
        validation.check_boolean(value)
        if self.get_consume() != value:
            self._trigger_options_changed()
        self._consume = value

    def get_random(self) -> bool:
        if False:
            while True:
                i = 10
        'Get random mode.\n\n        :class:`True`\n            Tracks are selected at random from the tracklist.\n        :class:`False`\n            Tracks are played in the order of the tracklist.\n        '
        return self._random

    def set_random(self, value: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Set random mode.\n\n        :class:`True`\n            Tracks are selected at random from the tracklist.\n        :class:`False`\n            Tracks are played in the order of the tracklist.\n        '
        validation.check_boolean(value)
        if self.get_random() != value:
            self._trigger_options_changed()
        if value:
            self._shuffled = self.get_tl_tracks()
            random.shuffle(self._shuffled)
        self._random = value

    def get_repeat(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Get repeat mode.\n\n        :class:`True`\n            The tracklist is played repeatedly.\n        :class:`False`\n            The tracklist is played once.\n        '
        return self._repeat

    def set_repeat(self, value: bool) -> None:
        if False:
            print('Hello World!')
        'Set repeat mode.\n\n        To repeat a single track, set both ``repeat`` and ``single``.\n\n        :class:`True`\n            The tracklist is played repeatedly.\n        :class:`False`\n            The tracklist is played once.\n        '
        validation.check_boolean(value)
        if self.get_repeat() != value:
            self._trigger_options_changed()
        self._repeat = value

    def get_single(self) -> bool:
        if False:
            while True:
                i = 10
        'Get single mode.\n\n        :class:`True`\n            Playback is stopped after current song, unless in ``repeat`` mode.\n        :class:`False`\n            Playback continues after current song.\n        '
        return self._single

    def set_single(self, value: bool) -> None:
        if False:
            return 10
        'Set single mode.\n\n        :class:`True`\n            Playback is stopped after current song, unless in ``repeat`` mode.\n        :class:`False`\n            Playback continues after current song.\n        '
        validation.check_boolean(value)
        if self.get_single() != value:
            self._trigger_options_changed()
        self._single = value

    def index(self, tl_track: TlTrack | None=None, tlid: int | None=None) -> int | None:
        if False:
            return 10
        'The position of the given track in the tracklist.\n\n        If neither *tl_track* or *tlid* is given we return the index of\n        the currently playing track.\n\n        :param tl_track: the track to find the index of\n        :param tlid: TLID of the track to find the index of\n\n        .. versionadded:: 1.1\n            The *tlid* parameter\n        '
        if tl_track is not None:
            validation.check_instance(tl_track, TlTrack)
        if tlid is not None:
            validation.check_integer(tlid, min=1)
        if tl_track is None and tlid is None:
            tl_track = self.core.playback.get_current_tl_track()
        if tl_track is not None:
            try:
                return self._tl_tracks.index(tl_track)
            except ValueError:
                pass
        elif tlid is not None:
            for (i, tl_track) in enumerate(self._tl_tracks):
                if tl_track.tlid == tlid:
                    return i
        return None

    def get_eot_tlid(self) -> int | None:
        if False:
            while True:
                i = 10
        'The TLID of the track that will be played after the current track.\n\n        Not necessarily the same TLID as returned by :meth:`get_next_tlid`.\n\n        .. versionadded:: 1.1\n        '
        current_tl_track = self.core.playback.get_current_tl_track()
        with deprecation.ignore('core.tracklist.eot_track'):
            eot_tl_track = self.eot_track(current_tl_track)
        return getattr(eot_tl_track, 'tlid', None)

    def eot_track(self, tl_track: TlTrack | None) -> TlTrack | None:
        if False:
            i = 10
            return i + 15
        'The track that will be played after the given track.\n\n        Not necessarily the same track as :meth:`next_track`.\n\n        .. deprecated:: 3.0\n            Use :meth:`get_eot_tlid` instead.\n\n        :param tl_track: the reference track\n        '
        deprecation.warn('core.tracklist.eot_track')
        if tl_track is not None:
            validation.check_instance(tl_track, TlTrack)
        if self.get_single() and self.get_repeat():
            return tl_track
        if self.get_single():
            return None
        return self.next_track(tl_track)

    def get_next_tlid(self) -> int | None:
        if False:
            i = 10
            return i + 15
        'The tlid of the track that will be played if calling\n        :meth:`mopidy.core.PlaybackController.next()`.\n\n        For normal playback this is the next track in the tracklist. If repeat\n        is enabled the next track can loop around the tracklist. When random is\n        enabled this should be a random track, all tracks should be played once\n        before the tracklist repeats.\n\n        .. versionadded:: 1.1\n        '
        current_tl_track = self.core.playback.get_current_tl_track()
        with deprecation.ignore('core.tracklist.next_track'):
            next_tl_track = self.next_track(current_tl_track)
        return getattr(next_tl_track, 'tlid', None)

    def next_track(self, tl_track: TlTrack | None) -> TlTrack | None:
        if False:
            i = 10
            return i + 15
        'The track that will be played if calling\n        :meth:`mopidy.core.PlaybackController.next()`.\n\n        For normal playback this is the next track in the tracklist. If repeat\n        is enabled the next track can loop around the tracklist. When random is\n        enabled this should be a random track, all tracks should be played once\n        before the tracklist repeats.\n\n        .. deprecated:: 3.0\n            Use :meth:`get_next_tlid` instead.\n\n        :param tl_track: the reference track\n        '
        deprecation.warn('core.tracklist.next_track')
        if tl_track is not None:
            validation.check_instance(tl_track, TlTrack)
        if not self._tl_tracks:
            return None
        if self.get_random() and (not self._shuffled) and (self.get_repeat() or not tl_track):
            logger.debug('Shuffling tracks')
            self._shuffled = self._tl_tracks[:]
            random.shuffle(self._shuffled)
        if self.get_random():
            if self._shuffled:
                return self._shuffled[0]
            return None
        next_index = self.index(tl_track)
        if next_index is None:
            next_index = 0
        else:
            next_index += 1
        if self.get_repeat():
            if self.get_consume() and len(self._tl_tracks) == 1:
                return None
            next_index %= len(self._tl_tracks)
        elif next_index >= len(self._tl_tracks):
            return None
        return self._tl_tracks[next_index]

    def get_previous_tlid(self) -> int | None:
        if False:
            return 10
        'Returns the TLID of the track that will be played if calling\n        :meth:`mopidy.core.PlaybackController.previous()`.\n\n        For normal playback this is the previous track in the tracklist. If\n        random and/or consume is enabled it should return the current track\n        instead.\n\n        .. versionadded:: 1.1\n        '
        current_tl_track = self.core.playback.get_current_tl_track()
        with deprecation.ignore('core.tracklist.previous_track'):
            previous_tl_track = self.previous_track(current_tl_track)
        return getattr(previous_tl_track, 'tlid', None)

    def previous_track(self, tl_track: TlTrack | None) -> TlTrack | None:
        if False:
            while True:
                i = 10
        'Returns the track that will be played if calling\n        :meth:`mopidy.core.PlaybackController.previous()`.\n\n        For normal playback this is the previous track in the tracklist. If\n        random and/or consume is enabled it should return the current track\n        instead.\n\n        .. deprecated:: 3.0\n            Use :meth:`get_previous_tlid` instead.\n\n        :param tl_track: the reference track\n        '
        deprecation.warn('core.tracklist.previous_track')
        if tl_track is not None:
            validation.check_instance(tl_track, TlTrack)
        if self.get_repeat() or self.get_consume() or self.get_random():
            return tl_track
        position = self.index(tl_track)
        if position in (None, 0):
            return None
        return self._tl_tracks[position - 1]

    def add(self, tracks: Iterable[Track] | None=None, at_position: int | None=None, uris: Iterable[Uri] | None=None) -> list[TlTrack]:
        if False:
            i = 10
            return i + 15
        'Add tracks to the tracklist.\n\n        If ``uris`` is given instead of ``tracks``, the URIs are\n        looked up in the library and the resulting tracks are added to the\n        tracklist.\n\n        If ``at_position`` is given, the tracks are inserted at the given\n        position in the tracklist. If ``at_position`` is not given, the tracks\n        are appended to the end of the tracklist.\n\n        Triggers the :meth:`mopidy.core.CoreListener.tracklist_changed` event.\n\n        :param tracks: tracks to add\n        :param at_position: position in tracklist to add tracks\n        :param uris: list of URIs for tracks to add\n\n        .. versionadded:: 1.0\n            The ``uris`` argument.\n\n        .. deprecated:: 1.0\n            The ``tracks`` argument. Use ``uris``.\n        '
        if sum((o is not None for o in [tracks, uris])) != 1:
            raise ValueError('Exactly one of "tracks" or "uris" must be set')
        if tracks is not None:
            validation.check_instances(tracks, Track)
        if uris is not None:
            validation.check_uris(uris)
        validation.check_integer(at_position or 0)
        if tracks:
            deprecation.warn('core.tracklist.add:tracks_arg')
        if tracks is None:
            tracks = []
            assert uris is not None
            track_map = self.core.library.lookup(uris=uris)
            for uri in uris:
                tracks.extend(track_map[uri])
        tl_tracks = []
        max_length = self.core._config['core']['max_tracklist_length']
        for track in tracks:
            if self.get_length() >= max_length:
                raise exceptions.TracklistFull(f'Tracklist may contain at most {max_length:d} tracks.')
            tl_track = TlTrack(self._next_tlid, track)
            self._next_tlid += 1
            if at_position is not None:
                self._tl_tracks.insert(at_position, tl_track)
                at_position += 1
            else:
                self._tl_tracks.append(tl_track)
            tl_tracks.append(tl_track)
        if tl_tracks:
            self._increase_version()
        return tl_tracks

    def clear(self) -> None:
        if False:
            return 10
        'Clear the tracklist.\n\n        Triggers the :meth:`mopidy.core.CoreListener.tracklist_changed` event.\n        '
        self._tl_tracks = []
        self._increase_version()

    def filter(self, criteria: Query[TracklistField]) -> list[TlTrack]:
        if False:
            while True:
                i = 10
        "Filter the tracklist by the given criteria.\n\n        Each rule in the criteria consists of a model field and a list of\n        values to compare it against. If the model field matches any of the\n        values, it may be returned.\n\n        Only tracks that match all the given criteria are returned.\n\n        Examples::\n\n            # Returns tracks with TLIDs 1, 2, 3, or 4 (tracklist ID)\n            filter({'tlid': [1, 2, 3, 4]})\n\n            # Returns track with URIs 'xyz' or 'abc'\n            filter({'uri': ['xyz', 'abc']})\n\n            # Returns track with a matching TLIDs (1, 3 or 6) and a\n            # matching URI ('xyz' or 'abc')\n            filter({'tlid': [1, 3, 6], 'uri': ['xyz', 'abc']})\n\n        :param criteria: one or more rules to match by\n        "
        tlids = criteria.pop('tlid', [])
        validation.check_query(criteria, validation.TRACKLIST_FIELDS.keys())
        validation.check_instances(tlids, int)
        matches = self._tl_tracks
        for (key, values) in criteria.items():
            matches = [ct for ct in matches if getattr(ct.track, key) in values]
        if tlids:
            matches = [ct for ct in matches if ct.tlid in tlids]
        return matches

    def move(self, start: int, end: int, to_position: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Move the tracks in the slice ``[start:end]`` to ``to_position``.\n\n        Triggers the :meth:`mopidy.core.CoreListener.tracklist_changed` event.\n\n        :param start: position of first track to move\n        :param end: position after last track to move\n        :param to_position: new position for the tracks\n        '
        if start == end:
            end += 1
        tl_tracks = self._tl_tracks
        if start >= end:
            raise AssertionError('start must be smaller than end')
        if start < 0:
            raise AssertionError('start must be at least zero')
        if end > len(tl_tracks):
            raise AssertionError('end can not be larger than tracklist length')
        if to_position < 0:
            raise AssertionError('to_position must be at least zero')
        if to_position > len(tl_tracks):
            raise AssertionError('to_position can not be larger than tracklist length')
        new_tl_tracks = tl_tracks[:start] + tl_tracks[end:]
        for tl_track in tl_tracks[start:end]:
            new_tl_tracks.insert(to_position, tl_track)
            to_position += 1
        self._tl_tracks = new_tl_tracks
        self._increase_version()

    def remove(self, criteria: Query[TracklistField]) -> list[TlTrack]:
        if False:
            return 10
        'Remove the matching tracks from the tracklist.\n\n        Uses :meth:`filter()` to lookup the tracks to remove.\n\n        Triggers the :meth:`mopidy.core.CoreListener.tracklist_changed` event.\n\n        Returns the removed tracks.\n\n        :param criteria: one or more rules to match by\n        '
        tl_tracks = self.filter(criteria)
        for tl_track in tl_tracks:
            position = self._tl_tracks.index(tl_track)
            del self._tl_tracks[position]
        self._increase_version()
        return tl_tracks

    def shuffle(self, start: int | None=None, end: int | None=None) -> None:
        if False:
            return 10
        'Shuffles the entire tracklist. If ``start`` and ``end`` is given only\n        shuffles the slice ``[start:end]``.\n\n        Triggers the :meth:`mopidy.core.CoreListener.tracklist_changed` event.\n\n        :param start: position of first track to shuffle\n        :param end: position after last track to shuffle\n        '
        tl_tracks = self._tl_tracks
        if start is not None and end is not None and (start >= end):
            raise AssertionError('start must be smaller than end')
        if start is not None and start < 0:
            raise AssertionError('start must be at least zero')
        if end is not None and end > len(tl_tracks):
            raise AssertionError('end can not be larger than tracklist length')
        before = tl_tracks[:start or 0]
        shuffled = tl_tracks[start:end]
        after = tl_tracks[end or len(tl_tracks):]
        random.shuffle(shuffled)
        self._tl_tracks = before + shuffled + after
        self._increase_version()

    def slice(self, start: int, end: int) -> list[TlTrack]:
        if False:
            i = 10
            return i + 15
        'Returns a slice of the tracklist, limited by the given start and end\n        positions.\n\n        :param start: position of first track to include in slice\n        :param end: position after last track to include in slice\n        '
        return self._tl_tracks[start:end]

    def _mark_playing(self, tl_track: TlTrack) -> None:
        if False:
            i = 10
            return i + 15
        'Internal method for :class:`mopidy.core.PlaybackController`.'
        if self.get_random() and tl_track in self._shuffled:
            self._shuffled.remove(tl_track)

    def _mark_unplayable(self, tl_track: TlTrack | None) -> None:
        if False:
            return 10
        'Internal method for :class:`mopidy.core.PlaybackController`.'
        logger.warning('Track is not playable: %s', tl_track.track.uri if tl_track else None)
        if self.get_consume() and tl_track is not None:
            self.remove({'tlid': [tl_track.tlid]})
        if self.get_random() and tl_track in self._shuffled:
            self._shuffled.remove(tl_track)

    def _mark_played(self, tl_track: TlTrack | None) -> bool:
        if False:
            while True:
                i = 10
        'Internal method for :class:`mopidy.core.PlaybackController`.'
        if self.get_consume() and tl_track is not None:
            self.remove({'tlid': [tl_track.tlid]})
            return True
        return False

    def _trigger_tracklist_changed(self) -> None:
        if False:
            print('Hello World!')
        if self.get_random():
            self._shuffled = self._tl_tracks[:]
            random.shuffle(self._shuffled)
        else:
            self._shuffled = []
        logger.debug('Triggering event: tracklist_changed()')
        listener.CoreListener.send('tracklist_changed')

    def _trigger_options_changed(self) -> None:
        if False:
            i = 10
            return i + 15
        logger.debug('Triggering options changed event')
        listener.CoreListener.send('options_changed')

    def _save_state(self) -> TracklistState:
        if False:
            return 10
        return TracklistState(tl_tracks=self._tl_tracks, next_tlid=self._next_tlid, consume=self.get_consume(), random=self.get_random(), repeat=self.get_repeat(), single=self.get_single())

    def _load_state(self, state: TracklistState, coverage: Iterable[str]) -> None:
        if False:
            print('Hello World!')
        if state:
            if 'mode' in coverage:
                self.set_consume(state.consume)
                self.set_random(state.random)
                self.set_repeat(state.repeat)
                self.set_single(state.single)
            if 'tracklist' in coverage:
                self._next_tlid = max(state.next_tlid, self._next_tlid)
                self._tl_tracks = list(state.tl_tracks)
                self._increase_version()

class TracklistControllerProxy:
    get_tl_tracks = proxy_method(TracklistController.get_tl_tracks)
    get_tracks = proxy_method(TracklistController.get_tracks)
    get_length = proxy_method(TracklistController.get_length)
    get_version = proxy_method(TracklistController.get_version)
    get_consume = proxy_method(TracklistController.get_consume)
    set_consume = proxy_method(TracklistController.set_consume)
    get_random = proxy_method(TracklistController.get_random)
    set_random = proxy_method(TracklistController.set_random)
    get_repeat = proxy_method(TracklistController.get_repeat)
    set_repeat = proxy_method(TracklistController.set_repeat)
    get_single = proxy_method(TracklistController.get_single)
    set_single = proxy_method(TracklistController.set_single)
    index = proxy_method(TracklistController.index)
    get_eot_tlid = proxy_method(TracklistController.get_eot_tlid)
    eot_track = proxy_method(TracklistController.eot_track)
    get_next_tlid = proxy_method(TracklistController.get_next_tlid)
    next_track = proxy_method(TracklistController.next_track)
    get_previous_tlid = proxy_method(TracklistController.get_previous_tlid)
    previous_track = proxy_method(TracklistController.previous_track)
    add = proxy_method(TracklistController.add)
    clear = proxy_method(TracklistController.clear)
    filter = proxy_method(TracklistController.filter)
    move = proxy_method(TracklistController.move)
    remove = proxy_method(TracklistController.remove)
    shuffle = proxy_method(TracklistController.shuffle)
    slice = proxy_method(TracklistController.slice)