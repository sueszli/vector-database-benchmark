import re
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from mycroft.messagebus.message import Message
from .mycroft_skill import MycroftSkill
from .audioservice import AudioService

class CPSMatchLevel(Enum):
    EXACT = 1
    MULTI_KEY = 2
    TITLE = 3
    ARTIST = 4
    CATEGORY = 5
    GENERIC = 6

class CPSTrackStatus(IntEnum):
    DISAMBIGUATION = 1
    PLAYING = 20
    PLAYING_AUDIOSERVICE = 21
    PLAYING_GUI = 22
    PLAYING_ENCLOSURE = 23
    QUEUED = 30
    QUEUED_AUDIOSERVICE = 31
    QUEUED_GUI = 32
    QUEUED_ENCLOSURE = 33
    PAUSED = 40
    STALLED = 60
    BUFFERING = 61
    END_OF_MEDIA = 90

class CommonPlaySkill(MycroftSkill, ABC):
    """ To integrate with the common play infrastructure of Mycroft
    skills should use this base class and override the two methods
    `CPS_match_query_phrase` (for checking if the skill can play the
    utterance) and `CPS_start` for launching the media.

    The class makes the skill available to queries from the
    mycroft-playback-control skill and no special vocab for starting playback
    is needed.
    """

    def __init__(self, name=None, bus=None):
        if False:
            i = 10
            return i + 15
        super().__init__(name, bus)
        self.audioservice = None
        self.play_service_string = None
        spoken = name or self.__class__.__name__
        self.spoken_name = re.sub('([a-z])([A-Z])', '\\g<1> \\g<2>', spoken.replace('Skill', ''))

    def bind(self, bus):
        if False:
            return 10
        'Overrides the normal bind method.\n\n        Adds handlers for play:query and play:start messages allowing\n        interaction with the playback control skill.\n\n        This is called automatically during setup, and\n        need not otherwise be used.\n        '
        if bus:
            super().bind(bus)
            self.audioservice = AudioService(self.bus)
            self.add_event('play:query', self.__handle_play_query)
            self.add_event('play:start', self.__handle_play_start)

    def __handle_play_query(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Query skill if it can start playback from given phrase.'
        search_phrase = message.data['phrase']
        self.bus.emit(message.response({'phrase': search_phrase, 'skill_id': self.skill_id, 'searching': True}))
        result = self.CPS_match_query_phrase(search_phrase)
        if result:
            match = result[0]
            level = result[1]
            callback = result[2] if len(result) > 2 else None
            confidence = self.__calc_confidence(match, search_phrase, level)
            self.bus.emit(message.response({'phrase': search_phrase, 'skill_id': self.skill_id, 'callback_data': callback, 'service_name': self.spoken_name, 'conf': confidence}))
        else:
            self.bus.emit(message.response({'phrase': search_phrase, 'skill_id': self.skill_id, 'searching': False}))

    def __calc_confidence(self, match, phrase, level):
        if False:
            while True:
                i = 10
        'Translate confidence level and match to a 0-1 value.\n\n        "play pandora"\n        "play pandora is my girlfriend"\n        "play tom waits on pandora"\n\n        Assume the more of the words that get consumed, the better the match\n\n        Args:\n            match (str): Matching string\n            phrase (str): original input phrase\n            level (CPSMatchLevel): match level\n        '
        consumed_pct = len(match.split()) / len(phrase.split())
        if consumed_pct > 1.0:
            consumed_pct = 1.0 / consumed_pct
        bonus = consumed_pct / 20.0
        if level == CPSMatchLevel.EXACT:
            return 1.0
        elif level == CPSMatchLevel.MULTI_KEY:
            return 0.9 + bonus
        elif level == CPSMatchLevel.TITLE:
            return 0.8 + bonus
        elif level == CPSMatchLevel.ARTIST:
            return 0.7 + bonus
        elif level == CPSMatchLevel.CATEGORY:
            return 0.6 + bonus
        elif level == CPSMatchLevel.GENERIC:
            return 0.5 + bonus
        else:
            return 0.0

    def __handle_play_start(self, message):
        if False:
            return 10
        'Bus handler for starting playback using the skill.'
        if message.data['skill_id'] != self.skill_id:
            return
        phrase = message.data['phrase']
        data = message.data.get('callback_data')
        if self.audioservice.is_playing:
            self.audioservice.stop()
        self.bus.emit(message.forward('mycroft.stop'))
        self.play_service_string = phrase
        self.make_active()
        self.CPS_start(phrase, data)

    def CPS_play(self, *args, **kwargs):
        if False:
            return 10
        'Begin playback of a media file or stream\n\n        Normally this method will be invoked with somthing like:\n           self.CPS_play(url)\n        Advanced use can also include keyword arguments, such as:\n           self.CPS_play(url, repeat=True)\n\n        Args:\n            same as the Audioservice.play method\n        '
        if 'utterance' not in kwargs:
            kwargs['utterance'] = self.play_service_string
        self.audioservice.play(*args, **kwargs)
        self.CPS_send_status(uri=args[0], status=CPSTrackStatus.PLAYING_AUDIOSERVICE)

    def stop(self):
        if False:
            while True:
                i = 10
        'Stop anything playing on the audioservice.'
        if self.audioservice.is_playing:
            self.audioservice.stop()
            return True
        else:
            return False

    @abstractmethod
    def CPS_match_query_phrase(self, phrase):
        if False:
            for i in range(10):
                print('nop')
        'Analyze phrase to see if it is a play-able phrase with this skill.\n\n        Args:\n            phrase (str): User phrase uttered after "Play", e.g. "some music"\n\n        Returns:\n            (match, CPSMatchLevel[, callback_data]) or None: Tuple containing\n                 a string with the appropriate matching phrase, the PlayMatch\n                 type, and optionally data to return in the callback if the\n                 match is selected.\n        '
        return None

    @abstractmethod
    def CPS_start(self, phrase, data):
        if False:
            i = 10
            return i + 15
        'Begin playing whatever is specified in \'phrase\'\n\n        Args:\n            phrase (str): User phrase uttered after "Play", e.g. "some music"\n            data (dict): Callback data specified in match_query_phrase()\n        '
        pass

    def CPS_extend_timeout(self, timeout=5):
        if False:
            while True:
                i = 10
        'Request Common Play Framework to wait another {timeout} seconds\n        for an answer from this skill.\n\n        Args:\n            timeout (int): Number of seconds\n        '
        self.bus.emit(Message('play:query.response', {'phrase': self.play_service_string, 'searching': True, 'timeout': timeout, 'skill_id': self.skill_id}))

    def CPS_send_status(self, artist='', track='', album='', image='', uri='', track_length=None, elapsed_time=None, playlist_position=None, status=CPSTrackStatus.DISAMBIGUATION, **kwargs):
        if False:
            i = 10
            return i + 15
        "Inform system of playback status.\n\n        If a skill is handling playback and wants the playback control to be\n        aware of it's current status it can emit this message indicating that\n        it's performing playback and can provide some standard info.\n\n        All parameters are optional so any can be left out. Also if extra\n        non-standard parameters are added, they too will be sent in the message\n        data.\n\n        Args:\n            artist (str): Current track artist\n            track (str): Track name\n            album (str): Album title\n            image (str): url for image to show\n            uri (str): uri for track\n            track_length (float): track length in seconds\n            elapsed_time (float): current offset into track in seconds\n            playlist_position (int): Position in playlist of current track\n        "
        data = {'skill': self.name, 'uri': uri, 'artist': artist, 'album': album, 'track': track, 'image': image, 'track_length': track_length, 'elapsed_time': elapsed_time, 'playlist_position': playlist_position, 'status': status}
        data = {**data, **kwargs}
        self.bus.emit(Message('play:status', data))

    def CPS_send_tracklist(self, tracklist):
        if False:
            return 10
        'Inform system of playlist track info.\n\n        Provides track data for playlist\n\n        Args:\n            tracklist (list/dict): Tracklist data\n        '
        tracklist = tracklist or []
        if not isinstance(tracklist, list):
            tracklist = [tracklist]
        for (idx, track) in enumerate(tracklist):
            self.CPS_send_status(playlist_position=idx, **track)