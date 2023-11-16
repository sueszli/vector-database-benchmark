from os.path import abspath
from mycroft.messagebus.message import Message

def ensure_uri(s):
    if False:
        return 10
    "Interprete paths as file:// uri's.\n\n    Args:\n        s: string to be checked\n\n    Returns:\n        if s is uri, s is returned otherwise file:// is prepended\n    "
    if isinstance(s, str):
        if '://' not in s:
            return 'file://' + abspath(s)
        else:
            return s
    elif isinstance(s, (tuple, list)):
        if '://' not in s[0]:
            return ('file://' + abspath(s[0]), s[1])
        else:
            return s
    else:
        raise ValueError('Invalid track')

class AudioService:
    """AudioService class for interacting with the audio subsystem

    Args:
        bus: Mycroft messagebus connection
    """

    def __init__(self, bus):
        if False:
            return 10
        self.bus = bus

    def queue(self, tracks=None):
        if False:
            i = 10
            return i + 15
        "Queue up a track to playing playlist.\n\n        Args:\n            tracks: track uri or list of track uri's\n                    Each track can be added as a tuple with (uri, mime)\n                    to give a hint of the mime type to the system\n        "
        tracks = tracks or []
        if isinstance(tracks, (str, tuple)):
            tracks = [tracks]
        elif not isinstance(tracks, list):
            raise ValueError
        tracks = [ensure_uri(t) for t in tracks]
        self.bus.emit(Message('mycroft.audio.service.queue', data={'tracks': tracks}))

    def play(self, tracks=None, utterance=None, repeat=None):
        if False:
            i = 10
            return i + 15
        "Start playback.\n\n        Args:\n            tracks: track uri or list of track uri's\n                    Each track can be added as a tuple with (uri, mime)\n                    to give a hint of the mime type to the system\n            utterance: forward utterance for further processing by the\n                        audio service.\n            repeat: if the playback should be looped\n        "
        repeat = repeat or False
        tracks = tracks or []
        utterance = utterance or ''
        if isinstance(tracks, (str, tuple)):
            tracks = [tracks]
        elif not isinstance(tracks, list):
            raise ValueError
        tracks = [ensure_uri(t) for t in tracks]
        self.bus.emit(Message('mycroft.audio.service.play', data={'tracks': tracks, 'utterance': utterance, 'repeat': repeat}))

    def stop(self):
        if False:
            return 10
        'Stop the track.'
        self.bus.emit(Message('mycroft.audio.service.stop'))

    def next(self):
        if False:
            print('Hello World!')
        'Change to next track.'
        self.bus.emit(Message('mycroft.audio.service.next'))

    def prev(self):
        if False:
            while True:
                i = 10
        'Change to previous track.'
        self.bus.emit(Message('mycroft.audio.service.prev'))

    def pause(self):
        if False:
            for i in range(10):
                print('nop')
        'Pause playback.'
        self.bus.emit(Message('mycroft.audio.service.pause'))

    def resume(self):
        if False:
            i = 10
            return i + 15
        'Resume paused playback.'
        self.bus.emit(Message('mycroft.audio.service.resume'))

    def seek(self, seconds=1):
        if False:
            print('Hello World!')
        'Seek X seconds.\n\n        Args:\n            seconds (int): number of seconds to seek, if negative rewind\n        '
        if seconds < 0:
            self.seek_backward(abs(seconds))
        else:
            self.seek_forward(seconds)

    def seek_forward(self, seconds=1):
        if False:
            i = 10
            return i + 15
        'Skip ahead X seconds.\n\n        Args:\n            seconds (int): number of seconds to skip\n        '
        self.bus.emit(Message('mycroft.audio.service.seek_forward', {'seconds': seconds}))

    def seek_backward(self, seconds=1):
        if False:
            i = 10
            return i + 15
        'Rewind X seconds\n\n         Args:\n            seconds (int): number of seconds to rewind\n        '
        self.bus.emit(Message('mycroft.audio.service.seek_backward', {'seconds': seconds}))

    def track_info(self):
        if False:
            for i in range(10):
                print('nop')
        'Request information of current playing track.\n\n        Returns:\n            Dict with track info.\n        '
        info = self.bus.wait_for_response(Message('mycroft.audio.service.track_info'), reply_type='mycroft.audio.service.track_info_reply', timeout=1)
        return info.data if info else {}

    def available_backends(self):
        if False:
            print('Hello World!')
        'Return available audio backends.\n\n        Returns:\n            dict with backend names as keys\n        '
        msg = Message('mycroft.audio.service.list_backends')
        response = self.bus.wait_for_response(msg)
        return response.data if response else {}

    @property
    def is_playing(self):
        if False:
            i = 10
            return i + 15
        'True if the audioservice is playing, else False.'
        return self.track_info() != {}