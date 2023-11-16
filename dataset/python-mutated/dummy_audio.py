"""A dummy audio actor for use in tests.

This class implements the audio API in the simplest way possible. It is used in
tests of the core and backends.
"""
import pykka
from mopidy import audio

def create_proxy(config=None, mixer=None):
    if False:
        return 10
    return DummyAudio.start(config, mixer).proxy()

class DummyAudio(pykka.ThreadingActor):

    def __init__(self, config=None, mixer=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.state = audio.PlaybackState.STOPPED
        self._volume = 0
        self._position = 0
        self._source_setup_callback = None
        self._about_to_finish_callback = None
        self._uri = None
        self._stream_changed = False
        self._live_stream = False
        self._tags = {}
        self._bad_uris = set()

    def set_uri(self, uri, live_stream=False, download=False):
        if False:
            print('Hello World!')
        assert self._uri is None, 'prepare change not called before set'
        self._position = 0
        self._uri = uri
        self._stream_changed = True
        self._live_stream = live_stream
        self._tags = {}

    def get_position(self):
        if False:
            print('Hello World!')
        return self._position

    def set_position(self, position):
        if False:
            print('Hello World!')
        self._position = position
        audio.AudioListener.send('position_changed', position=position)
        return True

    def start_playback(self):
        if False:
            return 10
        return self._change_state(audio.PlaybackState.PLAYING)

    def pause_playback(self):
        if False:
            print('Hello World!')
        return self._change_state(audio.PlaybackState.PAUSED)

    def prepare_change(self):
        if False:
            for i in range(10):
                print('nop')
        self._uri = None
        self._source_setup_callback = None
        return True

    def stop_playback(self):
        if False:
            i = 10
            return i + 15
        return self._change_state(audio.PlaybackState.STOPPED)

    def get_volume(self):
        if False:
            print('Hello World!')
        return self._volume

    def set_volume(self, volume):
        if False:
            return 10
        self._volume = volume
        return True

    def get_current_tags(self):
        if False:
            return 10
        return self._tags

    def set_source_setup_callback(self, callback):
        if False:
            print('Hello World!')
        self._source_setup_callback = callback

    def set_about_to_finish_callback(self, callback):
        if False:
            i = 10
            return i + 15
        self._about_to_finish_callback = callback

    def enable_sync_handler(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def wait_for_state_change(self):
        if False:
            i = 10
            return i + 15
        pass

    def _change_state(self, new_state):
        if False:
            while True:
                i = 10
        if not self._uri:
            return False
        if new_state == audio.PlaybackState.STOPPED and self._uri:
            self._stream_changed = True
            self._uri = None
        if self._stream_changed:
            self._stream_changed = False
            audio.AudioListener.send('stream_changed', uri=self._uri)
        if self._uri is not None:
            audio.AudioListener.send('position_changed', position=0)
        (old_state, self.state) = (self.state, new_state)
        audio.AudioListener.send('state_changed', old_state=old_state, new_state=new_state, target_state=None)
        if new_state == audio.PlaybackState.PLAYING:
            self._tags['audio-codec'] = ['fake info...']
            audio.AudioListener.send('tags_changed', tags=['audio-codec'])
        return self._uri not in self._bad_uris

    def trigger_fake_playback_failure(self, uri):
        if False:
            return 10
        self._bad_uris.add(uri)

    def trigger_fake_tags_changed(self, tags):
        if False:
            for i in range(10):
                print('nop')
        self._tags.update(tags)
        audio.AudioListener.send('tags_changed', tags=self._tags.keys())

    def get_source_setup_callback(self):
        if False:
            print('Hello World!')

        def wrapper():
            if False:
                for i in range(10):
                    print('nop')
            if self._source_setup_callback:
                self._source_setup_callback()
        return wrapper

    def get_about_to_finish_callback(self):
        if False:
            while True:
                i = 10

        def wrapper():
            if False:
                print('Hello World!')
            if self._about_to_finish_callback:
                self.prepare_change()
                self._about_to_finish_callback()
            if not self._uri or not self._about_to_finish_callback:
                self._tags = {}
                audio.AudioListener.send('reached_end_of_stream')
            else:
                audio.AudioListener.send('position_changed', position=0)
                audio.AudioListener.send('stream_changed', uri=self._uri)
        return wrapper