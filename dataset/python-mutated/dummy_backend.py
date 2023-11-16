"""A dummy backend for use in tests.

This backend implements the backend API in the simplest way possible.  It is
used in tests of the frontends.
"""
import pykka
from mopidy import backend
from mopidy.models import Playlist, Ref, SearchResult

def create_proxy(config=None, audio=None):
    if False:
        for i in range(10):
            print('nop')
    return DummyBackend.start(config=config, audio=audio).proxy()

class DummyBackend(pykka.ThreadingActor, backend.Backend):

    def __init__(self, config, audio):
        if False:
            print('Hello World!')
        super().__init__()
        self.library = DummyLibraryProvider(backend=self)
        if audio:
            self.playback = backend.PlaybackProvider(audio=audio, backend=self)
        else:
            self.playback = DummyPlaybackProvider(audio=audio, backend=self)
        self.playlists = DummyPlaylistsProvider(backend=self)
        self.uri_schemes = ['dummy']

class DummyLibraryProvider(backend.LibraryProvider):
    root_directory = Ref.directory(uri='dummy:/', name='dummy')

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.dummy_library = []
        self.dummy_get_distinct_result = {}
        self.dummy_browse_result = {}
        self.dummy_find_exact_result = SearchResult()
        self.dummy_search_result = SearchResult()

    def browse(self, path):
        if False:
            return 10
        return self.dummy_browse_result.get(path, [])

    def get_distinct(self, field, query=None):
        if False:
            return 10
        return self.dummy_get_distinct_result.get(field, set())

    def lookup(self, uri):
        if False:
            i = 10
            return i + 15
        uri = Ref.track(uri=uri).uri
        return [t for t in self.dummy_library if uri == t.uri]

    def refresh(self, uri=None):
        if False:
            print('Hello World!')
        pass

    def search(self, query=None, uris=None, exact=False):
        if False:
            print('Hello World!')
        if exact:
            return self.dummy_find_exact_result
        return self.dummy_search_result

class DummyPlaybackProvider(backend.PlaybackProvider):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self._uri = None
        self._time_position = 0

    def pause(self):
        if False:
            i = 10
            return i + 15
        return True

    def play(self):
        if False:
            i = 10
            return i + 15
        return self._uri and self._uri != 'dummy:error'

    def change_track(self, track):
        if False:
            while True:
                i = 10
        "Pass a track with URI 'dummy:error' to force failure"
        self._uri = track.uri
        self._time_position = 0
        return True

    def prepare_change(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def resume(self):
        if False:
            while True:
                i = 10
        return True

    def seek(self, time_position):
        if False:
            i = 10
            return i + 15
        self._time_position = time_position
        return True

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        self._uri = None
        return True

    def get_time_position(self):
        if False:
            while True:
                i = 10
        return self._time_position

class DummyPlaylistsProvider(backend.PlaylistsProvider):

    def __init__(self, backend):
        if False:
            return 10
        super().__init__(backend)
        self._playlists = []
        self._allow_save = True

    def set_dummy_playlists(self, playlists):
        if False:
            i = 10
            return i + 15
        'For tests using the dummy provider through an actor proxy.'
        self._playlists = playlists

    def set_allow_save(self, enabled):
        if False:
            i = 10
            return i + 15
        self._allow_save = enabled

    def as_list(self):
        if False:
            i = 10
            return i + 15
        return [Ref.playlist(uri=pl.uri, name=pl.name) for pl in self._playlists]

    def get_items(self, uri):
        if False:
            i = 10
            return i + 15
        playlist = self.lookup(uri)
        if playlist is None:
            return None
        return [Ref.track(uri=t.uri, name=t.name) for t in playlist.tracks]

    def lookup(self, uri):
        if False:
            while True:
                i = 10
        uri = Ref.playlist(uri=uri).uri
        for playlist in self._playlists:
            if playlist.uri == uri:
                return playlist
        return None

    def refresh(self):
        if False:
            print('Hello World!')
        pass

    def create(self, name):
        if False:
            for i in range(10):
                print('nop')
        playlist = Playlist(name=name, uri=f'dummy:{name}')
        self._playlists.append(playlist)
        return playlist

    def delete(self, uri):
        if False:
            while True:
                i = 10
        playlist = self.lookup(uri)
        if playlist:
            self._playlists.remove(playlist)

    def save(self, playlist):
        if False:
            return 10
        if not self._allow_save:
            return None
        old_playlist = self.lookup(playlist.uri)
        if old_playlist is not None:
            index = self._playlists.index(old_playlist)
            self._playlists[index] = playlist
        else:
            self._playlists.append(playlist)
        return playlist