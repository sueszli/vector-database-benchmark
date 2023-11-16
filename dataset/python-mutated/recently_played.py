from feeluown.library import ModelType
from feeluown.utils.utils import DedupList

class RecentlyPlayed:
    """
    RecentlyPlayed records recently played models, currently including songs
    and videos. Maybe artists and albums will be recorded in the future.
    """

    def __init__(self, playlist):
        if False:
            print('Hello World!')
        self._songs = DedupList()
        playlist.song_changed_v2.connect(self._on_song_played)

    def init_from_models(self, models):
        if False:
            for i in range(10):
                print('nop')
        for model in models:
            if ModelType(model.meta.model_type) is ModelType.song:
                self._songs.append(model)

    def list_songs(self):
        if False:
            i = 10
            return i + 15
        'List recently played songs (list of BriefSongModel).\n        '
        return list(self._songs.copy())

    def _on_song_played(self, song, _):
        if False:
            return 10
        if song is None:
            return
        if song in self._songs:
            self._songs.remove(song)
        if len(self._songs) >= 100:
            self._songs.pop()
        self._songs.insert(0, song)