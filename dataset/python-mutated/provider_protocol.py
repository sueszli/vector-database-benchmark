from typing import runtime_checkable, Protocol, List, Tuple, Optional, Dict
from abc import abstractmethod
from feeluown.media import Quality, Media
from .models import BriefCommentModel, SongModel, VideoModel, AlbumModel, ArtistModel, PlaylistModel, UserModel, ModelType
from .model_protocol import BriefArtistProtocol, BriefSongProtocol, SongProtocol, BriefVideoProtocol, VideoProtocol, LyricProtocol
from .flags import Flags as PF
__all__ = ('SupportsAlbumGet', 'SupportsAlbumSongsReader', 'SupportsArtistAlbumsReader', 'SupportsArtistGet', 'SupportsArtistSongsReader', 'SupportsCurrentUser', 'SupportsPlaylistAddSong', 'SupportsPlaylistGet', 'SupportsPlaylistCreateByName', 'SupportsPlaylistDelete', 'SupportsPlaylistRemoveSong', 'SupportsPlaylistSongsReader', 'SupportsSongGet', 'SupportsSongHotComments', 'SupportsSongLyric', 'SupportsSongMV', 'SupportsSongMultiQuality', 'SupportsSongSimilar', 'SupportsSongWebUrl', 'SupportsVideoGet', 'SupportsVideoMultiQuality')
ID = str
_FlagProtocolMapping: Dict[Tuple[ModelType, PF], type] = {}

def check_flag(provider, model_type: ModelType, flag: PF) -> bool:
    if False:
        i = 10
        return i + 15
    'Check if provider supports X'
    if flag is PF.model_v2:
        try:
            use_model_v2 = provider.use_model_v2(model_type)
        except AttributeError:
            return False
        return use_model_v2
    protocol_cls = _FlagProtocolMapping[model_type, flag]
    return isinstance(provider, protocol_cls)

def eq(model_type: ModelType, flag: PF):
    if False:
        while True:
            i = 10
    'Decorate a protocol class and associate it with a provider flag'

    def wrapper(cls):
        if False:
            i = 10
            return i + 15
        _FlagProtocolMapping[model_type, flag] = cls
        return cls
    return wrapper

@eq(ModelType.song, PF.get)
@runtime_checkable
class SupportsSongGet(Protocol):

    @abstractmethod
    def song_get(self, identifier: ID) -> SongModel:
        if False:
            return 10
        '\n        :raises ModelNotFound: model not found by the identifier\n        :raises ProviderIOError:\n        '
        raise NotImplementedError

@eq(ModelType.song, PF.similar)
@runtime_checkable
class SupportsSongSimilar(Protocol):

    @abstractmethod
    def song_list_similar(self, song: BriefSongProtocol) -> List[BriefSongProtocol]:
        if False:
            i = 10
            return i + 15
        'List similar songs\n        '
        raise NotImplementedError

@eq(ModelType.song, PF.multi_quality)
@runtime_checkable
class SupportsSongMultiQuality(Protocol):

    @abstractmethod
    def song_list_quality(self, song: BriefSongProtocol) -> List[Quality.Audio]:
        if False:
            i = 10
            return i + 15
        'List all possible qualities\n\n        Please ensure all the qualities are valid. `song_get_media(song, quality)`\n        must not return None with a valid quality.\n        '
        raise NotImplementedError

    @abstractmethod
    def song_select_media(self, song: BriefSongProtocol, policy=None) -> Tuple[Media, Quality.Audio]:
        if False:
            i = 10
            return i + 15
        'Select a media by the quality sorting policy\n\n        If the song has some valid medias, this method can always return one of them.\n        '
        raise NotImplementedError

    @abstractmethod
    def song_get_media(self, song: BriefVideoProtocol, quality: Quality.Audio) -> Optional[Media]:
        if False:
            while True:
                i = 10
        "Get song's media by a specified quality\n\n        :return: when quality is invalid, return None\n        "
        raise NotImplementedError

@eq(ModelType.song, PF.hot_comments)
@runtime_checkable
class SupportsSongHotComments(Protocol):

    def song_list_hot_comments(self, song: BriefSongProtocol) -> List[BriefCommentModel]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

@eq(ModelType.song, PF.web_url)
@runtime_checkable
class SupportsSongWebUrl(Protocol):

    def song_get_web_url(self, song: BriefSongProtocol) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

@eq(ModelType.song, PF.lyric)
@runtime_checkable
class SupportsSongLyric(Protocol):

    def song_get_lyric(self, song: BriefSongProtocol) -> Optional[LyricProtocol]:
        if False:
            print('Hello World!')
        'Get music video of the song\n        '
        raise NotImplementedError

@eq(ModelType.song, PF.mv)
@runtime_checkable
class SupportsSongMV(Protocol):

    def song_get_mv(self, song: BriefSongProtocol) -> Optional[VideoProtocol]:
        if False:
            i = 10
            return i + 15
        'Get music video of the song\n\n        '
        raise NotImplementedError

@eq(ModelType.album, PF.get)
@runtime_checkable
class SupportsAlbumGet(Protocol):

    @abstractmethod
    def album_get(self, identifier: ID) -> AlbumModel:
        if False:
            return 10
        '\n        :raises ModelNotFound: model not found by the identifier\n        :raises ProviderIOError:\n        '
        raise NotImplementedError

@eq(ModelType.album, PF.songs_rd)
@runtime_checkable
class SupportsAlbumSongsReader(Protocol):

    @abstractmethod
    def album_create_songs_rd(self, album) -> List[SongProtocol]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

@eq(ModelType.artist, PF.get)
@runtime_checkable
class SupportsArtistGet(Protocol):

    @abstractmethod
    def artist_get(self, identifier: ID) -> ArtistModel:
        if False:
            print('Hello World!')
        '\n        :raises ModelNotFound: model not found by the identifier\n        :raises ProviderIOError:\n        '
        raise NotImplementedError

@eq(ModelType.artist, PF.songs_rd)
@runtime_checkable
class SupportsArtistSongsReader(Protocol):

    @abstractmethod
    def artist_create_songs_rd(self, artist: BriefArtistProtocol):
        if False:
            print('Hello World!')
        'Create songs reader of the artist\n        '
        raise NotImplementedError

@eq(ModelType.artist, PF.albums_rd)
@runtime_checkable
class SupportsArtistAlbumsReader(Protocol):

    @abstractmethod
    def artist_create_albums_rd(self, artist: BriefArtistProtocol):
        if False:
            return 10
        'Create albums reader of the artist\n        '
        raise NotImplementedError

@runtime_checkable
class SupportsArtistContributedAlbumsReader(Protocol):

    @abstractmethod
    def artist_create_contributed_albums_rd(self, artist: BriefArtistProtocol):
        if False:
            return 10
        'Create contributed albums reader of the artist\n        '
        raise NotImplementedError

@eq(ModelType.video, PF.get)
@runtime_checkable
class SupportsVideoGet(Protocol):

    @abstractmethod
    def video_get(self, identifier: ID) -> VideoModel:
        if False:
            print('Hello World!')
        '\n        :raises ModelNotFound: model not found by the identifier\n        :raises ProviderIOError:\n        '
        raise NotImplementedError

@eq(ModelType.video, PF.multi_quality)
@runtime_checkable
class SupportsVideoMultiQuality(Protocol):

    @abstractmethod
    def video_list_quality(self, video: BriefVideoProtocol) -> List[Quality.Video]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def video_select_media(self, video: BriefVideoProtocol, policy=None) -> Tuple[Media, Quality.Video]:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def video_get_media(self, video: BriefVideoProtocol, quality) -> Optional[Media]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

@eq(ModelType.playlist, PF.get)
@runtime_checkable
class SupportsPlaylistGet(Protocol):

    @abstractmethod
    def playlist_get(self, identifier: ID) -> PlaylistModel:
        if False:
            i = 10
            return i + 15
        '\n        :raises ModelNotFound: model not found by the identifier\n        :raises ProviderIOError:\n        '
        raise NotImplementedError

@runtime_checkable
class SupportsPlaylistCreateByName(Protocol):

    @abstractmethod
    def playlist_create_by_name(self, name) -> PlaylistModel:
        if False:
            i = 10
            return i + 15
        'Create playlist for user logged in.\n\n        :raises NoUserLoggedIn:\n        :raises ProviderIOError:\n        '

@runtime_checkable
class SupportsPlaylistDelete(Protocol):

    @abstractmethod
    def playlist_delete(self, identifier: ID) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        :raises ModelNotFound: model not found by the identifier\n        :raises ProviderIOError:\n        '
        raise NotImplementedError

@eq(ModelType.playlist, PF.songs_rd)
@runtime_checkable
class SupportsPlaylistSongsReader(Protocol):

    @abstractmethod
    def playlist_create_songs_rd(self, playlist):
        if False:
            print('Hello World!')
        raise NotImplementedError

@eq(ModelType.playlist, PF.add_song)
@runtime_checkable
class SupportsPlaylistAddSong(Protocol):

    @abstractmethod
    def playlist_add_song(self, playlist, song) -> bool:
        if False:
            return 10
        raise NotImplementedError

@eq(ModelType.playlist, PF.remove_song)
@runtime_checkable
class SupportsPlaylistRemoveSong(Protocol):

    @abstractmethod
    def playlist_remove_song(self, playlist, song) -> bool:
        if False:
            while True:
                i = 10
        raise NotImplementedError

@eq(ModelType.none, PF.current_user)
@runtime_checkable
class SupportsCurrentUser(Protocol):

    @abstractmethod
    def has_current_user(self) -> bool:
        if False:
            return 10
        'Check if there is a logged in user.'

    @abstractmethod
    def get_current_user(self) -> UserModel:
        if False:
            while True:
                i = 10
        'Get current logged in user\n\n        :raises NoUserLoggedIn: there is no logged in user.\n        '