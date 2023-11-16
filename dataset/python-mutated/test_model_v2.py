import pytest
import pydantic
from feeluown.library import SongModel, BriefAlbumModel, BriefArtistModel, BriefSongModel

def test_use_pydantic_from_orm(song):
    if False:
        i = 10
        return i + 15
    with pytest.raises(Exception):
        BriefSongModel.from_orm(song)

def test_create_song_model_basic():
    if False:
        for i in range(10):
            print('nop')
    identifier = '1'
    brief_album = BriefAlbumModel(identifier='1', source='x', name='Film', artists_name='Audrey')
    brief_artist = BriefArtistModel(identifier='1', source='x', name='Audrey')
    song = SongModel(identifier=identifier, source='x', title='Moon', album=brief_album, artists=[brief_artist], duration=240000)
    assert song.artists_name == 'Audrey'

def test_create_model_with_extra_field():
    if False:
        return 10
    with pytest.raises(pydantic.ValidationError):
        BriefSongModel(identifier=1, source='x', unk=0)

def test_song_model_is_hashable():
    if False:
        while True:
            i = 10
    '\n    Song model must be hashable.\n    '
    song = BriefSongModel(identifier=1, source='x')
    hash(song)