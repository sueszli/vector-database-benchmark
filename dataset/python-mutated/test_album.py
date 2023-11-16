import pytest
from spotdl.types.album import Album

def test_album_init():
    if False:
        return 10
    '\n    Test if Playlist class is initialized correctly.\n    '
    Album(name='test', url='test', songs=[], artist={'name': 'test'}, urls=[])

def test_album_wrong_init():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if Playlist class raises exception when initialized with wrong parameters.\n    '
    with pytest.raises(TypeError):
        Album(name='test', url='test')

@pytest.mark.vcr()
def test_album_from_url():
    if False:
        while True:
            i = 10
    '\n    Test if Album class can be initialized from url.\n    '
    album = Album.from_url('https://open.spotify.com/album/4MQnUDGXmHOvnsWCpzeqWT')
    assert album.name == 'NCS: The Best of 2017'
    assert album.url == 'https://open.spotify.com/album/4MQnUDGXmHOvnsWCpzeqWT'
    assert album.artist['name'] == 'Various Artists'
    assert len(album.songs) == 16

@pytest.mark.vcr()
def test_album_length():
    if False:
        i = 10
        return i + 15
    '\n    Tests if Album.length works correctly.\n    '
    album = Album.from_url('https://open.spotify.com/album/4MQnUDGXmHOvnsWCpzeqWT')
    assert album.length == 16