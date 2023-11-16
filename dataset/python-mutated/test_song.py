import pytest
from spotdl.types.album import Album
from spotdl.types.song import Song

def test_song_init():
    if False:
        i = 10
        return i + 15
    '\n    Test if Song class is initialized correctly.\n    '
    song = Song(name='test', artists=['test'], album_id='test', album_name='test', album_artist='test', genres=['test'], disc_number=1, duration=1, year=1, date='test', track_number=1, tracks_count=1, isrc='test', song_id='test', cover_url='test', explicit=True, download_url='test', artist='test', copyright_text='test', disc_count=1, publisher='test', url='test', popularity=1)
    assert song.name == 'test'
    assert song.artists == ['test']
    assert song.album_id == 'test'
    assert song.album_name == 'test'
    assert song.album_artist == 'test'
    assert song.genres == ['test']
    assert song.disc_number == 1
    assert song.duration == 1
    assert song.year == 1
    assert song.date == 'test'
    assert song.track_number == 1
    assert song.tracks_count == 1
    assert song.isrc == 'test'
    assert song.song_id == 'test'
    assert song.cover_url == 'test'
    assert song.explicit is True
    assert song.download_url == 'test'
    assert song.popularity == 1

def test_song_wrong_init():
    if False:
        return 10
    '\n    Tests if Song class raises exception when initialized with wrong parameters.\n    '
    with pytest.raises(TypeError):
        Song(name='test', artists=['test'], album_name='test', album_artist=1, genres=['test'], disc_number=1, duration=1, year=1, date='test')

@pytest.mark.vcr()
def test_song_from_url():
    if False:
        while True:
            i = 10
    '\n    Tests if Song.from_url() works correctly.\n    '
    song = Song.from_url('https://open.spotify.com/track/1t2qKa8K72IBC8yQlhD9bU')
    assert song.name == 'Ropes'
    assert song.artists == ['Dirty Palm', 'Chandler Jewels']
    assert song.album_name == 'Ropes'
    assert song.album_artist == 'Dirty Palm'
    assert song.genres == ['gaming edm', 'melbourne bounce international']
    assert song.disc_number == 1
    assert song.duration == 188
    assert song.year == 2021
    assert song.date == '2021-10-28'
    assert song.track_number == 1
    assert song.tracks_count == 1
    assert song.isrc == 'GB2LD2110301'
    assert song.song_id == '1t2qKa8K72IBC8yQlhD9bU'
    assert song.cover_url == 'https://i.scdn.co/image/ab67616d0000b273fe2cb38e4d2412dbb0e54332'
    assert song.explicit == False
    assert song.download_url == None
    assert song.popularity == 0

def test_song_from_data_dump():
    if False:
        return 10
    '\n    Tests if Song.from_data_dump() works correctly.\n    '
    song = Song.from_data_dump('\n        {\n            "name": "Ropes",\n            "artists": ["Dirty Palm", "Chandler Jewels"],\n            "album_id": "4SN9kQlguIcjPtMNQJwD30",\n            "album_name": "Ropes",\n            "album_artist": "Dirty Palm",\n            "genres": ["gaming edm", "melbourne bounce international"],\n            "disc_number": 1,\n            "duration": 188,\n            "year": 2021,\n            "date": "2021-10-28",\n            "track_number": 1,\n            "tracks_count": 1,\n            "isrc": "GB2LD2110301",\n            "song_id": "1t2qKa8K72IBC8yQlhD9bU",\n            "cover_url": "https://i.scdn.co/image/ab67616d0000b273fe2cb38e4d2412dbb0e54332",\n            "explicit": false,\n            "download_url": null,\n            "artist" : "Dirty Palm",\n            "disc_count": 1,\n            "copyright_text": "",\n            "publisher": "",\n            "url": "https://open.spotify.com/track/1t2qKa8K72IBC8yQlhD9bU",\n            "popularity": 0\n        }\n        ')
    assert song.name == 'Ropes'
    assert song.artists == ['Dirty Palm', 'Chandler Jewels']
    assert song.album_name == 'Ropes'
    assert song.album_artist == 'Dirty Palm'
    assert song.genres == ['gaming edm', 'melbourne bounce international']
    assert song.disc_number == 1
    assert song.duration == 188
    assert song.year == 2021
    assert song.date == '2021-10-28'
    assert song.track_number == 1
    assert song.tracks_count == 1
    assert song.isrc == 'GB2LD2110301'
    assert song.song_id == '1t2qKa8K72IBC8yQlhD9bU'
    assert song.cover_url == 'https://i.scdn.co/image/ab67616d0000b273fe2cb38e4d2412dbb0e54332'
    assert song.explicit is False
    assert song.download_url is None
    assert song.popularity == 0

def test_song_from_data_dump_wrong_type():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests if Song.from_data_dump() raises exception when wrong type is passed.\n    '
    with pytest.raises(TypeError):
        Song.from_data_dump(1)

def test_song_from_dict():
    if False:
        i = 10
        return i + 15
    '\n    Tests if Song.from_dict() works correctly.\n    '
    song = Song.from_dict({'name': 'Ropes', 'artists': ['Dirty Palm', 'Chandler Jewels'], 'album_id': '4SN9kQlguIasvwv', 'album_name': 'Ropes', 'album_artist': 'Dirty Palm', 'genres': ['gaming edm', 'melbourne bounce international'], 'disc_number': 1, 'duration': 188, 'year': 2021, 'date': '2021-10-28', 'track_number': 1, 'tracks_count': 1, 'isrc': 'GB2LD2110301', 'song_id': '1t2qKa8K72IBC8yQlhD9bU', 'cover_url': 'https://i.scdn.co/image/ab67616d0000b273fe2cb38e4d2412dbb0e54332', 'explicit': False, 'download_url': None, 'artist': 'Dirty Palm', 'disc_count': 1, 'copyright_text': '', 'publisher': '', 'url': 'https://open.spotify.com/track/1t2qKa8K72IBC8yQlhD9bU', 'popularity': 0})
    assert song.name == 'Ropes'
    assert song.artists == ['Dirty Palm', 'Chandler Jewels']
    assert song.album_name == 'Ropes'
    assert song.album_artist == 'Dirty Palm'
    assert song.genres == ['gaming edm', 'melbourne bounce international']
    assert song.disc_number == 1
    assert song.duration == 188
    assert song.year == 2021
    assert song.date == '2021-10-28'
    assert song.track_number == 1
    assert song.tracks_count == 1
    assert song.isrc == 'GB2LD2110301'
    assert song.song_id == '1t2qKa8K72IBC8yQlhD9bU'
    assert song.cover_url == 'https://i.scdn.co/image/ab67616d0000b273fe2cb38e4d2412dbb0e54332'
    assert song.explicit == False
    assert song.popularity == 0