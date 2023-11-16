import shutil
from unittest.mock import Mock
import pytest
from aiohttp import ClientResponseError
from libtorrent import bencode
from tribler.core.components.libtorrent.torrentdef import TorrentDef, TorrentDefNoMetainfo
from tribler.core.tests.tools.common import TESTS_DATA_DIR, TORRENT_UBUNTU_FILE
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.utilities import bdecode_compat
TRACKER = 'http://www.tribler.org/announce'
VIDEO_FILE_NAME = 'video.avi'

def test_tdef_init():
    if False:
        print('Hello World!')
    '\n    Test initializing a TorrentDef object\n    '
    tdef_params = TorrentDef(torrent_parameters={'announce': 'http://test.com'})
    assert 'announce' in tdef_params.torrent_parameters

def test_create_invalid_tdef():
    if False:
        print('Hello World!')
    '\n    Test whether creating invalid TorrentDef objects result in ValueErrors\n    '
    invalid_metainfo = {}
    with pytest.raises(ValueError):
        TorrentDef.load_from_memory(bencode(invalid_metainfo))
    invalid_metainfo = {b'info': {}}
    with pytest.raises(ValueError):
        TorrentDef.load_from_memory(bencode(invalid_metainfo))

def test_add_content_dir(tdef):
    if False:
        return 10
    '\n    Test whether adding a single content directory with two files is working correctly\n    '
    torrent_dir = TESTS_DATA_DIR / 'contentdir'
    tdef.add_content(torrent_dir / 'file.txt')
    tdef.add_content(torrent_dir / 'otherfile.txt')
    tdef.save()
    metainfo = tdef.get_metainfo()
    assert len(metainfo[b'info'][b'files']) == 2

def test_add_single_file(tdef):
    if False:
        while True:
            i = 10
    '\n    Test whether adding a single file to a torrent is working correctly\n    '
    torrent_dir = TESTS_DATA_DIR / 'contentdir'
    tdef.add_content(torrent_dir / 'file.txt')
    tdef.save()
    metainfo = tdef.get_metainfo()
    assert metainfo[b'info'][b'name'] == b'file.txt'

def test_get_name_utf8_unknown(tdef):
    if False:
        while True:
            i = 10
    '\n    Test whether we can succesfully get the UTF-8 name\n    '
    tdef.set_name(b'\xa1\xc0')
    tdef.torrent_parameters[b'encoding'] = 'euc_kr'
    assert tdef.get_name_utf8() == '÷'

def test_get_name_utf8(tdef):
    if False:
        i = 10
        return i + 15
    '\n    Check whether we can successfully get the UTF-8 encoded torrent name when using a different encoding\n    '
    tdef.set_name(b'\xa1\xc0')
    assert tdef.get_name_utf8() == '¡À'

def test_add_content_piece_length(tdef):
    if False:
        while True:
            i = 10
    '\n    Add a single file with piece length to a TorrentDef\n    '
    fn = TESTS_DATA_DIR / VIDEO_FILE_NAME
    tdef.add_content(fn)
    tdef.set_piece_length(2 ** 16)
    tdef.save()
    metainfo = tdef.get_metainfo()
    assert metainfo[b'info'][b'piece length'] == 2 ** 16

def test_is_private(tdef):
    if False:
        return 10
    tdef.metainfo = {b'info': {b'private': 0}}
    assert tdef.is_private() is False
    tdef.metainfo = {b'info': {b'private': 1}}
    assert tdef.is_private() is True
    tdef.metainfo = {b'info': {b'private': b'i1e'}}
    assert tdef.is_private() is False
    tdef.metainfo = {b'info': {b'private': b'i0e'}}
    assert tdef.is_private() is False

async def test_is_private_loaded_from_existing_torrent():
    """
    Test whether the private field from an existing torrent is correctly read
    """
    privatefn = TESTS_DATA_DIR / 'private.torrent'
    publicfn = TESTS_DATA_DIR / 'bak_single.torrent'
    t1 = await TorrentDef.load(privatefn)
    t2 = await TorrentDef.load(publicfn)
    assert t1.is_private()
    assert not t2.is_private()

async def test_load_from_url(file_server, tmpdir):
    shutil.copyfile(TORRENT_UBUNTU_FILE, tmpdir / 'ubuntu.torrent')
    torrent_url = 'http://localhost:%d/ubuntu.torrent' % file_server
    torrent_def = await TorrentDef.load_from_url(torrent_url)
    assert torrent_def.get_metainfo() == (await TorrentDef.load(TORRENT_UBUNTU_FILE)).get_metainfo()
    assert torrent_def.infohash == (await TorrentDef.load(TORRENT_UBUNTU_FILE)).infohash

async def test_load_from_url_404(file_server, tmpdir):
    torrent_url = 'http://localhost:%d/ubuntu.torrent' % file_server
    try:
        await TorrentDef.load_from_url(torrent_url)
    except ClientResponseError as e:
        assert e.status == 404

def test_torrent_encoding(tdef):
    if False:
        for i in range(10):
            print('nop')
    assert tdef.get_encoding() == 'utf-8'
    tdef.set_encoding(b'my_fancy_encoding')
    assert tdef.get_encoding() == 'my_fancy_encoding'

def test_set_tracker_invalid_url(tdef):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        tdef.set_tracker('http/tracker.org')

def test_set_tracker_strip_slash(tdef):
    if False:
        while True:
            i = 10
    tdef.set_tracker('http://tracker.org/')
    assert tdef.torrent_parameters[b'announce'] == 'http://tracker.org'

def test_set_tracker(tdef):
    if False:
        print('Hello World!')
    assert len(tdef.get_trackers()) == 0
    tdef.set_tracker('http://tracker.org')
    assert tdef.get_trackers() == {'http://tracker.org'}

def test_get_trackers(tdef):
    if False:
        while True:
            i = 10
    '\n    Test that `get_trackers` returns flat set of trackers\n    '
    tdef.get_tracker_hierarchy = Mock(return_value=[['t1', 't2'], ['t3'], ['t4']])
    trackers = tdef.get_trackers()
    assert trackers == {'t1', 't2', 't3', 't4'}

def test_get_nr_pieces(tdef):
    if False:
        print('Hello World!')
    '\n    Test getting the number of pieces from a TorrentDef\n    '
    assert tdef.get_nr_pieces() == 0
    tdef.metainfo = {b'info': {b'pieces': b'a' * 40}}
    assert tdef.get_nr_pieces() == 2

def test_is_multifile(tdef):
    if False:
        while True:
            i = 10
    '\n    Test whether a TorrentDef is correctly classified as multifile torrent\n    '
    assert not tdef.is_multifile_torrent()
    tdef.metainfo = {}
    assert not tdef.is_multifile_torrent()
    tdef.metainfo = {b'info': {b'files': [b'a']}}
    assert tdef.is_multifile_torrent()

def test_set_piece_length_invalid_type(tdef):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        tdef.set_piece_length('20')

def test_get_piece_length(tdef):
    if False:
        i = 10
        return i + 15
    assert tdef.get_piece_length() == 0

def test_load_from_dict():
    if False:
        return 10
    with open(TESTS_DATA_DIR / 'bak_single.torrent', mode='rb') as torrent_file:
        encoded_metainfo = torrent_file.read()
    assert TorrentDef.load_from_dict(bdecode_compat(encoded_metainfo))

def test_torrent_no_metainfo():
    if False:
        i = 10
        return i + 15
    tdef = TorrentDefNoMetainfo(b'12345678901234567890', VIDEO_FILE_NAME, 'http://google.com')
    assert tdef.get_name() == VIDEO_FILE_NAME
    assert tdef.get_infohash() == b'12345678901234567890'
    assert tdef.get_length() == 0
    assert not tdef.get_metainfo()
    assert tdef.get_url() == 'http://google.com'
    assert not tdef.is_multifile_torrent()
    assert tdef.get_name_as_unicode() == VIDEO_FILE_NAME
    assert not tdef.get_files()
    assert tdef.get_files_with_length() == []
    assert len(tdef.get_trackers()) == 0
    assert not tdef.is_private()
    assert tdef.get_name_utf8() == 'video.avi'
    assert tdef.get_nr_pieces() == 0
    torrent2 = TorrentDefNoMetainfo(b'12345678901234567890', VIDEO_FILE_NAME, 'magnet:')
    assert len(torrent2.get_trackers()) == 0

def test_get_length(tdef):
    if False:
        while True:
            i = 10
    '\n    Test whether a TorrentDef has 0 length by default.\n    '
    assert not tdef.get_length()

def test_get_index(tdef):
    if False:
        print('Hello World!')
    '\n    Test whether we can successfully get the index of a file in a torrent.\n    '
    tdef.metainfo = {b'info': {b'files': [{b'path': [b'a.txt'], b'length': 123}]}}
    assert tdef.get_index_of_file_in_files('a.txt') == 0
    with pytest.raises(ValueError):
        tdef.get_index_of_file_in_files(b'b.txt')
    with pytest.raises(ValueError):
        tdef.get_index_of_file_in_files(None)
    tdef.metainfo = {b'info': {b'files': [{b'path': [b'a.txt'], b'path.utf-8': [b'b.txt'], b'length': 123}]}}
    assert tdef.get_index_of_file_in_files('b.txt') == 0
    tdef.metainfo = None
    with pytest.raises(ValueError):
        tdef.get_index_of_file_in_files('b.txt')

def test_get_name_as_unicode(tdef):
    if False:
        for i in range(10):
            print('nop')
    name_bytes = b'\xe8\xaf\xad\xe8\xa8\x80\xe5\xa4\x84\xe7\x90\x86'
    name_unicode = name_bytes.decode()
    tdef.metainfo = {b'info': {b'name.utf-8': name_bytes}}
    assert tdef.get_name_as_unicode() == name_unicode
    tdef.metainfo = {b'info': {b'name': name_bytes}}
    assert tdef.get_name_as_unicode() == name_unicode
    tdef.metainfo = {b'info': {b'name': b'test\xff' + name_bytes}}
    assert tdef.get_name_as_unicode() == 'test' + '?' * len(b'\xff' + name_bytes)

def test_filter_characters(tdef):
    if False:
        return 10
    '\n    Test `_filter_characters` sanitizes its input\n    '
    name_bytes = b'\xe8\xaf\xad\xe8\xa8\x80\xe5\xa4\x84\xe7\x90\x86'
    name = name_bytes
    name_sanitized = '?' * len(name)
    assert tdef._filter_characters(name) == name_sanitized
    name = b'test\xff' + name_bytes
    name_sanitized = 'test' + '?' * len(b'\xff' + name_bytes)
    assert tdef._filter_characters(name) == name_sanitized

def test_get_files_with_length(tdef):
    if False:
        print('Hello World!')
    name_bytes = b'\xe8\xaf\xad\xe8\xa8\x80\xe5\xa4\x84\xe7\x90\x86'
    name_unicode = name_bytes.decode()
    tdef.metainfo = {b'info': {b'files': [{b'path.utf-8': [name_bytes], b'length': 123}, {b'path.utf-8': [b'file.txt'], b'length': 456}]}}
    assert tdef.get_files_with_length() == [(Path(name_unicode), 123), (Path('file.txt'), 456)]
    tdef.metainfo = {b'info': {b'files': [{b'path': [name_bytes], b'length': 123}, {b'path': [b'file.txt'], b'length': 456}]}}
    assert tdef.get_files_with_length() == [(Path(name_unicode), 123), (Path('file.txt'), 456)]
    tdef.metainfo = {b'info': {b'files': [{b'path': [b'test\xff' + name_bytes], b'length': 123}, {b'path': [b'file.txt'], b'length': 456}]}}
    assert tdef.get_files_with_length() == [(Path('test?????????????'), 123), (Path('file.txt'), 456)]
    tdef.metainfo = {b'info': {b'files': [{b'path.utf-8': [b'test\xff' + name_bytes], b'length': 123}, {b'path': [b'file.txt'], b'length': 456}]}}
    assert tdef.get_files_with_length() == [(Path('file.txt'), 456)]