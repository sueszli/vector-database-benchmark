from pathlib import PurePosixPath, PureWindowsPath
from unittest.mock import patch
import pytest
from tribler.core.utilities.rest_utils import FILE_SCHEME, HTTP_SCHEME, MAGNET_SCHEME, path_to_url, scheme_from_url, url_is_valid_file, url_to_path
POSIX_PATH_URL = [('/path/to/file', 'file:///path/to/file'), ('/path/to/file with space', 'file:///path/to/file%20with%20space'), ('/path/to/%20%21file', 'file:///path/to/%2520%2521file'), ('//path/to/file', 'file:////path/to/file')]
POSIX_URL_CORNER_CASES = [('file:/path', '/path'), ('file://localhost/path', '/path')]
WIN_PATH_URL = [('C:\\path\\to\\file', 'file:///C:/path/to/file'), ('C:\\path\\to\\file with space', 'file:///C:/path/to/file%20with%20space'), ('C:\\%20%21file', 'file:///C:/%2520%2521file')]
WIN_URL_CORNER_CASES = [('file://server/share/path', '\\\\server\\share\\path')]
SCHEMES = [('file:///path/to/file', FILE_SCHEME), ('magnet:link', MAGNET_SCHEME), ('http://en.wikipedia.org', HTTP_SCHEME)]

@pytest.mark.parametrize('path, url', POSIX_PATH_URL)
@patch('os.name', 'posix')
def test_round_trip_posix(path, url):
    if False:
        i = 10
        return i + 15
    assert path_to_url(path, _path_cls=PurePosixPath) == url
    assert url_to_path(url, _path_cls=PurePosixPath) == path

@pytest.mark.parametrize('url, path', POSIX_URL_CORNER_CASES)
@patch('os.name', 'posix')
def test_posix_corner_cases(url, path):
    if False:
        print('Hello World!')
    assert url_to_path(url, _path_cls=PurePosixPath) == path

@pytest.mark.parametrize('path, url', WIN_PATH_URL)
@patch('os.name', 'nt')
def test_round_trip_win(path, url):
    if False:
        return 10
    assert path_to_url(path, _path_cls=PureWindowsPath) == url
    assert url_to_path(url, _path_cls=PureWindowsPath) == path

@pytest.mark.parametrize('url, path', WIN_URL_CORNER_CASES)
@patch('os.name', 'nt')
def test_win_corner_cases(url, path):
    if False:
        return 10
    assert url_to_path(url, _path_cls=PureWindowsPath) == path

@pytest.mark.parametrize('path, scheme', SCHEMES)
def test_scheme_from_uri(path, scheme):
    if False:
        for i in range(10):
            print('nop')
    assert scheme_from_url(path) == scheme

def test_uri_is_valid_file(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    file_path = tmpdir / '1.txt'
    file_path.write('test')
    file_uri = path_to_url(file_path)
    assert url_is_valid_file(file_uri)
    assert not url_is_valid_file(file_uri + '/*')