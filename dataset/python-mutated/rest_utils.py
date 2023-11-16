import os
from pathlib import Path
from typing import Any, Union
from yarl import URL
MAGNET_SCHEME = 'magnet'
FILE_SCHEME = 'file'
HTTP_SCHEME = 'http'
HTTPS_SCHEME = 'https'

def path_to_url(file_path: Union[str, Any], _path_cls=Path) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Convert path to url\n\n    Example:\n        '/path/to/file' -> 'file:///path/to/file'\n    "
    return _path_cls(file_path).as_uri()

def url_to_path(file_url: str, _path_cls=Path) -> str:
    if False:
        print('Hello World!')
    "Convert url to path\n\n    Example:\n        'file:///path/to/file' -> '/path/to/file'\n    "

    def url_to_path_win():
        if False:
            i = 10
            return i + 15
        if url.host:
            (_, share, *segments) = url.parts
            return str(_path_cls(f'\\\\{url.host}\\{share}', *segments))
        path = url.path.lstrip('/')
        return str(_path_cls(path))
    url = URL(file_url)
    if os.name == 'nt':
        return url_to_path_win()
    return str(_path_cls(url.path))

def scheme_from_url(url: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Get scheme from URL\n\n    Examples:\n        'file:///some/file' -> 'file'\n        'magnet:link' -> 'magnet'\n        'http://en.wikipedia.org' -> 'http'\n    "
    return URL(url).scheme

def url_is_valid_file(file_url: str) -> bool:
    if False:
        print('Hello World!')
    file_path = url_to_path(file_url)
    try:
        return Path(file_path).is_file()
    except OSError:
        return False