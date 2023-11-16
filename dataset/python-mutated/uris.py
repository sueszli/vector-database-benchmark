"""A collection of URI utilities with logic built on the VSCode URI library.

https://github.com/Microsoft/vscode-uri/blob/e59cab84f5df6265aed18ae5f43552d3eef13bb9/lib/index.ts
"""
import re
from urllib import parse
from pylsp import IS_WIN
RE_DRIVE_LETTER_PATH = re.compile('^\\/[a-zA-Z]:')

def urlparse(uri):
    if False:
        print('Hello World!')
    'Parse and decode the parts of a URI.'
    (scheme, netloc, path, params, query, fragment) = parse.urlparse(uri)
    return (parse.unquote(scheme), parse.unquote(netloc), parse.unquote(path), parse.unquote(params), parse.unquote(query), parse.unquote(fragment))

def urlunparse(parts):
    if False:
        while True:
            i = 10
    'Unparse and encode parts of a URI.'
    (scheme, netloc, path, params, query, fragment) = parts
    if RE_DRIVE_LETTER_PATH.match(path):
        quoted_path = path[:3] + parse.quote(path[3:])
    else:
        quoted_path = parse.quote(path)
    return parse.urlunparse((parse.quote(scheme), parse.quote(netloc), quoted_path, parse.quote(params), parse.quote(query), parse.quote(fragment)))

def to_fs_path(uri):
    if False:
        return 10
    'Returns the filesystem path of the given URI.\n\n    Will handle UNC paths and normalize windows drive letters to lower-case. Also\n    uses the platform specific path separator. Will *not* validate the path for\n    invalid characters and semantics. Will *not* look at the scheme of this URI.\n    '
    (scheme, netloc, path, _params, _query, _fragment) = urlparse(uri)
    if netloc and path and (scheme == 'file'):
        value = '//{}{}'.format(netloc, path)
    elif RE_DRIVE_LETTER_PATH.match(path):
        value = path[1].lower() + path[2:]
    else:
        value = path
    if IS_WIN:
        value = value.replace('/', '\\')
    return value

def from_fs_path(path):
    if False:
        for i in range(10):
            print('nop')
    'Returns a URI for the given filesystem path.'
    scheme = 'file'
    (params, query, fragment) = ('', '', '')
    (path, netloc) = _normalize_win_path(path)
    return urlunparse((scheme, netloc, path, params, query, fragment))

def uri_with(uri, scheme=None, netloc=None, path=None, params=None, query=None, fragment=None):
    if False:
        while True:
            i = 10
    'Return a URI with the given part(s) replaced.\n\n    Parts are decoded / encoded.\n    '
    (old_scheme, old_netloc, old_path, old_params, old_query, old_fragment) = urlparse(uri)
    (path, _netloc) = _normalize_win_path(path)
    return urlunparse((scheme or old_scheme, netloc or old_netloc, path or old_path, params or old_params, query or old_query, fragment or old_fragment))

def _normalize_win_path(path):
    if False:
        print('Hello World!')
    netloc = ''
    if IS_WIN:
        path = path.replace('\\', '/')
    if path[:2] == '//':
        idx = path.index('/', 2)
        if idx == -1:
            netloc = path[2:]
        else:
            netloc = path[2:idx]
            path = path[idx:]
    if not path.startswith('/'):
        path = '/' + path
    if RE_DRIVE_LETTER_PATH.match(path):
        path = path[0] + path[1].lower() + path[2:]
    return (path, netloc)