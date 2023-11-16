from __future__ import absolute_import
from collections import namedtuple
import urllib3

class HTTPError(Exception):
    """Base exception used by this module."""
    pass

class LocationValueError(ValueError, HTTPError):
    """Raised when there is something wrong with a given URL input."""
    pass

class LocationParseError(LocationValueError):
    """Raised when get_host or similar fails to parse the URL input."""

    def __init__(self, location):
        if False:
            i = 10
            return i + 15
        message = 'Failed to parse: %s' % location
        HTTPError.__init__(self, message)
        self.location = location
url_attrs = ['scheme', 'auth', 'host', 'port', 'path', 'query', 'fragment']
NORMALIZABLE_SCHEMES = ('http', 'https', None)

class Url(namedtuple('Url', url_attrs)):
    """
    Datastructure for representing an HTTP URL. Used as a return value for
    :func:`parse_url`. Both the scheme and host are normalized as they are
    both case-insensitive according to RFC 3986.
    """
    __slots__ = ()

    def __new__(cls, scheme=None, auth=None, host=None, port=None, path=None, query=None, fragment=None):
        if False:
            print('Hello World!')
        if path and (not path.startswith('/')):
            path = '/' + path
        if scheme:
            scheme = scheme.lower()
        if host and scheme in NORMALIZABLE_SCHEMES:
            host = host.lower()
        return super(Url, cls).__new__(cls, scheme, auth, host, port, path, query, fragment)

    @property
    def hostname(self):
        if False:
            print('Hello World!')
        "For backwards-compatibility with urlparse. We're nice like that."
        return self.host

    @property
    def request_uri(self):
        if False:
            i = 10
            return i + 15
        'Absolute path including the query string.'
        uri = self.path or '/'
        if self.query is not None:
            uri += '?' + self.query
        return uri

    @property
    def netloc(self):
        if False:
            while True:
                i = 10
        'Network location including host and port'
        if self.port:
            return '%s:%d' % (self.host, self.port)
        return self.host

    @property
    def url(self):
        if False:
            print('Hello World!')
        "\n        Convert self into a url\n\n        This function should more or less round-trip with :func:`.parse_url`. The\n        returned url may not be exactly the same as the url inputted to\n        :func:`.parse_url`, but it should be equivalent by the RFC (e.g., urls\n        with a blank port will have : removed).\n\n        Example: ::\n\n            >>> U = parse_url('http://google.com/mail/')\n            >>> U.url\n            'http://google.com/mail/'\n            >>> Url('http', 'username:password', 'host.com', 80,\n            ... '/path', 'query', 'fragment').url\n            'http://username:password@host.com:80/path?query#fragment'\n        "
        (scheme, auth, host, port, path, query, fragment) = self
        url = ''
        if scheme is not None:
            url += scheme + '://'
        if auth is not None:
            url += auth + '@'
        if host is not None:
            url += host
        if port is not None:
            url += ':' + str(port)
        if path is not None:
            url += path
        if query is not None:
            url += '?' + query
        if fragment is not None:
            url += '#' + fragment
        return url

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.url

def patched_parse_url(url):
    if False:
        print('Hello World!')
    "\n    Given a url, return a parsed :class:`.Url` namedtuple. Best-effort is\n    performed to parse incomplete urls. Fields not provided will be None.\n\n    Partly backwards-compatible with :mod:`urlparse`.\n\n    Example::\n\n        >>> parse_url('http://google.com/mail/')\n        Url(scheme='http', host='google.com', port=None, path='/mail/', ...)\n        >>> parse_url('google.com:80')\n        Url(scheme=None, host='google.com', port=80, path=None, ...)\n        >>> parse_url('/foo?bar')\n        Url(scheme=None, host=None, port=None, path='/foo', query='bar', ...)\n    "

    def split_first(s, delims):
        if False:
            while True:
                i = 10
        "\n        Given a string and an iterable of delimiters, split on the first found\n        delimiter. Return two split parts and the matched delimiter.\n\n        If not found, then the first part is the full input string.\n\n        Example::\n\n            >>> split_first('foo/bar?baz', '?/=')\n            ('foo', 'bar?baz', '/')\n            >>> split_first('foo/bar?baz', '123')\n            ('foo/bar?baz', '', None)\n\n        Scales linearly with number of delims. Not ideal for large number of delims.\n        "
        min_idx = None
        min_delim = None
        for d in delims:
            idx = s.find(d)
            if idx < 0:
                continue
            if min_idx is None or idx < min_idx:
                min_idx = idx
                min_delim = d
        if min_idx is None or min_idx < 0:
            return (s, '', None)
        return (s[:min_idx], s[min_idx + 1:], min_delim)
    if not url:
        return Url()
    scheme = None
    auth = None
    host = None
    port = None
    path = None
    fragment = None
    query = None
    if '://' in url:
        (scheme, url) = url.split('://', 1)
    (url, path_, delim) = split_first(url, ['/', '?', '#'])
    if delim:
        path = delim + path_
    if '@' in url:
        (auth, url) = url.rsplit('@', 1)
    if url and url[0] == '[':
        (host, url) = url.split(']', 1)
        host += ']'
    if ':' in url:
        (_host, port) = url.split(':', 1)
        if not host:
            host = _host
        if port:
            if not port.isdigit():
                raise LocationParseError(url)
            try:
                port = int(port)
            except ValueError:
                raise LocationParseError(url)
        else:
            port = None
    elif not host and url:
        host = url
    if not path:
        return Url(scheme, auth, host, port, path, query, fragment)
    if '#' in path:
        (path, fragment) = path.split('#', 1)
    if '?' in path:
        (path, query) = path.split('?', 1)
    return Url(scheme, auth, host, port, path, query, fragment)

def patch_urllib3_parse_url():
    if False:
        return 10
    try:
        urllib3.util.parse_url.__code__ = patched_parse_url.__code__
    except Exception:
        pass