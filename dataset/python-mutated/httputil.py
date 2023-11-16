"""HTTP utility code shared by clients and servers.

This module also defines the `HTTPServerRequest` class which is exposed
via `tornado.web.RequestHandler.request`.
"""
import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type
responses
import typing
from typing import Tuple, Iterable, List, Mapping, Iterator, Dict, Union, Optional, Awaitable, Generator, AnyStr
if typing.TYPE_CHECKING:
    from typing import Deque
    from asyncio import Future
    import unittest

@lru_cache(1000)
def _normalize_header(name: str) -> str:
    if False:
        return 10
    'Map a header name to Http-Header-Case.\n\n    >>> _normalize_header("coNtent-TYPE")\n    \'Content-Type\'\n    '
    return '-'.join([w.capitalize() for w in name.split('-')])

class HTTPHeaders(collections.abc.MutableMapping):
    """A dictionary that maintains ``Http-Header-Case`` for all keys.

    Supports multiple values per key via a pair of new methods,
    `add()` and `get_list()`.  The regular dictionary interface
    returns a single value per key, with multiple values joined by a
    comma.

    >>> h = HTTPHeaders({"content-type": "text/html"})
    >>> list(h.keys())
    ['Content-Type']
    >>> h["Content-Type"]
    'text/html'

    >>> h.add("Set-Cookie", "A=B")
    >>> h.add("Set-Cookie", "C=D")
    >>> h["set-cookie"]
    'A=B,C=D'
    >>> h.get_list("set-cookie")
    ['A=B', 'C=D']

    >>> for (k,v) in sorted(h.get_all()):
    ...    print('%s: %s' % (k,v))
    ...
    Content-Type: text/html
    Set-Cookie: A=B
    Set-Cookie: C=D
    """

    @typing.overload
    def __init__(self, __arg: Mapping[str, List[str]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @typing.overload
    def __init__(self, __arg: Mapping[str, str]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @typing.overload
    def __init__(self, *args: Tuple[str, str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @typing.overload
    def __init__(self, **kwargs: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __init__(self, *args: typing.Any, **kwargs: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._dict = {}
        self._as_list = {}
        self._last_key = None
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], HTTPHeaders):
            for (k, v) in args[0].get_all():
                self.add(k, v)
        else:
            self.update(*args, **kwargs)

    def add(self, name: str, value: str) -> None:
        if False:
            i = 10
            return i + 15
        'Adds a new value for the given key.'
        norm_name = _normalize_header(name)
        self._last_key = norm_name
        if norm_name in self:
            self._dict[norm_name] = native_str(self[norm_name]) + ',' + native_str(value)
            self._as_list[norm_name].append(value)
        else:
            self[norm_name] = value

    def get_list(self, name: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Returns all values for the given header as a list.'
        norm_name = _normalize_header(name)
        return self._as_list.get(norm_name, [])

    def get_all(self) -> Iterable[Tuple[str, str]]:
        if False:
            return 10
        'Returns an iterable of all (name, value) pairs.\n\n        If a header has multiple values, multiple pairs will be\n        returned with the same name.\n        '
        for (name, values) in self._as_list.items():
            for value in values:
                yield (name, value)

    def parse_line(self, line: str) -> None:
        if False:
            return 10
        'Updates the dictionary with a single header line.\n\n        >>> h = HTTPHeaders()\n        >>> h.parse_line("Content-Type: text/html")\n        >>> h.get(\'content-type\')\n        \'text/html\'\n        '
        if line[0].isspace():
            if self._last_key is None:
                raise HTTPInputError('first header line cannot start with whitespace')
            new_part = ' ' + line.lstrip()
            self._as_list[self._last_key][-1] += new_part
            self._dict[self._last_key] += new_part
        else:
            try:
                (name, value) = line.split(':', 1)
            except ValueError:
                raise HTTPInputError('no colon in header line')
            self.add(name, value.strip())

    @classmethod
    def parse(cls, headers: str) -> 'HTTPHeaders':
        if False:
            i = 10
            return i + 15
        'Returns a dictionary from HTTP header text.\n\n        >>> h = HTTPHeaders.parse("Content-Type: text/html\\r\\nContent-Length: 42\\r\\n")\n        >>> sorted(h.items())\n        [(\'Content-Length\', \'42\'), (\'Content-Type\', \'text/html\')]\n\n        .. versionchanged:: 5.1\n\n           Raises `HTTPInputError` on malformed headers instead of a\n           mix of `KeyError`, and `ValueError`.\n\n        '
        h = cls()
        for line in headers.split('\n'):
            if line.endswith('\r'):
                line = line[:-1]
            if line:
                h.parse_line(line)
        return h

    def __setitem__(self, name: str, value: str) -> None:
        if False:
            print('Hello World!')
        norm_name = _normalize_header(name)
        self._dict[norm_name] = value
        self._as_list[norm_name] = [value]

    def __getitem__(self, name: str) -> str:
        if False:
            print('Hello World!')
        return self._dict[_normalize_header(name)]

    def __delitem__(self, name: str) -> None:
        if False:
            print('Hello World!')
        norm_name = _normalize_header(name)
        del self._dict[norm_name]
        del self._as_list[norm_name]

    def __len__(self) -> int:
        if False:
            return 10
        return len(self._dict)

    def __iter__(self) -> Iterator[typing.Any]:
        if False:
            i = 10
            return i + 15
        return iter(self._dict)

    def copy(self) -> 'HTTPHeaders':
        if False:
            print('Hello World!')
        return HTTPHeaders(self)
    __copy__ = copy

    def __str__(self) -> str:
        if False:
            return 10
        lines = []
        for (name, value) in self.get_all():
            lines.append('%s: %s\n' % (name, value))
        return ''.join(lines)
    __unicode__ = __str__

class HTTPServerRequest(object):
    """A single HTTP request.

    All attributes are type `str` unless otherwise noted.

    .. attribute:: method

       HTTP request method, e.g. "GET" or "POST"

    .. attribute:: uri

       The requested uri.

    .. attribute:: path

       The path portion of `uri`

    .. attribute:: query

       The query portion of `uri`

    .. attribute:: version

       HTTP version specified in request, e.g. "HTTP/1.1"

    .. attribute:: headers

       `.HTTPHeaders` dictionary-like object for request headers.  Acts like
       a case-insensitive dictionary with additional methods for repeated
       headers.

    .. attribute:: body

       Request body, if present, as a byte string.

    .. attribute:: remote_ip

       Client's IP address as a string.  If ``HTTPServer.xheaders`` is set,
       will pass along the real IP address provided by a load balancer
       in the ``X-Real-Ip`` or ``X-Forwarded-For`` header.

    .. versionchanged:: 3.1
       The list format of ``X-Forwarded-For`` is now supported.

    .. attribute:: protocol

       The protocol used, either "http" or "https".  If ``HTTPServer.xheaders``
       is set, will pass along the protocol used by a load balancer if
       reported via an ``X-Scheme`` header.

    .. attribute:: host

       The requested hostname, usually taken from the ``Host`` header.

    .. attribute:: arguments

       GET/POST arguments are available in the arguments property, which
       maps arguments names to lists of values (to support multiple values
       for individual names). Names are of type `str`, while arguments
       are byte strings.  Note that this is different from
       `.RequestHandler.get_argument`, which returns argument values as
       unicode strings.

    .. attribute:: query_arguments

       Same format as ``arguments``, but contains only arguments extracted
       from the query string.

       .. versionadded:: 3.2

    .. attribute:: body_arguments

       Same format as ``arguments``, but contains only arguments extracted
       from the request body.

       .. versionadded:: 3.2

    .. attribute:: files

       File uploads are available in the files property, which maps file
       names to lists of `.HTTPFile`.

    .. attribute:: connection

       An HTTP request is attached to a single HTTP connection, which can
       be accessed through the "connection" attribute. Since connections
       are typically kept open in HTTP/1.1, multiple requests can be handled
       sequentially on a single connection.

    .. versionchanged:: 4.0
       Moved from ``tornado.httpserver.HTTPRequest``.
    """
    path = None
    query = None
    _body_future = None

    def __init__(self, method: Optional[str]=None, uri: Optional[str]=None, version: str='HTTP/1.0', headers: Optional[HTTPHeaders]=None, body: Optional[bytes]=None, host: Optional[str]=None, files: Optional[Dict[str, List['HTTPFile']]]=None, connection: Optional['HTTPConnection']=None, start_line: Optional['RequestStartLine']=None, server_connection: Optional[object]=None) -> None:
        if False:
            print('Hello World!')
        if start_line is not None:
            (method, uri, version) = start_line
        self.method = method
        self.uri = uri
        self.version = version
        self.headers = headers or HTTPHeaders()
        self.body = body or b''
        context = getattr(connection, 'context', None)
        self.remote_ip = getattr(context, 'remote_ip', None)
        self.protocol = getattr(context, 'protocol', 'http')
        self.host = host or self.headers.get('Host') or '127.0.0.1'
        self.host_name = split_host_and_port(self.host.lower())[0]
        self.files = files or {}
        self.connection = connection
        self.server_connection = server_connection
        self._start_time = time.time()
        self._finish_time = None
        if uri is not None:
            (self.path, sep, self.query) = uri.partition('?')
        self.arguments = parse_qs_bytes(self.query, keep_blank_values=True)
        self.query_arguments = copy.deepcopy(self.arguments)
        self.body_arguments = {}

    @property
    def cookies(self) -> Dict[str, http.cookies.Morsel]:
        if False:
            for i in range(10):
                print('nop')
        'A dictionary of ``http.cookies.Morsel`` objects.'
        if not hasattr(self, '_cookies'):
            self._cookies = http.cookies.SimpleCookie()
            if 'Cookie' in self.headers:
                try:
                    parsed = parse_cookie(self.headers['Cookie'])
                except Exception:
                    pass
                else:
                    for (k, v) in parsed.items():
                        try:
                            self._cookies[k] = v
                        except Exception:
                            pass
        return self._cookies

    def full_url(self) -> str:
        if False:
            print('Hello World!')
        'Reconstructs the full URL for this request.'
        return self.protocol + '://' + self.host + self.uri

    def request_time(self) -> float:
        if False:
            return 10
        'Returns the amount of time it took for this request to execute.'
        if self._finish_time is None:
            return time.time() - self._start_time
        else:
            return self._finish_time - self._start_time

    def get_ssl_certificate(self, binary_form: bool=False) -> Union[None, Dict, bytes]:
        if False:
            while True:
                i = 10
        'Returns the client\'s SSL certificate, if any.\n\n        To use client certificates, the HTTPServer\'s\n        `ssl.SSLContext.verify_mode` field must be set, e.g.::\n\n            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)\n            ssl_ctx.load_cert_chain("foo.crt", "foo.key")\n            ssl_ctx.load_verify_locations("cacerts.pem")\n            ssl_ctx.verify_mode = ssl.CERT_REQUIRED\n            server = HTTPServer(app, ssl_options=ssl_ctx)\n\n        By default, the return value is a dictionary (or None, if no\n        client certificate is present).  If ``binary_form`` is true, a\n        DER-encoded form of the certificate is returned instead.  See\n        SSLSocket.getpeercert() in the standard library for more\n        details.\n        http://docs.python.org/library/ssl.html#sslsocket-objects\n        '
        try:
            if self.connection is None:
                return None
            return self.connection.stream.socket.getpeercert(binary_form=binary_form)
        except SSLError:
            return None

    def _parse_body(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        parse_body_arguments(self.headers.get('Content-Type', ''), self.body, self.body_arguments, self.files, self.headers)
        for (k, v) in self.body_arguments.items():
            self.arguments.setdefault(k, []).extend(v)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        attrs = ('protocol', 'host', 'method', 'uri', 'version', 'remote_ip')
        args = ', '.join(['%s=%r' % (n, getattr(self, n)) for n in attrs])
        return '%s(%s)' % (self.__class__.__name__, args)

class HTTPInputError(Exception):
    """Exception class for malformed HTTP requests or responses
    from remote sources.

    .. versionadded:: 4.0
    """
    pass

class HTTPOutputError(Exception):
    """Exception class for errors in HTTP output.

    .. versionadded:: 4.0
    """
    pass

class HTTPServerConnectionDelegate(object):
    """Implement this interface to handle requests from `.HTTPServer`.

    .. versionadded:: 4.0
    """

    def start_request(self, server_conn: object, request_conn: 'HTTPConnection') -> 'HTTPMessageDelegate':
        if False:
            i = 10
            return i + 15
        'This method is called by the server when a new request has started.\n\n        :arg server_conn: is an opaque object representing the long-lived\n            (e.g. tcp-level) connection.\n        :arg request_conn: is a `.HTTPConnection` object for a single\n            request/response exchange.\n\n        This method should return a `.HTTPMessageDelegate`.\n        '
        raise NotImplementedError()

    def on_close(self, server_conn: object) -> None:
        if False:
            return 10
        'This method is called when a connection has been closed.\n\n        :arg server_conn: is a server connection that has previously been\n            passed to ``start_request``.\n        '
        pass

class HTTPMessageDelegate(object):
    """Implement this interface to handle an HTTP request or response.

    .. versionadded:: 4.0
    """

    def headers_received(self, start_line: Union['RequestStartLine', 'ResponseStartLine'], headers: HTTPHeaders) -> Optional[Awaitable[None]]:
        if False:
            return 10
        'Called when the HTTP headers have been received and parsed.\n\n        :arg start_line: a `.RequestStartLine` or `.ResponseStartLine`\n            depending on whether this is a client or server message.\n        :arg headers: a `.HTTPHeaders` instance.\n\n        Some `.HTTPConnection` methods can only be called during\n        ``headers_received``.\n\n        May return a `.Future`; if it does the body will not be read\n        until it is done.\n        '
        pass

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        if False:
            return 10
        'Called when a chunk of data has been received.\n\n        May return a `.Future` for flow control.\n        '
        pass

    def finish(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Called after the last chunk of data has been received.'
        pass

    def on_connection_close(self) -> None:
        if False:
            i = 10
            return i + 15
        'Called if the connection is closed without finishing the request.\n\n        If ``headers_received`` is called, either ``finish`` or\n        ``on_connection_close`` will be called, but not both.\n        '
        pass

class HTTPConnection(object):
    """Applications use this interface to write their responses.

    .. versionadded:: 4.0
    """

    def write_headers(self, start_line: Union['RequestStartLine', 'ResponseStartLine'], headers: HTTPHeaders, chunk: Optional[bytes]=None) -> 'Future[None]':
        if False:
            i = 10
            return i + 15
        'Write an HTTP header block.\n\n        :arg start_line: a `.RequestStartLine` or `.ResponseStartLine`.\n        :arg headers: a `.HTTPHeaders` instance.\n        :arg chunk: the first (optional) chunk of data.  This is an optimization\n            so that small responses can be written in the same call as their\n            headers.\n\n        The ``version`` field of ``start_line`` is ignored.\n\n        Returns a future for flow control.\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` argument was removed.\n        '
        raise NotImplementedError()

    def write(self, chunk: bytes) -> 'Future[None]':
        if False:
            for i in range(10):
                print('nop')
        'Writes a chunk of body data.\n\n        Returns a future for flow control.\n\n        .. versionchanged:: 6.0\n\n           The ``callback`` argument was removed.\n        '
        raise NotImplementedError()

    def finish(self) -> None:
        if False:
            return 10
        'Indicates that the last body data has been written.'
        raise NotImplementedError()

def url_concat(url: str, args: Union[None, Dict[str, str], List[Tuple[str, str]], Tuple[Tuple[str, str], ...]]) -> str:
    if False:
        while True:
            i = 10
    'Concatenate url and arguments regardless of whether\n    url has existing query parameters.\n\n    ``args`` may be either a dictionary or a list of key-value pairs\n    (the latter allows for multiple values with the same key.\n\n    >>> url_concat("http://example.com/foo", dict(c="d"))\n    \'http://example.com/foo?c=d\'\n    >>> url_concat("http://example.com/foo?a=b", dict(c="d"))\n    \'http://example.com/foo?a=b&c=d\'\n    >>> url_concat("http://example.com/foo?a=b", [("c", "d"), ("c", "d2")])\n    \'http://example.com/foo?a=b&c=d&c=d2\'\n    '
    if args is None:
        return url
    parsed_url = urlparse(url)
    if isinstance(args, dict):
        parsed_query = parse_qsl(parsed_url.query, keep_blank_values=True)
        parsed_query.extend(args.items())
    elif isinstance(args, list) or isinstance(args, tuple):
        parsed_query = parse_qsl(parsed_url.query, keep_blank_values=True)
        parsed_query.extend(args)
    else:
        err = "'args' parameter should be dict, list or tuple. Not {0}".format(type(args))
        raise TypeError(err)
    final_query = urlencode(parsed_query)
    url = urlunparse((parsed_url[0], parsed_url[1], parsed_url[2], parsed_url[3], final_query, parsed_url[5]))
    return url

class HTTPFile(ObjectDict):
    """Represents a file uploaded via a form.

    For backwards compatibility, its instance attributes are also
    accessible as dictionary keys.

    * ``filename``
    * ``body``
    * ``content_type``
    """
    filename: str
    body: bytes
    content_type: str

def _parse_request_range(range_header: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if False:
        while True:
            i = 10
    'Parses a Range header.\n\n    Returns either ``None`` or tuple ``(start, end)``.\n    Note that while the HTTP headers use inclusive byte positions,\n    this method returns indexes suitable for use in slices.\n\n    >>> start, end = _parse_request_range("bytes=1-2")\n    >>> start, end\n    (1, 3)\n    >>> [0, 1, 2, 3, 4][start:end]\n    [1, 2]\n    >>> _parse_request_range("bytes=6-")\n    (6, None)\n    >>> _parse_request_range("bytes=-6")\n    (-6, None)\n    >>> _parse_request_range("bytes=-0")\n    (None, 0)\n    >>> _parse_request_range("bytes=")\n    (None, None)\n    >>> _parse_request_range("foo=42")\n    >>> _parse_request_range("bytes=1-2,6-10")\n\n    Note: only supports one range (ex, ``bytes=1-2,6-10`` is not allowed).\n\n    See [0] for the details of the range header.\n\n    [0]: http://greenbytes.de/tech/webdav/draft-ietf-httpbis-p5-range-latest.html#byte.ranges\n    '
    (unit, _, value) = range_header.partition('=')
    (unit, value) = (unit.strip(), value.strip())
    if unit != 'bytes':
        return None
    (start_b, _, end_b) = value.partition('-')
    try:
        start = _int_or_none(start_b)
        end = _int_or_none(end_b)
    except ValueError:
        return None
    if end is not None:
        if start is None:
            if end != 0:
                start = -end
                end = None
        else:
            end += 1
    return (start, end)

def _get_content_range(start: Optional[int], end: Optional[int], total: int) -> str:
    if False:
        i = 10
        return i + 15
    'Returns a suitable Content-Range header:\n\n    >>> print(_get_content_range(None, 1, 4))\n    bytes 0-0/4\n    >>> print(_get_content_range(1, 3, 4))\n    bytes 1-2/4\n    >>> print(_get_content_range(None, None, 4))\n    bytes 0-3/4\n    '
    start = start or 0
    end = (end or total) - 1
    return 'bytes %s-%s/%s' % (start, end, total)

def _int_or_none(val: str) -> Optional[int]:
    if False:
        return 10
    val = val.strip()
    if val == '':
        return None
    return int(val)

def parse_body_arguments(content_type: str, body: bytes, arguments: Dict[str, List[bytes]], files: Dict[str, List[HTTPFile]], headers: Optional[HTTPHeaders]=None) -> None:
    if False:
        return 10
    'Parses a form request body.\n\n    Supports ``application/x-www-form-urlencoded`` and\n    ``multipart/form-data``.  The ``content_type`` parameter should be\n    a string and ``body`` should be a byte string.  The ``arguments``\n    and ``files`` parameters are dictionaries that will be updated\n    with the parsed contents.\n    '
    if content_type.startswith('application/x-www-form-urlencoded'):
        if headers and 'Content-Encoding' in headers:
            gen_log.warning('Unsupported Content-Encoding: %s', headers['Content-Encoding'])
            return
        try:
            uri_arguments = parse_qs_bytes(body, keep_blank_values=True)
        except Exception as e:
            gen_log.warning('Invalid x-www-form-urlencoded body: %s', e)
            uri_arguments = {}
        for (name, values) in uri_arguments.items():
            if values:
                arguments.setdefault(name, []).extend(values)
    elif content_type.startswith('multipart/form-data'):
        if headers and 'Content-Encoding' in headers:
            gen_log.warning('Unsupported Content-Encoding: %s', headers['Content-Encoding'])
            return
        try:
            fields = content_type.split(';')
            for field in fields:
                (k, sep, v) = field.strip().partition('=')
                if k == 'boundary' and v:
                    parse_multipart_form_data(utf8(v), body, arguments, files)
                    break
            else:
                raise ValueError('multipart boundary not found')
        except Exception as e:
            gen_log.warning('Invalid multipart/form-data: %s', e)

def parse_multipart_form_data(boundary: bytes, data: bytes, arguments: Dict[str, List[bytes]], files: Dict[str, List[HTTPFile]]) -> None:
    if False:
        while True:
            i = 10
    'Parses a ``multipart/form-data`` body.\n\n    The ``boundary`` and ``data`` parameters are both byte strings.\n    The dictionaries given in the arguments and files parameters\n    will be updated with the contents of the body.\n\n    .. versionchanged:: 5.1\n\n       Now recognizes non-ASCII filenames in RFC 2231/5987\n       (``filename*=``) format.\n    '
    if boundary.startswith(b'"') and boundary.endswith(b'"'):
        boundary = boundary[1:-1]
    final_boundary_index = data.rfind(b'--' + boundary + b'--')
    if final_boundary_index == -1:
        gen_log.warning('Invalid multipart/form-data: no final boundary')
        return
    parts = data[:final_boundary_index].split(b'--' + boundary + b'\r\n')
    for part in parts:
        if not part:
            continue
        eoh = part.find(b'\r\n\r\n')
        if eoh == -1:
            gen_log.warning('multipart/form-data missing headers')
            continue
        headers = HTTPHeaders.parse(part[:eoh].decode('utf-8'))
        disp_header = headers.get('Content-Disposition', '')
        (disposition, disp_params) = _parse_header(disp_header)
        if disposition != 'form-data' or not part.endswith(b'\r\n'):
            gen_log.warning('Invalid multipart/form-data')
            continue
        value = part[eoh + 4:-2]
        if not disp_params.get('name'):
            gen_log.warning('multipart/form-data value missing name')
            continue
        name = disp_params['name']
        if disp_params.get('filename'):
            ctype = headers.get('Content-Type', 'application/unknown')
            files.setdefault(name, []).append(HTTPFile(filename=disp_params['filename'], body=value, content_type=ctype))
        else:
            arguments.setdefault(name, []).append(value)

def format_timestamp(ts: Union[int, float, tuple, time.struct_time, datetime.datetime]) -> str:
    if False:
        while True:
            i = 10
    "Formats a timestamp in the format used by HTTP.\n\n    The argument may be a numeric timestamp as returned by `time.time`,\n    a time tuple as returned by `time.gmtime`, or a `datetime.datetime`\n    object. Naive `datetime.datetime` objects are assumed to represent\n    UTC; aware objects are converted to UTC before formatting.\n\n    >>> format_timestamp(1359312200)\n    'Sun, 27 Jan 2013 18:43:20 GMT'\n    "
    if isinstance(ts, (int, float)):
        time_num = ts
    elif isinstance(ts, (tuple, time.struct_time)):
        time_num = calendar.timegm(ts)
    elif isinstance(ts, datetime.datetime):
        time_num = calendar.timegm(ts.utctimetuple())
    else:
        raise TypeError('unknown timestamp type: %r' % ts)
    return email.utils.formatdate(time_num, usegmt=True)
RequestStartLine = collections.namedtuple('RequestStartLine', ['method', 'path', 'version'])
_http_version_re = re.compile('^HTTP/1\\.[0-9]$')

def parse_request_start_line(line: str) -> RequestStartLine:
    if False:
        return 10
    'Returns a (method, path, version) tuple for an HTTP 1.x request line.\n\n    The response is a `collections.namedtuple`.\n\n    >>> parse_request_start_line("GET /foo HTTP/1.1")\n    RequestStartLine(method=\'GET\', path=\'/foo\', version=\'HTTP/1.1\')\n    '
    try:
        (method, path, version) = line.split(' ')
    except ValueError:
        raise HTTPInputError('Malformed HTTP request line')
    if not _http_version_re.match(version):
        raise HTTPInputError('Malformed HTTP version in HTTP Request-Line: %r' % version)
    return RequestStartLine(method, path, version)
ResponseStartLine = collections.namedtuple('ResponseStartLine', ['version', 'code', 'reason'])
_http_response_line_re = re.compile('(HTTP/1.[0-9]) ([0-9]+) ([^\\r]*)')

def parse_response_start_line(line: str) -> ResponseStartLine:
    if False:
        return 10
    'Returns a (version, code, reason) tuple for an HTTP 1.x response line.\n\n    The response is a `collections.namedtuple`.\n\n    >>> parse_response_start_line("HTTP/1.1 200 OK")\n    ResponseStartLine(version=\'HTTP/1.1\', code=200, reason=\'OK\')\n    '
    line = native_str(line)
    match = _http_response_line_re.match(line)
    if not match:
        raise HTTPInputError('Error parsing response start line')
    return ResponseStartLine(match.group(1), int(match.group(2)), match.group(3))

def _parseparam(s: str) -> Generator[str, None, None]:
    if False:
        i = 10
        return i + 15
    while s[:1] == ';':
        s = s[1:]
        end = s.find(';')
        while end > 0 and (s.count('"', 0, end) - s.count('\\"', 0, end)) % 2:
            end = s.find(';', end + 1)
        if end < 0:
            end = len(s)
        f = s[:end]
        yield f.strip()
        s = s[end:]

def _parse_header(line: str) -> Tuple[str, Dict[str, str]]:
    if False:
        i = 10
        return i + 15
    'Parse a Content-type like header.\n\n    Return the main content-type and a dictionary of options.\n\n    >>> d = "form-data; foo=\\"b\\\\\\\\a\\\\\\"r\\"; file*=utf-8\'\'T%C3%A4st"\n    >>> ct, d = _parse_header(d)\n    >>> ct\n    \'form-data\'\n    >>> d[\'file\'] == r\'T\\u00e4st\'.encode(\'ascii\').decode(\'unicode_escape\')\n    True\n    >>> d[\'foo\']\n    \'b\\\\a"r\'\n    '
    parts = _parseparam(';' + line)
    key = next(parts)
    params = [('Dummy', 'value')]
    for p in parts:
        i = p.find('=')
        if i >= 0:
            name = p[:i].strip().lower()
            value = p[i + 1:].strip()
            params.append((name, native_str(value)))
    decoded_params = email.utils.decode_params(params)
    decoded_params.pop(0)
    pdict = {}
    for (name, decoded_value) in decoded_params:
        value = email.utils.collapse_rfc2231_value(decoded_value)
        if len(value) >= 2 and value[0] == '"' and (value[-1] == '"'):
            value = value[1:-1]
        pdict[name] = value
    return (key, pdict)

def _encode_header(key: str, pdict: Dict[str, str]) -> str:
    if False:
        return 10
    "Inverse of _parse_header.\n\n    >>> _encode_header('permessage-deflate',\n    ...     {'client_max_window_bits': 15, 'client_no_context_takeover': None})\n    'permessage-deflate; client_max_window_bits=15; client_no_context_takeover'\n    "
    if not pdict:
        return key
    out = [key]
    for (k, v) in sorted(pdict.items()):
        if v is None:
            out.append(k)
        else:
            out.append('%s=%s' % (k, v))
    return '; '.join(out)

def encode_username_password(username: Union[str, bytes], password: Union[str, bytes]) -> bytes:
    if False:
        while True:
            i = 10
    'Encodes a username/password pair in the format used by HTTP auth.\n\n    The return value is a byte string in the form ``username:password``.\n\n    .. versionadded:: 5.1\n    '
    if isinstance(username, unicode_type):
        username = unicodedata.normalize('NFC', username)
    if isinstance(password, unicode_type):
        password = unicodedata.normalize('NFC', password)
    return utf8(username) + b':' + utf8(password)

def doctests():
    if False:
        return 10
    import doctest
    return doctest.DocTestSuite()
_netloc_re = re.compile('^(.+):(\\d+)$')

def split_host_and_port(netloc: str) -> Tuple[str, Optional[int]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns ``(host, port)`` tuple from ``netloc``.\n\n    Returned ``port`` will be ``None`` if not present.\n\n    .. versionadded:: 4.1\n    '
    match = _netloc_re.match(netloc)
    if match:
        host = match.group(1)
        port = int(match.group(2))
    else:
        host = netloc
        port = None
    return (host, port)

def qs_to_qsl(qs: Dict[str, List[AnyStr]]) -> Iterable[Tuple[str, AnyStr]]:
    if False:
        for i in range(10):
            print('nop')
    'Generator converting a result of ``parse_qs`` back to name-value pairs.\n\n    .. versionadded:: 5.0\n    '
    for (k, vs) in qs.items():
        for v in vs:
            yield (k, v)
_OctalPatt = re.compile('\\\\[0-3][0-7][0-7]')
_QuotePatt = re.compile('[\\\\].')
_nulljoin = ''.join

def _unquote_cookie(s: str) -> str:
    if False:
        print('Hello World!')
    "Handle double quotes and escaping in cookie values.\n\n    This method is copied verbatim from the Python 3.5 standard\n    library (http.cookies._unquote) so we don't have to depend on\n    non-public interfaces.\n    "
    if s is None or len(s) < 2:
        return s
    if s[0] != '"' or s[-1] != '"':
        return s
    s = s[1:-1]
    i = 0
    n = len(s)
    res = []
    while 0 <= i < n:
        o_match = _OctalPatt.search(s, i)
        q_match = _QuotePatt.search(s, i)
        if not o_match and (not q_match):
            res.append(s[i:])
            break
        j = k = -1
        if o_match:
            j = o_match.start(0)
        if q_match:
            k = q_match.start(0)
        if q_match and (not o_match or k < j):
            res.append(s[i:k])
            res.append(s[k + 1])
            i = k + 2
        else:
            res.append(s[i:j])
            res.append(chr(int(s[j + 1:j + 4], 8)))
            i = j + 4
    return _nulljoin(res)

def parse_cookie(cookie: str) -> Dict[str, str]:
    if False:
        return 10
    "Parse a ``Cookie`` HTTP header into a dict of name/value pairs.\n\n    This function attempts to mimic browser cookie parsing behavior;\n    it specifically does not follow any of the cookie-related RFCs\n    (because browsers don't either).\n\n    The algorithm used is identical to that used by Django version 1.9.10.\n\n    .. versionadded:: 4.4.2\n    "
    cookiedict = {}
    for chunk in cookie.split(str(';')):
        if str('=') in chunk:
            (key, val) = chunk.split(str('='), 1)
        else:
            (key, val) = (str(''), chunk)
        (key, val) = (key.strip(), val.strip())
        if key or val:
            cookiedict[key] = _unquote_cookie(val)
    return cookiedict