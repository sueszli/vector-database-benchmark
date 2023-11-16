import asyncio
import dataclasses
import datetime
import io
import re
import socket
import string
import tempfile
import types
from http.cookies import SimpleCookie
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Final, Iterator, Mapping, MutableMapping, Optional, Pattern, Set, Tuple, Union, cast
from urllib.parse import parse_qsl
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import _SENTINEL, ETAG_ANY, LIST_QUOTED_ETAG_RE, ChainMapProxy, ETag, HeadersMixin, is_expected_content_type, parse_http_date, reify, sentinel, set_result
from .http_parser import RawRequestMessage
from .http_writer import HttpVersion
from .multipart import BodyPartReader, MultipartReader
from .streams import EmptyStreamReader, StreamReader
from .typedefs import DEFAULT_JSON_DECODER, JSONDecoder, LooseHeaders, RawHeaders, StrOrURL
from .web_exceptions import HTTPBadRequest, HTTPRequestEntityTooLarge, HTTPUnsupportedMediaType
from .web_response import StreamResponse
__all__ = ('BaseRequest', 'FileField', 'Request')
if TYPE_CHECKING:
    from .web_app import Application
    from .web_protocol import RequestHandler
    from .web_urldispatcher import UrlMappingMatchInfo

@dataclasses.dataclass(frozen=True)
class FileField:
    name: str
    filename: str
    file: io.BufferedReader
    content_type: str
    headers: 'CIMultiDictProxy[str]'
_TCHAR: Final[str] = string.digits + string.ascii_letters + "!#$%&'*+.^_`|~-"
_TOKEN: Final[str] = f'[{_TCHAR}]+'
_QDTEXT: Final[str] = '[{}]'.format(''.join((chr(c) for c in (9, 32, 33) + tuple(range(35, 127)))))
_QUOTED_PAIR: Final[str] = '\\\\[\\t !-~]'
_QUOTED_STRING: Final[str] = '"(?:{quoted_pair}|{qdtext})*"'.format(qdtext=_QDTEXT, quoted_pair=_QUOTED_PAIR)
_FORWARDED_PAIR: Final[str] = '({token})=({token}|{quoted_string})(:\\d{{1,4}})?'.format(token=_TOKEN, quoted_string=_QUOTED_STRING)
_QUOTED_PAIR_REPLACE_RE: Final[Pattern[str]] = re.compile('\\\\([\\t !-~])')
_FORWARDED_PAIR_RE: Final[Pattern[str]] = re.compile(_FORWARDED_PAIR)

class BaseRequest(MutableMapping[str, Any], HeadersMixin):
    POST_METHODS = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT, hdrs.METH_TRACE, hdrs.METH_DELETE}
    __slots__ = ('_message', '_protocol', '_payload_writer', '_payload', '_headers', '_method', '_version', '_rel_url', '_post', '_read_bytes', '_state', '_cache', '_task', '_client_max_size', '_loop', '_transport_sslcontext', '_transport_peername', '_disconnection_waiters', '__weakref__')

    def __init__(self, message: RawRequestMessage, payload: StreamReader, protocol: 'RequestHandler', payload_writer: AbstractStreamWriter, task: 'asyncio.Task[None]', loop: asyncio.AbstractEventLoop, *, client_max_size: int=1024 ** 2, state: Optional[Dict[str, Any]]=None, scheme: Optional[str]=None, host: Optional[str]=None, remote: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        if state is None:
            state = {}
        self._message = message
        self._protocol = protocol
        self._payload_writer = payload_writer
        self._payload = payload
        self._headers = message.headers
        self._method = message.method
        self._version = message.version
        self._cache: Dict[str, Any] = {}
        url = message.url
        if url.is_absolute():
            self._cache['url'] = url
            self._cache['host'] = url.host
            self._cache['scheme'] = url.scheme
            self._rel_url = url.relative()
        else:
            self._rel_url = message.url
        self._post: Optional[MultiDictProxy[Union[str, bytes, FileField]]] = None
        self._read_bytes: Optional[bytes] = None
        self._state = state
        self._task = task
        self._client_max_size = client_max_size
        self._loop = loop
        self._disconnection_waiters: Set[asyncio.Future[None]] = set()
        transport = self._protocol.transport
        assert transport is not None
        self._transport_sslcontext = transport.get_extra_info('sslcontext')
        self._transport_peername = transport.get_extra_info('peername')
        if scheme is not None:
            self._cache['scheme'] = scheme
        if host is not None:
            self._cache['host'] = host
        if remote is not None:
            self._cache['remote'] = remote

    def clone(self, *, method: Union[str, _SENTINEL]=sentinel, rel_url: Union[StrOrURL, _SENTINEL]=sentinel, headers: Union[LooseHeaders, _SENTINEL]=sentinel, scheme: Union[str, _SENTINEL]=sentinel, host: Union[str, _SENTINEL]=sentinel, remote: Union[str, _SENTINEL]=sentinel, client_max_size: Union[int, _SENTINEL]=sentinel) -> 'BaseRequest':
        if False:
            i = 10
            return i + 15
        'Clone itself with replacement some attributes.\n\n        Creates and returns a new instance of Request object. If no parameters\n        are given, an exact copy is returned. If a parameter is not passed, it\n        will reuse the one from the current request object.\n        '
        if self._read_bytes:
            raise RuntimeError('Cannot clone request after reading its content')
        dct: Dict[str, Any] = {}
        if method is not sentinel:
            dct['method'] = method
        if rel_url is not sentinel:
            new_url: URL = URL(rel_url)
            dct['url'] = new_url
            dct['path'] = str(new_url)
        if headers is not sentinel:
            new_headers = CIMultiDictProxy(CIMultiDict(headers))
            dct['headers'] = new_headers
            dct['raw_headers'] = tuple(((k.encode('utf-8'), v.encode('utf-8')) for (k, v) in new_headers.items()))
        message = self._message._replace(**dct)
        kwargs: Dict[str, str] = {}
        if scheme is not sentinel:
            kwargs['scheme'] = scheme
        if host is not sentinel:
            kwargs['host'] = host
        if remote is not sentinel:
            kwargs['remote'] = remote
        if client_max_size is sentinel:
            client_max_size = self._client_max_size
        return self.__class__(message, self._payload, self._protocol, self._payload_writer, self._task, self._loop, client_max_size=client_max_size, state=self._state.copy(), **kwargs)

    @property
    def task(self) -> 'asyncio.Task[None]':
        if False:
            print('Hello World!')
        return self._task

    @property
    def protocol(self) -> 'RequestHandler':
        if False:
            while True:
                i = 10
        return self._protocol

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        if False:
            i = 10
            return i + 15
        if self._protocol is None:
            return None
        return self._protocol.transport

    @property
    def writer(self) -> AbstractStreamWriter:
        if False:
            print('Hello World!')
        return self._payload_writer

    @property
    def client_max_size(self) -> int:
        if False:
            print('Hello World!')
        return self._client_max_size

    @reify
    def rel_url(self) -> URL:
        if False:
            for i in range(10):
                print('nop')
        return self._rel_url

    def __getitem__(self, key: str) -> Any:
        if False:
            while True:
                i = 10
        return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._state[key] = value

    def __delitem__(self, key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        del self._state[key]

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._state)

    def __iter__(self) -> Iterator[str]:
        if False:
            for i in range(10):
                print('nop')
        return iter(self._state)

    @reify
    def secure(self) -> bool:
        if False:
            while True:
                i = 10
        'A bool indicating if the request is handled with SSL.'
        return self.scheme == 'https'

    @reify
    def forwarded(self) -> Tuple[Mapping[str, str], ...]:
        if False:
            for i in range(10):
                print('nop')
        "A tuple containing all parsed Forwarded header(s).\n\n        Makes an effort to parse Forwarded headers as specified by RFC 7239:\n\n        - It adds one (immutable) dictionary per Forwarded 'field-value', ie\n          per proxy. The element corresponds to the data in the Forwarded\n          field-value added by the first proxy encountered by the client. Each\n          subsequent item corresponds to those added by later proxies.\n        - It checks that every value has valid syntax in general as specified\n          in section 4: either a 'token' or a 'quoted-string'.\n        - It un-escapes found escape sequences.\n        - It does NOT validate 'by' and 'for' contents as specified in section\n          6.\n        - It does NOT validate 'host' contents (Host ABNF).\n        - It does NOT validate 'proto' contents for valid URI scheme names.\n\n        Returns a tuple containing one or more immutable dicts\n        "
        elems = []
        for field_value in self._message.headers.getall(hdrs.FORWARDED, ()):
            length = len(field_value)
            pos = 0
            need_separator = False
            elem: Dict[str, str] = {}
            elems.append(types.MappingProxyType(elem))
            while 0 <= pos < length:
                match = _FORWARDED_PAIR_RE.match(field_value, pos)
                if match is not None:
                    if need_separator:
                        pos = field_value.find(',', pos)
                    else:
                        (name, value, port) = match.groups()
                        if value[0] == '"':
                            value = _QUOTED_PAIR_REPLACE_RE.sub('\\1', value[1:-1])
                        if port:
                            value += port
                        elem[name.lower()] = value
                        pos += len(match.group(0))
                        need_separator = True
                elif field_value[pos] == ',':
                    need_separator = False
                    elem = {}
                    elems.append(types.MappingProxyType(elem))
                    pos += 1
                elif field_value[pos] == ';':
                    need_separator = False
                    pos += 1
                elif field_value[pos] in ' \t':
                    pos += 1
                else:
                    pos = field_value.find(',', pos)
        return tuple(elems)

    @reify
    def scheme(self) -> str:
        if False:
            while True:
                i = 10
        "A string representing the scheme of the request.\n\n        Hostname is resolved in this order:\n\n        - overridden value by .clone(scheme=new_scheme) call.\n        - type of connection to peer: HTTPS if socket is SSL, HTTP otherwise.\n\n        'http' or 'https'.\n        "
        if self._transport_sslcontext:
            return 'https'
        else:
            return 'http'

    @reify
    def method(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Read only property for getting HTTP method.\n\n        The value is upper-cased str like 'GET', 'POST', 'PUT' etc.\n        "
        return self._method

    @reify
    def version(self) -> HttpVersion:
        if False:
            for i in range(10):
                print('nop')
        'Read only property for getting HTTP version of request.\n\n        Returns aiohttp.protocol.HttpVersion instance.\n        '
        return self._version

    @reify
    def host(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Hostname of the request.\n\n        Hostname is resolved in this order:\n\n        - overridden value by .clone(host=new_host) call.\n        - HOST HTTP header\n        - socket.getfqdn() value\n        '
        host = self._message.headers.get(hdrs.HOST)
        if host is not None:
            return host
        return socket.getfqdn()

    @reify
    def remote(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'Remote IP of client initiated HTTP request.\n\n        The IP is resolved in this order:\n\n        - overridden value by .clone(remote=new_remote) call.\n        - peername of opened socket\n        '
        if self._transport_peername is None:
            return None
        if isinstance(self._transport_peername, (list, tuple)):
            return str(self._transport_peername[0])
        return str(self._transport_peername)

    @reify
    def url(self) -> URL:
        if False:
            return 10
        url = URL.build(scheme=self.scheme, host=self.host)
        return url.join(self._rel_url)

    @reify
    def path(self) -> str:
        if False:
            while True:
                i = 10
        'The URL including *PATH INFO* without the host or scheme.\n\n        E.g., ``/app/blog``\n        '
        return self._rel_url.path

    @reify
    def path_qs(self) -> str:
        if False:
            while True:
                i = 10
        'The URL including PATH_INFO and the query string.\n\n        E.g, /app/blog?id=10\n        '
        return str(self._rel_url)

    @reify
    def raw_path(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The URL including raw *PATH INFO* without the host or scheme.\n\n        Warning, the path is unquoted and may contains non valid URL characters\n\n        E.g., ``/my%2Fpath%7Cwith%21some%25strange%24characters``\n        '
        return self._message.path

    @reify
    def query(self) -> MultiDictProxy[str]:
        if False:
            i = 10
            return i + 15
        'A multidict with all the variables in the query string.'
        return MultiDictProxy(self._rel_url.query)

    @reify
    def query_string(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The query string in the URL.\n\n        E.g., id=10\n        '
        return self._rel_url.query_string

    @reify
    def headers(self) -> 'CIMultiDictProxy[str]':
        if False:
            i = 10
            return i + 15
        'A case-insensitive multidict proxy with all headers.'
        return self._headers

    @reify
    def raw_headers(self) -> RawHeaders:
        if False:
            for i in range(10):
                print('nop')
        'A sequence of pairs for all headers.'
        return self._message.raw_headers

    @reify
    def if_modified_since(self) -> Optional[datetime.datetime]:
        if False:
            return 10
        'The value of If-Modified-Since HTTP header, or None.\n\n        This header is represented as a `datetime` object.\n        '
        return parse_http_date(self.headers.get(hdrs.IF_MODIFIED_SINCE))

    @reify
    def if_unmodified_since(self) -> Optional[datetime.datetime]:
        if False:
            i = 10
            return i + 15
        'The value of If-Unmodified-Since HTTP header, or None.\n\n        This header is represented as a `datetime` object.\n        '
        return parse_http_date(self.headers.get(hdrs.IF_UNMODIFIED_SINCE))

    @staticmethod
    def _etag_values(etag_header: str) -> Iterator[ETag]:
        if False:
            while True:
                i = 10
        'Extract `ETag` objects from raw header.'
        if etag_header == ETAG_ANY:
            yield ETag(is_weak=False, value=ETAG_ANY)
        else:
            for match in LIST_QUOTED_ETAG_RE.finditer(etag_header):
                (is_weak, value, garbage) = match.group(2, 3, 4)
                if garbage:
                    break
                yield ETag(is_weak=bool(is_weak), value=value)

    @classmethod
    def _if_match_or_none_impl(cls, header_value: Optional[str]) -> Optional[Tuple[ETag, ...]]:
        if False:
            while True:
                i = 10
        if not header_value:
            return None
        return tuple(cls._etag_values(header_value))

    @reify
    def if_match(self) -> Optional[Tuple[ETag, ...]]:
        if False:
            while True:
                i = 10
        'The value of If-Match HTTP header, or None.\n\n        This header is represented as a `tuple` of `ETag` objects.\n        '
        return self._if_match_or_none_impl(self.headers.get(hdrs.IF_MATCH))

    @reify
    def if_none_match(self) -> Optional[Tuple[ETag, ...]]:
        if False:
            return 10
        'The value of If-None-Match HTTP header, or None.\n\n        This header is represented as a `tuple` of `ETag` objects.\n        '
        return self._if_match_or_none_impl(self.headers.get(hdrs.IF_NONE_MATCH))

    @reify
    def if_range(self) -> Optional[datetime.datetime]:
        if False:
            i = 10
            return i + 15
        'The value of If-Range HTTP header, or None.\n\n        This header is represented as a `datetime` object.\n        '
        return parse_http_date(self.headers.get(hdrs.IF_RANGE))

    @reify
    def keep_alive(self) -> bool:
        if False:
            while True:
                i = 10
        'Is keepalive enabled by client?'
        return not self._message.should_close

    @reify
    def cookies(self) -> Mapping[str, str]:
        if False:
            while True:
                i = 10
        'Return request cookies.\n\n        A read-only dictionary-like object.\n        '
        raw = self.headers.get(hdrs.COOKIE, '')
        parsed = SimpleCookie(raw)
        return MappingProxyType({key: val.value for (key, val) in parsed.items()})

    @reify
    def http_range(self) -> slice:
        if False:
            print('Hello World!')
        'The content of Range HTTP header.\n\n        Return a slice instance.\n\n        '
        rng = self._headers.get(hdrs.RANGE)
        (start, end) = (None, None)
        if rng is not None:
            try:
                pattern = '^bytes=(\\d*)-(\\d*)$'
                (start, end) = re.findall(pattern, rng)[0]
            except IndexError:
                raise ValueError('range not in acceptable format')
            end = int(end) if end else None
            start = int(start) if start else None
            if start is None and end is not None:
                start = -end
                end = None
            if start is not None and end is not None:
                end += 1
                if start >= end:
                    raise ValueError('start cannot be after end')
            if start is end is None:
                raise ValueError('No start or end of range specified')
        return slice(start, end, 1)

    @reify
    def content(self) -> StreamReader:
        if False:
            while True:
                i = 10
        'Return raw payload stream.'
        return self._payload

    @property
    def can_read_body(self) -> bool:
        if False:
            i = 10
            return i + 15
        "Return True if request's HTTP BODY can be read, False otherwise."
        return not self._payload.at_eof()

    @reify
    def body_exists(self) -> bool:
        if False:
            return 10
        'Return True if request has HTTP BODY, False otherwise.'
        return type(self._payload) is not EmptyStreamReader

    async def release(self) -> None:
        """Release request.

        Eat unread part of HTTP BODY if present.
        """
        while not self._payload.at_eof():
            await self._payload.readany()

    async def read(self) -> bytes:
        """Read request body if present.

        Returns bytes object with full request content.
        """
        if self._read_bytes is None:
            body = bytearray()
            while True:
                chunk = await self._payload.readany()
                body.extend(chunk)
                if self._client_max_size:
                    body_size = len(body)
                    if body_size > self._client_max_size:
                        raise HTTPRequestEntityTooLarge(max_size=self._client_max_size, actual_size=body_size)
                if not chunk:
                    break
            self._read_bytes = bytes(body)
        return self._read_bytes

    async def text(self) -> str:
        """Return BODY as text using encoding from .charset."""
        bytes_body = await self.read()
        encoding = self.charset or 'utf-8'
        try:
            return bytes_body.decode(encoding)
        except LookupError:
            raise HTTPUnsupportedMediaType()

    async def json(self, *, loads: JSONDecoder=DEFAULT_JSON_DECODER, content_type: Optional[str]='application/json') -> Any:
        """Return BODY as JSON."""
        body = await self.text()
        if content_type:
            if not is_expected_content_type(self.content_type, content_type):
                raise HTTPBadRequest(text='Attempt to decode JSON with unexpected mimetype: %s' % self.content_type)
        return loads(body)

    async def multipart(self) -> MultipartReader:
        """Return async iterator to process BODY as multipart."""
        return MultipartReader(self._headers, self._payload)

    async def post(self) -> 'MultiDictProxy[Union[str, bytes, FileField]]':
        """Return POST parameters."""
        if self._post is not None:
            return self._post
        if self._method not in self.POST_METHODS:
            self._post = MultiDictProxy(MultiDict())
            return self._post
        content_type = self.content_type
        if content_type not in ('', 'application/x-www-form-urlencoded', 'multipart/form-data'):
            self._post = MultiDictProxy(MultiDict())
            return self._post
        out: MultiDict[Union[str, bytes, FileField]] = MultiDict()
        if content_type == 'multipart/form-data':
            multipart = await self.multipart()
            max_size = self._client_max_size
            field = await multipart.next()
            while field is not None:
                size = 0
                field_ct = field.headers.get(hdrs.CONTENT_TYPE)
                if isinstance(field, BodyPartReader):
                    assert field.name is not None
                    if field.filename:
                        tmp = tempfile.TemporaryFile()
                        chunk = await field.read_chunk(size=2 ** 16)
                        while chunk:
                            chunk = field.decode(chunk)
                            tmp.write(chunk)
                            size += len(chunk)
                            if 0 < max_size < size:
                                tmp.close()
                                raise HTTPRequestEntityTooLarge(max_size=max_size, actual_size=size)
                            chunk = await field.read_chunk(size=2 ** 16)
                        tmp.seek(0)
                        if field_ct is None:
                            field_ct = 'application/octet-stream'
                        ff = FileField(field.name, field.filename, cast(io.BufferedReader, tmp), field_ct, field.headers)
                        out.add(field.name, ff)
                    else:
                        value = await field.read(decode=True)
                        if field_ct is None or field_ct.startswith('text/'):
                            charset = field.get_charset(default='utf-8')
                            out.add(field.name, value.decode(charset))
                        else:
                            out.add(field.name, value)
                        size += len(value)
                        if 0 < max_size < size:
                            raise HTTPRequestEntityTooLarge(max_size=max_size, actual_size=size)
                else:
                    raise ValueError('To decode nested multipart you need to use custom reader')
                field = await multipart.next()
        else:
            data = await self.read()
            if data:
                charset = self.charset or 'utf-8'
                bytes_query = data.rstrip()
                try:
                    query = bytes_query.decode(charset)
                except LookupError:
                    raise HTTPUnsupportedMediaType()
                out.extend(parse_qsl(qs=query, keep_blank_values=True, encoding=charset))
        self._post = MultiDictProxy(out)
        return self._post

    def get_extra_info(self, name: str, default: Any=None) -> Any:
        if False:
            i = 10
            return i + 15
        'Extra info from protocol transport'
        protocol = self._protocol
        if protocol is None:
            return default
        transport = protocol.transport
        if transport is None:
            return default
        return transport.get_extra_info(name, default)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        ascii_encodable_path = self.path.encode('ascii', 'backslashreplace').decode('ascii')
        return '<{} {} {} >'.format(self.__class__.__name__, self._method, ascii_encodable_path)

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return id(self) == id(other)

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    async def _prepare_hook(self, response: StreamResponse) -> None:
        return

    def _cancel(self, exc: BaseException) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._payload.set_exception(exc)
        for fut in self._disconnection_waiters:
            set_result(fut, None)

    def _finish(self) -> None:
        if False:
            while True:
                i = 10
        for fut in self._disconnection_waiters:
            fut.cancel()
        if self._post is None or self.content_type != 'multipart/form-data':
            return
        for (file_name, file_field_object) in self._post.items():
            if not isinstance(file_field_object, FileField):
                continue
            file_field_object.file.close()

    async def wait_for_disconnection(self) -> None:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._disconnection_waiters.add(fut)
        try:
            await fut
        finally:
            self._disconnection_waiters.remove(fut)

class Request(BaseRequest):
    __slots__ = ('_match_info',)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._match_info: Optional[UrlMappingMatchInfo] = None

    def clone(self, *, method: Union[str, _SENTINEL]=sentinel, rel_url: Union[StrOrURL, _SENTINEL]=sentinel, headers: Union[LooseHeaders, _SENTINEL]=sentinel, scheme: Union[str, _SENTINEL]=sentinel, host: Union[str, _SENTINEL]=sentinel, remote: Union[str, _SENTINEL]=sentinel, client_max_size: Union[int, _SENTINEL]=sentinel) -> 'Request':
        if False:
            print('Hello World!')
        ret = super().clone(method=method, rel_url=rel_url, headers=headers, scheme=scheme, host=host, remote=remote, client_max_size=client_max_size)
        new_ret = cast(Request, ret)
        new_ret._match_info = self._match_info
        return new_ret

    @reify
    def match_info(self) -> 'UrlMappingMatchInfo':
        if False:
            i = 10
            return i + 15
        'Result of route resolving.'
        match_info = self._match_info
        assert match_info is not None
        return match_info

    @property
    def app(self) -> 'Application':
        if False:
            i = 10
            return i + 15
        'Application instance.'
        match_info = self._match_info
        assert match_info is not None
        return match_info.current_app

    @property
    def config_dict(self) -> ChainMapProxy:
        if False:
            return 10
        match_info = self._match_info
        assert match_info is not None
        lst = match_info.apps
        app = self.app
        idx = lst.index(app)
        sublist = list(reversed(lst[:idx + 1]))
        return ChainMapProxy(sublist)

    async def _prepare_hook(self, response: StreamResponse) -> None:
        match_info = self._match_info
        if match_info is None:
            return
        for app in match_info._apps:
            await app.on_response_prepare.send(self, response)