import abc
import asyncio
import base64
import hashlib
import keyword
import os
import re
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Container, Dict, Final, Generator, Iterable, Iterator, List, Mapping, NoReturn, Optional, Pattern, Set, Sized, Tuple, Type, TypedDict, Union, cast
from yarl import URL, __version__ as yarl_version
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import HTTPException, HTTPExpectationFailed, HTTPForbidden, HTTPMethodNotAllowed, HTTPNotFound
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef
__all__ = ('UrlDispatcher', 'UrlMappingMatchInfo', 'AbstractResource', 'Resource', 'PlainResource', 'DynamicResource', 'AbstractRoute', 'ResourceRoute', 'StaticResource', 'View')
if TYPE_CHECKING:
    from .web_app import Application
    BaseDict = Dict[str, str]
else:
    BaseDict = dict
YARL_VERSION: Final[Tuple[int, ...]] = tuple(map(int, yarl_version.split('.')[:2]))
HTTP_METHOD_RE: Final[Pattern[str]] = re.compile("^[0-9A-Za-z!#\\$%&'\\*\\+\\-\\.\\^_`\\|~]+$")
ROUTE_RE: Final[Pattern[str]] = re.compile('(\\{[_a-zA-Z][^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\})')
PATH_SEP: Final[str] = re.escape('/')
_ExpectHandler = Callable[[Request], Awaitable[Optional[StreamResponse]]]
_Resolve = Tuple[Optional['UrlMappingMatchInfo'], Set[str]]

class _InfoDict(TypedDict, total=False):
    path: str
    formatter: str
    pattern: Pattern[str]
    directory: Path
    prefix: str
    routes: Mapping[str, 'AbstractRoute']
    app: 'Application'
    domain: str
    rule: 'AbstractRuleMatching'
    http_exception: HTTPException

class AbstractResource(Sized, Iterable['AbstractRoute']):

    def __init__(self, *, name: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._name = name

    @property
    def name(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._name

    @property
    @abc.abstractmethod
    def canonical(self) -> str:
        if False:
            while True:
                i = 10
        "Exposes the resource's canonical path.\n\n        For example '/foo/bar/{name}'\n\n        "

    @abc.abstractmethod
    def url_for(self, **kwargs: str) -> URL:
        if False:
            i = 10
            return i + 15
        'Construct url for resource with additional params.'

    @abc.abstractmethod
    async def resolve(self, request: Request) -> _Resolve:
        """Resolve resource.

        Return (UrlMappingMatchInfo, allowed_methods) pair.
        """

    @abc.abstractmethod
    def add_prefix(self, prefix: str) -> None:
        if False:
            while True:
                i = 10
        'Add a prefix to processed URLs.\n\n        Required for subapplications support.\n        '

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        if False:
            print('Hello World!')
        'Return a dict with additional info useful for introspection'

    def freeze(self) -> None:
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def raw_match(self, path: str) -> bool:
        if False:
            while True:
                i = 10
        'Perform a raw match against path'

class AbstractRoute(abc.ABC):

    def __init__(self, method: str, handler: Union[Handler, Type[AbstractView]], *, expect_handler: Optional[_ExpectHandler]=None, resource: Optional[AbstractResource]=None) -> None:
        if False:
            while True:
                i = 10
        if expect_handler is None:
            expect_handler = _default_expect_handler
        assert asyncio.iscoroutinefunction(expect_handler), f'Coroutine is expected, got {expect_handler!r}'
        method = method.upper()
        if not HTTP_METHOD_RE.match(method):
            raise ValueError(f'{method} is not allowed HTTP method')
        if asyncio.iscoroutinefunction(handler):
            pass
        elif isinstance(handler, type) and issubclass(handler, AbstractView):
            pass
        else:
            raise TypeError('Only async functions are allowed as web-handlers , got {!r}'.format(handler))
        self._method = method
        self._handler = handler
        self._expect_handler = expect_handler
        self._resource = resource

    @property
    def method(self) -> str:
        if False:
            print('Hello World!')
        return self._method

    @property
    def handler(self) -> Handler:
        if False:
            for i in range(10):
                print('nop')
        return self._handler

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        if False:
            return 10
        "Optional route's name, always equals to resource's name."

    @property
    def resource(self) -> Optional[AbstractResource]:
        if False:
            i = 10
            return i + 15
        return self._resource

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        if False:
            i = 10
            return i + 15
        'Return a dict with additional info useful for introspection'

    @abc.abstractmethod
    def url_for(self, *args: str, **kwargs: str) -> URL:
        if False:
            print('Hello World!')
        'Construct url for route with additional params.'

    async def handle_expect_header(self, request: Request) -> Optional[StreamResponse]:
        return await self._expect_handler(request)

class UrlMappingMatchInfo(BaseDict, AbstractMatchInfo):

    def __init__(self, match_dict: Dict[str, str], route: AbstractRoute):
        if False:
            print('Hello World!')
        super().__init__(match_dict)
        self._route = route
        self._apps: List[Application] = []
        self._current_app: Optional[Application] = None
        self._frozen = False

    @property
    def handler(self) -> Handler:
        if False:
            i = 10
            return i + 15
        return self._route.handler

    @property
    def route(self) -> AbstractRoute:
        if False:
            return 10
        return self._route

    @property
    def expect_handler(self) -> _ExpectHandler:
        if False:
            while True:
                i = 10
        return self._route.handle_expect_header

    @property
    def http_exception(self) -> Optional[HTTPException]:
        if False:
            i = 10
            return i + 15
        return None

    def get_info(self) -> _InfoDict:
        if False:
            i = 10
            return i + 15
        return self._route.get_info()

    @property
    def apps(self) -> Tuple['Application', ...]:
        if False:
            i = 10
            return i + 15
        return tuple(self._apps)

    def add_app(self, app: 'Application') -> None:
        if False:
            while True:
                i = 10
        if self._frozen:
            raise RuntimeError('Cannot change apps stack after .freeze() call')
        if self._current_app is None:
            self._current_app = app
        self._apps.insert(0, app)

    @property
    def current_app(self) -> 'Application':
        if False:
            print('Hello World!')
        app = self._current_app
        assert app is not None
        return app

    @contextmanager
    def set_current_app(self, app: 'Application') -> Generator[None, None, None]:
        if False:
            while True:
                i = 10
        if DEBUG:
            if app not in self._apps:
                raise RuntimeError('Expected one of the following apps {!r}, got {!r}'.format(self._apps, app))
        prev = self._current_app
        self._current_app = app
        try:
            yield
        finally:
            self._current_app = prev

    def freeze(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._frozen = True

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<MatchInfo {super().__repr__()}: {self._route}>'

class MatchInfoError(UrlMappingMatchInfo):

    def __init__(self, http_exception: HTTPException) -> None:
        if False:
            i = 10
            return i + 15
        self._exception = http_exception
        super().__init__({}, SystemRoute(self._exception))

    @property
    def http_exception(self) -> HTTPException:
        if False:
            while True:
                i = 10
        return self._exception

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '<MatchInfoError {}: {}>'.format(self._exception.status, self._exception.reason)

async def _default_expect_handler(request: Request) -> None:
    """Default handler for Expect header.

    Just send "100 Continue" to client.
    raise HTTPExpectationFailed if value of header is not "100-continue"
    """
    expect = request.headers.get(hdrs.EXPECT, '')
    if request.version == HttpVersion11:
        if expect.lower() == '100-continue':
            await request.writer.write(b'HTTP/1.1 100 Continue\r\n\r\n')
        else:
            raise HTTPExpectationFailed(text='Unknown Expect: %s' % expect)

class Resource(AbstractResource):

    def __init__(self, *, name: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(name=name)
        self._routes: List[ResourceRoute] = []

    def add_route(self, method: str, handler: Union[Type[AbstractView], Handler], *, expect_handler: Optional[_ExpectHandler]=None) -> 'ResourceRoute':
        if False:
            return 10
        for route_obj in self._routes:
            if route_obj.method == method or route_obj.method == hdrs.METH_ANY:
                raise RuntimeError('Added route will never be executed, method {route.method} is already registered'.format(route=route_obj))
        route_obj = ResourceRoute(method, handler, self, expect_handler=expect_handler)
        self.register_route(route_obj)
        return route_obj

    def register_route(self, route: 'ResourceRoute') -> None:
        if False:
            while True:
                i = 10
        assert isinstance(route, ResourceRoute), f'Instance of Route class is required, got {route!r}'
        self._routes.append(route)

    async def resolve(self, request: Request) -> _Resolve:
        allowed_methods: Set[str] = set()
        match_dict = self._match(request.rel_url.raw_path)
        if match_dict is None:
            return (None, allowed_methods)
        for route_obj in self._routes:
            route_method = route_obj.method
            allowed_methods.add(route_method)
            if route_method == request.method or route_method == hdrs.METH_ANY:
                return (UrlMappingMatchInfo(match_dict, route_obj), allowed_methods)
        else:
            return (None, allowed_methods)

    @abc.abstractmethod
    def _match(self, path: str) -> Optional[Dict[str, str]]:
        if False:
            print('Hello World!')
        pass

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self._routes)

    def __iter__(self) -> Iterator['ResourceRoute']:
        if False:
            i = 10
            return i + 15
        return iter(self._routes)

class PlainResource(Resource):

    def __init__(self, path: str, *, name: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(name=name)
        assert not path or path.startswith('/')
        self._path = path

    @property
    def canonical(self) -> str:
        if False:
            return 10
        return self._path

    def freeze(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self._path:
            self._path = '/'

    def add_prefix(self, prefix: str) -> None:
        if False:
            return 10
        assert prefix.startswith('/')
        assert not prefix.endswith('/')
        assert len(prefix) > 1
        self._path = prefix + self._path

    def _match(self, path: str) -> Optional[Dict[str, str]]:
        if False:
            i = 10
            return i + 15
        if self._path == path:
            return {}
        else:
            return None

    def raw_match(self, path: str) -> bool:
        if False:
            while True:
                i = 10
        return self._path == path

    def get_info(self) -> _InfoDict:
        if False:
            print('Hello World!')
        return {'path': self._path}

    def url_for(self) -> URL:
        if False:
            while True:
                i = 10
        return URL.build(path=self._path, encoded=True)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        name = "'" + self.name + "' " if self.name is not None else ''
        return f'<PlainResource {name} {self._path}>'

class DynamicResource(Resource):
    DYN = re.compile('\\{(?P<var>[_a-zA-Z][_a-zA-Z0-9]*)\\}')
    DYN_WITH_RE = re.compile('\\{(?P<var>[_a-zA-Z][_a-zA-Z0-9]*):(?P<re>.+)\\}')
    GOOD = '[^{}/]+'

    def __init__(self, path: str, *, name: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(name=name)
        pattern = ''
        formatter = ''
        for part in ROUTE_RE.split(path):
            match = self.DYN.fullmatch(part)
            if match:
                pattern += '(?P<{}>{})'.format(match.group('var'), self.GOOD)
                formatter += '{' + match.group('var') + '}'
                continue
            match = self.DYN_WITH_RE.fullmatch(part)
            if match:
                pattern += '(?P<{var}>{re})'.format(**match.groupdict())
                formatter += '{' + match.group('var') + '}'
                continue
            if '{' in part or '}' in part:
                raise ValueError(f"Invalid path '{path}'['{part}']")
            part = _requote_path(part)
            formatter += part
            pattern += re.escape(part)
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Bad pattern '{pattern}': {exc}") from None
        assert compiled.pattern.startswith(PATH_SEP)
        assert formatter.startswith('/')
        self._pattern = compiled
        self._formatter = formatter

    @property
    def canonical(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._formatter

    def add_prefix(self, prefix: str) -> None:
        if False:
            print('Hello World!')
        assert prefix.startswith('/')
        assert not prefix.endswith('/')
        assert len(prefix) > 1
        self._pattern = re.compile(re.escape(prefix) + self._pattern.pattern)
        self._formatter = prefix + self._formatter

    def _match(self, path: str) -> Optional[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        match = self._pattern.fullmatch(path)
        if match is None:
            return None
        else:
            return {key: _unquote_path(value) for (key, value) in match.groupdict().items()}

    def raw_match(self, path: str) -> bool:
        if False:
            i = 10
            return i + 15
        return self._formatter == path

    def get_info(self) -> _InfoDict:
        if False:
            return 10
        return {'formatter': self._formatter, 'pattern': self._pattern}

    def url_for(self, **parts: str) -> URL:
        if False:
            print('Hello World!')
        url = self._formatter.format_map({k: _quote_path(v) for (k, v) in parts.items()})
        return URL.build(path=url, encoded=True)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        name = "'" + self.name + "' " if self.name is not None else ''
        return '<DynamicResource {name} {formatter}>'.format(name=name, formatter=self._formatter)

class PrefixResource(AbstractResource):

    def __init__(self, prefix: str, *, name: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        assert not prefix or prefix.startswith('/'), prefix
        assert prefix in ('', '/') or not prefix.endswith('/'), prefix
        super().__init__(name=name)
        self._prefix = _requote_path(prefix)
        self._prefix2 = self._prefix + '/'

    @property
    def canonical(self) -> str:
        if False:
            while True:
                i = 10
        return self._prefix

    def add_prefix(self, prefix: str) -> None:
        if False:
            print('Hello World!')
        assert prefix.startswith('/')
        assert not prefix.endswith('/')
        assert len(prefix) > 1
        self._prefix = prefix + self._prefix
        self._prefix2 = self._prefix + '/'

    def raw_match(self, prefix: str) -> bool:
        if False:
            while True:
                i = 10
        return False

class StaticResource(PrefixResource):
    VERSION_KEY = 'v'

    def __init__(self, prefix: str, directory: PathLike, *, name: Optional[str]=None, expect_handler: Optional[_ExpectHandler]=None, chunk_size: int=256 * 1024, show_index: bool=False, follow_symlinks: bool=False, append_version: bool=False) -> None:
        if False:
            return 10
        super().__init__(prefix, name=name)
        try:
            directory = Path(directory)
            if str(directory).startswith('~'):
                directory = Path(os.path.expanduser(str(directory)))
            directory = directory.resolve()
            if not directory.is_dir():
                raise ValueError('Not a directory')
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"No directory exists at '{directory}'") from error
        self._directory = directory
        self._show_index = show_index
        self._chunk_size = chunk_size
        self._follow_symlinks = follow_symlinks
        self._expect_handler = expect_handler
        self._append_version = append_version
        self._routes = {'GET': ResourceRoute('GET', self._handle, self, expect_handler=expect_handler), 'HEAD': ResourceRoute('HEAD', self._handle, self, expect_handler=expect_handler)}

    def url_for(self, *, filename: PathLike, append_version: Optional[bool]=None) -> URL:
        if False:
            i = 10
            return i + 15
        if append_version is None:
            append_version = self._append_version
        filename = str(filename).lstrip('/')
        url = URL.build(path=self._prefix, encoded=True)
        if YARL_VERSION < (1, 6):
            url = url / filename.replace('%', '%25')
        else:
            url = url / filename
        if append_version:
            try:
                filepath = self._directory.joinpath(filename).resolve()
                if not self._follow_symlinks:
                    filepath.relative_to(self._directory)
            except (ValueError, FileNotFoundError):
                return url
            if filepath.is_file():
                with filepath.open('rb') as f:
                    file_bytes = f.read()
                h = self._get_file_hash(file_bytes)
                url = url.with_query({self.VERSION_KEY: h})
                return url
        return url

    @staticmethod
    def _get_file_hash(byte_array: bytes) -> str:
        if False:
            i = 10
            return i + 15
        m = hashlib.sha256()
        m.update(byte_array)
        b64 = base64.urlsafe_b64encode(m.digest())
        return b64.decode('ascii')

    def get_info(self) -> _InfoDict:
        if False:
            return 10
        return {'directory': self._directory, 'prefix': self._prefix, 'routes': self._routes}

    def set_options_route(self, handler: Handler) -> None:
        if False:
            return 10
        if 'OPTIONS' in self._routes:
            raise RuntimeError('OPTIONS route was set already')
        self._routes['OPTIONS'] = ResourceRoute('OPTIONS', handler, self, expect_handler=self._expect_handler)

    async def resolve(self, request: Request) -> _Resolve:
        path = request.rel_url.raw_path
        method = request.method
        allowed_methods = set(self._routes)
        if not path.startswith(self._prefix2) and path != self._prefix:
            return (None, set())
        if method not in allowed_methods:
            return (None, allowed_methods)
        match_dict = {'filename': _unquote_path(path[len(self._prefix) + 1:])}
        return (UrlMappingMatchInfo(match_dict, self._routes[method]), allowed_methods)

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._routes)

    def __iter__(self) -> Iterator[AbstractRoute]:
        if False:
            i = 10
            return i + 15
        return iter(self._routes.values())

    async def _handle(self, request: Request) -> StreamResponse:
        rel_url = request.match_info['filename']
        try:
            filename = Path(rel_url)
            if filename.anchor:
                raise HTTPForbidden()
            filepath = self._directory.joinpath(filename).resolve()
            if not self._follow_symlinks:
                filepath.relative_to(self._directory)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPNotFound() from error
        except HTTPForbidden:
            raise
        except Exception as error:
            request.app.logger.exception(error)
            raise HTTPNotFound() from error
        if filepath.is_dir():
            if self._show_index:
                try:
                    return Response(text=self._directory_as_html(filepath), content_type='text/html')
                except PermissionError:
                    raise HTTPForbidden()
            else:
                raise HTTPForbidden()
        elif filepath.is_file():
            return FileResponse(filepath, chunk_size=self._chunk_size)
        else:
            raise HTTPNotFound

    def _directory_as_html(self, filepath: Path) -> str:
        if False:
            while True:
                i = 10
        assert filepath.is_dir()
        relative_path_to_dir = filepath.relative_to(self._directory).as_posix()
        index_of = f'Index of /{relative_path_to_dir}'
        h1 = f'<h1>{index_of}</h1>'
        index_list = []
        dir_index = filepath.iterdir()
        for _file in sorted(dir_index):
            rel_path = _file.relative_to(self._directory).as_posix()
            file_url = self._prefix + '/' + rel_path
            if _file.is_dir():
                file_name = f'{_file.name}/'
            else:
                file_name = _file.name
            index_list.append('<li><a href="{url}">{name}</a></li>'.format(url=file_url, name=file_name))
        ul = '<ul>\n{}\n</ul>'.format('\n'.join(index_list))
        body = f'<body>\n{h1}\n{ul}\n</body>'
        head_str = f'<head>\n<title>{index_of}</title>\n</head>'
        html = f'<html>\n{head_str}\n{body}\n</html>'
        return html

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        name = "'" + self.name + "'" if self.name is not None else ''
        return '<StaticResource {name} {path} -> {directory!r}>'.format(name=name, path=self._prefix, directory=self._directory)

class PrefixedSubAppResource(PrefixResource):

    def __init__(self, prefix: str, app: 'Application') -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(prefix)
        self._app = app
        for resource in app.router.resources():
            resource.add_prefix(prefix)

    def add_prefix(self, prefix: str) -> None:
        if False:
            i = 10
            return i + 15
        super().add_prefix(prefix)
        for resource in self._app.router.resources():
            resource.add_prefix(prefix)

    def url_for(self, *args: str, **kwargs: str) -> URL:
        if False:
            return 10
        raise RuntimeError('.url_for() is not supported by sub-application root')

    def get_info(self) -> _InfoDict:
        if False:
            i = 10
            return i + 15
        return {'app': self._app, 'prefix': self._prefix}

    async def resolve(self, request: Request) -> _Resolve:
        if not request.url.raw_path.startswith(self._prefix2) and request.url.raw_path != self._prefix:
            return (None, set())
        match_info = await self._app.router.resolve(request)
        match_info.add_app(self._app)
        if isinstance(match_info.http_exception, HTTPMethodNotAllowed):
            methods = match_info.http_exception.allowed_methods
        else:
            methods = set()
        return (match_info, methods)

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._app.router.routes())

    def __iter__(self) -> Iterator[AbstractRoute]:
        if False:
            while True:
                i = 10
        return iter(self._app.router.routes())

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<PrefixedSubAppResource {prefix} -> {app!r}>'.format(prefix=self._prefix, app=self._app)

class AbstractRuleMatching(abc.ABC):

    @abc.abstractmethod
    async def match(self, request: Request) -> bool:
        """Return bool if the request satisfies the criteria"""

    @abc.abstractmethod
    def get_info(self) -> _InfoDict:
        if False:
            return 10
        'Return a dict with additional info useful for introspection'

    @property
    @abc.abstractmethod
    def canonical(self) -> str:
        if False:
            return 10
        'Return a str'

class Domain(AbstractRuleMatching):
    re_part = re.compile('(?!-)[a-z\\d-]{1,63}(?<!-)')

    def __init__(self, domain: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._domain = self.validation(domain)

    @property
    def canonical(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._domain

    def validation(self, domain: str) -> str:
        if False:
            while True:
                i = 10
        if not isinstance(domain, str):
            raise TypeError('Domain must be str')
        domain = domain.rstrip('.').lower()
        if not domain:
            raise ValueError('Domain cannot be empty')
        elif '://' in domain:
            raise ValueError('Scheme not supported')
        url = URL('http://' + domain)
        assert url.raw_host is not None
        if not all((self.re_part.fullmatch(x) for x in url.raw_host.split('.'))):
            raise ValueError('Domain not valid')
        if url.port == 80:
            return url.raw_host
        return f'{url.raw_host}:{url.port}'

    async def match(self, request: Request) -> bool:
        host = request.headers.get(hdrs.HOST)
        if not host:
            return False
        return self.match_domain(host)

    def match_domain(self, host: str) -> bool:
        if False:
            i = 10
            return i + 15
        return host.lower() == self._domain

    def get_info(self) -> _InfoDict:
        if False:
            print('Hello World!')
        return {'domain': self._domain}

class MaskDomain(Domain):
    re_part = re.compile('(?!-)[a-z\\d\\*-]{1,63}(?<!-)')

    def __init__(self, domain: str) -> None:
        if False:
            return 10
        super().__init__(domain)
        mask = self._domain.replace('.', '\\.').replace('*', '.*')
        self._mask = re.compile(mask)

    @property
    def canonical(self) -> str:
        if False:
            print('Hello World!')
        return self._mask.pattern

    def match_domain(self, host: str) -> bool:
        if False:
            while True:
                i = 10
        return self._mask.fullmatch(host) is not None

class MatchedSubAppResource(PrefixedSubAppResource):

    def __init__(self, rule: AbstractRuleMatching, app: 'Application') -> None:
        if False:
            while True:
                i = 10
        AbstractResource.__init__(self)
        self._prefix = ''
        self._app = app
        self._rule = rule

    @property
    def canonical(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._rule.canonical

    def get_info(self) -> _InfoDict:
        if False:
            return 10
        return {'app': self._app, 'rule': self._rule}

    async def resolve(self, request: Request) -> _Resolve:
        if not await self._rule.match(request):
            return (None, set())
        match_info = await self._app.router.resolve(request)
        match_info.add_app(self._app)
        if isinstance(match_info.http_exception, HTTPMethodNotAllowed):
            methods = match_info.http_exception.allowed_methods
        else:
            methods = set()
        return (match_info, methods)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '<MatchedSubAppResource -> {app!r}>'.format(app=self._app)

class ResourceRoute(AbstractRoute):
    """A route with resource"""

    def __init__(self, method: str, handler: Union[Handler, Type[AbstractView]], resource: AbstractResource, *, expect_handler: Optional[_ExpectHandler]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(method, handler, expect_handler=expect_handler, resource=resource)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<ResourceRoute [{method}] {resource} -> {handler!r}'.format(method=self.method, resource=self._resource, handler=self.handler)

    @property
    def name(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if self._resource is None:
            return None
        return self._resource.name

    def url_for(self, *args: str, **kwargs: str) -> URL:
        if False:
            print('Hello World!')
        'Construct url for route with additional params.'
        assert self._resource is not None
        return self._resource.url_for(*args, **kwargs)

    def get_info(self) -> _InfoDict:
        if False:
            while True:
                i = 10
        assert self._resource is not None
        return self._resource.get_info()

class SystemRoute(AbstractRoute):

    def __init__(self, http_exception: HTTPException) -> None:
        if False:
            print('Hello World!')
        super().__init__(hdrs.METH_ANY, self._handle)
        self._http_exception = http_exception

    def url_for(self, *args: str, **kwargs: str) -> URL:
        if False:
            while True:
                i = 10
        raise RuntimeError('.url_for() is not allowed for SystemRoute')

    @property
    def name(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return None

    def get_info(self) -> _InfoDict:
        if False:
            while True:
                i = 10
        return {'http_exception': self._http_exception}

    async def _handle(self, request: Request) -> StreamResponse:
        raise self._http_exception

    @property
    def status(self) -> int:
        if False:
            print('Hello World!')
        return self._http_exception.status

    @property
    def reason(self) -> str:
        if False:
            print('Hello World!')
        return self._http_exception.reason

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<SystemRoute {self.status}: {self.reason}>'.format(self=self)

class View(AbstractView):

    async def _iter(self) -> StreamResponse:
        if self.request.method not in hdrs.METH_ALL:
            self._raise_allowed_methods()
        method: Optional[Callable[[], Awaitable[StreamResponse]]] = getattr(self, self.request.method.lower(), None)
        if method is None:
            self._raise_allowed_methods()
        return await method()

    def __await__(self) -> Generator[Any, None, StreamResponse]:
        if False:
            while True:
                i = 10
        return self._iter().__await__()

    def _raise_allowed_methods(self) -> NoReturn:
        if False:
            print('Hello World!')
        allowed_methods = {m for m in hdrs.METH_ALL if hasattr(self, m.lower())}
        raise HTTPMethodNotAllowed(self.request.method, allowed_methods)

class ResourcesView(Sized, Iterable[AbstractResource], Container[AbstractResource]):

    def __init__(self, resources: List[AbstractResource]) -> None:
        if False:
            return 10
        self._resources = resources

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._resources)

    def __iter__(self) -> Iterator[AbstractResource]:
        if False:
            while True:
                i = 10
        yield from self._resources

    def __contains__(self, resource: object) -> bool:
        if False:
            print('Hello World!')
        return resource in self._resources

class RoutesView(Sized, Iterable[AbstractRoute], Container[AbstractRoute]):

    def __init__(self, resources: List[AbstractResource]):
        if False:
            i = 10
            return i + 15
        self._routes: List[AbstractRoute] = []
        for resource in resources:
            for route in resource:
                self._routes.append(route)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self._routes)

    def __iter__(self) -> Iterator[AbstractRoute]:
        if False:
            i = 10
            return i + 15
        yield from self._routes

    def __contains__(self, route: object) -> bool:
        if False:
            print('Hello World!')
        return route in self._routes

class UrlDispatcher(AbstractRouter, Mapping[str, AbstractResource]):
    NAME_SPLIT_RE = re.compile('[.:-]')

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._resources: List[AbstractResource] = []
        self._named_resources: Dict[str, AbstractResource] = {}

    async def resolve(self, request: Request) -> UrlMappingMatchInfo:
        method = request.method
        allowed_methods: Set[str] = set()
        for resource in self._resources:
            (match_dict, allowed) = await resource.resolve(request)
            if match_dict is not None:
                return match_dict
            else:
                allowed_methods |= allowed
        if allowed_methods:
            return MatchInfoError(HTTPMethodNotAllowed(method, allowed_methods))
        else:
            return MatchInfoError(HTTPNotFound())

    def __iter__(self) -> Iterator[str]:
        if False:
            return 10
        return iter(self._named_resources)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self._named_resources)

    def __contains__(self, resource: object) -> bool:
        if False:
            i = 10
            return i + 15
        return resource in self._named_resources

    def __getitem__(self, name: str) -> AbstractResource:
        if False:
            while True:
                i = 10
        return self._named_resources[name]

    def resources(self) -> ResourcesView:
        if False:
            while True:
                i = 10
        return ResourcesView(self._resources)

    def routes(self) -> RoutesView:
        if False:
            i = 10
            return i + 15
        return RoutesView(self._resources)

    def named_resources(self) -> Mapping[str, AbstractResource]:
        if False:
            print('Hello World!')
        return MappingProxyType(self._named_resources)

    def register_resource(self, resource: AbstractResource) -> None:
        if False:
            while True:
                i = 10
        assert isinstance(resource, AbstractResource), f'Instance of AbstractResource class is required, got {resource!r}'
        if self.frozen:
            raise RuntimeError('Cannot register a resource into frozen router.')
        name = resource.name
        if name is not None:
            parts = self.NAME_SPLIT_RE.split(name)
            for part in parts:
                if keyword.iskeyword(part):
                    raise ValueError(f'Incorrect route name {name!r}, python keywords cannot be used for route name')
                if not part.isidentifier():
                    raise ValueError('Incorrect route name {!r}, the name should be a sequence of python identifiers separated by dash, dot or column'.format(name))
            if name in self._named_resources:
                raise ValueError('Duplicate {!r}, already handled by {!r}'.format(name, self._named_resources[name]))
            self._named_resources[name] = resource
        self._resources.append(resource)

    def add_resource(self, path: str, *, name: Optional[str]=None) -> Resource:
        if False:
            print('Hello World!')
        if path and (not path.startswith('/')):
            raise ValueError('path should be started with / or be empty')
        if self._resources:
            resource = self._resources[-1]
            if resource.name == name and resource.raw_match(path):
                return cast(Resource, resource)
        if not ('{' in path or '}' in path or ROUTE_RE.search(path)):
            resource = PlainResource(_requote_path(path), name=name)
            self.register_resource(resource)
            return resource
        resource = DynamicResource(path, name=name)
        self.register_resource(resource)
        return resource

    def add_route(self, method: str, path: str, handler: Union[Handler, Type[AbstractView]], *, name: Optional[str]=None, expect_handler: Optional[_ExpectHandler]=None) -> AbstractRoute:
        if False:
            print('Hello World!')
        resource = self.add_resource(path, name=name)
        return resource.add_route(method, handler, expect_handler=expect_handler)

    def add_static(self, prefix: str, path: PathLike, *, name: Optional[str]=None, expect_handler: Optional[_ExpectHandler]=None, chunk_size: int=256 * 1024, show_index: bool=False, follow_symlinks: bool=False, append_version: bool=False) -> AbstractResource:
        if False:
            for i in range(10):
                print('nop')
        'Add static files view.\n\n        prefix - url prefix\n        path - folder with files\n\n        '
        assert prefix.startswith('/')
        if prefix.endswith('/'):
            prefix = prefix[:-1]
        resource = StaticResource(prefix, path, name=name, expect_handler=expect_handler, chunk_size=chunk_size, show_index=show_index, follow_symlinks=follow_symlinks, append_version=append_version)
        self.register_resource(resource)
        return resource

    def add_head(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        if False:
            i = 10
            return i + 15
        'Shortcut for add_route with method HEAD.'
        return self.add_route(hdrs.METH_HEAD, path, handler, **kwargs)

    def add_options(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        if False:
            return 10
        'Shortcut for add_route with method OPTIONS.'
        return self.add_route(hdrs.METH_OPTIONS, path, handler, **kwargs)

    def add_get(self, path: str, handler: Handler, *, name: Optional[str]=None, allow_head: bool=True, **kwargs: Any) -> AbstractRoute:
        if False:
            i = 10
            return i + 15
        'Shortcut for add_route with method GET.\n\n        If allow_head is true, another\n        route is added allowing head requests to the same endpoint.\n        '
        resource = self.add_resource(path, name=name)
        if allow_head:
            resource.add_route(hdrs.METH_HEAD, handler, **kwargs)
        return resource.add_route(hdrs.METH_GET, handler, **kwargs)

    def add_post(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        if False:
            i = 10
            return i + 15
        'Shortcut for add_route with method POST.'
        return self.add_route(hdrs.METH_POST, path, handler, **kwargs)

    def add_put(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        if False:
            print('Hello World!')
        'Shortcut for add_route with method PUT.'
        return self.add_route(hdrs.METH_PUT, path, handler, **kwargs)

    def add_patch(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        if False:
            i = 10
            return i + 15
        'Shortcut for add_route with method PATCH.'
        return self.add_route(hdrs.METH_PATCH, path, handler, **kwargs)

    def add_delete(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        if False:
            for i in range(10):
                print('nop')
        'Shortcut for add_route with method DELETE.'
        return self.add_route(hdrs.METH_DELETE, path, handler, **kwargs)

    def add_view(self, path: str, handler: Type[AbstractView], **kwargs: Any) -> AbstractRoute:
        if False:
            while True:
                i = 10
        'Shortcut for add_route with ANY methods for a class-based view.'
        return self.add_route(hdrs.METH_ANY, path, handler, **kwargs)

    def freeze(self) -> None:
        if False:
            print('Hello World!')
        super().freeze()
        for resource in self._resources:
            resource.freeze()

    def add_routes(self, routes: Iterable[AbstractRouteDef]) -> List[AbstractRoute]:
        if False:
            for i in range(10):
                print('nop')
        'Append routes to route table.\n\n        Parameter should be a sequence of RouteDef objects.\n\n        Returns a list of registered AbstractRoute instances.\n        '
        registered_routes = []
        for route_def in routes:
            registered_routes.extend(route_def.register(self))
        return registered_routes

def _quote_path(value: str) -> str:
    if False:
        i = 10
        return i + 15
    if YARL_VERSION < (1, 6):
        value = value.replace('%', '%25')
    return URL.build(path=value, encoded=False).raw_path

def _unquote_path(value: str) -> str:
    if False:
        while True:
            i = 10
    return URL.build(path=value, encoded=True).path

def _requote_path(value: str) -> str:
    if False:
        while True:
            i = 10
    result = _quote_path(value)
    if '%' in value:
        result = result.replace('%25', '%')
    return result