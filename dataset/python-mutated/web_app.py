import asyncio
import logging
import warnings
from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Type, TypeVar, Union, cast, final, overload
from aiosignal import Signal
from frozenlist import FrozenList
from . import hdrs
from .helpers import AppKey
from .log import web_logger
from .typedefs import Middleware
from .web_exceptions import NotAppKeyWarning
from .web_middlewares import _fix_request_current_app
from .web_request import Request
from .web_response import StreamResponse
from .web_routedef import AbstractRouteDef
from .web_urldispatcher import AbstractResource, AbstractRoute, Domain, MaskDomain, MatchedSubAppResource, PrefixedSubAppResource, UrlDispatcher
__all__ = ('Application', 'CleanupError')
if TYPE_CHECKING:
    _AppSignal = Signal[Callable[['Application'], Awaitable[None]]]
    _RespPrepareSignal = Signal[Callable[[Request, StreamResponse], Awaitable[None]]]
    _Middlewares = FrozenList[Middleware]
    _MiddlewaresHandlers = Sequence[Middleware]
    _Subapps = List['Application']
else:
    _AppSignal = Signal
    _RespPrepareSignal = Signal
    _Handler = Callable
    _Middlewares = FrozenList
    _MiddlewaresHandlers = Sequence
    _Subapps = List
_T = TypeVar('_T')
_U = TypeVar('_U')

@final
class Application(MutableMapping[Union[str, AppKey[Any]], Any]):
    __slots__ = ('logger', '_debug', '_router', '_loop', '_handler_args', '_middlewares', '_middlewares_handlers', '_run_middlewares', '_state', '_frozen', '_pre_frozen', '_subapps', '_on_response_prepare', '_on_startup', '_on_shutdown', '_on_cleanup', '_client_max_size', '_cleanup_ctx')

    def __init__(self, *, logger: logging.Logger=web_logger, middlewares: Iterable[Middleware]=(), handler_args: Optional[Mapping[str, Any]]=None, client_max_size: int=1024 ** 2, debug: Any=...) -> None:
        if False:
            while True:
                i = 10
        if debug is not ...:
            warnings.warn('debug argument is no-op since 4.0 and scheduled for removal in 5.0', DeprecationWarning, stacklevel=2)
        self._router = UrlDispatcher()
        self._handler_args = handler_args
        self.logger = logger
        self._middlewares: _Middlewares = FrozenList(middlewares)
        self._middlewares_handlers: _MiddlewaresHandlers = tuple()
        self._run_middlewares: Optional[bool] = None
        self._state: Dict[Union[AppKey[Any], str], object] = {}
        self._frozen = False
        self._pre_frozen = False
        self._subapps: _Subapps = []
        self._on_response_prepare: _RespPrepareSignal = Signal(self)
        self._on_startup: _AppSignal = Signal(self)
        self._on_shutdown: _AppSignal = Signal(self)
        self._on_cleanup: _AppSignal = Signal(self)
        self._cleanup_ctx = CleanupContext()
        self._on_startup.append(self._cleanup_ctx._on_startup)
        self._on_cleanup.append(self._cleanup_ctx._on_cleanup)
        self._client_max_size = client_max_size

    def __init_subclass__(cls: Type['Application']) -> None:
        if False:
            i = 10
            return i + 15
        raise TypeError('Inheritance class {} from web.Application is forbidden'.format(cls.__name__))

    def __eq__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self is other

    @overload
    def __getitem__(self, key: AppKey[_T]) -> _T:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __getitem__(self, key: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __getitem__(self, key: Union[str, AppKey[_T]]) -> Any:
        if False:
            print('Hello World!')
        return self._state[key]

    def _check_frozen(self) -> None:
        if False:
            print('Hello World!')
        if self._frozen:
            raise RuntimeError('Changing state of started or joined application is forbidden')

    @overload
    def __setitem__(self, key: AppKey[_T], value: _T) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __setitem__(self, key: str, value: Any) -> None:
        if False:
            while True:
                i = 10
        ...

    def __setitem__(self, key: Union[str, AppKey[_T]], value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._check_frozen()
        if not isinstance(key, AppKey):
            warnings.warn('It is recommended to use web.AppKey instances for keys.\n' + 'https://docs.aiohttp.org/en/stable/web_advanced.html' + '#application-s-config', category=NotAppKeyWarning, stacklevel=2)
        self._state[key] = value

    def __delitem__(self, key: Union[str, AppKey[_T]]) -> None:
        if False:
            return 10
        self._check_frozen()
        del self._state[key]

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._state)

    def __iter__(self) -> Iterator[Union[str, AppKey[Any]]]:
        if False:
            for i in range(10):
                print('nop')
        return iter(self._state)

    @overload
    def get(self, key: AppKey[_T], default: None=...) -> Optional[_T]:
        if False:
            print('Hello World!')
        ...

    @overload
    def get(self, key: AppKey[_T], default: _U) -> Union[_T, _U]:
        if False:
            return 10
        ...

    @overload
    def get(self, key: str, default: Any=...) -> Any:
        if False:
            i = 10
            return i + 15
        ...

    def get(self, key: Union[str, AppKey[_T]], default: Any=None) -> Any:
        if False:
            print('Hello World!')
        return self._state.get(key, default)

    def _set_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        if False:
            return 10
        warnings.warn('_set_loop() is no-op since 4.0 and scheduled for removal in 5.0', DeprecationWarning, stacklevel=2)

    @property
    def pre_frozen(self) -> bool:
        if False:
            print('Hello World!')
        return self._pre_frozen

    def pre_freeze(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._pre_frozen:
            return
        self._pre_frozen = True
        self._middlewares.freeze()
        self._router.freeze()
        self._on_response_prepare.freeze()
        self._cleanup_ctx.freeze()
        self._on_startup.freeze()
        self._on_shutdown.freeze()
        self._on_cleanup.freeze()
        self._middlewares_handlers = tuple(self._prepare_middleware())
        self._run_middlewares = True if self.middlewares else False
        for subapp in self._subapps:
            subapp.pre_freeze()
            self._run_middlewares = self._run_middlewares or subapp._run_middlewares

    @property
    def frozen(self) -> bool:
        if False:
            return 10
        return self._frozen

    def freeze(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._frozen:
            return
        self.pre_freeze()
        self._frozen = True
        for subapp in self._subapps:
            subapp.freeze()

    @property
    def debug(self) -> bool:
        if False:
            return 10
        warnings.warn('debug property is deprecated since 4.0and scheduled for removal in 5.0', DeprecationWarning, stacklevel=2)
        return asyncio.get_event_loop().get_debug()

    def _reg_subapp_signals(self, subapp: 'Application') -> None:
        if False:
            while True:
                i = 10

        def reg_handler(signame: str) -> None:
            if False:
                while True:
                    i = 10
            subsig = getattr(subapp, signame)

            async def handler(app: 'Application') -> None:
                await subsig.send(subapp)
            appsig = getattr(self, signame)
            appsig.append(handler)
        reg_handler('on_startup')
        reg_handler('on_shutdown')
        reg_handler('on_cleanup')

    def add_subapp(self, prefix: str, subapp: 'Application') -> AbstractResource:
        if False:
            i = 10
            return i + 15
        if not isinstance(prefix, str):
            raise TypeError('Prefix must be str')
        prefix = prefix.rstrip('/')
        if not prefix:
            raise ValueError('Prefix cannot be empty')
        factory = partial(PrefixedSubAppResource, prefix, subapp)
        return self._add_subapp(factory, subapp)

    def _add_subapp(self, resource_factory: Callable[[], AbstractResource], subapp: 'Application') -> AbstractResource:
        if False:
            i = 10
            return i + 15
        if self.frozen:
            raise RuntimeError('Cannot add sub application to frozen application')
        if subapp.frozen:
            raise RuntimeError('Cannot add frozen application')
        resource = resource_factory()
        self.router.register_resource(resource)
        self._reg_subapp_signals(subapp)
        self._subapps.append(subapp)
        subapp.pre_freeze()
        return resource

    def add_domain(self, domain: str, subapp: 'Application') -> AbstractResource:
        if False:
            print('Hello World!')
        if not isinstance(domain, str):
            raise TypeError('Domain must be str')
        elif '*' in domain:
            rule: Domain = MaskDomain(domain)
        else:
            rule = Domain(domain)
        factory = partial(MatchedSubAppResource, rule, subapp)
        return self._add_subapp(factory, subapp)

    def add_routes(self, routes: Iterable[AbstractRouteDef]) -> List[AbstractRoute]:
        if False:
            while True:
                i = 10
        return self.router.add_routes(routes)

    @property
    def on_response_prepare(self) -> _RespPrepareSignal:
        if False:
            print('Hello World!')
        return self._on_response_prepare

    @property
    def on_startup(self) -> _AppSignal:
        if False:
            while True:
                i = 10
        return self._on_startup

    @property
    def on_shutdown(self) -> _AppSignal:
        if False:
            return 10
        return self._on_shutdown

    @property
    def on_cleanup(self) -> _AppSignal:
        if False:
            return 10
        return self._on_cleanup

    @property
    def cleanup_ctx(self) -> 'CleanupContext':
        if False:
            for i in range(10):
                print('nop')
        return self._cleanup_ctx

    @property
    def router(self) -> UrlDispatcher:
        if False:
            for i in range(10):
                print('nop')
        return self._router

    @property
    def middlewares(self) -> _Middlewares:
        if False:
            for i in range(10):
                print('nop')
        return self._middlewares

    async def startup(self) -> None:
        """Causes on_startup signal

        Should be called in the event loop along with the request handler.
        """
        await self.on_startup.send(self)

    async def shutdown(self) -> None:
        """Causes on_shutdown signal

        Should be called before cleanup()
        """
        await self.on_shutdown.send(self)

    async def cleanup(self) -> None:
        """Causes on_cleanup signal

        Should be called after shutdown()
        """
        if self.on_cleanup.frozen:
            await self.on_cleanup.send(self)
        else:
            await self._cleanup_ctx._on_cleanup(self)

    def _prepare_middleware(self) -> Iterator[Middleware]:
        if False:
            print('Hello World!')
        yield from reversed(self._middlewares)
        yield _fix_request_current_app(self)

    async def _handle(self, request: Request) -> StreamResponse:
        match_info = await self._router.resolve(request)
        match_info.add_app(self)
        match_info.freeze()
        resp = None
        request._match_info = match_info
        expect = request.headers.get(hdrs.EXPECT)
        if expect:
            resp = await match_info.expect_handler(request)
            await request.writer.drain()
        if resp is None:
            handler = match_info.handler
            if self._run_middlewares:
                for app in match_info.apps[::-1]:
                    assert app.pre_frozen, 'middleware handlers are not ready'
                    for m in app._middlewares_handlers:
                        handler = update_wrapper(partial(m, handler=handler), handler)
            resp = await handler(request)
        return resp

    def __call__(self) -> 'Application':
        if False:
            return 10
        'gunicorn compatibility'
        return self

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'<Application 0x{id(self):x}>'

    def __bool__(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

class CleanupError(RuntimeError):

    @property
    def exceptions(self) -> List[BaseException]:
        if False:
            return 10
        return cast(List[BaseException], self.args[1])
if TYPE_CHECKING:
    _CleanupContextBase = FrozenList[Callable[[Application], AsyncIterator[None]]]
else:
    _CleanupContextBase = FrozenList

class CleanupContext(_CleanupContextBase):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._exits: List[AsyncIterator[None]] = []

    async def _on_startup(self, app: Application) -> None:
        for cb in self:
            it = cb(app).__aiter__()
            await it.__anext__()
            self._exits.append(it)

    async def _on_cleanup(self, app: Application) -> None:
        errors = []
        for it in reversed(self._exits):
            try:
                await it.__anext__()
            except StopAsyncIteration:
                pass
            except Exception as exc:
                errors.append(exc)
            else:
                errors.append(RuntimeError(f"{it!r} has more than one 'yield'"))
        if errors:
            if len(errors) == 1:
                raise errors[0]
            else:
                raise CleanupError('Multiple errors on cleanup stage', errors)