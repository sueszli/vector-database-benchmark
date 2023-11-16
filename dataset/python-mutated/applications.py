from __future__ import annotations
import typing
import warnings
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket
AppType = typing.TypeVar('AppType', bound='Starlette')

class Starlette:
    """
    Creates an application instance.

    **Parameters:**

    * **debug** - Boolean indicating if debug tracebacks should be returned on errors.
    * **routes** - A list of routes to serve incoming HTTP and WebSocket requests.
    * **middleware** - A list of middleware to run for every request. A starlette
    application will always automatically include two middleware classes.
    `ServerErrorMiddleware` is added as the very outermost middleware, to handle
    any uncaught errors occurring anywhere in the entire stack.
    `ExceptionMiddleware` is added as the very innermost middleware, to deal
    with handled exception cases occurring in the routing or endpoints.
    * **exception_handlers** - A mapping of either integer status codes,
    or exception class types onto callables which handle the exceptions.
    Exception handler callables should be of the form
    `handler(request, exc) -> response` and may be either standard functions, or
    async functions.
    * **on_startup** - A list of callables to run on application startup.
    Startup handler callables do not take any arguments, and may be be either
    standard functions, or async functions.
    * **on_shutdown** - A list of callables to run on application shutdown.
    Shutdown handler callables do not take any arguments, and may be be either
    standard functions, or async functions.
    * **lifespan** - A lifespan context function, which can be used to perform
    startup and shutdown tasks. This is a newer style that replaces the
    `on_startup` and `on_shutdown` handlers. Use one or the other, not both.
    """

    def __init__(self: 'AppType', debug: bool=False, routes: typing.Sequence[BaseRoute] | None=None, middleware: typing.Sequence[Middleware] | None=None, exception_handlers: typing.Mapping[typing.Any, ExceptionHandler] | None=None, on_startup: typing.Sequence[typing.Callable[[], typing.Any]] | None=None, on_shutdown: typing.Sequence[typing.Callable[[], typing.Any]] | None=None, lifespan: typing.Optional[Lifespan['AppType']]=None) -> None:
        if False:
            i = 10
            return i + 15
        assert lifespan is None or (on_startup is None and on_shutdown is None), "Use either 'lifespan' or 'on_startup'/'on_shutdown', not both."
        self.debug = debug
        self.state = State()
        self.router = Router(routes, on_startup=on_startup, on_shutdown=on_shutdown, lifespan=lifespan)
        self.exception_handlers = {} if exception_handlers is None else dict(exception_handlers)
        self.user_middleware = [] if middleware is None else list(middleware)
        self.middleware_stack: typing.Optional[ASGIApp] = None

    def build_middleware_stack(self) -> ASGIApp:
        if False:
            print('Hello World!')
        debug = self.debug
        error_handler = None
        exception_handlers: typing.Dict[typing.Any, typing.Callable[[Request, Exception], Response]] = {}
        for (key, value) in self.exception_handlers.items():
            if key in (500, Exception):
                error_handler = value
            else:
                exception_handlers[key] = value
        middleware = [Middleware(ServerErrorMiddleware, handler=error_handler, debug=debug)] + self.user_middleware + [Middleware(ExceptionMiddleware, handlers=exception_handlers, debug=debug)]
        app = self.router
        for (cls, options) in reversed(middleware):
            app = cls(app=app, **options)
        return app

    @property
    def routes(self) -> typing.List[BaseRoute]:
        if False:
            while True:
                i = 10
        return self.router.routes

    def url_path_for(self, name: str, /, **path_params: typing.Any) -> URLPath:
        if False:
            print('Hello World!')
        return self.router.url_path_for(name, **path_params)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope['app'] = self
        if self.middleware_stack is None:
            self.middleware_stack = self.build_middleware_stack()
        await self.middleware_stack(scope, receive, send)

    def on_event(self, event_type: str) -> typing.Callable:
        if False:
            print('Hello World!')
        return self.router.on_event(event_type)

    def mount(self, path: str, app: ASGIApp, name: str | None=None) -> None:
        if False:
            while True:
                i = 10
        self.router.mount(path, app=app, name=name)

    def host(self, host: str, app: ASGIApp, name: str | None=None) -> None:
        if False:
            while True:
                i = 10
        self.router.host(host, app=app, name=name)

    def add_middleware(self, middleware_class: type, **options: typing.Any) -> None:
        if False:
            while True:
                i = 10
        if self.middleware_stack is not None:
            raise RuntimeError('Cannot add middleware after an application has started')
        self.user_middleware.insert(0, Middleware(middleware_class, **options))

    def add_exception_handler(self, exc_class_or_status_code: int | typing.Type[Exception], handler: ExceptionHandler) -> None:
        if False:
            i = 10
            return i + 15
        self.exception_handlers[exc_class_or_status_code] = handler

    def add_event_handler(self, event_type: str, func: typing.Callable) -> None:
        if False:
            i = 10
            return i + 15
        self.router.add_event_handler(event_type, func)

    def add_route(self, path: str, route: typing.Callable[[Request], typing.Awaitable[Response] | Response], methods: typing.Optional[typing.List[str]]=None, name: typing.Optional[str]=None, include_in_schema: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.router.add_route(path, route, methods=methods, name=name, include_in_schema=include_in_schema)

    def add_websocket_route(self, path: str, route: typing.Callable[[WebSocket], typing.Awaitable[None]], name: str | None=None) -> None:
        if False:
            return 10
        self.router.add_websocket_route(path, route, name=name)

    def exception_handler(self, exc_class_or_status_code: int | typing.Type[Exception]) -> typing.Callable:
        if False:
            while True:
                i = 10
        warnings.warn('The `exception_handler` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/exceptions/ for the recommended approach.', DeprecationWarning)

        def decorator(func: typing.Callable) -> typing.Callable:
            if False:
                return 10
            self.add_exception_handler(exc_class_or_status_code, func)
            return func
        return decorator

    def route(self, path: str, methods: typing.List[str] | None=None, name: str | None=None, include_in_schema: bool=True) -> typing.Callable:
        if False:
            return 10
        '\n        We no longer document this decorator style API, and its usage is discouraged.\n        Instead you should use the following approach:\n\n        >>> routes = [Route(path, endpoint=...), ...]\n        >>> app = Starlette(routes=routes)\n        '
        warnings.warn('The `route` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/routing/ for the recommended approach.', DeprecationWarning)

        def decorator(func: typing.Callable) -> typing.Callable:
            if False:
                i = 10
                return i + 15
            self.router.add_route(path, func, methods=methods, name=name, include_in_schema=include_in_schema)
            return func
        return decorator

    def websocket_route(self, path: str, name: str | None=None) -> typing.Callable:
        if False:
            return 10
        '\n        We no longer document this decorator style API, and its usage is discouraged.\n        Instead you should use the following approach:\n\n        >>> routes = [WebSocketRoute(path, endpoint=...), ...]\n        >>> app = Starlette(routes=routes)\n        '
        warnings.warn('The `websocket_route` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/routing/#websocket-routing for the recommended approach.', DeprecationWarning)

        def decorator(func: typing.Callable) -> typing.Callable:
            if False:
                for i in range(10):
                    print('nop')
            self.router.add_websocket_route(path, func, name=name)
            return func
        return decorator

    def middleware(self, middleware_type: str) -> typing.Callable:
        if False:
            return 10
        '\n        We no longer document this decorator style API, and its usage is discouraged.\n        Instead you should use the following approach:\n\n        >>> middleware = [Middleware(...), ...]\n        >>> app = Starlette(middleware=middleware)\n        '
        warnings.warn('The `middleware` decorator is deprecated, and will be removed in version 1.0.0. Refer to https://www.starlette.io/middleware/#using-middleware for recommended approach.', DeprecationWarning)
        assert middleware_type == 'http', 'Currently only middleware("http") is supported.'

        def decorator(func: typing.Callable) -> typing.Callable:
            if False:
                i = 10
                return i + 15
            self.add_middleware(BaseHTTPMiddleware, dispatch=func)
            return func
        return decorator