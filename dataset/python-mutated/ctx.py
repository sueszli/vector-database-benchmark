from __future__ import annotations
import contextvars
import sys
import typing as t
from functools import update_wrapper
from types import TracebackType
from werkzeug.exceptions import HTTPException
from . import typing as ft
from .globals import _cv_app
from .globals import _cv_request
from .signals import appcontext_popped
from .signals import appcontext_pushed
if t.TYPE_CHECKING:
    from .app import Flask
    from .sessions import SessionMixin
    from .wrappers import Request
_sentinel = object()

class _AppCtxGlobals:
    """A plain object. Used as a namespace for storing data during an
    application context.

    Creating an app context automatically creates this object, which is
    made available as the :data:`g` proxy.

    .. describe:: 'key' in g

        Check whether an attribute is present.

        .. versionadded:: 0.10

    .. describe:: iter(g)

        Return an iterator over the attribute names.

        .. versionadded:: 0.10
    """

    def __getattr__(self, name: str) -> t.Any:
        if False:
            while True:
                i = 10
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: t.Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.__dict__[name] = value

    def __delattr__(self, name: str) -> None:
        if False:
            print('Hello World!')
        try:
            del self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, name: str, default: t.Any | None=None) -> t.Any:
        if False:
            for i in range(10):
                print('nop')
        'Get an attribute by name, or a default value. Like\n        :meth:`dict.get`.\n\n        :param name: Name of attribute to get.\n        :param default: Value to return if the attribute is not present.\n\n        .. versionadded:: 0.10\n        '
        return self.__dict__.get(name, default)

    def pop(self, name: str, default: t.Any=_sentinel) -> t.Any:
        if False:
            while True:
                i = 10
        'Get and remove an attribute by name. Like :meth:`dict.pop`.\n\n        :param name: Name of attribute to pop.\n        :param default: Value to return if the attribute is not present,\n            instead of raising a ``KeyError``.\n\n        .. versionadded:: 0.11\n        '
        if default is _sentinel:
            return self.__dict__.pop(name)
        else:
            return self.__dict__.pop(name, default)

    def setdefault(self, name: str, default: t.Any=None) -> t.Any:
        if False:
            for i in range(10):
                print('nop')
        'Get the value of an attribute if it is present, otherwise\n        set and return a default value. Like :meth:`dict.setdefault`.\n\n        :param name: Name of attribute to get.\n        :param default: Value to set and return if the attribute is not\n            present.\n\n        .. versionadded:: 0.11\n        '
        return self.__dict__.setdefault(name, default)

    def __contains__(self, item: str) -> bool:
        if False:
            return 10
        return item in self.__dict__

    def __iter__(self) -> t.Iterator[str]:
        if False:
            return 10
        return iter(self.__dict__)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        ctx = _cv_app.get(None)
        if ctx is not None:
            return f"<flask.g of '{ctx.app.name}'>"
        return object.__repr__(self)

def after_this_request(f: ft.AfterRequestCallable) -> ft.AfterRequestCallable:
    if False:
        return 10
    "Executes a function after this request.  This is useful to modify\n    response objects.  The function is passed the response object and has\n    to return the same or a new one.\n\n    Example::\n\n        @app.route('/')\n        def index():\n            @after_this_request\n            def add_header(response):\n                response.headers['X-Foo'] = 'Parachute'\n                return response\n            return 'Hello World!'\n\n    This is more useful if a function other than the view function wants to\n    modify a response.  For instance think of a decorator that wants to add\n    some headers without converting the return value into a response object.\n\n    .. versionadded:: 0.9\n    "
    ctx = _cv_request.get(None)
    if ctx is None:
        raise RuntimeError("'after_this_request' can only be used when a request context is active, such as in a view function.")
    ctx._after_request_functions.append(f)
    return f

def copy_current_request_context(f: t.Callable) -> t.Callable:
    if False:
        while True:
            i = 10
    "A helper function that decorates a function to retain the current\n    request context.  This is useful when working with greenlets.  The moment\n    the function is decorated a copy of the request context is created and\n    then pushed when the function is called.  The current session is also\n    included in the copied request context.\n\n    Example::\n\n        import gevent\n        from flask import copy_current_request_context\n\n        @app.route('/')\n        def index():\n            @copy_current_request_context\n            def do_some_work():\n                # do some work here, it can access flask.request or\n                # flask.session like you would otherwise in the view function.\n                ...\n            gevent.spawn(do_some_work)\n            return 'Regular response'\n\n    .. versionadded:: 0.10\n    "
    ctx = _cv_request.get(None)
    if ctx is None:
        raise RuntimeError("'copy_current_request_context' can only be used when a request context is active, such as in a view function.")
    ctx = ctx.copy()

    def wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        with ctx:
            return ctx.app.ensure_sync(f)(*args, **kwargs)
    return update_wrapper(wrapper, f)

def has_request_context() -> bool:
    if False:
        print('Hello World!')
    'If you have code that wants to test if a request context is there or\n    not this function can be used.  For instance, you may want to take advantage\n    of request information if the request object is available, but fail\n    silently if it is unavailable.\n\n    ::\n\n        class User(db.Model):\n\n            def __init__(self, username, remote_addr=None):\n                self.username = username\n                if remote_addr is None and has_request_context():\n                    remote_addr = request.remote_addr\n                self.remote_addr = remote_addr\n\n    Alternatively you can also just test any of the context bound objects\n    (such as :class:`request` or :class:`g`) for truthness::\n\n        class User(db.Model):\n\n            def __init__(self, username, remote_addr=None):\n                self.username = username\n                if remote_addr is None and request:\n                    remote_addr = request.remote_addr\n                self.remote_addr = remote_addr\n\n    .. versionadded:: 0.7\n    '
    return _cv_request.get(None) is not None

def has_app_context() -> bool:
    if False:
        while True:
            i = 10
    'Works like :func:`has_request_context` but for the application\n    context.  You can also just do a boolean check on the\n    :data:`current_app` object instead.\n\n    .. versionadded:: 0.9\n    '
    return _cv_app.get(None) is not None

class AppContext:
    """The app context contains application-specific information. An app
    context is created and pushed at the beginning of each request if
    one is not already active. An app context is also pushed when
    running CLI commands.
    """

    def __init__(self, app: Flask) -> None:
        if False:
            return 10
        self.app = app
        self.url_adapter = app.create_url_adapter(None)
        self.g: _AppCtxGlobals = app.app_ctx_globals_class()
        self._cv_tokens: list[contextvars.Token] = []

    def push(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Binds the app context to the current context.'
        self._cv_tokens.append(_cv_app.set(self))
        appcontext_pushed.send(self.app, _async_wrapper=self.app.ensure_sync)

    def pop(self, exc: BaseException | None=_sentinel) -> None:
        if False:
            print('Hello World!')
        'Pops the app context.'
        try:
            if len(self._cv_tokens) == 1:
                if exc is _sentinel:
                    exc = sys.exc_info()[1]
                self.app.do_teardown_appcontext(exc)
        finally:
            ctx = _cv_app.get()
            _cv_app.reset(self._cv_tokens.pop())
        if ctx is not self:
            raise AssertionError(f'Popped wrong app context. ({ctx!r} instead of {self!r})')
        appcontext_popped.send(self.app, _async_wrapper=self.app.ensure_sync)

    def __enter__(self) -> AppContext:
        if False:
            i = 10
            return i + 15
        self.push()
        return self

    def __exit__(self, exc_type: type | None, exc_value: BaseException | None, tb: TracebackType | None) -> None:
        if False:
            return 10
        self.pop(exc_value)

class RequestContext:
    """The request context contains per-request information. The Flask
    app creates and pushes it at the beginning of the request, then pops
    it at the end of the request. It will create the URL adapter and
    request object for the WSGI environment provided.

    Do not attempt to use this class directly, instead use
    :meth:`~flask.Flask.test_request_context` and
    :meth:`~flask.Flask.request_context` to create this object.

    When the request context is popped, it will evaluate all the
    functions registered on the application for teardown execution
    (:meth:`~flask.Flask.teardown_request`).

    The request context is automatically popped at the end of the
    request. When using the interactive debugger, the context will be
    restored so ``request`` is still accessible. Similarly, the test
    client can preserve the context after the request ends. However,
    teardown functions may already have closed some resources such as
    database connections.
    """

    def __init__(self, app: Flask, environ: dict, request: Request | None=None, session: SessionMixin | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.app = app
        if request is None:
            request = app.request_class(environ)
            request.json_module = app.json
        self.request: Request = request
        self.url_adapter = None
        try:
            self.url_adapter = app.create_url_adapter(self.request)
        except HTTPException as e:
            self.request.routing_exception = e
        self.flashes: list[tuple[str, str]] | None = None
        self.session: SessionMixin | None = session
        self._after_request_functions: list[ft.AfterRequestCallable] = []
        self._cv_tokens: list[tuple[contextvars.Token, AppContext | None]] = []

    def copy(self) -> RequestContext:
        if False:
            return 10
        'Creates a copy of this request context with the same request object.\n        This can be used to move a request context to a different greenlet.\n        Because the actual request object is the same this cannot be used to\n        move a request context to a different thread unless access to the\n        request object is locked.\n\n        .. versionadded:: 0.10\n\n        .. versionchanged:: 1.1\n           The current session object is used instead of reloading the original\n           data. This prevents `flask.session` pointing to an out-of-date object.\n        '
        return self.__class__(self.app, environ=self.request.environ, request=self.request, session=self.session)

    def match_request(self) -> None:
        if False:
            while True:
                i = 10
        'Can be overridden by a subclass to hook into the matching\n        of the request.\n        '
        try:
            result = self.url_adapter.match(return_rule=True)
            (self.request.url_rule, self.request.view_args) = result
        except HTTPException as e:
            self.request.routing_exception = e

    def push(self) -> None:
        if False:
            print('Hello World!')
        app_ctx = _cv_app.get(None)
        if app_ctx is None or app_ctx.app is not self.app:
            app_ctx = self.app.app_context()
            app_ctx.push()
        else:
            app_ctx = None
        self._cv_tokens.append((_cv_request.set(self), app_ctx))
        if self.session is None:
            session_interface = self.app.session_interface
            self.session = session_interface.open_session(self.app, self.request)
            if self.session is None:
                self.session = session_interface.make_null_session(self.app)
        if self.url_adapter is not None:
            self.match_request()

    def pop(self, exc: BaseException | None=_sentinel) -> None:
        if False:
            i = 10
            return i + 15
        'Pops the request context and unbinds it by doing that.  This will\n        also trigger the execution of functions registered by the\n        :meth:`~flask.Flask.teardown_request` decorator.\n\n        .. versionchanged:: 0.9\n           Added the `exc` argument.\n        '
        clear_request = len(self._cv_tokens) == 1
        try:
            if clear_request:
                if exc is _sentinel:
                    exc = sys.exc_info()[1]
                self.app.do_teardown_request(exc)
                request_close = getattr(self.request, 'close', None)
                if request_close is not None:
                    request_close()
        finally:
            ctx = _cv_request.get()
            (token, app_ctx) = self._cv_tokens.pop()
            _cv_request.reset(token)
            if clear_request:
                ctx.request.environ['werkzeug.request'] = None
            if app_ctx is not None:
                app_ctx.pop(exc)
            if ctx is not self:
                raise AssertionError(f'Popped wrong request context. ({ctx!r} instead of {self!r})')

    def __enter__(self) -> RequestContext:
        if False:
            print('Hello World!')
        self.push()
        return self

    def __exit__(self, exc_type: type | None, exc_value: BaseException | None, tb: TracebackType | None) -> None:
        if False:
            print('Hello World!')
        self.pop(exc_value)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<{type(self).__name__} {self.request.url!r} [{self.request.method}] of {self.app.name}>'