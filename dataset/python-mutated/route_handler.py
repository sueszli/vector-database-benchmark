from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping
from litestar.exceptions import ImproperlyConfiguredException
from litestar.handlers import BaseRouteHandler
from litestar.types.builtin_types import NoneType
from litestar.utils.predicates import is_async_callable
if TYPE_CHECKING:
    from litestar.types import Dependencies, ExceptionHandler, Guard, Middleware

class WebsocketRouteHandler(BaseRouteHandler):
    """Websocket route handler decorator.

    Use this decorator to decorate websocket handler functions.
    """

    def __init__(self, path: str | None | list[str] | None=None, *, dependencies: Dependencies | None=None, exception_handlers: dict[int | type[Exception], ExceptionHandler] | None=None, guards: list[Guard] | None=None, middleware: list[Middleware] | None=None, name: str | None=None, opt: dict[str, Any] | None=None, signature_namespace: Mapping[str, Any] | None=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize ``WebsocketRouteHandler``\n\n        Args:\n            path: A path fragment for the route handler function or a sequence of path fragments. If not given defaults\n                to ``/``\n            dependencies: A string keyed mapping of dependency :class:`Provider <.di.Provide>` instances.\n            exception_handlers: A mapping of status codes and/or exception types to handler functions.\n            guards: A sequence of :class:`Guard <.types.Guard>` callables.\n            middleware: A sequence of :class:`Middleware <.types.Middleware>`.\n            name: A string identifying the route handler.\n            opt: A string keyed mapping of arbitrary values that can be accessed in :class:`Guards <.types.Guard>` or\n                wherever you have access to :class:`Request <.connection.Request>` or\n                :class:`ASGI Scope <.types.Scope>`.\n            signature_namespace: A mapping of names to types for use in forward reference resolution during signature modelling.\n            type_encoders: A mapping of types to callables that transform them into types supported for serialization.\n            **kwargs: Any additional kwarg - will be set in the opt dictionary.\n        '
        super().__init__(path=path, dependencies=dependencies, exception_handlers=exception_handlers, guards=guards, middleware=middleware, name=name, opt=opt, signature_namespace=signature_namespace, **kwargs)

    def _validate_handler_function(self) -> None:
        if False:
            while True:
                i = 10
        "Validate the route handler function once it's set by inspecting its return annotations."
        super()._validate_handler_function()
        if not self.parsed_fn_signature.return_type.is_subclass_of(NoneType):
            raise ImproperlyConfiguredException("Websocket handler functions should return 'None'")
        if 'socket' not in self.parsed_fn_signature.parameters:
            raise ImproperlyConfiguredException("Websocket handlers must set a 'socket' kwarg")
        for param in ('request', 'body', 'data'):
            if param in self.parsed_fn_signature.parameters:
                raise ImproperlyConfiguredException(f'The {param} kwarg is not supported with websocket handlers')
        if not is_async_callable(self.fn):
            raise ImproperlyConfiguredException("Functions decorated with 'websocket' must be async functions")
websocket = WebsocketRouteHandler