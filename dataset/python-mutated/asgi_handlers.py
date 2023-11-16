from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from litestar.exceptions import ImproperlyConfiguredException
from litestar.handlers.base import BaseRouteHandler
from litestar.types.builtin_types import NoneType
from litestar.utils.predicates import is_async_callable
__all__ = ('ASGIRouteHandler', 'asgi')
if TYPE_CHECKING:
    from litestar.types import ExceptionHandlersMap, Guard, MaybePartial

class ASGIRouteHandler(BaseRouteHandler):
    """ASGI Route Handler decorator.

    Use this decorator to decorate ASGI applications.
    """
    __slots__ = ('is_mount', 'is_static')

    def __init__(self, path: str | Sequence[str] | None=None, *, exception_handlers: ExceptionHandlersMap | None=None, guards: Sequence[Guard] | None=None, name: str | None=None, opt: Mapping[str, Any] | None=None, is_mount: bool=False, is_static: bool=False, signature_namespace: Mapping[str, Any] | None=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Initialize ``ASGIRouteHandler``.\n\n        Args:\n            exception_handlers: A mapping of status codes and/or exception types to handler functions.\n            guards: A sequence of :class:`Guard <.types.Guard>` callables.\n            name: A string identifying the route handler.\n            opt: A string key mapping of arbitrary values that can be accessed in :class:`Guards <.types.Guard>` or\n                wherever you have access to :class:`Request <.connection.Request>` or\n                :class:`ASGI Scope <.types.Scope>`.\n            path: A path fragment for the route handler function or a list of path fragments. If not given defaults to\n                ``/``\n            is_mount: A boolean dictating whether the handler's paths should be regarded as mount paths. Mount path\n                accept any arbitrary paths that begin with the defined prefixed path. For example, a mount with the path\n                ``/some-path/`` will accept requests for ``/some-path/`` and any sub path under this, e.g.\n                ``/some-path/sub-path/`` etc.\n            is_static: A boolean dictating whether the handler's paths should be regarded as static paths. Static paths\n                are used to deliver static files.\n            signature_namespace: A mapping of names to types for use in forward reference resolution during signature modelling.\n            type_encoders: A mapping of types to callables that transform them into types supported for serialization.\n            **kwargs: Any additional kwarg - will be set in the opt dictionary.\n        "
        self.is_mount = is_mount or is_static
        self.is_static = is_static
        super().__init__(path, exception_handlers=exception_handlers, guards=guards, name=name, opt=opt, signature_namespace=signature_namespace, **kwargs)

    def _validate_handler_function(self) -> None:
        if False:
            i = 10
            return i + 15
        "Validate the route handler function once it's set by inspecting its return annotations."
        super()._validate_handler_function()
        if not self.parsed_fn_signature.return_type.is_subclass_of(NoneType):
            raise ImproperlyConfiguredException("ASGI handler functions should return 'None'")
        if any((key not in self.parsed_fn_signature.parameters for key in ('scope', 'send', 'receive'))):
            raise ImproperlyConfiguredException("ASGI handler functions should define 'scope', 'send' and 'receive' arguments")
        if not is_async_callable(self.fn):
            raise ImproperlyConfiguredException("Functions decorated with 'asgi' must be async functions")
asgi = ASGIRouteHandler