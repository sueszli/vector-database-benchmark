import abc
import dataclasses
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Type, Union, overload
from . import hdrs
from .abc import AbstractView
from .typedefs import Handler, PathLike
if TYPE_CHECKING:
    from .web_request import Request
    from .web_response import StreamResponse
    from .web_urldispatcher import AbstractRoute, UrlDispatcher
else:
    Request = StreamResponse = UrlDispatcher = AbstractRoute = None
__all__ = ('AbstractRouteDef', 'RouteDef', 'StaticDef', 'RouteTableDef', 'head', 'options', 'get', 'post', 'patch', 'put', 'delete', 'route', 'view', 'static')

class AbstractRouteDef(abc.ABC):

    @abc.abstractmethod
    def register(self, router: UrlDispatcher) -> List[AbstractRoute]:
        if False:
            print('Hello World!')
        pass
_HandlerType = Union[Type[AbstractView], Handler]

@dataclasses.dataclass(frozen=True, repr=False)
class RouteDef(AbstractRouteDef):
    method: str
    path: str
    handler: _HandlerType
    kwargs: Dict[str, Any]

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        info = []
        for (name, value) in sorted(self.kwargs.items()):
            info.append(f', {name}={value!r}')
        return '<RouteDef {method} {path} -> {handler.__name__!r}{info}>'.format(method=self.method, path=self.path, handler=self.handler, info=''.join(info))

    def register(self, router: UrlDispatcher) -> List[AbstractRoute]:
        if False:
            for i in range(10):
                print('nop')
        if self.method in hdrs.METH_ALL:
            reg = getattr(router, 'add_' + self.method.lower())
            return [reg(self.path, self.handler, **self.kwargs)]
        else:
            return [router.add_route(self.method, self.path, self.handler, **self.kwargs)]

@dataclasses.dataclass(frozen=True, repr=False)
class StaticDef(AbstractRouteDef):
    prefix: str
    path: PathLike
    kwargs: Dict[str, Any]

    def __repr__(self) -> str:
        if False:
            return 10
        info = []
        for (name, value) in sorted(self.kwargs.items()):
            info.append(f', {name}={value!r}')
        return '<StaticDef {prefix} -> {path}{info}>'.format(prefix=self.prefix, path=self.path, info=''.join(info))

    def register(self, router: UrlDispatcher) -> List[AbstractRoute]:
        if False:
            print('Hello World!')
        resource = router.add_static(self.prefix, self.path, **self.kwargs)
        routes = resource.get_info().get('routes', {})
        return list(routes.values())

def route(method: str, path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    if False:
        while True:
            i = 10
    return RouteDef(method, path, handler, kwargs)

def head(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    if False:
        i = 10
        return i + 15
    return route(hdrs.METH_HEAD, path, handler, **kwargs)

def options(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    if False:
        i = 10
        return i + 15
    return route(hdrs.METH_OPTIONS, path, handler, **kwargs)

def get(path: str, handler: _HandlerType, *, name: Optional[str]=None, allow_head: bool=True, **kwargs: Any) -> RouteDef:
    if False:
        i = 10
        return i + 15
    return route(hdrs.METH_GET, path, handler, name=name, allow_head=allow_head, **kwargs)

def post(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    if False:
        for i in range(10):
            print('nop')
    return route(hdrs.METH_POST, path, handler, **kwargs)

def put(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    if False:
        i = 10
        return i + 15
    return route(hdrs.METH_PUT, path, handler, **kwargs)

def patch(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    if False:
        while True:
            i = 10
    return route(hdrs.METH_PATCH, path, handler, **kwargs)

def delete(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    if False:
        i = 10
        return i + 15
    return route(hdrs.METH_DELETE, path, handler, **kwargs)

def view(path: str, handler: Type[AbstractView], **kwargs: Any) -> RouteDef:
    if False:
        i = 10
        return i + 15
    return route(hdrs.METH_ANY, path, handler, **kwargs)

def static(prefix: str, path: PathLike, **kwargs: Any) -> StaticDef:
    if False:
        for i in range(10):
            print('nop')
    return StaticDef(prefix, path, kwargs)
_Deco = Callable[[_HandlerType], _HandlerType]

class RouteTableDef(Sequence[AbstractRouteDef]):
    """Route definition table"""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._items: List[AbstractRouteDef] = []

    def __repr__(self) -> str:
        if False:
            return 10
        return f'<RouteTableDef count={len(self._items)}>'

    @overload
    def __getitem__(self, index: int) -> AbstractRouteDef:
        if False:
            return 10
        ...

    @overload
    def __getitem__(self, index: slice) -> List[AbstractRouteDef]:
        if False:
            return 10
        ...

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return self._items[index]

    def __iter__(self) -> Iterator[AbstractRouteDef]:
        if False:
            print('Hello World!')
        return iter(self._items)

    def __len__(self) -> int:
        if False:
            return 10
        return len(self._items)

    def __contains__(self, item: object) -> bool:
        if False:
            while True:
                i = 10
        return item in self._items

    def route(self, method: str, path: str, **kwargs: Any) -> _Deco:
        if False:
            i = 10
            return i + 15

        def inner(handler: _HandlerType) -> _HandlerType:
            if False:
                for i in range(10):
                    print('nop')
            self._items.append(RouteDef(method, path, handler, kwargs))
            return handler
        return inner

    def head(self, path: str, **kwargs: Any) -> _Deco:
        if False:
            for i in range(10):
                print('nop')
        return self.route(hdrs.METH_HEAD, path, **kwargs)

    def get(self, path: str, **kwargs: Any) -> _Deco:
        if False:
            return 10
        return self.route(hdrs.METH_GET, path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> _Deco:
        if False:
            return 10
        return self.route(hdrs.METH_POST, path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> _Deco:
        if False:
            print('Hello World!')
        return self.route(hdrs.METH_PUT, path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> _Deco:
        if False:
            i = 10
            return i + 15
        return self.route(hdrs.METH_PATCH, path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> _Deco:
        if False:
            print('Hello World!')
        return self.route(hdrs.METH_DELETE, path, **kwargs)

    def options(self, path: str, **kwargs: Any) -> _Deco:
        if False:
            for i in range(10):
                print('nop')
        return self.route(hdrs.METH_OPTIONS, path, **kwargs)

    def view(self, path: str, **kwargs: Any) -> _Deco:
        if False:
            print('Hello World!')
        return self.route(hdrs.METH_ANY, path, **kwargs)

    def static(self, prefix: str, path: PathLike, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._items.append(StaticDef(prefix, path, kwargs))