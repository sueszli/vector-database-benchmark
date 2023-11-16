from __future__ import annotations
import sys
from asyncio import BaseTransport
from typing import TYPE_CHECKING, Any, AnyStr, Optional
if TYPE_CHECKING:
    from sanic.http.constants import HTTP
    from sanic.models.asgi import ASGIScope
if sys.version_info < (3, 8):
    Range = Any
    HTMLProtocol = Any
else:
    from typing import Protocol

    class HTMLProtocol(Protocol):

        def __html__(self) -> AnyStr:
            if False:
                print('Hello World!')
            ...

        def _repr_html_(self) -> AnyStr:
            if False:
                i = 10
                return i + 15
            ...

    class Range(Protocol):
        start: Optional[int]
        end: Optional[int]
        size: Optional[int]
        total: Optional[int]
        __slots__ = ()

class TransportProtocol(BaseTransport):
    scope: ASGIScope
    version: HTTP
    __slots__ = ()