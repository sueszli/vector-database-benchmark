import builtins
import types
import typing
from collections.abc import Awaitable
from types import TracebackType
from typing import Any, Type
import _typeshed
import typing_extensions
from _typeshed import Unused

class GoodOne:

    def __exit__(self, *args: object) -> None:
        if False:
            print('Hello World!')
        ...

    async def __aexit__(self, *args) -> str:
        ...

class GoodTwo:

    def __exit__(self, typ: type[builtins.BaseException] | None, *args: builtins.object) -> bool | None:
        if False:
            while True:
                i = 10
        ...

    async def __aexit__(self, /, typ: Type[BaseException] | None, *args: object, **kwargs) -> bool:
        ...

class GoodThree:

    def __exit__(self, __typ: typing.Type[BaseException] | None, exc: BaseException | None, *args: object) -> None:
        if False:
            return 10
        ...

    async def __aexit__(self, typ: typing_extensions.Type[BaseException] | None, __exc: BaseException | None, *args: object) -> None:
        ...

class GoodFour:

    def __exit__(self, typ: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        if False:
            while True:
                i = 10
        ...

    async def __aexit__(self, typ: type[BaseException] | None, exc: BaseException | None, tb: types.TracebackType | None, *args: list[None]) -> None:
        ...

class GoodFive:

    def __exit__(self, typ: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None, weird_extra_arg: int=..., *args: int, **kwargs: str) -> None:
        if False:
            return 10
        ...

    async def __aexit__(self, typ: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> Awaitable[None]:
        ...

class GoodSix:

    def __exit__(self, typ: object, exc: builtins.object, tb: object) -> None:
        if False:
            return 10
        ...

    async def __aexit__(self, typ: object, exc: object, tb: builtins.object) -> None:
        ...

class GoodSeven:

    def __exit__(self, *args: Unused) -> bool:
        if False:
            print('Hello World!')
        ...

    async def __aexit__(self, typ: Type[BaseException] | None, *args: _typeshed.Unused) -> Awaitable[None]:
        ...

class GoodEight:

    def __exit__(self, __typ: typing.Type[BaseException] | None, exc: BaseException | None, *args: _typeshed.Unused) -> bool:
        if False:
            print('Hello World!')
        ...

    async def __aexit__(self, typ: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None, weird_extra_arg: int=..., *args: Unused, **kwargs: Unused) -> Awaitable[None]:
        ...

class GoodNine:

    def __exit__(self, __typ: typing.Union[typing.Type[BaseException], None], exc: typing.Union[BaseException, None], *args: _typeshed.Unused) -> bool:
        if False:
            return 10
        ...

    async def __aexit__(self, typ: typing.Union[typing.Type[BaseException], None], exc: typing.Union[BaseException, None], tb: typing.Union[TracebackType, None], weird_extra_arg: int=..., *args: Unused, **kwargs: Unused) -> Awaitable[None]:
        ...

class GoodTen:

    def __exit__(self, __typ: typing.Optional[typing.Type[BaseException]], exc: typing.Optional[BaseException], *args: _typeshed.Unused) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

    async def __aexit__(self, typ: typing.Optional[typing.Type[BaseException]], exc: typing.Optional[BaseException], tb: typing.Optional[TracebackType], weird_extra_arg: int=..., *args: Unused, **kwargs: Unused) -> Awaitable[None]:
        ...

class BadOne:

    def __exit__(self, *args: Any) -> None:
        if False:
            print('Hello World!')
        ...

    async def __aexit__(self) -> None:
        ...

class BadTwo:

    def __exit__(self, typ, exc, tb, weird_extra_arg) -> None:
        if False:
            while True:
                i = 10
        ...

    async def __aexit__(self, typ, exc, tb, *, weird_extra_arg) -> None:
        ...

class BadThree:

    def __exit__(self, typ: type[BaseException], exc: BaseException | None, tb: TracebackType | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    async def __aexit__(self, __typ: type[BaseException] | None, __exc: BaseException, __tb: TracebackType) -> bool | None:
        ...

class BadFour:

    def __exit__(self, typ: typing.Optional[type[BaseException]], exc: typing.Union[BaseException, None], tb: TracebackType) -> None:
        if False:
            print('Hello World!')
        ...

    async def __aexit__(self, __typ: type[BaseException] | None, __exc: BaseException | None, __tb: typing.Union[TracebackType, None, int]) -> bool | None:
        ...

class BadFive:

    def __exit__(self, typ: BaseException | None, *args: list[str]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

    async def __aexit__(self, /, typ: type[BaseException] | None, *args: Any) -> Awaitable[None]:
        ...

class BadSix:

    def __exit__(self, typ, exc, tb, weird_extra_arg, extra_arg2=None) -> None:
        if False:
            print('Hello World!')
        ...

    async def __aexit__(self, typ, exc, tb, *, weird_extra_arg) -> None:
        ...