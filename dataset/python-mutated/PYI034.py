import abc
import builtins
import collections.abc
import enum
import typing
from abc import ABCMeta, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from enum import EnumMeta
from typing import Any, overload
import typing_extensions
from _typeshed import Self
from typing_extensions import final

class Bad(object):

    def __new__(cls, *args: Any, **kwargs: Any) -> Bad:
        if False:
            while True:
                i = 10
        ...

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __str__(self) -> builtins.str:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        ...

    def __ne__(self, other: typing.Any) -> typing.Any:
        if False:
            return 10
        ...

    def __enter__(self) -> Bad:
        if False:
            for i in range(10):
                print('nop')
        ...

    async def __aenter__(self) -> Bad:
        ...

    def __iadd__(self, other: Bad) -> Bad:
        if False:
            i = 10
            return i + 15
        ...

class AlsoBad(int, builtins.object):
    ...

class Good:

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        if False:
            for i in range(10):
                print('nop')
        ...

    @abstractmethod
    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        ...

    def __eq__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __ne__(self, obj: object) -> int:
        if False:
            return 10
        ...

    def __enter__(self: Self) -> Self:
        if False:
            while True:
                i = 10
        ...

    async def __aenter__(self: Self) -> Self:
        ...

    def __ior__(self: Self, other: Self) -> Self:
        if False:
            return 10
        ...

class Fine:

    @overload
    def __new__(cls, foo: int) -> FineSubclass:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __new__(cls, *args: Any, **kwargs: Any) -> Fine:
        if False:
            while True:
                i = 10
        ...

    @abc.abstractmethod
    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        if False:
            return 10
        ...

    def __eq__(self, other: Any, strange_extra_arg: list[str]) -> Any:
        if False:
            i = 10
            return i + 15
        ...

    def __ne__(self, *, kw_only_other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __enter__(self) -> None:
        if False:
            print('Hello World!')
        ...

    async def __aenter__(self) -> bool:
        ...

class FineSubclass(Fine):
    ...

class StrangeButAcceptable(str):

    @typing_extensions.overload
    def __new__(cls, foo: int) -> StrangeButAcceptableSubclass:
        if False:
            for i in range(10):
                print('nop')
        ...

    @typing_extensions.overload
    def __new__(cls, *args: Any, **kwargs: Any) -> StrangeButAcceptable:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __str__(self) -> StrangeButAcceptable:
        if False:
            while True:
                i = 10
        ...

    def __repr__(self) -> StrangeButAcceptable:
        if False:
            return 10
        ...

class StrangeButAcceptableSubclass(StrangeButAcceptable):
    ...

class FineAndDandy:

    def __str__(self, weird_extra_arg) -> str:
        if False:
            return 10
        ...

    def __repr__(self, weird_extra_arg_with_default=...) -> str:
        if False:
            while True:
                i = 10
        ...

@final
class WillNotBeSubclassed:

    def __new__(cls, *args: Any, **kwargs: Any) -> WillNotBeSubclassed:
        if False:
            i = 10
            return i + 15
        ...

    def __enter__(self) -> WillNotBeSubclassed:
        if False:
            return 10
        ...

    async def __aenter__(self) -> WillNotBeSubclassed:
        ...

class InvalidButPluginDoesNotCrash:

    def __new__() -> InvalidButPluginDoesNotCrash:
        if False:
            i = 10
            return i + 15
        ...

    def __enter__() -> InvalidButPluginDoesNotCrash:
        if False:
            return 10
        ...

    async def __aenter__() -> InvalidButPluginDoesNotCrash:
        ...

class BadIterator1(Iterator[int]):

    def __iter__(self) -> Iterator[int]:
        if False:
            print('Hello World!')
        ...

class BadIterator2(typing.Iterator[int]):

    def __iter__(self) -> Iterator[int]:
        if False:
            while True:
                i = 10
        ...

class BadIterator3(typing.Iterator[int]):

    def __iter__(self) -> collections.abc.Iterator[int]:
        if False:
            while True:
                i = 10
        ...

class BadIterator4(Iterator[int]):

    def __iter__(self) -> Iterable[int]:
        if False:
            return 10
        ...

class IteratorReturningIterable:

    def __iter__(self) -> Iterable[str]:
        if False:
            return 10
        ...

class BadAsyncIterator(collections.abc.AsyncIterator[str]):

    def __aiter__(self) -> typing.AsyncIterator[str]:
        if False:
            return 10
        ...

class AsyncIteratorReturningAsyncIterable:

    def __aiter__(self) -> AsyncIterable[str]:
        if False:
            print('Hello World!')
        ...

class MetaclassInWhichSelfCannotBeUsed(type):

    def __new__(cls) -> MetaclassInWhichSelfCannotBeUsed:
        if False:
            while True:
                i = 10
        ...

    def __enter__(self) -> MetaclassInWhichSelfCannotBeUsed:
        if False:
            print('Hello World!')
        ...

    async def __aenter__(self) -> MetaclassInWhichSelfCannotBeUsed:
        ...

    def __isub__(self, other: MetaclassInWhichSelfCannotBeUsed) -> MetaclassInWhichSelfCannotBeUsed:
        if False:
            print('Hello World!')
        ...

class MetaclassInWhichSelfCannotBeUsed2(EnumMeta):

    def __new__(cls) -> MetaclassInWhichSelfCannotBeUsed2:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __enter__(self) -> MetaclassInWhichSelfCannotBeUsed2:
        if False:
            return 10
        ...

    async def __aenter__(self) -> MetaclassInWhichSelfCannotBeUsed2:
        ...

    def __isub__(self, other: MetaclassInWhichSelfCannotBeUsed2) -> MetaclassInWhichSelfCannotBeUsed2:
        if False:
            print('Hello World!')
        ...

class MetaclassInWhichSelfCannotBeUsed3(enum.EnumType):

    def __new__(cls) -> MetaclassInWhichSelfCannotBeUsed3:
        if False:
            print('Hello World!')
        ...

    def __enter__(self) -> MetaclassInWhichSelfCannotBeUsed3:
        if False:
            i = 10
            return i + 15
        ...

    async def __aenter__(self) -> MetaclassInWhichSelfCannotBeUsed3:
        ...

    def __isub__(self, other: MetaclassInWhichSelfCannotBeUsed3) -> MetaclassInWhichSelfCannotBeUsed3:
        if False:
            print('Hello World!')
        ...

class MetaclassInWhichSelfCannotBeUsed4(ABCMeta):

    def __new__(cls) -> MetaclassInWhichSelfCannotBeUsed4:
        if False:
            return 10
        ...

    def __enter__(self) -> MetaclassInWhichSelfCannotBeUsed4:
        if False:
            while True:
                i = 10
        ...

    async def __aenter__(self) -> MetaclassInWhichSelfCannotBeUsed4:
        ...

    def __isub__(self, other: MetaclassInWhichSelfCannotBeUsed4) -> MetaclassInWhichSelfCannotBeUsed4:
        if False:
            i = 10
            return i + 15
        ...

class Abstract(Iterator[str]):

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        ...

    @abstractmethod
    def __enter__(self) -> Abstract:
        if False:
            return 10
        ...

    @abstractmethod
    async def __aenter__(self) -> Abstract:
        ...

class GoodIterator(Iterator[str]):

    def __iter__(self: Self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        ...

class GoodAsyncIterator(AsyncIterator[int]):

    def __aiter__(self: Self) -> Self:
        if False:
            for i in range(10):
                print('nop')
        ...

class DoesNotInheritFromIterator:

    def __iter__(self) -> DoesNotInheritFromIterator:
        if False:
            return 10
        ...

class Unannotated:

    def __new__(cls, *args, **kwargs):
        if False:
            return 10
        ...

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        ...

    def __aiter__(self):
        if False:
            i = 10
            return i + 15
        ...

    async def __aenter__(self):
        ...

    def __repr__(self):
        if False:
            print('Hello World!')
        ...

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        ...

    def __eq__(self):
        if False:
            for i in range(10):
                print('nop')
        ...

    def __ne__(self):
        if False:
            print('Hello World!')
        ...

    def __iadd__(self):
        if False:
            for i in range(10):
                print('nop')
        ...

    def __ior__(self):
        if False:
            i = 10
            return i + 15
        ...

def __repr__(self) -> str:
    if False:
        while True:
            i = 10
    ...

def __str__(self) -> str:
    if False:
        print('Hello World!')
    ...

def __eq__(self, other: Any) -> bool:
    if False:
        while True:
            i = 10
    ...

def __ne__(self, other: Any) -> bool:
    if False:
        i = 10
        return i + 15
    ...

def __imul__(self, other: Any) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    ...