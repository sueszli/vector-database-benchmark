import collections.abc
import typing
from collections.abc import Iterator, Iterable

class NoReturn:

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        ...

class TypingIterableTReturn:

    def __iter__(self) -> typing.Iterable[int]:
        if False:
            while True:
                i = 10
        ...

    def not_iter(self) -> typing.Iterable[int]:
        if False:
            for i in range(10):
                print('nop')
        ...

class TypingIterableReturn:

    def __iter__(self) -> typing.Iterable:
        if False:
            while True:
                i = 10
        ...

    def not_iter(self) -> typing.Iterable:
        if False:
            return 10
        ...

class CollectionsIterableTReturn:

    def __iter__(self) -> collections.abc.Iterable[int]:
        if False:
            return 10
        ...

    def not_iter(self) -> collections.abc.Iterable[int]:
        if False:
            print('Hello World!')
        ...

class CollectionsIterableReturn:

    def __iter__(self) -> collections.abc.Iterable:
        if False:
            while True:
                i = 10
        ...

    def not_iter(self) -> collections.abc.Iterable:
        if False:
            i = 10
            return i + 15
        ...

class IterableReturn:

    def __iter__(self) -> Iterable:
        if False:
            i = 10
            return i + 15
        ...

class IteratorReturn:

    def __iter__(self) -> Iterator:
        if False:
            for i in range(10):
                print('nop')
        ...

class IteratorTReturn:

    def __iter__(self) -> Iterator[int]:
        if False:
            while True:
                i = 10
        ...

class TypingIteratorReturn:

    def __iter__(self) -> typing.Iterator:
        if False:
            i = 10
            return i + 15
        ...

class TypingIteratorTReturn:

    def __iter__(self) -> typing.Iterator[int]:
        if False:
            return 10
        ...

class CollectionsIteratorReturn:

    def __iter__(self) -> collections.abc.Iterator:
        if False:
            return 10
        ...

class CollectionsIteratorTReturn:

    def __iter__(self) -> collections.abc.Iterator[int]:
        if False:
            return 10
        ...

class TypingAsyncIterableTReturn:

    def __aiter__(self) -> typing.AsyncIterable[int]:
        if False:
            for i in range(10):
                print('nop')
        ...

class TypingAsyncIterableReturn:

    def __aiter__(self) -> typing.AsyncIterable:
        if False:
            i = 10
            return i + 15
        ...