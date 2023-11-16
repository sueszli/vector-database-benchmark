from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterable, cast
from typing_extensions import ClassVar, override
T = TypeVar('T')

class LazyProxy(Generic[T], ABC):
    """Implements data methods to pretend that an instance is another instance.

    This includes forwarding attribute access and othe methods.
    """
    should_cache: ClassVar[bool] = False

    def __init__(self) -> None:
        if False:
            return 10
        self.__proxied: T | None = None

    def __getattr__(self, attr: str) -> object:
        if False:
            while True:
                i = 10
        return getattr(self.__get_proxied__(), attr)

    @override
    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return repr(self.__get_proxied__())

    @override
    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return str(self.__get_proxied__())

    @override
    def __dir__(self) -> Iterable[str]:
        if False:
            i = 10
            return i + 15
        return self.__get_proxied__().__dir__()

    @property
    @override
    def __class__(self) -> type:
        if False:
            print('Hello World!')
        return self.__get_proxied__().__class__

    def __get_proxied__(self) -> T:
        if False:
            for i in range(10):
                print('nop')
        if not self.should_cache:
            return self.__load__()
        proxied = self.__proxied
        if proxied is not None:
            return proxied
        self.__proxied = proxied = self.__load__()
        return proxied

    def __set_proxied__(self, value: T) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.__proxied = value

    def __as_proxied__(self) -> T:
        if False:
            print('Hello World!')
        'Helper method that returns the current proxy, typed as the loaded object'
        return cast(T, self)

    @abstractmethod
    def __load__(self) -> T:
        if False:
            i = 10
            return i + 15
        ...