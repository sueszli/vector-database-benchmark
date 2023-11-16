from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any
EnvElementInfo = namedtuple('EnvElementInfo', ['shape', 'value'])

class IEnvElement(ABC):

    @abstractmethod
    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @property
    @abstractmethod
    def info(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class EnvElement(IEnvElement):
    _instance = None
    _name = 'EnvElement'

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        self._init(*args, **kwargs)
        self._check()

    @abstractmethod
    def _init(*args, **kwargs) -> None:
        if False:
            return 10
        raise NotImplementedError

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '{}: {}'.format(self._name, self._details())

    @abstractmethod
    def _details(self) -> str:
        if False:
            return 10
        raise NotImplementedError

    def _check(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        flag = [hasattr(self, '_shape'), hasattr(self, '_value')]
        assert all(flag), 'this class {} is not a legal subclass of EnvElement({})'.format(self.__class__, flag)

    @property
    def info(self) -> 'EnvElementInfo':
        if False:
            while True:
                i = 10
        return EnvElementInfo(shape=self._shape, value=self._value)