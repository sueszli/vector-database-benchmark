from abc import abstractmethod
from typing import Any
from .env_element import EnvElement, IEnvElement, EnvElementInfo
from ..env.base_env import BaseEnv

class IEnvElementRunner(IEnvElement):

    @abstractmethod
    def get(self, engine: BaseEnv) -> Any:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

class EnvElementRunner(IEnvElementRunner):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        self._init(*args, **kwargs)
        self._check()

    @abstractmethod
    def _init(self, *args, **kwargs) -> None:
        if False:
            return 10
        raise NotImplementedError

    def _check(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        flag = [hasattr(self, '_core'), isinstance(self._core, EnvElement)]
        assert all(flag), flag

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return repr(self._core)

    @property
    def info(self) -> 'EnvElementInfo':
        if False:
            while True:
                i = 10
        return self._core.info