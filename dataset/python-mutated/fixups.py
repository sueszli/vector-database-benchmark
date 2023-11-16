import abc
import typing
from typing import Iterable
if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
else:

    class _AppT:
        ...
__all__ = ['FixupT']

class FixupT(abc.ABC):
    app: _AppT

    @abc.abstractmethod
    def __init__(self, app: _AppT) -> None:
        if False:
            print('Hello World!')
        ...

    @abc.abstractmethod
    def enabled(self) -> bool:
        if False:
            i = 10
            return i + 15
        ...

    @abc.abstractmethod
    def autodiscover_modules(self) -> Iterable[str]:
        if False:
            while True:
                i = 10
        ...

    @abc.abstractmethod
    def on_worker_init(self) -> None:
        if False:
            i = 10
            return i + 15
        ...