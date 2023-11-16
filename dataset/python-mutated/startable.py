from abc import ABC, abstractmethod

class StartableBase(ABC):
    """Abstract base class for Thread- and Process-like objects."""
    __slots__ = ()

    @abstractmethod
    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError
__all__ = ['StartableBase']