from __future__ import annotations
from abc import abstractmethod

class BaseUser:
    """User model interface."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        if False:
            i = 10
            return i + 15
        ...

    @abstractmethod
    def get_id(self) -> str:
        if False:
            while True:
                i = 10
        ...