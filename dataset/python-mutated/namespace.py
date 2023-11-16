"""Provides isolated namespace of skip tensors."""
import abc
from functools import total_ordering
from typing import Any
import uuid
__all__ = ['Namespace']

@total_ordering
class Namespace(metaclass=abc.ABCMeta):
    """Namespace for isolating skip tensors used by :meth:`isolate()
    <torchpipe.skip.skippable.Skippable.isolate>`.
    """
    __slots__ = ('id',)

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.id = uuid.uuid4()

    def __repr__(self) -> str:
        if False:
            return 10
        return f"<Namespace '{self.id}'>"

    def __hash__(self) -> int:
        if False:
            return 10
        return hash(self.id)

    def __lt__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(other, Namespace):
            return self.id < other.id
        return False

    def __eq__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        if isinstance(other, Namespace):
            return self.id == other.id
        return False
Namespace.register(type(None))