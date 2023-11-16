from __future__ import annotations
from abc import ABC
from typing import Any, Generator, Generic, Iterable, Mapping, TypeVar
from multidict import MultiDict as BaseMultiDict
from multidict import MultiDictProxy, MultiMapping
from litestar.datastructures.upload_file import UploadFile
__all__ = ('FormMultiDict', 'ImmutableMultiDict', 'MultiDict', 'MultiMixin')
T = TypeVar('T')

class MultiMixin(Generic[T], MultiMapping[T], ABC):
    """Mixin providing common methods for multi dicts, used by :class:`ImmutableMultiDict` and :class:`MultiDict`"""

    def dict(self) -> dict[str, list[Any]]:
        if False:
            while True:
                i = 10
        'Return the multi-dict as a dict of lists.\n\n        Returns:\n            A dict of lists\n        '
        return {k: self.getall(k) for k in set(self.keys())}

    def multi_items(self) -> Generator[tuple[str, T], None, None]:
        if False:
            i = 10
            return i + 15
        'Get all keys and values, including duplicates.\n\n        Returns:\n            A list of tuples containing key-value pairs\n        '
        for key in set(self):
            for value in self.getall(key):
                yield (key, value)

class MultiDict(BaseMultiDict[T], MultiMixin[T], Generic[T]):
    """MultiDict, using :class:`MultiDict <multidict.MultiDictProxy>`."""

    def __init__(self, args: MultiMapping | Mapping[str, T] | Iterable[tuple[str, T]] | None=None) -> None:
        if False:
            while True:
                i = 10
        'Initialize ``MultiDict`` from a`MultiMapping``, :class:`Mapping <typing.Mapping>` or an iterable of tuples.\n\n        Args:\n            args: Mapping-like structure to create the ``MultiDict`` from\n        '
        super().__init__(args or {})

    def immutable(self) -> ImmutableMultiDict[T]:
        if False:
            i = 10
            return i + 15
        'Create an.\n\n        :class:`ImmutableMultiDict` view.\n\n        Returns:\n            An immutable multi dict\n        '
        return ImmutableMultiDict[T](self)

class ImmutableMultiDict(MultiDictProxy[T], MultiMixin[T], Generic[T]):
    """Immutable MultiDict, using class:`MultiDictProxy <multidict.MultiDictProxy>`."""

    def __init__(self, args: MultiMapping | Mapping[str, Any] | Iterable[tuple[str, Any]] | None=None) -> None:
        if False:
            return 10
        'Initialize ``ImmutableMultiDict`` from a.\n\n        ``MultiMapping``, :class:`Mapping <typing.Mapping>` or an iterable of tuples.\n\n        Args:\n            args: Mapping-like structure to create the ``ImmutableMultiDict`` from\n        '
        super().__init__(BaseMultiDict(args or {}))

    def mutable_copy(self) -> MultiDict[T]:
        if False:
            i = 10
            return i + 15
        'Create a mutable copy as a.\n\n        :class:`MultiDict`\n\n        Returns:\n            A mutable multi dict\n        '
        return MultiDict(list(self.multi_items()))

class FormMultiDict(ImmutableMultiDict[Any]):
    """MultiDict for form data."""

    async def close(self) -> None:
        """Close all files in the multi-dict.

        Returns:
            None
        """
        for (_, value) in self.multi_items():
            if isinstance(value, UploadFile):
                await value.close()