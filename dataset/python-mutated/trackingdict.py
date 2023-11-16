"""This module contains a mutable mapping that keeps track of the keys that where accessed.

.. versionadded:: 20.0

Warning:
    Contents of this module are intended to be used internally by the library and *not* by the
    user. Changes to this module are not considered breaking changes and may not be documented in
    the changelog.
"""
from collections import UserDict
from typing import Final, Generic, List, Mapping, Optional, Set, Tuple, TypeVar, Union
from telegram._utils.defaultvalue import DEFAULT_NONE, DefaultValue
_VT = TypeVar('_VT')
_KT = TypeVar('_KT')
_T = TypeVar('_T')

class TrackingDict(UserDict, Generic[_KT, _VT]):
    """Mutable mapping that keeps track of which keys where accessed with write access.
    Read-access is not tracked.

    Note:
        * ``setdefault()`` and ``pop`` are considered writing only depending on whether the
            key is present
        * deleting values is considered writing
    """
    DELETED: Final = object()
    'Special marker indicating that an entry was deleted.'
    __slots__ = ('_write_access_keys',)

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._write_access_keys: Set[_KT] = set()

    def __setitem__(self, key: _KT, value: _VT) -> None:
        if False:
            print('Hello World!')
        self.__track_write(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: _KT) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.__track_write(key)
        super().__delitem__(key)

    def __track_write(self, key: Union[_KT, Set[_KT]]) -> None:
        if False:
            print('Hello World!')
        if isinstance(key, set):
            self._write_access_keys |= key
        else:
            self._write_access_keys.add(key)

    def pop_accessed_keys(self) -> Set[_KT]:
        if False:
            print('Hello World!')
        'Returns all keys that were write-accessed since the last time this method was called.'
        out = self._write_access_keys
        self._write_access_keys = set()
        return out

    def pop_accessed_write_items(self) -> List[Tuple[_KT, _VT]]:
        if False:
            print('Hello World!')
        '\n        Returns all keys & corresponding values as set of tuples that were write-accessed since\n        the last time this method was called. If a key was deleted, the value will be\n        :attr:`DELETED`.\n        '
        keys = self.pop_accessed_keys()
        return [(key, self[key] if key in self else self.DELETED) for key in keys]

    def mark_as_accessed(self, key: _KT) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Use this method have the key returned again in the next call to\n        :meth:`pop_accessed_write_items` or :meth:`pop_accessed_keys`\n        '
        self._write_access_keys.add(key)

    def update_no_track(self, mapping: Mapping[_KT, _VT]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Like ``update``, but doesn't count towards write access."
        for (key, value) in mapping.items():
            self.data[key] = value

    def pop(self, key: _KT, default: _VT=DEFAULT_NONE) -> _VT:
        if False:
            i = 10
            return i + 15
        if key in self:
            self.__track_write(key)
        if isinstance(default, DefaultValue):
            return super().pop(key)
        return super().pop(key, default=default)

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        self.__track_write(set(super().keys()))
        super().clear()

    def setdefault(self: 'TrackingDict[_KT, _T]', key: _KT, default: Optional[_T]=None) -> _T:
        if False:
            while True:
                i = 10
        if key in self:
            return self[key]
        self.__track_write(key)
        self[key] = default
        return default