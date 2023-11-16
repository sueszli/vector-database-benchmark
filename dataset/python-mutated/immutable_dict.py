"""Immutable mapping."""
import collections.abc

class ImmutableDict(collections.abc.Mapping):
    """Immutable `Mapping`."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._dict = dict(*args, **kwargs)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self._dict[key]

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return key in self._dict

    def __iter__(self):
        if False:
            return 10
        return iter(self._dict)

    def __len__(self):
        if False:
            return 10
        return len(self._dict)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'ImmutableDict({self._dict})'
    __supported_by_tf_nest__ = True