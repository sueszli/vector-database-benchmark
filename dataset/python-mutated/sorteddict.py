"""Sorted Dict
==============

:doc:`Sorted Containers<index>` is an Apache2 licensed Python sorted
collections library, written in pure-Python, and fast as C-extensions. The
:doc:`introduction<introduction>` is the best way to get started.

Sorted dict implementations:

.. currentmodule:: sortedcontainers

* :class:`SortedDict`
* :class:`SortedKeysView`
* :class:`SortedItemsView`
* :class:`SortedValuesView`

"""
import sys
import warnings
from itertools import chain
from .sortedlist import SortedList, recursive_repr
from .sortedset import SortedSet
try:
    from collections.abc import ItemsView, KeysView, Mapping, ValuesView, Sequence
except ImportError:
    from collections import ItemsView, KeysView, Mapping, ValuesView, Sequence

class SortedDict(dict):
    """Sorted dict is a sorted mutable mapping.

    Sorted dict keys are maintained in sorted order. The design of sorted dict
    is simple: sorted dict inherits from dict to store items and maintains a
    sorted list of keys.

    Sorted dict keys must be hashable and comparable. The hash and total
    ordering of keys must not change while they are stored in the sorted dict.

    Mutable mapping methods:

    * :func:`SortedDict.__getitem__` (inherited from dict)
    * :func:`SortedDict.__setitem__`
    * :func:`SortedDict.__delitem__`
    * :func:`SortedDict.__iter__`
    * :func:`SortedDict.__len__` (inherited from dict)

    Methods for adding items:

    * :func:`SortedDict.setdefault`
    * :func:`SortedDict.update`

    Methods for removing items:

    * :func:`SortedDict.clear`
    * :func:`SortedDict.pop`
    * :func:`SortedDict.popitem`

    Methods for looking up items:

    * :func:`SortedDict.__contains__` (inherited from dict)
    * :func:`SortedDict.get` (inherited from dict)
    * :func:`SortedDict.peekitem`

    Methods for views:

    * :func:`SortedDict.keys`
    * :func:`SortedDict.items`
    * :func:`SortedDict.values`

    Methods for miscellany:

    * :func:`SortedDict.copy`
    * :func:`SortedDict.fromkeys`
    * :func:`SortedDict.__reversed__`
    * :func:`SortedDict.__eq__` (inherited from dict)
    * :func:`SortedDict.__ne__` (inherited from dict)
    * :func:`SortedDict.__repr__`
    * :func:`SortedDict._check`

    Sorted list methods available (applies to keys):

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.count`
    * :func:`SortedList.index`
    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList._reset`

    Additional sorted list methods available, if key-function used:

    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Sorted dicts may only be compared for equality and inequality.

    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Initialize sorted dict instance.\n\n        Optional key-function argument defines a callable that, like the `key`\n        argument to the built-in `sorted` function, extracts a comparison key\n        from each dictionary key. If no function is specified, the default\n        compares the dictionary keys directly. The key-function argument must\n        be provided as a positional argument and must come before all other\n        arguments.\n\n        Optional iterable argument provides an initial sequence of pairs to\n        initialize the sorted dict. Each pair in the sequence defines the key\n        and corresponding value. If a key is seen more than once, the last\n        value associated with it is stored in the new sorted dict.\n\n        Optional mapping argument provides an initial mapping of items to\n        initialize the sorted dict.\n\n        If keyword arguments are given, the keywords themselves, with their\n        associated values, are added as items to the dictionary. If a key is\n        specified both in the positional argument and as a keyword argument,\n        the value associated with the keyword is stored in the\n        sorted dict.\n\n        Sorted dict keys must be hashable, per the requirement for Python's\n        dictionaries. Keys (or the result of the key-function) must also be\n        comparable, per the requirement for sorted lists.\n\n        >>> d = {'alpha': 1, 'beta': 2}\n        >>> SortedDict([('alpha', 1), ('beta', 2)]) == d\n        True\n        >>> SortedDict({'alpha': 1, 'beta': 2}) == d\n        True\n        >>> SortedDict(alpha=1, beta=2) == d\n        True\n\n        "
        if args and (args[0] is None or callable(args[0])):
            _key = self._key = args[0]
            args = args[1:]
        else:
            _key = self._key = None
        self._list = SortedList(key=_key)
        _list = self._list
        self._list_add = _list.add
        self._list_clear = _list.clear
        self._list_iter = _list.__iter__
        self._list_reversed = _list.__reversed__
        self._list_pop = _list.pop
        self._list_remove = _list.remove
        self._list_update = _list.update
        self.bisect_left = _list.bisect_left
        self.bisect = _list.bisect_right
        self.bisect_right = _list.bisect_right
        self.index = _list.index
        self.irange = _list.irange
        self.islice = _list.islice
        self._reset = _list._reset
        if _key is not None:
            self.bisect_key_left = _list.bisect_key_left
            self.bisect_key_right = _list.bisect_key_right
            self.bisect_key = _list.bisect_key
            self.irange_key = _list.irange_key
        self._update(*args, **kwargs)

    @property
    def key(self):
        if False:
            return 10
        'Function used to extract comparison key from keys.\n\n        Sorted dict compares keys directly when the key function is none.\n\n        '
        return self._key

    @property
    def iloc(self):
        if False:
            i = 10
            return i + 15
        'Cached reference of sorted keys view.\n\n        Deprecated in version 2 of Sorted Containers. Use\n        :func:`SortedDict.keys` instead.\n\n        '
        try:
            return self._iloc
        except AttributeError:
            warnings.warn('sorted_dict.iloc is deprecated. Use SortedDict.keys() instead.', DeprecationWarning, stacklevel=2)
            _iloc = self._iloc = SortedKeysView(self)
            return _iloc

    def clear(self):
        if False:
            print('Hello World!')
        'Remove all items from sorted dict.\n\n        Runtime complexity: `O(n)`\n\n        '
        dict.clear(self)
        self._list_clear()

    def __delitem__(self, key):
        if False:
            return 10
        "Remove item from sorted dict identified by `key`.\n\n        ``sd.__delitem__(key)`` <==> ``del sd[key]``\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})\n        >>> del sd['b']\n        >>> sd\n        SortedDict({'a': 1, 'c': 3})\n        >>> del sd['z']\n        Traceback (most recent call last):\n          ...\n        KeyError: 'z'\n\n        :param key: `key` for item lookup\n        :raises KeyError: if key not found\n\n        "
        dict.__delitem__(self, key)
        self._list_remove(key)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        'Return an iterator over the keys of the sorted dict.\n\n        ``sd.__iter__()`` <==> ``iter(sd)``\n\n        Iterating the sorted dict while adding or deleting items may raise a\n        :exc:`RuntimeError` or fail to iterate over all keys.\n\n        '
        return self._list_iter()

    def __reversed__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a reverse iterator over the keys of the sorted dict.\n\n        ``sd.__reversed__()`` <==> ``reversed(sd)``\n\n        Iterating the sorted dict while adding or deleting items may raise a\n        :exc:`RuntimeError` or fail to iterate over all keys.\n\n        '
        return self._list_reversed()

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        "Store item in sorted dict with `key` and corresponding `value`.\n\n        ``sd.__setitem__(key, value)`` <==> ``sd[key] = value``\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sd = SortedDict()\n        >>> sd['c'] = 3\n        >>> sd['a'] = 1\n        >>> sd['b'] = 2\n        >>> sd\n        SortedDict({'a': 1, 'b': 2, 'c': 3})\n\n        :param key: key for item\n        :param value: value for item\n\n        "
        if key not in self:
            self._list_add(key)
        dict.__setitem__(self, key, value)
    _setitem = __setitem__

    def __or__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Mapping):
            return NotImplemented
        items = chain(self.items(), other.items())
        return self.__class__(self._key, items)

    def __ror__(self, other):
        if False:
            return 10
        if not isinstance(other, Mapping):
            return NotImplemented
        items = chain(other.items(), self.items())
        return self.__class__(self._key, items)

    def __ior__(self, other):
        if False:
            i = 10
            return i + 15
        self._update(other)
        return self

    def copy(self):
        if False:
            return 10
        'Return a shallow copy of the sorted dict.\n\n        Runtime complexity: `O(n)`\n\n        :return: new sorted dict\n\n        '
        return self.__class__(self._key, self.items())
    __copy__ = copy

    @classmethod
    def fromkeys(cls, iterable, value=None):
        if False:
            i = 10
            return i + 15
        'Return a new sorted dict initailized from `iterable` and `value`.\n\n        Items in the sorted dict have keys from `iterable` and values equal to\n        `value`.\n\n        Runtime complexity: `O(n*log(n))`\n\n        :return: new sorted dict\n\n        '
        return cls(((key, value) for key in iterable))

    def keys(self):
        if False:
            return 10
        "Return new sorted keys view of the sorted dict's keys.\n\n        See :class:`SortedKeysView` for details.\n\n        :return: new sorted keys view\n\n        "
        return SortedKeysView(self)

    def items(self):
        if False:
            while True:
                i = 10
        "Return new sorted items view of the sorted dict's items.\n\n        See :class:`SortedItemsView` for details.\n\n        :return: new sorted items view\n\n        "
        return SortedItemsView(self)

    def values(self):
        if False:
            return 10
        "Return new sorted values view of the sorted dict's values.\n\n        Note that the values view is sorted by key.\n\n        See :class:`SortedValuesView` for details.\n\n        :return: new sorted values view\n\n        "
        return SortedValuesView(self)
    if sys.hexversion < 50331648:

        def __make_raise_attributeerror(original, alternate):
            if False:
                while True:
                    i = 10
            message = 'SortedDict.{original}() is not implemented. Use SortedDict.{alternate}() instead.'.format(original=original, alternate=alternate)

            def method(self):
                if False:
                    return 10
                raise AttributeError(message)
            method.__name__ = original
            method.__doc__ = message
            return property(method)
        iteritems = __make_raise_attributeerror('iteritems', 'items')
        iterkeys = __make_raise_attributeerror('iterkeys', 'keys')
        itervalues = __make_raise_attributeerror('itervalues', 'values')
        viewitems = __make_raise_attributeerror('viewitems', 'items')
        viewkeys = __make_raise_attributeerror('viewkeys', 'keys')
        viewvalues = __make_raise_attributeerror('viewvalues', 'values')

    class _NotGiven(object):

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return '<not-given>'
    __not_given = _NotGiven()

    def pop(self, key, default=__not_given):
        if False:
            print('Hello World!')
        "Remove and return value for item identified by `key`.\n\n        If the `key` is not found then return `default` if given. If `default`\n        is not given then raise :exc:`KeyError`.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})\n        >>> sd.pop('c')\n        3\n        >>> sd.pop('z', 26)\n        26\n        >>> sd.pop('y')\n        Traceback (most recent call last):\n          ...\n        KeyError: 'y'\n\n        :param key: `key` for item\n        :param default: `default` value if key not found (optional)\n        :return: value for item\n        :raises KeyError: if `key` not found and `default` not given\n\n        "
        if key in self:
            self._list_remove(key)
            return dict.pop(self, key)
        else:
            if default is self.__not_given:
                raise KeyError(key)
            return default

    def popitem(self, index=-1):
        if False:
            for i in range(10):
                print('nop')
        "Remove and return ``(key, value)`` pair at `index` from sorted dict.\n\n        Optional argument `index` defaults to -1, the last item in the sorted\n        dict. Specify ``index=0`` for the first item in the sorted dict.\n\n        If the sorted dict is empty, raises :exc:`KeyError`.\n\n        If the `index` is out of range, raises :exc:`IndexError`.\n\n        Runtime complexity: `O(log(n))`\n\n        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})\n        >>> sd.popitem()\n        ('c', 3)\n        >>> sd.popitem(0)\n        ('a', 1)\n        >>> sd.popitem(100)\n        Traceback (most recent call last):\n          ...\n        IndexError: list index out of range\n\n        :param int index: `index` of item (default -1)\n        :return: key and value pair\n        :raises KeyError: if sorted dict is empty\n        :raises IndexError: if `index` out of range\n\n        "
        if not self:
            raise KeyError('popitem(): dictionary is empty')
        key = self._list_pop(index)
        value = dict.pop(self, key)
        return (key, value)

    def peekitem(self, index=-1):
        if False:
            i = 10
            return i + 15
        "Return ``(key, value)`` pair at `index` in sorted dict.\n\n        Optional argument `index` defaults to -1, the last item in the sorted\n        dict. Specify ``index=0`` for the first item in the sorted dict.\n\n        Unlike :func:`SortedDict.popitem`, the sorted dict is not modified.\n\n        If the `index` is out of range, raises :exc:`IndexError`.\n\n        Runtime complexity: `O(log(n))`\n\n        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})\n        >>> sd.peekitem()\n        ('c', 3)\n        >>> sd.peekitem(0)\n        ('a', 1)\n        >>> sd.peekitem(100)\n        Traceback (most recent call last):\n          ...\n        IndexError: list index out of range\n\n        :param int index: index of item (default -1)\n        :return: key and value pair\n        :raises IndexError: if `index` out of range\n\n        "
        key = self._list[index]
        return (key, self[key])

    def setdefault(self, key, default=None):
        if False:
            i = 10
            return i + 15
        "Return value for item identified by `key` in sorted dict.\n\n        If `key` is in the sorted dict then return its value. If `key` is not\n        in the sorted dict then insert `key` with value `default` and return\n        `default`.\n\n        Optional argument `default` defaults to none.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sd = SortedDict()\n        >>> sd.setdefault('a', 1)\n        1\n        >>> sd.setdefault('a', 10)\n        1\n        >>> sd\n        SortedDict({'a': 1})\n\n        :param key: key for item\n        :param default: value for item (default None)\n        :return: value for item identified by `key`\n\n        "
        if key in self:
            return self[key]
        dict.__setitem__(self, key, default)
        self._list_add(key)
        return default

    def update(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Update sorted dict with items from `args` and `kwargs`.\n\n        Overwrites existing items.\n\n        Optional arguments `args` and `kwargs` may be a mapping, an iterable of\n        pairs or keyword arguments. See :func:`SortedDict.__init__` for\n        details.\n\n        :param args: mapping or iterable of pairs\n        :param kwargs: keyword arguments mapping\n\n        '
        if not self:
            dict.update(self, *args, **kwargs)
            self._list_update(dict.__iter__(self))
            return
        if not kwargs and len(args) == 1 and isinstance(args[0], dict):
            pairs = args[0]
        else:
            pairs = dict(*args, **kwargs)
        if 10 * len(pairs) > len(self):
            dict.update(self, pairs)
            self._list_clear()
            self._list_update(dict.__iter__(self))
        else:
            for key in pairs:
                self._setitem(key, pairs[key])
    _update = update

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        'Support for pickle.\n\n        The tricks played with caching references in\n        :func:`SortedDict.__init__` confuse pickle so customize the reducer.\n\n        '
        items = dict.copy(self)
        return (type(self), (self._key, items))

    @recursive_repr()
    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return string representation of sorted dict.\n\n        ``sd.__repr__()`` <==> ``repr(sd)``\n\n        :return: string representation\n\n        '
        _key = self._key
        type_name = type(self).__name__
        key_arg = '' if _key is None else '{0!r}, '.format(_key)
        item_format = '{0!r}: {1!r}'.format
        items = ', '.join((item_format(key, self[key]) for key in self._list))
        return '{0}({1}{{{2}}})'.format(type_name, key_arg, items)

    def _check(self):
        if False:
            for i in range(10):
                print('nop')
        'Check invariants of sorted dict.\n\n        Runtime complexity: `O(n)`\n\n        '
        _list = self._list
        _list._check()
        assert len(self) == len(_list)
        assert all((key in self for key in _list))

def _view_delitem(self, index):
    if False:
        return 10
    "Remove item at `index` from sorted dict.\n\n    ``view.__delitem__(index)`` <==> ``del view[index]``\n\n    Supports slicing.\n\n    Runtime complexity: `O(log(n))` -- approximate.\n\n    >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})\n    >>> view = sd.keys()\n    >>> del view[0]\n    >>> sd\n    SortedDict({'b': 2, 'c': 3})\n    >>> del view[-1]\n    >>> sd\n    SortedDict({'b': 2})\n    >>> del view[:]\n    >>> sd\n    SortedDict({})\n\n    :param index: integer or slice for indexing\n    :raises IndexError: if index out of range\n\n    "
    _mapping = self._mapping
    _list = _mapping._list
    dict_delitem = dict.__delitem__
    if isinstance(index, slice):
        keys = _list[index]
        del _list[index]
        for key in keys:
            dict_delitem(_mapping, key)
    else:
        key = _list.pop(index)
        dict_delitem(_mapping, key)

class SortedKeysView(KeysView, Sequence):
    """Sorted keys view is a dynamic view of the sorted dict's keys.

    When the sorted dict's keys change, the view reflects those changes.

    The keys view implements the set and sequence abstract base classes.

    """
    __slots__ = ()

    @classmethod
    def _from_iterable(cls, it):
        if False:
            return 10
        return SortedSet(it)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        "Lookup key at `index` in sorted keys views.\n\n        ``skv.__getitem__(index)`` <==> ``skv[index]``\n\n        Supports slicing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})\n        >>> skv = sd.keys()\n        >>> skv[0]\n        'a'\n        >>> skv[-1]\n        'c'\n        >>> skv[:]\n        ['a', 'b', 'c']\n        >>> skv[100]\n        Traceback (most recent call last):\n          ...\n        IndexError: list index out of range\n\n        :param index: integer or slice for indexing\n        :return: key or list of keys\n        :raises IndexError: if index out of range\n\n        "
        return self._mapping._list[index]
    __delitem__ = _view_delitem

class SortedItemsView(ItemsView, Sequence):
    """Sorted items view is a dynamic view of the sorted dict's items.

    When the sorted dict's items change, the view reflects those changes.

    The items view implements the set and sequence abstract base classes.

    """
    __slots__ = ()

    @classmethod
    def _from_iterable(cls, it):
        if False:
            return 10
        return SortedSet(it)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        "Lookup item at `index` in sorted items view.\n\n        ``siv.__getitem__(index)`` <==> ``siv[index]``\n\n        Supports slicing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})\n        >>> siv = sd.items()\n        >>> siv[0]\n        ('a', 1)\n        >>> siv[-1]\n        ('c', 3)\n        >>> siv[:]\n        [('a', 1), ('b', 2), ('c', 3)]\n        >>> siv[100]\n        Traceback (most recent call last):\n          ...\n        IndexError: list index out of range\n\n        :param index: integer or slice for indexing\n        :return: item or list of items\n        :raises IndexError: if index out of range\n\n        "
        _mapping = self._mapping
        _mapping_list = _mapping._list
        if isinstance(index, slice):
            keys = _mapping_list[index]
            return [(key, _mapping[key]) for key in keys]
        key = _mapping_list[index]
        return (key, _mapping[key])
    __delitem__ = _view_delitem

class SortedValuesView(ValuesView, Sequence):
    """Sorted values view is a dynamic view of the sorted dict's values.

    When the sorted dict's values change, the view reflects those changes.

    The values view implements the sequence abstract base class.

    """
    __slots__ = ()

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        "Lookup value at `index` in sorted values view.\n\n        ``siv.__getitem__(index)`` <==> ``siv[index]``\n\n        Supports slicing.\n\n        Runtime complexity: `O(log(n))` -- approximate.\n\n        >>> sd = SortedDict({'a': 2, 'b': 1, 'c': 3})\n        >>> svv = sd.values()\n        >>> svv[0]\n        2\n        >>> svv[-1]\n        3\n        >>> svv[:]\n        [2, 1, 3]\n        >>> svv[100]\n        Traceback (most recent call last):\n          ...\n        IndexError: list index out of range\n\n        :param index: integer or slice for indexing\n        :return: value or list of values\n        :raises IndexError: if index out of range\n\n        "
        _mapping = self._mapping
        _mapping_list = _mapping._list
        if isinstance(index, slice):
            keys = _mapping_list[index]
            return [_mapping[key] for key in keys]
        key = _mapping_list[index]
        return _mapping[key]
    __delitem__ = _view_delitem