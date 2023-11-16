import collections
import operator
from functools import reduce
from collections.abc import Mapping
__all__ = ('merge', 'merge_with', 'valmap', 'keymap', 'itemmap', 'valfilter', 'keyfilter', 'itemfilter', 'assoc', 'dissoc', 'assoc_in', 'update_in', 'get_in')

def _get_factory(f, kwargs):
    if False:
        while True:
            i = 10
    factory = kwargs.pop('factory', dict)
    if kwargs:
        raise TypeError(f"{f.__name__}() got an unexpected keyword argument '{kwargs.popitem()[0]}'")
    return factory

def merge(*dicts, **kwargs):
    if False:
        print('Hello World!')
    " Merge a collection of dictionaries\n\n    >>> merge({1: 'one'}, {2: 'two'})\n    {1: 'one', 2: 'two'}\n\n    Later dictionaries have precedence\n\n    >>> merge({1: 2, 3: 4}, {3: 3, 4: 4})\n    {1: 2, 3: 3, 4: 4}\n\n    See Also:\n        merge_with\n    "
    if len(dicts) == 1 and (not isinstance(dicts[0], Mapping)):
        dicts = dicts[0]
    factory = _get_factory(merge, kwargs)
    rv = factory()
    for d in dicts:
        rv.update(d)
    return rv

def merge_with(func, *dicts, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' Merge dictionaries and apply function to combined values\n\n    A key may occur in more than one dict, and all values mapped from the key\n    will be passed to the function as a list, such as func([val1, val2, ...]).\n\n    >>> merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})\n    {1: 11, 2: 22}\n\n    >>> merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})  # doctest: +SKIP\n    {1: 1, 2: 2, 3: 30}\n\n    See Also:\n        merge\n    '
    if len(dicts) == 1 and (not isinstance(dicts[0], Mapping)):
        dicts = dicts[0]
    factory = _get_factory(merge_with, kwargs)
    result = factory()
    for d in dicts:
        for (k, v) in d.items():
            if k not in result:
                result[k] = [v]
            else:
                result[k].append(v)
    return valmap(func, result, factory)

def valmap(func, d, factory=dict):
    if False:
        i = 10
        return i + 15
    ' Apply function to values of dictionary\n\n    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}\n    >>> valmap(sum, bills)  # doctest: +SKIP\n    {\'Alice\': 65, \'Bob\': 45}\n\n    See Also:\n        keymap\n        itemmap\n    '
    rv = factory()
    rv.update(zip(d.keys(), map(func, d.values())))
    return rv

def keymap(func, d, factory=dict):
    if False:
        while True:
            i = 10
    ' Apply function to keys of dictionary\n\n    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}\n    >>> keymap(str.lower, bills)  # doctest: +SKIP\n    {\'alice\': [20, 15, 30], \'bob\': [10, 35]}\n\n    See Also:\n        valmap\n        itemmap\n    '
    rv = factory()
    rv.update(zip(map(func, d.keys()), d.values()))
    return rv

def itemmap(func, d, factory=dict):
    if False:
        print('Hello World!')
    ' Apply function to items of dictionary\n\n    >>> accountids = {"Alice": 10, "Bob": 20}\n    >>> itemmap(reversed, accountids)  # doctest: +SKIP\n    {10: "Alice", 20: "Bob"}\n\n    See Also:\n        keymap\n        valmap\n    '
    rv = factory()
    rv.update(map(func, d.items()))
    return rv

def valfilter(predicate, d, factory=dict):
    if False:
        for i in range(10):
            print('nop')
    ' Filter items in dictionary by value\n\n    >>> iseven = lambda x: x % 2 == 0\n    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}\n    >>> valfilter(iseven, d)\n    {1: 2, 3: 4}\n\n    See Also:\n        keyfilter\n        itemfilter\n        valmap\n    '
    rv = factory()
    for (k, v) in d.items():
        if predicate(v):
            rv[k] = v
    return rv

def keyfilter(predicate, d, factory=dict):
    if False:
        print('Hello World!')
    ' Filter items in dictionary by key\n\n    >>> iseven = lambda x: x % 2 == 0\n    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}\n    >>> keyfilter(iseven, d)\n    {2: 3, 4: 5}\n\n    See Also:\n        valfilter\n        itemfilter\n        keymap\n    '
    rv = factory()
    for (k, v) in d.items():
        if predicate(k):
            rv[k] = v
    return rv

def itemfilter(predicate, d, factory=dict):
    if False:
        return 10
    ' Filter items in dictionary by item\n\n    >>> def isvalid(item):\n    ...     k, v = item\n    ...     return k % 2 == 0 and v < 4\n\n    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}\n    >>> itemfilter(isvalid, d)\n    {2: 3}\n\n    See Also:\n        keyfilter\n        valfilter\n        itemmap\n    '
    rv = factory()
    for item in d.items():
        if predicate(item):
            (k, v) = item
            rv[k] = v
    return rv

def assoc(d, key, value, factory=dict):
    if False:
        while True:
            i = 10
    " Return a new dict with new key value pair\n\n    New dict has d[key] set to value. Does not modify the initial dictionary.\n\n    >>> assoc({'x': 1}, 'x', 2)\n    {'x': 2}\n    >>> assoc({'x': 1}, 'y', 3)   # doctest: +SKIP\n    {'x': 1, 'y': 3}\n    "
    d2 = factory()
    d2.update(d)
    d2[key] = value
    return d2

def dissoc(d, *keys, **kwargs):
    if False:
        print('Hello World!')
    " Return a new dict with the given key(s) removed.\n\n    New dict has d[key] deleted for each supplied key.\n    Does not modify the initial dictionary.\n\n    >>> dissoc({'x': 1, 'y': 2}, 'y')\n    {'x': 1}\n    >>> dissoc({'x': 1, 'y': 2}, 'y', 'x')\n    {}\n    >>> dissoc({'x': 1}, 'y') # Ignores missing keys\n    {'x': 1}\n    "
    factory = _get_factory(dissoc, kwargs)
    d2 = factory()
    if len(keys) < len(d) * 0.6:
        d2.update(d)
        for key in keys:
            if key in d2:
                del d2[key]
    else:
        remaining = set(d)
        remaining.difference_update(keys)
        for k in remaining:
            d2[k] = d[k]
    return d2

def assoc_in(d, keys, value, factory=dict):
    if False:
        while True:
            i = 10
    " Return a new dict with new, potentially nested, key value pair\n\n    >>> purchase = {'name': 'Alice',\n    ...             'order': {'items': ['Apple', 'Orange'],\n    ...                       'costs': [0.50, 1.25]},\n    ...             'credit card': '5555-1234-1234-1234'}\n    >>> assoc_in(purchase, ['order', 'costs'], [0.25, 1.00]) # doctest: +SKIP\n    {'credit card': '5555-1234-1234-1234',\n     'name': 'Alice',\n     'order': {'costs': [0.25, 1.00], 'items': ['Apple', 'Orange']}}\n    "
    return update_in(d, keys, lambda x: value, value, factory)

def update_in(d, keys, func, default=None, factory=dict):
    if False:
        for i in range(10):
            print('nop')
    ' Update value in a (potentially) nested dictionary\n\n    inputs:\n    d - dictionary on which to operate\n    keys - list or tuple giving the location of the value to be changed in d\n    func - function to operate on that value\n\n    If keys == [k0,..,kX] and d[k0]..[kX] == v, update_in returns a copy of the\n    original dictionary with v replaced by func(v), but does not mutate the\n    original dictionary.\n\n    If k0 is not a key in d, update_in creates nested dictionaries to the depth\n    specified by the keys, with the innermost value set to func(default).\n\n    >>> inc = lambda x: x + 1\n    >>> update_in({\'a\': 0}, [\'a\'], inc)\n    {\'a\': 1}\n\n    >>> transaction = {\'name\': \'Alice\',\n    ...                \'purchase\': {\'items\': [\'Apple\', \'Orange\'],\n    ...                             \'costs\': [0.50, 1.25]},\n    ...                \'credit card\': \'5555-1234-1234-1234\'}\n    >>> update_in(transaction, [\'purchase\', \'costs\'], sum) # doctest: +SKIP\n    {\'credit card\': \'5555-1234-1234-1234\',\n     \'name\': \'Alice\',\n     \'purchase\': {\'costs\': 1.75, \'items\': [\'Apple\', \'Orange\']}}\n\n    >>> # updating a value when k0 is not in d\n    >>> update_in({}, [1, 2, 3], str, default="bar")\n    {1: {2: {3: \'bar\'}}}\n    >>> update_in({1: \'foo\'}, [2, 3, 4], inc, 0)\n    {1: \'foo\', 2: {3: {4: 1}}}\n    '
    ks = iter(keys)
    k = next(ks)
    rv = inner = factory()
    rv.update(d)
    for key in ks:
        if k in d:
            d = d[k]
            dtemp = factory()
            dtemp.update(d)
        else:
            d = dtemp = factory()
        inner[k] = inner = dtemp
        k = key
    if k in d:
        inner[k] = func(d[k])
    else:
        inner[k] = func(default)
    return rv

def get_in(keys, coll, default=None, no_default=False):
    if False:
        return 10
    " Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.\n\n    If coll[i0][i1]...[iX] cannot be found, returns ``default``, unless\n    ``no_default`` is specified, then it raises KeyError or IndexError.\n\n    ``get_in`` is a generalization of ``operator.getitem`` for nested data\n    structures such as dictionaries and lists.\n\n    >>> transaction = {'name': 'Alice',\n    ...                'purchase': {'items': ['Apple', 'Orange'],\n    ...                             'costs': [0.50, 1.25]},\n    ...                'credit card': '5555-1234-1234-1234'}\n    >>> get_in(['purchase', 'items', 0], transaction)\n    'Apple'\n    >>> get_in(['name'], transaction)\n    'Alice'\n    >>> get_in(['purchase', 'total'], transaction)\n    >>> get_in(['purchase', 'items', 'apple'], transaction)\n    >>> get_in(['purchase', 'items', 10], transaction)\n    >>> get_in(['purchase', 'total'], transaction, 0)\n    0\n    >>> get_in(['y'], {}, no_default=True)\n    Traceback (most recent call last):\n        ...\n    KeyError: 'y'\n\n    See Also:\n        itertoolz.get\n        operator.getitem\n    "
    try:
        return reduce(operator.getitem, keys, coll)
    except (KeyError, IndexError, TypeError):
        if no_default:
            raise
        return default

def getter(index):
    if False:
        i = 10
        return i + 15
    if isinstance(index, list):
        if len(index) == 1:
            index = index[0]
            return lambda x: (x[index],)
        elif index:
            return operator.itemgetter(*index)
        else:
            return lambda x: ()
    else:
        return operator.itemgetter(index)

def groupby(key, seq):
    if False:
        print('Hello World!')
    " Group a collection by a key function\n\n    >>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']\n    >>> groupby(len, names)  # doctest: +SKIP\n    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}\n\n    >>> iseven = lambda x: x % 2 == 0\n    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP\n    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}\n\n    Non-callable keys imply grouping on a member.\n\n    >>> groupby('gender', [{'name': 'Alice', 'gender': 'F'},\n    ...                    {'name': 'Bob', 'gender': 'M'},\n    ...                    {'name': 'Charlie', 'gender': 'M'}]) # doctest:+SKIP\n    {'F': [{'gender': 'F', 'name': 'Alice'}],\n     'M': [{'gender': 'M', 'name': 'Bob'},\n           {'gender': 'M', 'name': 'Charlie'}]}\n\n    Not to be confused with ``itertools.groupby``\n\n    See Also:\n        countby\n    "
    if not callable(key):
        key = getter(key)
    d = collections.defaultdict(lambda : [].append)
    for item in seq:
        d[key(item)](item)
    rv = {}
    for (k, v) in d.items():
        rv[k] = v.__self__
    return rv

def first(seq):
    if False:
        return 10
    " The first element in a sequence\n\n    >>> first('ABC')\n    'A'\n    "
    return next(iter(seq))