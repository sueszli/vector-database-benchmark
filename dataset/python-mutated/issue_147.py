from sortedcontainers import SortedDict, SortedList
import gc

def check(f):
    if False:
        while True:
            i = 10
    print('start')
    a = f()
    t = type(a)
    print('post-setup')
    for obj in gc.get_objects():
        if type(obj) == t:
            print(obj)
    del a
    print('post-delete')
    for obj in gc.get_objects():
        if type(obj) == t:
            print(obj)
    gc.collect()
    print('post-collect')
    for obj in gc.get_objects():
        if type(obj) == t:
            print(obj)
    print('finish')
check(lambda : SortedDict({'a': 1, 'b': 2}))

class MyDict(dict):
    pass
check(lambda : MyDict({'a': 1, 'b': 2}))
check(lambda : SortedList([1, 2, 3]))
from functools import partial

class SortedDictSub(SortedDict):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Initialize sorted dict instance.\n        Optional key-function argument defines a callable that, like the `key`\n        argument to the built-in `sorted` function, extracts a comparison key\n        from each dictionary key. If no function is specified, the default\n        compares the dictionary keys directly. The key-function argument must\n        be provided as a positional argument and must come before all other\n        arguments.\n        Optional iterable argument provides an initial sequence of pairs to\n        initialize the sorted dict. Each pair in the sequence defines the key\n        and corresponding value. If a key is seen more than once, the last\n        value associated with it is stored in the new sorted dict.\n        Optional mapping argument provides an initial mapping of items to\n        initialize the sorted dict.\n        If keyword arguments are given, the keywords themselves, with their\n        associated values, are added as items to the dictionary. If a key is\n        specified both in the positional argument and as a keyword argument,\n        the value associated with the keyword is stored in the\n        sorted dict.\n        Sorted dict keys must be hashable, per the requirement for Python's\n        dictionaries. Keys (or the result of the key-function) must also be\n        comparable, per the requirement for sorted lists.\n        >>> d = {'alpha': 1, 'beta': 2}\n        >>> SortedDict([('alpha', 1), ('beta', 2)]) == d\n        True\n        >>> SortedDict({'alpha': 1, 'beta': 2}) == d\n        True\n        >>> SortedDict(alpha=1, beta=2) == d\n        True\n        "
        if args and (args[0] is None or callable(args[0])):
            _key = self._key = args[0]
            args = args[1:]
        else:
            _key = self._key = None
        self._list = SortedList(key=_key)
        _dict = super(SortedDict, self)
        self._dict_iter = partial(dict.__iter__, self)
        self._dict_update = partial(dict.update, self)
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
check(lambda : SortedDictSub({'a': 1, 'b': 2}))