import sys
from functools import cmp_to_key
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    setattr(collections, 'MutableMapping', collections.abc.MutableMapping)
import collections
from . import pbItem

def StringCmp(obj1, obj2):
    if False:
        while True:
            i = 10
    result = -1
    if obj1 > obj2:
        result = 1
    elif obj1 == obj2:
        result = 0
    return result

def KeySorter(obj1, obj2):
    if False:
        i = 10
        return i + 15
    result = 0
    if str(obj1) == 'isa':
        result = -1
    elif str(obj2) == 'isa':
        result = 1
    else:
        result = StringCmp(str(obj1), str(obj2))
    return result

class pbRoot(collections.abc.MutableMapping):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.store = dict()
        self.key_storage = list()
        self.update(dict(*args, **kwargs))

    def __internalKeyCheck(self, key):
        if False:
            while True:
                i = 10
        safe_key = key
        if isinstance(safe_key, str):
            safe_key = pbItem.pbItemResolver(safe_key, 'qstring')
        return safe_key

    def __getitem__(self, key):
        if False:
            return 10
        return self.store[key]

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        if key not in self.key_storage:
            self.key_storage.append(self.__internalKeyCheck(key))
        self.store[key] = value

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        if key in self.key_storage:
            self.key_storage.remove(key)
        del self.store[key]

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self.key_storage.__iter__()

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.key_storage.__len__()

    def __str__(self):
        if False:
            print('Hello World!')
        return self.store.__str__()

    def __contains__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return item in self.key_storage

    def __getattr__(self, attrib):
        if False:
            return 10
        return getattr(self.store, attrib)

    def __keytransform__(self, key):
        if False:
            print('Hello World!')
        result = key
        if isinstance(key, pbItem.pbItem):
            result = key.value
        return result

    def sortedKeys(self):
        if False:
            print('Hello World!')
        unsorted_keys = self.key_storage
        sorted_keys = sorted(unsorted_keys, key=cmp_to_key(KeySorter))
        can_sort = False
        if len(sorted_keys) > 0:
            all_dictionaries = all((isinstance(self[key].value, dict) or isinstance(self[key].value, pbRoot) for key in unsorted_keys))
            if all_dictionaries:
                can_sort = all((self[key].get('isa', None) is not None for key in unsorted_keys))
                if can_sort:
                    sorted_keys = sorted(unsorted_keys, key=lambda k: str(self[k]['isa']))
        return (can_sort, sorted_keys)