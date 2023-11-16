from collections import Counter
import collections.abc as _c
_iter_values = 'values'
_range = range
_string_type = str

class _kView(_c.KeysView):

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._mapping.iterkeys()

class _vView(_c.ValuesView):

    def __iter__(self):
        if False:
            return 10
        return self._mapping.itervalues()

class _iView(_c.ItemsView):

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._mapping.iteritems()

class VDFDict(dict):

    def __init__(self, data=None):
        if False:
            i = 10
            return i + 15
        "\n        This is a dictionary that supports duplicate keys and preserves insert order\n\n        ``data`` can be a ``dict``, or a sequence of key-value tuples. (e.g. ``[('key', 'value'),..]``)\n        The only supported type for key is str.\n\n        Get/set duplicates is done by tuples ``(index, key)``, where index is the duplicate index\n        for the specified key. (e.g. ``(0, 'key')``, ``(1, 'key')``...)\n\n        When the ``key`` is ``str``, instead of tuple, set will create a duplicate and get will look up ``(0, key)``\n        "
        self.__omap = []
        self.__kcount = Counter()
        if data is not None:
            if not isinstance(data, (list, dict)):
                raise ValueError('Expected data to be list of pairs or dict, got %s' % type(data))
            self.update(data)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        out = '%s(' % self.__class__.__name__
        out += '%s)' % repr(list(self.iteritems()))
        return out

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.__omap)

    @staticmethod
    def _verify_key_tuple(key):
        if False:
            print('Hello World!')
        if len(key) != 2:
            raise ValueError('Expected key tuple length to be 2, got %d' % len(key))
        if not isinstance(key[0], int):
            raise TypeError('Key index should be an int')
        if not isinstance(key[1], _string_type):
            raise TypeError('Key value should be a str')

    def _normalize_key(self, key):
        if False:
            while True:
                i = 10
        if isinstance(key, _string_type):
            key = (0, key)
        elif isinstance(key, tuple):
            self._verify_key_tuple(key)
        else:
            raise TypeError('Expected key to be a str or tuple, got %s' % type(key))
        return key

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        if isinstance(key, _string_type):
            key = (self.__kcount[key], key)
            self.__omap.append(key)
        elif isinstance(key, tuple):
            self._verify_key_tuple(key)
            if key not in self:
                raise KeyError("%s doesn't exist" % repr(key))
        else:
            raise TypeError('Expected either a str or tuple for key')
        super(VDFDict, self).__setitem__(key, value)
        self.__kcount[key[1]] += 1

    def __getitem__(self, key):
        if False:
            return 10
        return super(VDFDict, self).__getitem__(self._normalize_key(key))

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        key = self._normalize_key(key)
        result = super(VDFDict, self).__delitem__(key)
        start_idx = self.__omap.index(key)
        del self.__omap[start_idx]
        (dup_idx, skey) = key
        self.__kcount[skey] -= 1
        tail_count = self.__kcount[skey] - dup_idx
        if tail_count > 0:
            for idx in _range(start_idx, len(self.__omap)):
                if self.__omap[idx][1] == skey:
                    oldkey = self.__omap[idx]
                    newkey = (dup_idx, skey)
                    super(VDFDict, self).__setitem__(newkey, self[oldkey])
                    super(VDFDict, self).__delitem__(oldkey)
                    self.__omap[idx] = newkey
                    dup_idx += 1
                    tail_count -= 1
                    if tail_count == 0:
                        break
        if self.__kcount[skey] == 0:
            del self.__kcount[skey]
        return result

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.iterkeys())

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        return super(VDFDict, self).__contains__(self._normalize_key(key))

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, VDFDict):
            return list(self.items()) == list(other.items())
        else:
            return False

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def clear(self):
        if False:
            while True:
                i = 10
        super(VDFDict, self).clear()
        self.__kcount.clear()
        self.__omap = list()

    def get(self, key, *_args):
        if False:
            for i in range(10):
                print('nop')
        return super(VDFDict, self).get(self._normalize_key(key), *_args)

    def setdefault(self, key, default=None):
        if False:
            return 10
        if key not in self:
            self.__setitem__(key, default)
        return self.__getitem__(key)

    def pop(self, key):
        if False:
            i = 10
            return i + 15
        key = self._normalize_key(key)
        value = self.__getitem__(key)
        self.__delitem__(key)
        return value

    def popitem(self):
        if False:
            while True:
                i = 10
        if not self.__omap:
            raise KeyError('VDFDict is empty')
        key = self.__omap[-1]
        return (key[1], self.pop(key))

    def update(self, data=None, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(data, dict):
            data = data.items()
        elif not isinstance(data, list):
            raise TypeError('Expected data to be a list or dict, got %s' % type(data))
        for (key, value) in data:
            self.__setitem__(key, value)

    def iterkeys(self):
        if False:
            for i in range(10):
                print('nop')
        return (key[1] for key in self.__omap)

    def keys(self):
        if False:
            while True:
                i = 10
        return _kView(self)

    def itervalues(self):
        if False:
            i = 10
            return i + 15
        return (self[key] for key in self.__omap)

    def values(self):
        if False:
            return 10
        return _vView(self)

    def iteritems(self):
        if False:
            return 10
        return ((key[1], self[key]) for key in self.__omap)

    def items(self):
        if False:
            return 10
        return _iView(self)

    def get_all_for(self, key):
        if False:
            for i in range(10):
                print('nop')
        ' Returns all values of the given key '
        if not isinstance(key, _string_type):
            raise TypeError('Key needs to be a string.')
        return [self[idx, key] for idx in _range(self.__kcount[key])]

    def remove_all_for(self, key):
        if False:
            i = 10
            return i + 15
        ' Removes all items with the given key '
        if not isinstance(key, _string_type):
            raise TypeError('Key need to be a string.')
        for idx in _range(self.__kcount[key]):
            super(VDFDict, self).__delitem__((idx, key))
        self.__omap = list(filter(lambda x: x[1] != key, self.__omap))
        del self.__kcount[key]

    def has_duplicates(self):
        if False:
            print('Hello World!')
        '\n        Returns ``True`` if the dict contains keys with duplicates.\n        Recurses through any all keys with value that is ``VDFDict``.\n        '
        for n in getattr(self.__kcount, _iter_values)():
            if n != 1:
                return True

        def dict_recurse(obj):
            if False:
                while True:
                    i = 10
            for v in getattr(obj, _iter_values)():
                if isinstance(v, VDFDict) and v.has_duplicates():
                    return True
                elif isinstance(v, dict):
                    return dict_recurse(v)
            return False
        return dict_recurse(self)