import sys
if not sys.version_info < (2, 7):
    from collections import OrderedDict
else:
    from UserDict import DictMixin

    class OrderedDict(dict, DictMixin):

        def __init__(self, *args, **kwds):
            if False:
                return 10
            if len(args) > 1:
                raise TypeError('expected at most 1 arguments, got %d' % len(args))
            try:
                self.__end
            except AttributeError:
                self.clear()
            self.update(*args, **kwds)

        def clear(self):
            if False:
                print('Hello World!')
            self.__end = end = []
            end += [None, end, end]
            self.__map = {}
            dict.clear(self)

        def __setitem__(self, key, value):
            if False:
                print('Hello World!')
            if key not in self:
                end = self.__end
                curr = end[1]
                curr[2] = end[1] = self.__map[key] = [key, curr, end]
            dict.__setitem__(self, key, value)

        def __delitem__(self, key):
            if False:
                print('Hello World!')
            dict.__delitem__(self, key)
            (key, prev, next_) = self.__map.pop(key)
            prev[2] = next_
            next_[1] = prev

        def __iter__(self):
            if False:
                print('Hello World!')
            end = self.__end
            curr = end[2]
            while curr is not end:
                yield curr[0]
                curr = curr[2]

        def __reversed__(self):
            if False:
                while True:
                    i = 10
            end = self.__end
            curr = end[1]
            while curr is not end:
                yield curr[0]
                curr = curr[1]

        def popitem(self, last=True):
            if False:
                return 10
            if not self:
                raise KeyError('dictionary is empty')
            if last:
                key = reversed(self).next()
            else:
                key = iter(self).next()
            value = self.pop(key)
            return (key, value)

        def __reduce__(self):
            if False:
                print('Hello World!')
            items = [[k, self[k]] for k in self]
            tmp = (self.__map, self.__end)
            del self.__map, self.__end
            inst_dict = vars(self).copy()
            (self.__map, self.__end) = tmp
            if inst_dict:
                return (self.__class__, (items,), inst_dict)
            return (self.__class__, (items,))

        def keys(self):
            if False:
                while True:
                    i = 10
            return list(self)
        setdefault = DictMixin.setdefault
        update = DictMixin.update
        pop = DictMixin.pop
        values = DictMixin.values
        items = DictMixin.items
        iterkeys = DictMixin.iterkeys
        itervalues = DictMixin.itervalues
        iteritems = DictMixin.iteritems

        def __repr__(self):
            if False:
                while True:
                    i = 10
            if not self:
                return '%s()' % (self.__class__.__name__,)
            return '%s(%r)' % (self.__class__.__name__, self.items())

        def copy(self):
            if False:
                print('Hello World!')
            return self.__class__(self)

        @classmethod
        def fromkeys(cls, iterable, value=None):
            if False:
                print('Hello World!')
            d = cls()
            for key in iterable:
                d[key] = value
            return d

        def __eq__(self, other):
            if False:
                return 10
            if isinstance(other, OrderedDict):
                if len(self) != len(other):
                    return False
                for (p, q) in zip(self.items(), other.items()):
                    if p != q:
                        return False
                return True
            return dict.__eq__(self, other)

        def __ne__(self, other):
            if False:
                i = 10
                return i + 15
            return not self == other