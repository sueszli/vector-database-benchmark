from UserDict import DictMixin

class OrderedDict(dict, DictMixin):

    def __init__(self, *args, **kwds):
        if False:
            print('Hello World!')
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        try:
            self.__end
        except AttributeError:
            self.clear()
        self.update(*args, **kwds)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.__end = end = []
        end += [None, end, end]
        self.__map = {}
        dict.clear(self)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        if key not in self:
            end = self.__end
            curr = end[1]
            curr[2] = end[1] = self.__map[key] = [key, curr, end]
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        dict.__delitem__(self, key)
        (key, prev, next) = self.__map.pop(key)
        prev[2] = next
        next[1] = prev

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        end = self.__end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        if False:
            return 10
        end = self.__end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def popitem(self, last=True):
        if False:
            print('Hello World!')
        if not self:
            raise KeyError('dictionary is empty')
        if last:
            key = next(reversed(self))
        else:
            key = next(iter(self))
        value = self.pop(key)
        return (key, value)

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
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
            return 10
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
            print('Hello World!')
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.items()))

    def copy(self):
        if False:
            while True:
                i = 10
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        if False:
            for i in range(10):
                print('nop')
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, OrderedDict):
            return len(self) == len(other) and list(self.items()) == list(other.items())
        return dict.__eq__(self, other)

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other