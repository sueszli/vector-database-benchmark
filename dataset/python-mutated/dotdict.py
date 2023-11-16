from collections import OrderedDict
from .robottypes import is_dict_like

class DotDict(OrderedDict):

    def __init__(self, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        args = [self._convert_nested_initial_dicts(a) for a in args]
        kwds = self._convert_nested_initial_dicts(kwds)
        OrderedDict.__init__(self, *args, **kwds)

    def _convert_nested_initial_dicts(self, value):
        if False:
            for i in range(10):
                print('nop')
        items = value.items() if is_dict_like(value) else value
        return OrderedDict(((key, self._convert_nested_dicts(value)) for (key, value) in items))

    def _convert_nested_dicts(self, value):
        if False:
            print('Hello World!')
        if isinstance(value, DotDict):
            return value
        if is_dict_like(value):
            return DotDict(value)
        if isinstance(value, list):
            value[:] = [self._convert_nested_dicts(item) for item in value]
        return value

    def __getattr__(self, key):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if False:
            print('Hello World!')
        if not key.startswith('_OrderedDict__'):
            self[key] = value
        else:
            OrderedDict.__setattr__(self, key, value)

    def __delattr__(self, key):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.pop(key)
        except KeyError:
            OrderedDict.__delattr__(self, key)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return dict.__eq__(self, other)

    def __str__(self):
        if False:
            print('Hello World!')
        return '{%s}' % ', '.join(('%r: %r' % (key, self[key]) for key in self))
    __repr__ = dict.__repr__