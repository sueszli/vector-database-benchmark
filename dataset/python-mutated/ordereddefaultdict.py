from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from .py3 import iteritems

class OrderedDefaultdict(OrderedDict):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not args:
            self.default_factory = None
        else:
            if not (args[0] is None or callable(args[0])):
                raise TypeError('first argument must be callable or None')
            self.default_factory = args[0]
            args = args[1:]
        super(OrderedDefaultdict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        if False:
            print('Hello World!')
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = default = self.default_factory()
        return default

    def __reduce__(self):
        if False:
            print('Hello World!')
        args = (self.default_factory,) if self.default_factory else ()
        return (self.__class__, args, None, None, iteritems(self))