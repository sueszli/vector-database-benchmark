from __future__ import absolute_import, division, print_function, unicode_literals
from .metabase import MetaParams
from .utils.py3 import with_metaclass
__all__ = ['Filter']

class MetaFilter(MetaParams):
    pass

class Filter(with_metaclass(MetaParams, object)):
    _firsttime = True

    def __init__(self, data):
        if False:
            return 10
        pass

    def __call__(self, data):
        if False:
            print('Hello World!')
        if self._firsttime:
            self.nextstart(data)
            self._firsttime = False
        self.next(data)

    def nextstart(self, data):
        if False:
            i = 10
            return i + 15
        pass

    def next(self, data):
        if False:
            while True:
                i = 10
        pass