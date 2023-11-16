"""
Topic: 混入类
Desc : 如果单独使用Minxin类没有任何意义，但是当利用多继承和其他类配合后就有神奇效果了。
    Mixin也是多继承的主要用途。
"""

class LoggedMappingMixin:
    """
    Add logging to get/set/delete operations for debugging.
    """
    __slots__ = ()

    def __getitem__(self, key):
        if False:
            return 10
        print('Getting ' + str(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        print('Setting {} = {!r}'.format(key, value))
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        if False:
            while True:
                i = 10
        print('Deleting ' + str(key))
        return super().__delitem__(key)

class SetOnceMappingMixin:
    """
    Only allow a key to be set once.
    """
    __slots__ = ()

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        if key in self:
            raise KeyError(str(key) + ' already set')
        return super().__setitem__(key, value)

class StringKeysMappingMixin:
    """
    Restrict keys to strings only
    """
    __slots__ = ()

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(key, str):
            raise TypeError('keys must be strings')
        return super().__setitem__(key, value)

class LoggedDict(LoggedMappingMixin, dict):
    pass
d = LoggedDict()
d['x'] = 23
print(d['x'])
del d['x']
from collections import defaultdict

class SetOnceDefaultDict(SetOnceMappingMixin, defaultdict):
    pass
d = SetOnceDefaultDict(list)
d['x'].append(2)
d['x'].append(3)

def LoggedMapping(cls):
    if False:
        return 10
    '第二种方式：使用类装饰器'
    cls_getitem = cls.__getitem__
    cls_setitem = cls.__setitem__
    cls_delitem = cls.__delitem__

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        print('Getting ' + str(key))
        return cls_getitem(self, key)

    def __setitem__(self, key, value):
        if False:
            return 10
        print('Setting {} = {!r}'.format(key, value))
        return cls_setitem(self, key, value)

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        print('Deleting ' + str(key))
        return cls_delitem(self, key)
    cls.__getitem__ = __getitem__
    cls.__setitem__ = __setitem__
    cls.__delitem__ = __delitem__
    return cls

@LoggedMapping
class LoggedDict(dict):
    pass