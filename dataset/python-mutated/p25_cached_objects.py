"""
Topic: 创建缓存实例
Desc :
"""
import logging
a = logging.getLogger('foo')
b = logging.getLogger('bar')
print(a is b)
c = logging.getLogger('foo')
print(a is c)

class Spam:

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name
import weakref
_spam_cache = weakref.WeakValueDictionary()

def get_spam(name):
    if False:
        while True:
            i = 10
    if name not in _spam_cache:
        s = Spam(name)
        _spam_cache[name] = s
    else:
        s = _spam_cache[name]
    return s

class Spam1:
    _spam_cache = weakref.WeakValueDictionary()

    def __new__(cls, name):
        if False:
            while True:
                i = 10
        print('Spam1__new__')
        if name in cls._spam_cache:
            return cls._spam_cache[name]
        else:
            self = super().__new__(cls)
            cls._spam_cache[name] = self
            return self

    def __init__(self, name):
        if False:
            return 10
        print('Initializing Spam')
        self.name = name
s = Spam1('Dave')
t = Spam1('Dave')
print(s is t)

class CachedSpamManager:

    def __init__(self):
        if False:
            print('Hello World!')
        self._cache = weakref.WeakValueDictionary()

    def get_spam(self, name):
        if False:
            return 10
        if name not in self._cache:
            s = Spam(name)
            self._cache[name] = s
        else:
            s = self._cache[name]
        return s

    def clear(self):
        if False:
            print('Hello World!')
        self._cache.clear()

class Spam2:
    manager = CachedSpamManager()

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name

def get_spam(name):
    if False:
        return 10
    return Spam2.manager.get_spam(name)

class CachedSpamManager2:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._cache = weakref.WeakValueDictionary()

    def get_spam(self, name):
        if False:
            print('Hello World!')
        if name not in self._cache:
            temp = Spam3._new(name)
            self._cache[name] = temp
        else:
            temp = self._cache[name]
        return temp

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self._cache.clear()

class Spam3:

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise RuntimeError("Can't instantiate directly")

    @classmethod
    def _new(cls, name):
        if False:
            while True:
                i = 10
        self = cls.__new__(cls)
        self.name = name
        return self
print('------------------------------')
cachedSpamManager = CachedSpamManager2()
s = cachedSpamManager.get_spam('Dave')
t = cachedSpamManager.get_spam('Dave')
print(s is t)