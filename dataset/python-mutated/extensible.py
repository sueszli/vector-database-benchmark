from functools import wraps, lru_cache
__all__ = ['Extensible', 'cache', 'drawcache', 'drawcache_property']

class Extensible:
    _cache_clearers = []

    @classmethod
    def init(cls, membername, initfunc=lambda : None, copy=False):
        if False:
            i = 10
            return i + 15
        'Prepend equivalent of ``self.<membername> = initfunc()`` to ``<cls>.__init__``.  If *copy* is True, <membername> will be copied when object is copied.'

        def thisclass_hasattr(cls, k):
            if False:
                print('Hello World!')
            return getattr(cls, k, None) is not getattr(cls.__bases__[0], k, None)
        oldinit = thisclass_hasattr(cls, '__init__') and getattr(cls, '__init__')

        def newinit(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            if not hasattr(self, membername):
                setattr(self, membername, initfunc())
            if oldinit:
                oldinit(self, *args, **kwargs)
            else:
                super(cls, self).__init__(*args, **kwargs)
        cls.__init__ = wraps(oldinit)(newinit) if oldinit else newinit
        oldcopy = thisclass_hasattr(cls, '__copy__') and getattr(cls, '__copy__')

        def newcopy(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if oldcopy:
                ret = oldcopy(self, *args, **kwargs)
            else:
                ret = super(cls, self).__copy__(*args, **kwargs)
            setattr(ret, membername, getattr(self, membername) if copy and hasattr(self, membername) else initfunc())
            return ret
        cls.__copy__ = wraps(oldcopy)(newcopy) if oldcopy else newcopy

    @classmethod
    def superclasses(cls):
        if False:
            return 10
        yield cls
        yield from cls.__bases__
        for b in cls.__bases__:
            if hasattr(b, 'superclasses'):
                yield from b.superclasses()

    @classmethod
    def api(cls, func):
        if False:
            while True:
                i = 10
        oldfunc = getattr(cls, func.__name__, None)
        if oldfunc:
            func = wraps(oldfunc)(func)
        from visidata import vd
        func.importingModule = vd.importingModule
        setattr(cls, func.__name__, func)
        return func

    @classmethod
    def before(cls, beforefunc):
        if False:
            for i in range(10):
                print('nop')
        funcname = beforefunc.__name__
        oldfunc = getattr(cls, funcname, None)
        if not oldfunc:
            setattr(cls, funcname, beforefunc)

        @wraps(oldfunc)
        def wrappedfunc(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            beforefunc(*args, **kwargs)
            return oldfunc(*args, **kwargs)
        setattr(cls, funcname, wrappedfunc)
        return wrappedfunc

    @classmethod
    def after(cls, afterfunc):
        if False:
            i = 10
            return i + 15
        funcname = afterfunc.__name__
        oldfunc = getattr(cls, funcname, None)
        if not oldfunc:
            setattr(cls, funcname, afterfunc)

        @wraps(oldfunc)
        def wrappedfunc(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            r = oldfunc(*args, **kwargs)
            afterfunc(*args, **kwargs)
            return r
        setattr(cls, funcname, wrappedfunc)
        return wrappedfunc

    @classmethod
    def class_api(cls, func):
        if False:
            i = 10
            return i + 15
        name = func.__get__(None, dict).__func__.__name__
        oldfunc = getattr(cls, name, None)
        if oldfunc:
            func = wraps(oldfunc)(func)
        setattr(cls, name, func)
        return func

    @classmethod
    def property(cls, func):
        if False:
            for i in range(10):
                print('nop')

        @property
        @wraps(func)
        def dofunc(self):
            if False:
                while True:
                    i = 10
            return func(self)
        setattr(cls, func.__name__, dofunc)
        return dofunc

    @classmethod
    def lazy_property(cls, func):
        if False:
            return 10
        'Return ``func()`` on first access and cache result; return cached result thereafter.'
        name = '_' + func.__name__
        cls.init(name, lambda : None, copy=False)

        @property
        @wraps(func)
        def get_if_not(self):
            if False:
                while True:
                    i = 10
            if getattr(self, name, None) is None:
                setattr(self, name, func(self))
            return getattr(self, name)
        setattr(cls, func.__name__, get_if_not)
        return get_if_not

    @classmethod
    def cached_property(cls, func):
        if False:
            while True:
                i = 10
        'Return ``func()`` on first access, and cache result; return cached result until ``clearCaches()``.'

        @property
        @wraps(func)
        @lru_cache(maxsize=None)
        def get_if_not(self):
            if False:
                i = 10
                return i + 15
            return func(self)
        setattr(cls, func.__name__, get_if_not)
        Extensible._cache_clearers.append(get_if_not.fget.cache_clear)
        return get_if_not

    @classmethod
    def clear_all_caches(cls):
        if False:
            i = 10
            return i + 15
        for func in Extensible._cache_clearers:
            func()

def cache(func):
    if False:
        i = 10
        return i + 15
    'Return func(...) on first access, and cache result; return cached result until clearCaches().'

    @wraps(func)
    @lru_cache(maxsize=None)
    def call_if_not(self, *args, **kwargs):
        if False:
            return 10
        return func(self, *args, **kwargs)
    Extensible._cache_clearers.append(call_if_not.cache_clear)
    return call_if_not
drawcache = cache

def drawcache_property(func):
    if False:
        while True:
            i = 10
    return property(drawcache(func))