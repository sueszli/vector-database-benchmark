__all__ = ['synchronized', 'lazy_classproperty', 'lazy_property', 'classproperty', 'conditional']
from functools import wraps
from inspect import getsourcefile

class lazy_property(object):
    """ Decorator for a lazy property of an object, i.e., an object attribute
        that is determined by the result of a method call evaluated once. To
        reevaluate the property, simply delete the attribute on the object, and
        get it again.
    """

    def __init__(self, fget):
        if False:
            return 10
        self.fget = fget

    def __get__(self, obj, cls):
        if False:
            for i in range(10):
                print('nop')
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value

    @property
    def __doc__(self):
        if False:
            i = 10
            return i + 15
        return self.fget.__doc__

    @staticmethod
    def reset_all(obj):
        if False:
            for i in range(10):
                print('nop')
        ' Reset all lazy properties on the instance `obj`. '
        cls = type(obj)
        obj_dict = vars(obj)
        for name in obj_dict.keys():
            if isinstance(getattr(cls, name, None), lazy_property):
                obj_dict.pop(name)

class lazy_classproperty(lazy_property):
    """ Similar to :class:`lazy_property`, but for classes. """

    def __get__(self, obj, cls):
        if False:
            return 10
        val = self.fget(cls)
        setattr(cls, self.fget.__name__, val)
        return val

def conditional(condition, decorator):
    if False:
        i = 10
        return i + 15
    " Decorator for a conditionally applied decorator.\n\n        Example:\n\n           @conditional(get_config('use_cache'), ormcache)\n           def fn():\n               pass\n    "
    if condition:
        return decorator
    else:
        return lambda fn: fn

def synchronized(lock_attr='_lock'):
    if False:
        i = 10
        return i + 15

    def decorator(func):
        if False:
            return 10

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            lock = getattr(self, lock_attr)
            try:
                lock.acquire()
                return func(self, *args, **kwargs)
            finally:
                lock.release()
        return wrapper
    return decorator

def frame_codeinfo(fframe, back=0):
    if False:
        print('Hello World!')
    " Return a (filename, line) pair for a previous frame .\n        @return (filename, lineno) where lineno is either int or string==''\n    "
    try:
        if not fframe:
            return ('<unknown>', '')
        for i in range(back):
            fframe = fframe.f_back
        try:
            fname = getsourcefile(fframe)
        except TypeError:
            fname = '<builtin>'
        lineno = fframe.f_lineno or ''
        return (fname, lineno)
    except Exception:
        return ('<unknown>', '')

def compose(a, b):
    if False:
        for i in range(10):
            print('nop')
    ' Composes the callables ``a`` and ``b``. ``compose(a, b)(*args)`` is\n    equivalent to ``a(b(*args))``.\n\n    Can be used as a decorator by partially applying ``a``::\n\n         @partial(compose, a)\n         def b():\n            ...\n    '

    @wraps(b)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return a(b(*args, **kwargs))
    return wrapper

class _ClassProperty(property):

    def __get__(self, cls, owner):
        if False:
            print('Hello World!')
        return self.fget.__get__(None, owner)()

def classproperty(func):
    if False:
        for i in range(10):
            print('nop')
    return _ClassProperty(classmethod(func))