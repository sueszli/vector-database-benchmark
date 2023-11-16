import contextvars
import inspect
_in__init__ = contextvars.ContextVar('_immutable_in__init__', default=False)

class _Immutable:
    """Immutable mixin class"""
    __slots__ = ()

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        if _in__init__.get() is not self:
            raise TypeError("object doesn't support attribute assignment")
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if False:
            i = 10
            return i + 15
        if _in__init__.get() is not self:
            raise TypeError("object doesn't support attribute assignment")
        else:
            super().__delattr__(name)

def _immutable_init(f):
    if False:
        i = 10
        return i + 15

    def nf(*args, **kwargs):
        if False:
            return 10
        previous = _in__init__.set(args[0])
        try:
            f(*args, **kwargs)
        finally:
            _in__init__.reset(previous)
    nf.__signature__ = inspect.signature(f)
    return nf

def immutable(cls):
    if False:
        print('Hello World!')
    if _Immutable in cls.__mro__:
        cls.__init__ = _immutable_init(cls.__init__)
        if hasattr(cls, '__setstate__'):
            cls.__setstate__ = _immutable_init(cls.__setstate__)
        ncls = cls
    else:

        class ncls(_Immutable, cls):
            __slots__ = ()

            @_immutable_init
            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                super().__init__(*args, **kwargs)
            if hasattr(cls, '__setstate__'):

                @_immutable_init
                def __setstate__(self, *args, **kwargs):
                    if False:
                        i = 10
                        return i + 15
                    super().__setstate__(*args, **kwargs)
        ncls.__name__ = cls.__name__
        ncls.__qualname__ = cls.__qualname__
        ncls.__module__ = cls.__module__
    return ncls