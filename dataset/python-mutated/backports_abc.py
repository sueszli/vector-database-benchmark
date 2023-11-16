"""
Patch recently added ABCs into the standard lib module
``collections.abc`` (Py3) or ``collections`` (Py2).

Usage::

    import backports_abc
    backports_abc.patch()

or::

    try:
        from collections.abc import Generator
    except ImportError:
        from backports_abc import Generator
"""
try:
    import collections.abc as _collections_abc
except ImportError:
    import collections as _collections_abc

def get_mro(cls):
    if False:
        print('Hello World!')
    try:
        return cls.__mro__
    except AttributeError:
        return old_style_mro(cls)

def old_style_mro(cls):
    if False:
        for i in range(10):
            print('nop')
    yield cls
    for base in cls.__bases__:
        for c in old_style_mro(base):
            yield c

def mk_gen():
    if False:
        print('Hello World!')
    from abc import abstractmethod
    required_methods = ('__iter__', '__next__' if hasattr(iter(()), '__next__') else 'next', 'send', 'throw', 'close')

    class Generator(_collections_abc.Iterator):
        __slots__ = ()
        if '__next__' in required_methods:

            def __next__(self):
                if False:
                    return 10
                return self.send(None)
        else:

            def next(self):
                if False:
                    return 10
                return self.send(None)

        @abstractmethod
        def send(self, value):
            if False:
                print('Hello World!')
            raise StopIteration

        @abstractmethod
        def throw(self, typ, val=None, tb=None):
            if False:
                for i in range(10):
                    print('nop')
            if val is None:
                if tb is None:
                    raise typ
                val = typ()
            if tb is not None:
                val = val.with_traceback(tb)
            raise val

        def close(self):
            if False:
                for i in range(10):
                    print('nop')
            try:
                self.throw(GeneratorExit)
            except (GeneratorExit, StopIteration):
                pass
            else:
                raise RuntimeError('generator ignored GeneratorExit')

        @classmethod
        def __subclasshook__(cls, C):
            if False:
                for i in range(10):
                    print('nop')
            if cls is Generator:
                mro = get_mro(C)
                for method in required_methods:
                    for base in mro:
                        if method in base.__dict__:
                            break
                    else:
                        return NotImplemented
                return True
            return NotImplemented
    generator = type((lambda : (yield))())
    Generator.register(generator)
    return Generator

def mk_awaitable():
    if False:
        for i in range(10):
            print('nop')
    from abc import abstractmethod, ABCMeta

    @abstractmethod
    def __await__(self):
        if False:
            return 10
        yield

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            print('Hello World!')
        if cls is Awaitable:
            for B in get_mro(C):
                if '__await__' in B.__dict__:
                    if B.__dict__['__await__']:
                        return True
                    break
        return NotImplemented
    Awaitable = ABCMeta('Awaitable', (), {'__slots__': (), '__await__': __await__, '__subclasshook__': __subclasshook__})
    return Awaitable

def mk_coroutine():
    if False:
        print('Hello World!')
    from abc import abstractmethod

    class Coroutine(Awaitable):
        __slots__ = ()

        @abstractmethod
        def send(self, value):
            if False:
                i = 10
                return i + 15
            'Send a value into the coroutine.\n            Return next yielded value or raise StopIteration.\n            '
            raise StopIteration

        @abstractmethod
        def throw(self, typ, val=None, tb=None):
            if False:
                print('Hello World!')
            'Raise an exception in the coroutine.\n            Return next yielded value or raise StopIteration.\n            '
            if val is None:
                if tb is None:
                    raise typ
                val = typ()
            if tb is not None:
                val = val.with_traceback(tb)
            raise val

        def close(self):
            if False:
                print('Hello World!')
            'Raise GeneratorExit inside coroutine.\n            '
            try:
                self.throw(GeneratorExit)
            except (GeneratorExit, StopIteration):
                pass
            else:
                raise RuntimeError('coroutine ignored GeneratorExit')

        @classmethod
        def __subclasshook__(cls, C):
            if False:
                return 10
            if cls is Coroutine:
                mro = get_mro(C)
                for method in ('__await__', 'send', 'throw', 'close'):
                    for base in mro:
                        if method in base.__dict__:
                            break
                    else:
                        return NotImplemented
                return True
            return NotImplemented
    return Coroutine
try:
    Generator = _collections_abc.Generator
except AttributeError:
    Generator = mk_gen()
try:
    Awaitable = _collections_abc.Awaitable
except AttributeError:
    Awaitable = mk_awaitable()
try:
    Coroutine = _collections_abc.Coroutine
except AttributeError:
    Coroutine = mk_coroutine()
try:
    from inspect import isawaitable
except ImportError:

    def isawaitable(obj):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(obj, Awaitable)
PATCHED = {}

def patch(patch_inspect=True):
    if False:
        print('Hello World!')
    '\n    Main entry point for patching the ``collections.abc`` and ``inspect``\n    standard library modules.\n    '
    PATCHED['collections.abc.Generator'] = _collections_abc.Generator = Generator
    PATCHED['collections.abc.Coroutine'] = _collections_abc.Coroutine = Coroutine
    PATCHED['collections.abc.Awaitable'] = _collections_abc.Awaitable = Awaitable
    if patch_inspect:
        import inspect
        PATCHED['inspect.isawaitable'] = inspect.isawaitable = isawaitable