"""
PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020 Python Software Foundation;
All Rights Reserved" are retained in Python alone or in any derivative version
prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.
"""
'Utilities for with-statement contexts.  See PEP 343.'
import abc
import sys
import _collections_abc
from collections import deque
from functools import wraps
__all__ = ['asynccontextmanager', 'contextmanager', 'closing', 'nullcontext', 'AbstractContextManager', 'AbstractAsyncContextManager', 'AsyncExitStack', 'ContextDecorator', 'ExitStack', 'redirect_stdout', 'redirect_stderr', 'suppress']

class AbstractContextManager(abc.ABC):
    """An abstract base class for context managers."""

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        'Return `self` upon entering the runtime context.'
        return self

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        'Raise any exception triggered within the runtime context.'
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            i = 10
            return i + 15
        if cls is AbstractContextManager:
            return _collections_abc._check_methods(C, '__enter__', '__exit__')
        return NotImplemented

class AbstractAsyncContextManager(abc.ABC):
    """An abstract base class for asynchronous context managers."""

    async def __aenter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if False:
            i = 10
            return i + 15
        if cls is AbstractAsyncContextManager:
            return _collections_abc._check_methods(C, '__aenter__', '__aexit__')
        return NotImplemented

class ContextDecorator(object):
    """A base class or mixin that enables context managers to work as decorators."""

    def _recreate_cm(self):
        if False:
            while True:
                i = 10
        'Return a recreated instance of self.\n\n        Allows an otherwise one-shot context manager like\n        _GeneratorContextManager to support use as\n        a decorator via implicit recreation.\n\n        This is a private interface just for _GeneratorContextManager.\n        See issue #11647 for details.\n        '
        return self

    def __call__(self, func):
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def inner(*args, **kwds):
            if False:
                i = 10
                return i + 15
            with self._recreate_cm():
                return func(*args, **kwds)
        return inner

class _GeneratorContextManagerBase:
    """Shared functionality for @contextmanager and @asynccontextmanager."""

    def __init__(self, func, args, kwds):
        if False:
            while True:
                i = 10
        self.gen = func(*args, **kwds)
        (self.func, self.args, self.kwds) = (func, args, kwds)
        doc = getattr(func, '__doc__', None)
        if doc is None:
            doc = type(self).__doc__
        self.__doc__ = doc

class _GeneratorContextManager(_GeneratorContextManagerBase, AbstractContextManager, ContextDecorator):
    """Helper for @contextmanager decorator."""

    def _recreate_cm(self):
        if False:
            i = 10
            return i + 15
        return self.__class__(self.func, self.args, self.kwds)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        del self.args, self.kwds, self.func
        try:
            return next(self.gen)
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        if type is None:
            try:
                next(self.gen)
            except StopIteration:
                return False
            else:
                raise RuntimeError("generator didn't stop")
        else:
            if value is None:
                value = type()
            try:
                self.gen.throw(type, value, traceback)
            except StopIteration as exc:
                return exc is not value
            except RuntimeError as exc:
                if exc is value:
                    return False
                if type is StopIteration and exc.__cause__ is value:
                    return False
                raise
            except:
                if sys.exc_info()[1] is value:
                    return False
                raise
            raise RuntimeError("generator didn't stop after throw()")

class _AsyncGeneratorContextManager(_GeneratorContextManagerBase, AbstractAsyncContextManager):
    """Helper for @asynccontextmanager."""

    async def __aenter__(self):
        try:
            return await self.gen.__anext__()
        except StopAsyncIteration:
            raise RuntimeError("generator didn't yield") from None

    async def __aexit__(self, typ, value, traceback):
        if typ is None:
            try:
                await self.gen.__anext__()
            except StopAsyncIteration:
                return
            else:
                raise RuntimeError("generator didn't stop")
        else:
            if value is None:
                value = typ()
            try:
                await self.gen.athrow(typ, value, traceback)
                raise RuntimeError("generator didn't stop after throw()")
            except StopAsyncIteration as exc:
                return exc is not value
            except RuntimeError as exc:
                if exc is value:
                    return False
                if isinstance(value, (StopIteration, StopAsyncIteration)):
                    if exc.__cause__ is value:
                        return False
                raise
            except BaseException as exc:
                if exc is not value:
                    raise

def contextmanager(func):
    if False:
        print('Hello World!')
    '@contextmanager decorator.\n\n    Typical usage:\n\n        @contextmanager\n        def some_generator(<arguments>):\n            <setup>\n            try:\n                yield <value>\n            finally:\n                <cleanup>\n\n    This makes this:\n\n        with some_generator(<arguments>) as <variable>:\n            <body>\n\n    equivalent to this:\n\n        <setup>\n        try:\n            <variable> = <value>\n            <body>\n        finally:\n            <cleanup>\n    '

    @wraps(func)
    def helper(*args, **kwds):
        if False:
            i = 10
            return i + 15
        return _GeneratorContextManager(func, args, kwds)
    return helper

def asynccontextmanager(func):
    if False:
        for i in range(10):
            print('nop')
    '@asynccontextmanager decorator.\n\n    Typical usage:\n\n        @asynccontextmanager\n        async def some_async_generator(<arguments>):\n            <setup>\n            try:\n                yield <value>\n            finally:\n                <cleanup>\n\n    This makes this:\n\n        async with some_async_generator(<arguments>) as <variable>:\n            <body>\n\n    equivalent to this:\n\n        <setup>\n        try:\n            <variable> = <value>\n            <body>\n        finally:\n            <cleanup>\n    '

    @wraps(func)
    def helper(*args, **kwds):
        if False:
            i = 10
            return i + 15
        return _AsyncGeneratorContextManager(func, args, kwds)
    return helper

class closing(AbstractContextManager):
    """Context to automatically close something at the end of a block.

    Code like this:

        with closing(<module>.open(<arguments>)) as f:
            <block>

    is equivalent to this:

        f = <module>.open(<arguments>)
        try:
            <block>
        finally:
            f.close()

    """

    def __init__(self, thing):
        if False:
            return 10
        self.thing = thing

    def __enter__(self):
        if False:
            print('Hello World!')
        return self.thing

    def __exit__(self, *exc_info):
        if False:
            for i in range(10):
                print('nop')
        self.thing.close()

class _RedirectStream(AbstractContextManager):
    _stream = None

    def __init__(self, new_target):
        if False:
            while True:
                i = 10
        self._new_target = new_target
        self._old_targets = []

    def __enter__(self):
        if False:
            return 10
        self._old_targets.append(getattr(sys, self._stream))
        setattr(sys, self._stream, self._new_target)
        return self._new_target

    def __exit__(self, exctype, excinst, exctb):
        if False:
            i = 10
            return i + 15
        setattr(sys, self._stream, self._old_targets.pop())

class redirect_stdout(_RedirectStream):
    """Context manager for temporarily redirecting stdout to another file.

        # How to send help() to stderr
        with redirect_stdout(sys.stderr):
            help(dir)

        # How to write help() to a file
        with open('help.txt', 'w') as f:
            with redirect_stdout(f):
                help(pow)
    """
    _stream = 'stdout'

class redirect_stderr(_RedirectStream):
    """Context manager for temporarily redirecting stderr to another file."""
    _stream = 'stderr'

class suppress(AbstractContextManager):
    """Context manager to suppress specified exceptions

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

         with suppress(FileNotFoundError):
             os.remove(somefile)
         # Execution still resumes here if the file was already removed
    """

    def __init__(self, *exceptions):
        if False:
            for i in range(10):
                print('nop')
        self._exceptions = exceptions

    def __enter__(self):
        if False:
            while True:
                i = 10
        pass

    def __exit__(self, exctype, excinst, exctb):
        if False:
            i = 10
            return i + 15
        return exctype is not None and issubclass(exctype, self._exceptions)

class _BaseExitStack:
    """A base class for ExitStack and AsyncExitStack."""

    @staticmethod
    def _create_exit_wrapper(cm, cm_exit):
        if False:
            for i in range(10):
                print('nop')

        def _exit_wrapper(exc_type, exc, tb):
            if False:
                return 10
            return cm_exit(cm, exc_type, exc, tb)
        return _exit_wrapper

    @staticmethod
    def _create_cb_wrapper(*args, **kwds):
        if False:
            return 10
        (callback, *args) = args

        def _exit_wrapper(exc_type, exc, tb):
            if False:
                return 10
            callback(*args, **kwds)
        return _exit_wrapper

    def __init__(self):
        if False:
            while True:
                i = 10
        self._exit_callbacks = deque()

    def pop_all(self):
        if False:
            i = 10
            return i + 15
        'Preserve the context stack by transferring it to a new instance.'
        new_stack = type(self)()
        new_stack._exit_callbacks = self._exit_callbacks
        self._exit_callbacks = deque()
        return new_stack

    def push(self, exit):
        if False:
            i = 10
            return i + 15
        'Registers a callback with the standard __exit__ method signature.\n\n        Can suppress exceptions the same way __exit__ method can.\n        Also accepts any object with an __exit__ method (registering a call\n        to the method instead of the object itself).\n        '
        _cb_type = type(exit)
        try:
            exit_method = _cb_type.__exit__
        except AttributeError:
            self._push_exit_callback(exit)
        else:
            self._push_cm_exit(exit, exit_method)
        return exit

    def enter_context(self, cm):
        if False:
            print('Hello World!')
        'Enters the supplied context manager.\n\n        If successful, also pushes its __exit__ method as a callback and\n        returns the result of the __enter__ method.\n        '
        _cm_type = type(cm)
        _exit = _cm_type.__exit__
        result = _cm_type.__enter__(cm)
        self._push_cm_exit(cm, _exit)
        return result

    def callback(*args, **kwds):
        if False:
            print('Hello World!')
        'Registers an arbitrary callback and arguments.\n\n        Cannot suppress exceptions.\n        '
        if len(args) >= 2:
            (self, callback, *args) = args
        elif not args:
            raise TypeError("descriptor 'callback' of '_BaseExitStack' object needs an argument")
        elif 'callback' in kwds:
            callback = kwds.pop('callback')
            (self, *args) = args
        else:
            raise TypeError('callback expected at least 1 positional argument, got %d' % (len(args) - 1))
        _exit_wrapper = self._create_cb_wrapper(callback, *args, **kwds)
        _exit_wrapper.__wrapped__ = callback
        self._push_exit_callback(_exit_wrapper)
        return callback

    def _push_cm_exit(self, cm, cm_exit):
        if False:
            return 10
        'Helper to correctly register callbacks to __exit__ methods.'
        _exit_wrapper = self._create_exit_wrapper(cm, cm_exit)
        _exit_wrapper.__self__ = cm
        self._push_exit_callback(_exit_wrapper, True)

    def _push_exit_callback(self, callback, is_sync=True):
        if False:
            while True:
                i = 10
        self._exit_callbacks.append((is_sync, callback))

class ExitStack(_BaseExitStack, AbstractContextManager):
    """Context manager for dynamic management of a stack of exit callbacks.

    For example:
        with ExitStack() as stack:
            files = [stack.enter_context(open(fname)) for fname in filenames]
            # All opened files will automatically be closed at the end of
            # the with statement, even if attempts to open files later
            # in the list raise an exception.
    """

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, *exc_details):
        if False:
            i = 10
            return i + 15
        received_exc = exc_details[0] is not None
        frame_exc = sys.exc_info()[1]

        def _fix_exception_context(new_exc, old_exc):
            if False:
                while True:
                    i = 10
            while 1:
                exc_context = new_exc.__context__
                if exc_context is old_exc:
                    return
                if exc_context is None or exc_context is frame_exc:
                    break
                new_exc = exc_context
            new_exc.__context__ = old_exc
        suppressed_exc = False
        pending_raise = False
        while self._exit_callbacks:
            (is_sync, cb) = self._exit_callbacks.pop()
            assert is_sync
            try:
                if cb(*exc_details):
                    suppressed_exc = True
                    pending_raise = False
                    exc_details = (None, None, None)
            except:
                new_exc_details = sys.exc_info()
                _fix_exception_context(new_exc_details[1], exc_details[1])
                pending_raise = True
                exc_details = new_exc_details
        if pending_raise:
            try:
                fixed_ctx = exc_details[1].__context__
                raise exc_details[1]
            except BaseException:
                exc_details[1].__context__ = fixed_ctx
                raise
        return received_exc and suppressed_exc

    def close(self):
        if False:
            return 10
        'Immediately unwind the context stack.'
        self.__exit__(None, None, None)

class AsyncExitStack(_BaseExitStack, AbstractAsyncContextManager):
    """Async context manager for dynamic management of a stack of exit
    callbacks.

    For example:
        async with AsyncExitStack() as stack:
            connections = [await stack.enter_async_context(get_connection())
                for i in range(5)]
            # All opened connections will automatically be released at the
            # end of the async with statement, even if attempts to open a
            # connection later in the list raise an exception.
    """

    @staticmethod
    def _create_async_exit_wrapper(cm, cm_exit):
        if False:
            return 10

        async def _exit_wrapper(exc_type, exc, tb):
            return await cm_exit(cm, exc_type, exc, tb)
        return _exit_wrapper

    @staticmethod
    def _create_async_cb_wrapper(*args, **kwds):
        if False:
            return 10
        (callback, *args) = args

        async def _exit_wrapper(exc_type, exc, tb):
            await callback(*args, **kwds)
        return _exit_wrapper

    async def enter_async_context(self, cm):
        """Enters the supplied async context manager.

        If successful, also pushes its __aexit__ method as a callback and
        returns the result of the __aenter__ method.
        """
        _cm_type = type(cm)
        _exit = _cm_type.__aexit__
        result = await _cm_type.__aenter__(cm)
        self._push_async_cm_exit(cm, _exit)
        return result

    def push_async_exit(self, exit):
        if False:
            i = 10
            return i + 15
        'Registers a coroutine function with the standard __aexit__ method\n        signature.\n\n        Can suppress exceptions the same way __aexit__ method can.\n        Also accepts any object with an __aexit__ method (registering a call\n        to the method instead of the object itself).\n        '
        _cb_type = type(exit)
        try:
            exit_method = _cb_type.__aexit__
        except AttributeError:
            self._push_exit_callback(exit, False)
        else:
            self._push_async_cm_exit(exit, exit_method)
        return exit

    def push_async_callback(*args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'Registers an arbitrary coroutine function and arguments.\n\n        Cannot suppress exceptions.\n        '
        if len(args) >= 2:
            (self, callback, *args) = args
        elif not args:
            raise TypeError("descriptor 'push_async_callback' of 'AsyncExitStack' object needs an argument")
        elif 'callback' in kwds:
            callback = kwds.pop('callback')
            (self, *args) = args
        else:
            raise TypeError('push_async_callback expected at least 1 positional argument, got %d' % (len(args) - 1))
        _exit_wrapper = self._create_async_cb_wrapper(callback, *args, **kwds)
        _exit_wrapper.__wrapped__ = callback
        self._push_exit_callback(_exit_wrapper, False)
        return callback

    async def aclose(self):
        """Immediately unwind the context stack."""
        await self.__aexit__(None, None, None)

    def _push_async_cm_exit(self, cm, cm_exit):
        if False:
            return 10
        'Helper to correctly register coroutine function to __aexit__\n        method.'
        _exit_wrapper = self._create_async_exit_wrapper(cm, cm_exit)
        _exit_wrapper.__self__ = cm
        self._push_exit_callback(_exit_wrapper, False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_details):
        received_exc = exc_details[0] is not None
        frame_exc = sys.exc_info()[1]

        def _fix_exception_context(new_exc, old_exc):
            if False:
                i = 10
                return i + 15
            while 1:
                exc_context = new_exc.__context__
                if exc_context is old_exc:
                    return
                if exc_context is None or exc_context is frame_exc:
                    break
                new_exc = exc_context
            new_exc.__context__ = old_exc
        suppressed_exc = False
        pending_raise = False
        while self._exit_callbacks:
            (is_sync, cb) = self._exit_callbacks.pop()
            try:
                if is_sync:
                    cb_suppress = cb(*exc_details)
                else:
                    cb_suppress = await cb(*exc_details)
                if cb_suppress:
                    suppressed_exc = True
                    pending_raise = False
                    exc_details = (None, None, None)
            except:
                new_exc_details = sys.exc_info()
                _fix_exception_context(new_exc_details[1], exc_details[1])
                pending_raise = True
                exc_details = new_exc_details
        if pending_raise:
            try:
                fixed_ctx = exc_details[1].__context__
                raise exc_details[1]
            except BaseException:
                exc_details[1].__context__ = fixed_ctx
                raise
        return received_exc and suppressed_exc

class nullcontext(AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, enter_result=None):
        if False:
            for i in range(10):
                print('nop')
        self.enter_result = enter_result

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self.enter_result

    def __exit__(self, *excinfo):
        if False:
            print('Hello World!')
        pass