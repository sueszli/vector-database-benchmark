"""An I/O event loop for non-blocking sockets.

In Tornado 6.0, `.IOLoop` is a wrapper around the `asyncio` event loop, with a
slightly different interface. The `.IOLoop` interface is now provided primarily
for backwards compatibility; new code should generally use the `asyncio` event
loop interface directly. The `IOLoop.current` class method provides the
`IOLoop` instance corresponding to the running `asyncio` event loop.

"""
import asyncio
import concurrent.futures
import datetime
import functools
import numbers
import os
import sys
import time
import math
import random
import warnings
from inspect import isawaitable
from tornado.concurrent import Future, is_future, chain_future, future_set_exc_info, future_add_done_callback
from tornado.log import app_log
from tornado.util import Configurable, TimeoutError, import_object
import typing
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable
if typing.TYPE_CHECKING:
    from typing import Dict, List, Set
    from typing_extensions import Protocol
else:
    Protocol = object

class _Selectable(Protocol):

    def fileno(self) -> int:
        if False:
            return 10
        pass

    def close(self) -> None:
        if False:
            print('Hello World!')
        pass
_T = TypeVar('_T')
_S = TypeVar('_S', bound=_Selectable)

class IOLoop(Configurable):
    """An I/O event loop.

    As of Tornado 6.0, `IOLoop` is a wrapper around the `asyncio` event loop.

    Example usage for a simple TCP server:

    .. testcode::

        import asyncio
        import errno
        import functools
        import socket

        import tornado
        from tornado.iostream import IOStream

        async def handle_connection(connection, address):
            stream = IOStream(connection)
            message = await stream.read_until_close()
            print("message from client:", message.decode().strip())

        def connection_ready(sock, fd, events):
            while True:
                try:
                    connection, address = sock.accept()
                except BlockingIOError:
                    return
                connection.setblocking(0)
                io_loop = tornado.ioloop.IOLoop.current()
                io_loop.spawn_callback(handle_connection, connection, address)

        async def main():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setblocking(0)
            sock.bind(("", 8888))
            sock.listen(128)

            io_loop = tornado.ioloop.IOLoop.current()
            callback = functools.partial(connection_ready, sock)
            io_loop.add_handler(sock.fileno(), callback, io_loop.READ)
            await asyncio.Event().wait()

        if __name__ == "__main__":
            asyncio.run(main())

    .. testoutput::
       :hide:

    Most applications should not attempt to construct an `IOLoop` directly,
    and instead initialize the `asyncio` event loop and use `IOLoop.current()`.
    In some cases, such as in test frameworks when initializing an `IOLoop`
    to be run in a secondary thread, it may be appropriate to construct
    an `IOLoop` with ``IOLoop(make_current=False)``.

    In general, an `IOLoop` cannot survive a fork or be shared across processes
    in any way. When multiple processes are being used, each process should
    create its own `IOLoop`, which also implies that any objects which depend on
    the `IOLoop` (such as `.AsyncHTTPClient`) must also be created in the child
    processes. As a guideline, anything that starts processes (including the
    `tornado.process` and `multiprocessing` modules) should do so as early as
    possible, ideally the first thing the application does after loading its
    configuration, and *before* any calls to `.IOLoop.start` or `asyncio.run`.

    .. versionchanged:: 4.2
       Added the ``make_current`` keyword argument to the `IOLoop`
       constructor.

    .. versionchanged:: 5.0

       Uses the `asyncio` event loop by default. The ``IOLoop.configure`` method
       cannot be used on Python 3 except to redundantly specify the `asyncio`
       event loop.

    .. versionchanged:: 6.3
       ``make_current=True`` is now the default when creating an IOLoop -
       previously the default was to make the event loop current if there wasn't
       already a current one.
    """
    NONE = 0
    READ = 1
    WRITE = 4
    ERROR = 24
    _ioloop_for_asyncio = dict()
    _pending_tasks = set()

    @classmethod
    def configure(cls, impl: 'Union[None, str, Type[Configurable]]', **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        from tornado.platform.asyncio import BaseAsyncIOLoop
        if isinstance(impl, str):
            impl = import_object(impl)
        if isinstance(impl, type) and (not issubclass(impl, BaseAsyncIOLoop)):
            raise RuntimeError('only AsyncIOLoop is allowed when asyncio is available')
        super(IOLoop, cls).configure(impl, **kwargs)

    @staticmethod
    def instance() -> 'IOLoop':
        if False:
            print('Hello World!')
        "Deprecated alias for `IOLoop.current()`.\n\n        .. versionchanged:: 5.0\n\n           Previously, this method returned a global singleton\n           `IOLoop`, in contrast with the per-thread `IOLoop` returned\n           by `current()`. In nearly all cases the two were the same\n           (when they differed, it was generally used from non-Tornado\n           threads to communicate back to the main thread's `IOLoop`).\n           This distinction is not present in `asyncio`, so in order\n           to facilitate integration with that package `instance()`\n           was changed to be an alias to `current()`. Applications\n           using the cross-thread communications aspect of\n           `instance()` should instead set their own global variable\n           to point to the `IOLoop` they want to use.\n\n        .. deprecated:: 5.0\n        "
        return IOLoop.current()

    def install(self) -> None:
        if False:
            print('Hello World!')
        'Deprecated alias for `make_current()`.\n\n        .. versionchanged:: 5.0\n\n           Previously, this method would set this `IOLoop` as the\n           global singleton used by `IOLoop.instance()`. Now that\n           `instance()` is an alias for `current()`, `install()`\n           is an alias for `make_current()`.\n\n        .. deprecated:: 5.0\n        '
        self.make_current()

    @staticmethod
    def clear_instance() -> None:
        if False:
            print('Hello World!')
        'Deprecated alias for `clear_current()`.\n\n        .. versionchanged:: 5.0\n\n           Previously, this method would clear the `IOLoop` used as\n           the global singleton by `IOLoop.instance()`. Now that\n           `instance()` is an alias for `current()`,\n           `clear_instance()` is an alias for `clear_current()`.\n\n        .. deprecated:: 5.0\n\n        '
        IOLoop.clear_current()

    @typing.overload
    @staticmethod
    def current() -> 'IOLoop':
        if False:
            while True:
                i = 10
        pass

    @typing.overload
    @staticmethod
    def current(instance: bool=True) -> Optional['IOLoop']:
        if False:
            return 10
        pass

    @staticmethod
    def current(instance: bool=True) -> Optional['IOLoop']:
        if False:
            while True:
                i = 10
        "Returns the current thread's `IOLoop`.\n\n        If an `IOLoop` is currently running or has been marked as\n        current by `make_current`, returns that instance.  If there is\n        no current `IOLoop` and ``instance`` is true, creates one.\n\n        .. versionchanged:: 4.1\n           Added ``instance`` argument to control the fallback to\n           `IOLoop.instance()`.\n        .. versionchanged:: 5.0\n           On Python 3, control of the current `IOLoop` is delegated\n           to `asyncio`, with this and other methods as pass-through accessors.\n           The ``instance`` argument now controls whether an `IOLoop`\n           is created automatically when there is none, instead of\n           whether we fall back to `IOLoop.instance()` (which is now\n           an alias for this method). ``instance=False`` is deprecated,\n           since even if we do not create an `IOLoop`, this method\n           may initialize the asyncio loop.\n\n        .. deprecated:: 6.2\n           It is deprecated to call ``IOLoop.current()`` when no `asyncio`\n           event loop is running.\n        "
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            if not instance:
                return None
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            return IOLoop._ioloop_for_asyncio[loop]
        except KeyError:
            if instance:
                from tornado.platform.asyncio import AsyncIOMainLoop
                current = AsyncIOMainLoop()
            else:
                current = None
        return current

    def make_current(self) -> None:
        if False:
            return 10
        'Makes this the `IOLoop` for the current thread.\n\n        An `IOLoop` automatically becomes current for its thread\n        when it is started, but it is sometimes useful to call\n        `make_current` explicitly before starting the `IOLoop`,\n        so that code run at startup time can find the right\n        instance.\n\n        .. versionchanged:: 4.1\n           An `IOLoop` created while there is no current `IOLoop`\n           will automatically become current.\n\n        .. versionchanged:: 5.0\n           This method also sets the current `asyncio` event loop.\n\n        .. deprecated:: 6.2\n           Setting and clearing the current event loop through Tornado is\n           deprecated. Use ``asyncio.set_event_loop`` instead if you need this.\n        '
        warnings.warn('make_current is deprecated; start the event loop first', DeprecationWarning, stacklevel=2)
        self._make_current()

    def _make_current(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @staticmethod
    def clear_current() -> None:
        if False:
            while True:
                i = 10
        'Clears the `IOLoop` for the current thread.\n\n        Intended primarily for use by test frameworks in between tests.\n\n        .. versionchanged:: 5.0\n           This method also clears the current `asyncio` event loop.\n        .. deprecated:: 6.2\n        '
        warnings.warn('clear_current is deprecated', DeprecationWarning, stacklevel=2)
        IOLoop._clear_current()

    @staticmethod
    def _clear_current() -> None:
        if False:
            return 10
        old = IOLoop.current(instance=False)
        if old is not None:
            old._clear_current_hook()

    def _clear_current_hook(self) -> None:
        if False:
            print('Hello World!')
        'Instance method called when an IOLoop ceases to be current.\n\n        May be overridden by subclasses as a counterpart to make_current.\n        '
        pass

    @classmethod
    def configurable_base(cls) -> Type[Configurable]:
        if False:
            return 10
        return IOLoop

    @classmethod
    def configurable_default(cls) -> Type[Configurable]:
        if False:
            i = 10
            return i + 15
        from tornado.platform.asyncio import AsyncIOLoop
        return AsyncIOLoop

    def initialize(self, make_current: bool=True) -> None:
        if False:
            print('Hello World!')
        if make_current:
            self._make_current()

    def close(self, all_fds: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Closes the `IOLoop`, freeing any resources used.\n\n        If ``all_fds`` is true, all file descriptors registered on the\n        IOLoop will be closed (not just the ones created by the\n        `IOLoop` itself).\n\n        Many applications will only use a single `IOLoop` that runs for the\n        entire lifetime of the process.  In that case closing the `IOLoop`\n        is not necessary since everything will be cleaned up when the\n        process exits.  `IOLoop.close` is provided mainly for scenarios\n        such as unit tests, which create and destroy a large number of\n        ``IOLoops``.\n\n        An `IOLoop` must be completely stopped before it can be closed.  This\n        means that `IOLoop.stop()` must be called *and* `IOLoop.start()` must\n        be allowed to return before attempting to call `IOLoop.close()`.\n        Therefore the call to `close` will usually appear just after\n        the call to `start` rather than near the call to `stop`.\n\n        .. versionchanged:: 3.1\n           If the `IOLoop` implementation supports non-integer objects\n           for "file descriptors", those objects will have their\n           ``close`` method when ``all_fds`` is true.\n        '
        raise NotImplementedError()

    @typing.overload
    def add_handler(self, fd: int, handler: Callable[[int, int], None], events: int) -> None:
        if False:
            while True:
                i = 10
        pass

    @typing.overload
    def add_handler(self, fd: _S, handler: Callable[[_S, int], None], events: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def add_handler(self, fd: Union[int, _Selectable], handler: Callable[..., None], events: int) -> None:
        if False:
            i = 10
            return i + 15
        'Registers the given handler to receive the given events for ``fd``.\n\n        The ``fd`` argument may either be an integer file descriptor or\n        a file-like object with a ``fileno()`` and ``close()`` method.\n\n        The ``events`` argument is a bitwise or of the constants\n        ``IOLoop.READ``, ``IOLoop.WRITE``, and ``IOLoop.ERROR``.\n\n        When an event occurs, ``handler(fd, events)`` will be run.\n\n        .. versionchanged:: 4.0\n           Added the ability to pass file-like objects in addition to\n           raw file descriptors.\n        '
        raise NotImplementedError()

    def update_handler(self, fd: Union[int, _Selectable], events: int) -> None:
        if False:
            return 10
        'Changes the events we listen for ``fd``.\n\n        .. versionchanged:: 4.0\n           Added the ability to pass file-like objects in addition to\n           raw file descriptors.\n        '
        raise NotImplementedError()

    def remove_handler(self, fd: Union[int, _Selectable]) -> None:
        if False:
            while True:
                i = 10
        'Stop listening for events on ``fd``.\n\n        .. versionchanged:: 4.0\n           Added the ability to pass file-like objects in addition to\n           raw file descriptors.\n        '
        raise NotImplementedError()

    def start(self) -> None:
        if False:
            return 10
        'Starts the I/O loop.\n\n        The loop will run until one of the callbacks calls `stop()`, which\n        will make the loop stop after the current event iteration completes.\n        '
        raise NotImplementedError()

    def stop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Stop the I/O loop.\n\n        If the event loop is not currently running, the next call to `start()`\n        will return immediately.\n\n        Note that even after `stop` has been called, the `IOLoop` is not\n        completely stopped until `IOLoop.start` has also returned.\n        Some work that was scheduled before the call to `stop` may still\n        be run before the `IOLoop` shuts down.\n        '
        raise NotImplementedError()

    def run_sync(self, func: Callable, timeout: Optional[float]=None) -> Any:
        if False:
            return 10
        "Starts the `IOLoop`, runs the given function, and stops the loop.\n\n        The function must return either an awaitable object or\n        ``None``. If the function returns an awaitable object, the\n        `IOLoop` will run until the awaitable is resolved (and\n        `run_sync()` will return the awaitable's result). If it raises\n        an exception, the `IOLoop` will stop and the exception will be\n        re-raised to the caller.\n\n        The keyword-only argument ``timeout`` may be used to set\n        a maximum duration for the function.  If the timeout expires,\n        a `asyncio.TimeoutError` is raised.\n\n        This method is useful to allow asynchronous calls in a\n        ``main()`` function::\n\n            async def main():\n                # do stuff...\n\n            if __name__ == '__main__':\n                IOLoop.current().run_sync(main)\n\n        .. versionchanged:: 4.3\n           Returning a non-``None``, non-awaitable value is now an error.\n\n        .. versionchanged:: 5.0\n           If a timeout occurs, the ``func`` coroutine will be cancelled.\n\n        .. versionchanged:: 6.2\n           ``tornado.util.TimeoutError`` is now an alias to ``asyncio.TimeoutError``.\n        "
        future_cell = [None]

        def run() -> None:
            if False:
                print('Hello World!')
            try:
                result = func()
                if result is not None:
                    from tornado.gen import convert_yielded
                    result = convert_yielded(result)
            except Exception:
                fut = Future()
                future_cell[0] = fut
                future_set_exc_info(fut, sys.exc_info())
            else:
                if is_future(result):
                    future_cell[0] = result
                else:
                    fut = Future()
                    future_cell[0] = fut
                    fut.set_result(result)
            assert future_cell[0] is not None
            self.add_future(future_cell[0], lambda future: self.stop())
        self.add_callback(run)
        if timeout is not None:

            def timeout_callback() -> None:
                if False:
                    i = 10
                    return i + 15
                assert future_cell[0] is not None
                if not future_cell[0].cancel():
                    self.stop()
            timeout_handle = self.add_timeout(self.time() + timeout, timeout_callback)
        self.start()
        if timeout is not None:
            self.remove_timeout(timeout_handle)
        assert future_cell[0] is not None
        if future_cell[0].cancelled() or not future_cell[0].done():
            raise TimeoutError('Operation timed out after %s seconds' % timeout)
        return future_cell[0].result()

    def time(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        "Returns the current time according to the `IOLoop`'s clock.\n\n        The return value is a floating-point number relative to an\n        unspecified time in the past.\n\n        Historically, the IOLoop could be customized to use e.g.\n        `time.monotonic` instead of `time.time`, but this is not\n        currently supported and so this method is equivalent to\n        `time.time`.\n\n        "
        return time.time()

    def add_timeout(self, deadline: Union[float, datetime.timedelta], callback: Callable, *args: Any, **kwargs: Any) -> object:
        if False:
            for i in range(10):
                print('nop')
        "Runs the ``callback`` at the time ``deadline`` from the I/O loop.\n\n        Returns an opaque handle that may be passed to\n        `remove_timeout` to cancel.\n\n        ``deadline`` may be a number denoting a time (on the same\n        scale as `IOLoop.time`, normally `time.time`), or a\n        `datetime.timedelta` object for a deadline relative to the\n        current time.  Since Tornado 4.0, `call_later` is a more\n        convenient alternative for the relative case since it does not\n        require a timedelta object.\n\n        Note that it is not safe to call `add_timeout` from other threads.\n        Instead, you must use `add_callback` to transfer control to the\n        `IOLoop`'s thread, and then call `add_timeout` from there.\n\n        Subclasses of IOLoop must implement either `add_timeout` or\n        `call_at`; the default implementations of each will call\n        the other.  `call_at` is usually easier to implement, but\n        subclasses that wish to maintain compatibility with Tornado\n        versions prior to 4.0 must use `add_timeout` instead.\n\n        .. versionchanged:: 4.0\n           Now passes through ``*args`` and ``**kwargs`` to the callback.\n        "
        if isinstance(deadline, numbers.Real):
            return self.call_at(deadline, callback, *args, **kwargs)
        elif isinstance(deadline, datetime.timedelta):
            return self.call_at(self.time() + deadline.total_seconds(), callback, *args, **kwargs)
        else:
            raise TypeError('Unsupported deadline %r' % deadline)

    def call_later(self, delay: float, callback: Callable, *args: Any, **kwargs: Any) -> object:
        if False:
            i = 10
            return i + 15
        'Runs the ``callback`` after ``delay`` seconds have passed.\n\n        Returns an opaque handle that may be passed to `remove_timeout`\n        to cancel.  Note that unlike the `asyncio` method of the same\n        name, the returned object does not have a ``cancel()`` method.\n\n        See `add_timeout` for comments on thread-safety and subclassing.\n\n        .. versionadded:: 4.0\n        '
        return self.call_at(self.time() + delay, callback, *args, **kwargs)

    def call_at(self, when: float, callback: Callable, *args: Any, **kwargs: Any) -> object:
        if False:
            i = 10
            return i + 15
        'Runs the ``callback`` at the absolute time designated by ``when``.\n\n        ``when`` must be a number using the same reference point as\n        `IOLoop.time`.\n\n        Returns an opaque handle that may be passed to `remove_timeout`\n        to cancel.  Note that unlike the `asyncio` method of the same\n        name, the returned object does not have a ``cancel()`` method.\n\n        See `add_timeout` for comments on thread-safety and subclassing.\n\n        .. versionadded:: 4.0\n        '
        return self.add_timeout(when, callback, *args, **kwargs)

    def remove_timeout(self, timeout: object) -> None:
        if False:
            print('Hello World!')
        'Cancels a pending timeout.\n\n        The argument is a handle as returned by `add_timeout`.  It is\n        safe to call `remove_timeout` even if the callback has already\n        been run.\n        '
        raise NotImplementedError()

    def add_callback(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        "Calls the given callback on the next I/O loop iteration.\n\n        It is safe to call this method from any thread at any time,\n        except from a signal handler.  Note that this is the **only**\n        method in `IOLoop` that makes this thread-safety guarantee; all\n        other interaction with the `IOLoop` must be done from that\n        `IOLoop`'s thread.  `add_callback()` may be used to transfer\n        control from other threads to the `IOLoop`'s thread.\n        "
        raise NotImplementedError()

    def add_callback_from_signal(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Calls the given callback on the next I/O loop iteration.\n\n        Intended to be afe for use from a Python signal handler; should not be\n        used otherwise.\n\n        .. deprecated:: 6.4\n           Use ``asyncio.AbstractEventLoop.add_signal_handler`` instead.\n           This method is suspected to have been broken since Tornado 5.0 and\n           will be removed in version 7.0.\n        '
        raise NotImplementedError()

    def spawn_callback(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        'Calls the given callback on the next IOLoop iteration.\n\n        As of Tornado 6.0, this method is equivalent to `add_callback`.\n\n        .. versionadded:: 4.0\n        '
        self.add_callback(callback, *args, **kwargs)

    def add_future(self, future: 'Union[Future[_T], concurrent.futures.Future[_T]]', callback: Callable[['Future[_T]'], None]) -> None:
        if False:
            while True:
                i = 10
        'Schedules a callback on the ``IOLoop`` when the given\n        `.Future` is finished.\n\n        The callback is invoked with one argument, the\n        `.Future`.\n\n        This method only accepts `.Future` objects and not other\n        awaitables (unlike most of Tornado where the two are\n        interchangeable).\n        '
        if isinstance(future, Future):
            future.add_done_callback(lambda f: self._run_callback(functools.partial(callback, f)))
        else:
            assert is_future(future)
            future_add_done_callback(future, lambda f: self.add_callback(callback, f))

    def run_in_executor(self, executor: Optional[concurrent.futures.Executor], func: Callable[..., _T], *args: Any) -> 'Future[_T]':
        if False:
            for i in range(10):
                print('nop')
        "Runs a function in a ``concurrent.futures.Executor``. If\n        ``executor`` is ``None``, the IO loop's default executor will be used.\n\n        Use `functools.partial` to pass keyword arguments to ``func``.\n\n        .. versionadded:: 5.0\n        "
        if executor is None:
            if not hasattr(self, '_executor'):
                from tornado.process import cpu_count
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 5)
            executor = self._executor
        c_future = executor.submit(func, *args)
        t_future = Future()
        self.add_future(c_future, lambda f: chain_future(f, t_future))
        return t_future

    def set_default_executor(self, executor: concurrent.futures.Executor) -> None:
        if False:
            return 10
        'Sets the default executor to use with :meth:`run_in_executor`.\n\n        .. versionadded:: 5.0\n        '
        self._executor = executor

    def _run_callback(self, callback: Callable[[], Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Runs a callback with error handling.\n\n        .. versionchanged:: 6.0\n\n           CancelledErrors are no longer logged.\n        '
        try:
            ret = callback()
            if ret is not None:
                from tornado import gen
                try:
                    ret = gen.convert_yielded(ret)
                except gen.BadYieldError:
                    pass
                else:
                    self.add_future(ret, self._discard_future_result)
        except asyncio.CancelledError:
            pass
        except Exception:
            app_log.error('Exception in callback %r', callback, exc_info=True)

    def _discard_future_result(self, future: Future) -> None:
        if False:
            i = 10
            return i + 15
        'Avoid unhandled-exception warnings from spawned coroutines.'
        future.result()

    def split_fd(self, fd: Union[int, _Selectable]) -> Tuple[int, Union[int, _Selectable]]:
        if False:
            print('Hello World!')
        if isinstance(fd, int):
            return (fd, fd)
        return (fd.fileno(), fd)

    def close_fd(self, fd: Union[int, _Selectable]) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            if isinstance(fd, int):
                os.close(fd)
            else:
                fd.close()
        except OSError:
            pass

    def _register_task(self, f: Future) -> None:
        if False:
            return 10
        self._pending_tasks.add(f)

    def _unregister_task(self, f: Future) -> None:
        if False:
            return 10
        self._pending_tasks.discard(f)

class _Timeout(object):
    """An IOLoop timeout, a UNIX timestamp and a callback"""
    __slots__ = ['deadline', 'callback', 'tdeadline']

    def __init__(self, deadline: float, callback: Callable[[], None], io_loop: IOLoop) -> None:
        if False:
            print('Hello World!')
        if not isinstance(deadline, numbers.Real):
            raise TypeError('Unsupported deadline %r' % deadline)
        self.deadline = deadline
        self.callback = callback
        self.tdeadline = (deadline, next(io_loop._timeout_counter))

    def __lt__(self, other: '_Timeout') -> bool:
        if False:
            while True:
                i = 10
        return self.tdeadline < other.tdeadline

    def __le__(self, other: '_Timeout') -> bool:
        if False:
            while True:
                i = 10
        return self.tdeadline <= other.tdeadline

class PeriodicCallback(object):
    """Schedules the given callback to be called periodically.

    The callback is called every ``callback_time`` milliseconds when
    ``callback_time`` is a float. Note that the timeout is given in
    milliseconds, while most other time-related functions in Tornado use
    seconds. ``callback_time`` may alternatively be given as a
    `datetime.timedelta` object.

    If ``jitter`` is specified, each callback time will be randomly selected
    within a window of ``jitter * callback_time`` milliseconds.
    Jitter can be used to reduce alignment of events with similar periods.
    A jitter of 0.1 means allowing a 10% variation in callback time.
    The window is centered on ``callback_time`` so the total number of calls
    within a given interval should not be significantly affected by adding
    jitter.

    If the callback runs for longer than ``callback_time`` milliseconds,
    subsequent invocations will be skipped to get back on schedule.

    `start` must be called after the `PeriodicCallback` is created.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    .. versionchanged:: 5.1
       The ``jitter`` argument is added.

    .. versionchanged:: 6.2
       If the ``callback`` argument is a coroutine, and a callback runs for
       longer than ``callback_time``, subsequent invocations will be skipped.
       Previously this was only true for regular functions, not coroutines,
       which were "fire-and-forget" for `PeriodicCallback`.

       The ``callback_time`` argument now accepts `datetime.timedelta` objects,
       in addition to the previous numeric milliseconds.
    """

    def __init__(self, callback: Callable[[], Optional[Awaitable]], callback_time: Union[datetime.timedelta, float], jitter: float=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.callback = callback
        if isinstance(callback_time, datetime.timedelta):
            self.callback_time = callback_time / datetime.timedelta(milliseconds=1)
        else:
            if callback_time <= 0:
                raise ValueError('Periodic callback must have a positive callback_time')
            self.callback_time = callback_time
        self.jitter = jitter
        self._running = False
        self._timeout = None

    def start(self) -> None:
        if False:
            print('Hello World!')
        'Starts the timer.'
        self.io_loop = IOLoop.current()
        self._running = True
        self._next_timeout = self.io_loop.time()
        self._schedule_next()

    def stop(self) -> None:
        if False:
            while True:
                i = 10
        'Stops the timer.'
        self._running = False
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
            self._timeout = None

    def is_running(self) -> bool:
        if False:
            while True:
                i = 10
        'Returns ``True`` if this `.PeriodicCallback` has been started.\n\n        .. versionadded:: 4.1\n        '
        return self._running

    async def _run(self) -> None:
        if not self._running:
            return
        try:
            val = self.callback()
            if val is not None and isawaitable(val):
                await val
        except Exception:
            app_log.error('Exception in callback %r', self.callback, exc_info=True)
        finally:
            self._schedule_next()

    def _schedule_next(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._running:
            self._update_next(self.io_loop.time())
            self._timeout = self.io_loop.add_timeout(self._next_timeout, self._run)

    def _update_next(self, current_time: float) -> None:
        if False:
            while True:
                i = 10
        callback_time_sec = self.callback_time / 1000.0
        if self.jitter:
            callback_time_sec *= 1 + self.jitter * (random.random() - 0.5)
        if self._next_timeout <= current_time:
            self._next_timeout += (math.floor((current_time - self._next_timeout) / callback_time_sec) + 1) * callback_time_sec
        else:
            self._next_timeout += callback_time_sec