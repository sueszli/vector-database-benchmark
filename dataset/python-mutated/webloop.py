import asyncio
import contextvars
import inspect
import sys
import time
import traceback
from asyncio import Future, Task
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, overload
from .ffi import IN_BROWSER, create_once_callable
if IN_BROWSER:
    from js import setTimeout
T = TypeVar('T')
S = TypeVar('S')

class PyodideFuture(Future[T]):
    """A :py:class:`~asyncio.Future` with extra :js:meth:`~Promise.then`,
    :js:meth:`~Promise.catch`, and :js:meth:`finally_() <Promise.finally>` methods
    based on the Javascript promise API. :py:meth:`~asyncio.loop.create_future`
    returns these so in practice all futures encountered in Pyodide should be an
    instance of :py:class:`~pyodide.webloop.PyodideFuture`.
    """

    @overload
    def then(self, onfulfilled: None, onrejected: Callable[[BaseException], Awaitable[S]]) -> 'PyodideFuture[S]':
        if False:
            return 10
        ...

    @overload
    def then(self, onfulfilled: None, onrejected: Callable[[BaseException], S]) -> 'PyodideFuture[S]':
        if False:
            print('Hello World!')
        ...

    @overload
    def then(self, onfulfilled: Callable[[T], Awaitable[S]], onrejected: Callable[[BaseException], Awaitable[S]] | None=None) -> 'PyodideFuture[S]':
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def then(self, onfulfilled: Callable[[T], S], onrejected: Callable[[BaseException], S] | None=None) -> 'PyodideFuture[S]':
        if False:
            print('Hello World!')
        ...

    def then(self, onfulfilled: Callable[[T], S | Awaitable[S]] | None, onrejected: Callable[[BaseException], S | Awaitable[S]] | None=None) -> 'PyodideFuture[S]':
        if False:
            print('Hello World!')
        'When the Future is done, either execute onfulfilled with the result\n        or execute onrejected with the exception.\n\n        Returns a new Future which will be marked done when either the\n        onfulfilled or onrejected callback is completed. If the return value of\n        the executed callback is awaitable it will be awaited repeatedly until a\n        nonawaitable value is received. The returned Future will be resolved\n        with that value. If an error is raised, the returned Future will be\n        rejected with the error.\n\n        Parameters\n        ----------\n        onfulfilled:\n            A function called if the Future is fulfilled. This function receives\n            one argument, the fulfillment value.\n\n        onrejected:\n            A function called if the Future is rejected. This function receives\n            one argument, the rejection value.\n\n        Returns\n        -------\n            A new future to be resolved when the original future is done and the\n            appropriate callback is also done.\n        '
        result: PyodideFuture[S] = PyodideFuture()
        onfulfilled_: Callable[[T], S | Awaitable[S]]
        onrejected_: Callable[[BaseException], S | Awaitable[S]]
        if onfulfilled:
            onfulfilled_ = onfulfilled
        else:

            def onfulfilled_(x):
                if False:
                    while True:
                        i = 10
                return x
        if onrejected:
            onrejected_ = onrejected
        else:

            def onrejected_(x):
                if False:
                    i = 10
                    return i + 15
                raise x

        async def callback(fut: Future[T]) -> None:
            e = fut.exception()
            try:
                if e:
                    r = onrejected_(e)
                else:
                    r = onfulfilled_(fut.result())
                while inspect.isawaitable(r):
                    r = await r
            except Exception as result_exception:
                result.set_exception(result_exception)
                return
            result.set_result(r)

        def wrapper(fut: Future[T]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            asyncio.ensure_future(callback(fut))
        self.add_done_callback(wrapper)
        return result

    @overload
    def catch(self, onrejected: Callable[[BaseException], Awaitable[S]]) -> 'PyodideFuture[S]':
        if False:
            while True:
                i = 10
        ...

    @overload
    def catch(self, onrejected: Callable[[BaseException], S]) -> 'PyodideFuture[S]':
        if False:
            while True:
                i = 10
        ...

    def catch(self, onrejected: Callable[[BaseException], object]) -> 'PyodideFuture[Any]':
        if False:
            for i in range(10):
                print('nop')
        'Equivalent to ``then(None, onrejected)``'
        return self.then(None, onrejected)

    def finally_(self, onfinally: Callable[[], None]) -> 'PyodideFuture[T]':
        if False:
            while True:
                i = 10
        'When the future is either resolved or rejected, call ``onfinally`` with\n        no arguments.\n        '
        result: PyodideFuture[T] = PyodideFuture()

        async def callback(fut: Future[T]) -> None:
            exc = fut.exception()
            try:
                r = onfinally()
                while inspect.isawaitable(r):
                    r = await r
            except Exception as e:
                result.set_exception(e)
                return
            if exc:
                result.set_exception(exc)
            else:
                result.set_result(fut.result())

        def wrapper(fut: Future[T]) -> None:
            if False:
                i = 10
                return i + 15
            asyncio.ensure_future(callback(fut))
        self.add_done_callback(wrapper)
        return result

    def syncify(self):
        if False:
            for i in range(10):
                print('nop')
        from .ffi import create_proxy
        p = create_proxy(self)
        try:
            return p.syncify()
        finally:
            p.destroy()

class PyodideTask(Task[T], PyodideFuture[T]):
    """Inherits from both :py:class:`~asyncio.Task` and
    :py:class:`~pyodide.webloop.PyodideFuture`

    Instantiation is discouraged unless you are writing your own event loop.
    """
    pass

class WebLoop(asyncio.AbstractEventLoop):
    """A custom event loop for use in Pyodide.

    Schedules tasks on the browser event loop. Does no lifecycle management and
    runs forever.

    :py:meth:`~asyncio.loop.run_forever` and
    :py:meth:`~asyncio.loop.run_until_complete` cannot block like a normal event
    loop would because we only have one thread so blocking would stall the
    browser event loop and prevent anything from ever happening.

    We defer all work to the browser event loop using the :js:func:`setTimeout`
    function. To ensure that this event loop doesn't stall out UI and other
    browser handling, we want to make sure that each task is scheduled on the
    browser event loop as a task not as a microtask. ``setTimeout(callback, 0)``
    enqueues the callback as a task so it works well for our purposes.

    See the Python :external:doc:`library/asyncio-eventloop` documentation.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._task_factory = None
        asyncio._set_running_loop(self)
        self._exception_handler = None
        self._current_handle = None
        self._in_progress = 0
        self._no_in_progress_handler = None
        self._keyboard_interrupt_handler = None
        self._system_exit_handler = None

    def get_debug(self):
        if False:
            return 10
        return False

    def is_running(self) -> bool:
        if False:
            print('Hello World!')
        'Returns ``True`` if the event loop is running.\n\n        Always returns ``True`` because WebLoop has no lifecycle management.\n        '
        return True

    def is_closed(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns ``True`` if the event loop was closed.\n\n        Always returns ``False`` because WebLoop has no lifecycle management.\n        '
        return False

    def _check_closed(self):
        if False:
            return 10
        'Used in create_task.\n\n        Would raise an error if ``self.is_closed()``, but we are skipping all lifecycle stuff.\n        '
        pass

    def run_forever(self):
        if False:
            for i in range(10):
                print('nop')
        'Run the event loop forever. Does nothing in this implementation.\n\n        We cannot block like a normal event loop would\n        because we only have one thread so blocking would stall the browser event loop\n        and prevent anything from ever happening.\n        '
        pass

    def run_until_complete(self, future):
        if False:
            return 10
        'Run until future is done.\n\n        If the argument is a coroutine, it is wrapped in a Task.\n\n        The native event loop `run_until_complete` blocks until evaluation of the\n        future is complete and then returns the result of the future.\n        Since we cannot block, we just ensure that the future is scheduled and\n        return the future. This makes this method a bit useless. Instead, use\n        `future.add_done_callback(do_something_with_result)` or:\n        ```python\n        async def wrapper():\n            result = await future\n            do_something_with_result(result)\n        ```\n        '
        return asyncio.ensure_future(future)

    def call_soon(self, callback: Callable[..., Any], *args: Any, context: contextvars.Context | None=None) -> asyncio.Handle:
        if False:
            i = 10
            return i + 15
        'Arrange for a callback to be called as soon as possible.\n\n        Any positional arguments after the callback will be passed to\n        the callback when it is called.\n\n        This schedules the callback on the browser event loop using ``setTimeout(callback, 0)``.\n        '
        delay = 0
        return self.call_later(delay, callback, *args, context=context)

    def call_soon_threadsafe(self, callback: Callable[..., Any], *args: Any, context: contextvars.Context | None=None) -> asyncio.Handle:
        if False:
            while True:
                i = 10
        'Like ``call_soon()``, but thread-safe.\n\n        We have no threads so everything is "thread safe", and we just use ``call_soon``.\n        '
        return self.call_soon(callback, *args, context=context)

    def call_later(self, delay: float, callback: Callable[..., Any], *args: Any, context: contextvars.Context | None=None) -> asyncio.Handle:
        if False:
            for i in range(10):
                print('nop')
        'Arrange for a callback to be called at a given time.\n\n        Return a Handle: an opaque object with a cancel() method that\n        can be used to cancel the call.\n\n        The delay can be an int or float, expressed in seconds.  It is\n        always relative to the current time.\n\n        Each callback will be called exactly once.  If two callbacks\n        are scheduled for exactly the same time, it undefined which\n        will be called first.\n\n        Any positional arguments after the callback will be passed to\n        the callback when it is called.\n\n        This uses `setTimeout(callback, delay)`\n        '
        if delay < 0:
            raise ValueError("Can't schedule in the past")
        h = asyncio.Handle(callback, args, self, context=context)

        def run_handle():
            if False:
                print('Hello World!')
            if h.cancelled():
                return
            try:
                h._run()
            except SystemExit as e:
                if self._system_exit_handler:
                    self._system_exit_handler(e.code)
                else:
                    raise
            except KeyboardInterrupt:
                if self._keyboard_interrupt_handler:
                    self._keyboard_interrupt_handler()
                else:
                    raise
        setTimeout(create_once_callable(run_handle), delay * 1000)
        return h

    def _decrement_in_progress(self, *args):
        if False:
            return 10
        self._in_progress -= 1
        if self._no_in_progress_handler and self._in_progress == 0:
            self._no_in_progress_handler()

    def call_at(self, when: float, callback: Callable[..., Any], *args: Any, context: contextvars.Context | None=None) -> asyncio.Handle:
        if False:
            for i in range(10):
                print('nop')
        "Like ``call_later()``, but uses an absolute time.\n\n        Absolute time corresponds to the event loop's ``time()`` method.\n\n        This uses ``setTimeout(callback, when - cur_time)``\n        "
        cur_time = self.time()
        delay = when - cur_time
        return self.call_later(delay, callback, *args, context=context)

    def run_in_executor(self, executor, func, *args):
        if False:
            for i in range(10):
                print('nop')
        "Arrange for func to be called in the specified executor.\n\n        This is normally supposed to run func(*args) in a separate process or\n        thread and signal back to our event loop when it is done. It's possible\n        to make the executor, but if we actually try to submit any functions to\n        it, it will try to create a thread and throw an error. Best we can do is\n        to run func(args) in this thread and stick the result into a future.\n        "
        fut = self.create_future()
        try:
            fut.set_result(func(*args))
        except BaseException as e:
            fut.set_exception(e)
        return fut

    def create_future(self) -> asyncio.Future[Any]:
        if False:
            return 10
        self._in_progress += 1
        fut: PyodideFuture[Any] = PyodideFuture(loop=self)
        fut.add_done_callback(self._decrement_in_progress)
        'Create a Future object attached to the loop.'
        return fut

    def time(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        "Return the time according to the event loop's clock.\n\n        This is a float expressed in seconds since an epoch, but the\n        epoch, precision, accuracy and drift are unspecified and may\n        differ per event loop.\n\n        Copied from ``BaseEventLoop.time``\n        "
        return time.monotonic()

    def create_task(self, coro, *, name=None):
        if False:
            i = 10
            return i + 15
        'Schedule a coroutine object.\n\n        Return a task object.\n\n        Copied from ``BaseEventLoop.create_task``\n        '
        self._check_closed()
        if self._task_factory is None:
            task = PyodideTask(coro, loop=self, name=name)
            if task._source_traceback:
                del task._source_traceback[-1]
        else:
            task = self._task_factory(self, coro)
            asyncio.tasks._set_task_name(task, name)
        self._in_progress += 1
        task.add_done_callback(self._decrement_in_progress)
        return task

    def set_task_factory(self, factory):
        if False:
            while True:
                i = 10
        "Set a task factory that will be used by loop.create_task().\n\n        If factory is None the default task factory will be set.\n\n        If factory is a callable, it should have a signature matching\n        '(loop, coro)', where 'loop' will be a reference to the active\n        event loop, 'coro' will be a coroutine object.  The callable\n        must return a Future.\n\n        Copied from ``BaseEventLoop.set_task_factory``\n        "
        if factory is not None and (not callable(factory)):
            raise TypeError('task factory must be a callable or None')
        self._task_factory = factory

    def get_task_factory(self):
        if False:
            print('Hello World!')
        'Return a task factory, or None if the default one is in use.\n\n        Copied from ``BaseEventLoop.get_task_factory``\n        '
        return self._task_factory

    def get_exception_handler(self):
        if False:
            while True:
                i = 10
        'Return an exception handler, or None if the default one is in use.'
        return self._exception_handler

    def set_exception_handler(self, handler):
        if False:
            while True:
                i = 10
        "Set handler as the new event loop exception handler.\n\n        If handler is None, the default exception handler will be set.\n\n        If handler is a callable object, it should have a signature matching\n        '(loop, context)', where 'loop' will be a reference to the active event\n        loop, 'context' will be a dict object (see `call_exception_handler()`\n        documentation for details about context).\n        "
        if handler is not None and (not callable(handler)):
            raise TypeError(f'A callable object or None is expected, got {handler!r}')
        self._exception_handler = handler

    def default_exception_handler(self, context):
        if False:
            i = 10
            return i + 15
        'Default exception handler.\n\n        This is called when an exception occurs and no exception handler is set,\n        and can be called by a custom exception handler that wants to defer to\n        the default behavior. This default handler logs the error message and\n        other context-dependent information.\n\n\n        In debug mode, a truncated stack trace is also appended showing where\n        the given object (e.g. a handle or future or task) was created, if any.\n        The context parameter has the same meaning as in\n        `call_exception_handler()`.\n        '
        message = context.get('message')
        if not message:
            message = 'Unhandled exception in event loop'
        if 'source_traceback' not in context and self._current_handle is not None and self._current_handle._source_traceback:
            context['handle_traceback'] = self._current_handle._source_traceback
        log_lines = [message]
        for key in sorted(context):
            if key in {'message', 'exception'}:
                continue
            value = context[key]
            if key == 'source_traceback':
                tb = ''.join(traceback.format_list(value))
                value = 'Object created at (most recent call last):\n'
                value += tb.rstrip()
            elif key == 'handle_traceback':
                tb = ''.join(traceback.format_list(value))
                value = 'Handle created at (most recent call last):\n'
                value += tb.rstrip()
            else:
                value = repr(value)
            log_lines.append(f'{key}: {value}')
        print('\n'.join(log_lines), file=sys.stderr)

    def call_exception_handler(self, context):
        if False:
            i = 10
            return i + 15
        "Call the current event loop's exception handler.\n        The context argument is a dict containing the following keys:\n        - 'message': Error message;\n        - 'exception' (optional): Exception object;\n        - 'future' (optional): Future instance;\n        - 'task' (optional): Task instance;\n        - 'handle' (optional): Handle instance;\n        - 'protocol' (optional): Protocol instance;\n        - 'transport' (optional): Transport instance;\n        - 'socket' (optional): Socket instance;\n        - 'asyncgen' (optional): Asynchronous generator that caused\n                                 the exception.\n        New keys maybe introduced in the future.\n        Note: do not overload this method in an event loop subclass.\n        For custom exception handling, use the\n        `set_exception_handler()` method.\n        "
        if self._exception_handler is None:
            try:
                self.default_exception_handler(context)
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException:
                print('Exception in default exception handler', file=sys.stderr)
                traceback.print_exc()
        else:
            try:
                self._exception_handler(self, context)
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:
                try:
                    self.default_exception_handler({'message': 'Unhandled error in exception handler', 'exception': exc, 'context': context})
                except (SystemExit, KeyboardInterrupt):
                    raise
                except BaseException:
                    print('Exception in default exception handler while handling an unexpected error in custom exception handler', file=sys.stderr)
                    traceback.print_exc()

class WebLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """
    A simple event loop policy for managing :py:class:`WebLoop`-based event loops.
    """

    def __init__(self):
        if False:
            return 10
        self._default_loop = None

    def get_event_loop(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the current event loop'
        if self._default_loop:
            return self._default_loop
        return self.new_event_loop()

    def new_event_loop(self) -> WebLoop:
        if False:
            i = 10
            return i + 15
        'Create a new event loop'
        self._default_loop = WebLoop()
        return self._default_loop

    def set_event_loop(self, loop: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Set the current event loop'
        self._default_loop = loop

def _initialize_event_loop():
    if False:
        while True:
            i = 10
    from .ffi import IN_BROWSER
    if not IN_BROWSER:
        return
    import asyncio
    from .webloop import WebLoopPolicy
    policy = WebLoopPolicy()
    asyncio.set_event_loop_policy(policy)
    policy.get_event_loop()
__all__ = ['WebLoop', 'WebLoopPolicy', 'PyodideFuture', 'PyodideTask']