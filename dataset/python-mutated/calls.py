"""
Implementation of the `Call` data structure for transport of deferred function calls
and low-level management of call execution.
"""
import abc
import asyncio
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import inspect
import threading
from concurrent.futures._base import CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED, RUNNING
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, Tuple, TypeVar
from typing_extensions import ParamSpec
from prefect._internal.concurrency import logger
from prefect._internal.concurrency.cancellation import CancelledError, cancel_async_at, cancel_sync_at, get_deadline
from prefect._internal.concurrency.event_loop import get_running_loop
T = TypeVar('T')
P = ParamSpec('P')
current_call: contextvars.ContextVar['Call'] = contextvars.ContextVar('current_call')
_ASYNC_TASK_REFS = set()

def get_current_call() -> Optional['Call']:
    if False:
        i = 10
        return i + 15
    return current_call.get(None)

@contextlib.contextmanager
def set_current_call(call: 'Call'):
    if False:
        i = 10
        return i + 15
    token = current_call.set(call)
    try:
        yield
    finally:
        current_call.reset(token)

class Future(concurrent.futures.Future):
    """
    Extension of `concurrent.futures.Future` with support for cancellation of running
    futures.

    Used by `Call`.
    """

    def __init__(self, name: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._cancel_scope = None
        self._deadline = None
        self._cancel_callbacks = []
        self._name = name
        self._timed_out = False

    def set_running_or_notify_cancel(self, timeout: Optional[float]=None):
        if False:
            for i in range(10):
                print('nop')
        self._deadline = get_deadline(timeout)
        return super().set_running_or_notify_cancel()

    @contextlib.contextmanager
    def enforce_async_deadline(self):
        if False:
            print('Hello World!')
        with cancel_async_at(self._deadline, name=self._name) as self._cancel_scope:
            for callback in self._cancel_callbacks:
                self._cancel_scope.add_cancel_callback(callback)
            yield self._cancel_scope

    @contextlib.contextmanager
    def enforce_sync_deadline(self):
        if False:
            print('Hello World!')
        with cancel_sync_at(self._deadline, name=self._name) as self._cancel_scope:
            for callback in self._cancel_callbacks:
                self._cancel_scope.add_cancel_callback(callback)
            yield self._cancel_scope

    def add_cancel_callback(self, callback: Callable[[], None]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a callback to be enforced on cancellation.\n\n        Unlike "done" callbacks, this callback will be invoked _before_ the future is\n        cancelled. If added after the future is cancelled, nothing will happen.\n        '
        if self._cancel_scope:
            self._cancel_scope.add_cancel_callback(callback)
        self._cancel_callbacks.append(callback)

    def timedout(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        with self._condition:
            return self._timed_out

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        'Cancel the future if possible.\n\n        Returns True if the future was cancelled, False otherwise. A future cannot be\n        cancelled if it has already completed.\n        '
        with self._condition:
            if self._state in [RUNNING]:
                if self._cancel_scope is None:
                    return False
                elif not self._cancel_scope.cancelled():
                    if not self._cancel_scope.cancel():
                        return False
            if self._state in [FINISHED]:
                return False
            if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                return True
            if not self._cancel_scope:
                for callback in self._cancel_callbacks:
                    callback()
            self._state = CANCELLED
            self._condition.notify_all()
        self._invoke_callbacks()
        return True

    def result(self, timeout=None):
        if False:
            print('Hello World!')
        "Return the result of the call that the future represents.\n\n        Args:\n            timeout: The number of seconds to wait for the result if the future\n                isn't done. If None, then there is no limit on the wait time.\n\n        Returns:\n            The result of the call that the future represents.\n\n        Raises:\n            CancelledError: If the future was cancelled.\n            TimeoutError: If the future didn't finish executing before the given\n                timeout.\n            Exception: If the call raised then that exception will be raised.\n        "
        try:
            with self._condition:
                if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                    raise CancelledError()
                elif self._state == FINISHED:
                    return self.__get_result()
                self._condition.wait(timeout)
                if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:
                    raise CancelledError()
                elif self._state == FINISHED:
                    return self.__get_result()
                else:
                    raise TimeoutError()
        finally:
            self = None

@dataclasses.dataclass
class Call(Generic[T]):
    """
    A deferred function call.
    """
    future: Future
    fn: Callable[..., T]
    args: Tuple
    kwargs: Dict[str, Any]
    context: contextvars.Context
    timeout: float
    runner: Optional['Portal'] = None

    @classmethod
    def new(cls, __fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> 'Call[T]':
        if False:
            return 10
        return cls(future=Future(name=getattr(__fn, '__name__', str(__fn))), fn=__fn, args=args, kwargs=kwargs, context=contextvars.copy_context(), timeout=None)

    def set_timeout(self, timeout: Optional[float]=None) -> None:
        if False:
            return 10
        '\n        Set the timeout for the call.\n\n        The timeout begins when the call starts.\n        '
        if self.future.done() or self.future.running():
            raise RuntimeError('Timeouts cannot be added when the call has started.')
        self.timeout = timeout

    def set_runner(self, portal: 'Portal') -> None:
        if False:
            print('Hello World!')
        '\n        Update the portal used to run this call.\n        '
        if self.runner is not None:
            raise RuntimeError('The portal is already set for this call.')
        self.runner = portal

    def run(self) -> Optional[Awaitable[T]]:
        if False:
            print('Hello World!')
        '\n        Execute the call and place the result on the future.\n\n        All exceptions during execution of the call are captured.\n        '
        if not self.future.set_running_or_notify_cancel(self.timeout):
            logger.debug('Skipping execution of cancelled call %r', self)
            return None
        logger.debug('Running call %r in thread %r%s', self, threading.current_thread().name, f' with timeout of {self.timeout}s' if self.timeout is not None else '')
        coro = self.context.run(self._run_sync)
        if coro is not None:
            loop = get_running_loop()
            if loop:
                logger.debug('Scheduling coroutine for call %r in running loop %r', self, loop)
                task = self.context.run(loop.create_task, self._run_async(coro))
                _ASYNC_TASK_REFS.add(task)
                asyncio.ensure_future(task).add_done_callback(lambda _: _ASYNC_TASK_REFS.remove(task))
                return task
            else:
                logger.debug('Executing coroutine for call %r in new loop', self)
                return self.context.run(asyncio.run, self._run_async(coro))
        return None

    def result(self, timeout: Optional[float]=None) -> T:
        if False:
            while True:
                i = 10
        '\n        Wait for the result of the call.\n\n        Not safe for use from asynchronous contexts.\n        '
        return self.future.result(timeout=timeout)

    async def aresult(self):
        """
        Wait for the result of the call.

        For use from asynchronous contexts.
        """
        try:
            return await asyncio.wrap_future(self.future)
        except asyncio.CancelledError as exc:
            raise CancelledError() from exc

    def cancelled(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if the call was cancelled.\n        '
        return self.future.cancelled()

    def timedout(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Check if the call timed out.\n        '
        return self.future.timedout()

    def cancel(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.future.cancel()

    def _run_sync(self):
        if False:
            i = 10
            return i + 15
        cancel_scope = None
        try:
            with set_current_call(self):
                with self.future.enforce_sync_deadline() as cancel_scope:
                    try:
                        result = self.fn(*self.args, **self.kwargs)
                    finally:
                        self.args = None
                        self.kwargs = None
            if inspect.isawaitable(result):
                return result
        except CancelledError:
            if cancel_scope.timedout():
                self.future._timed_out = True
                self.future.cancel()
            elif cancel_scope.cancelled():
                self.future.cancel()
            else:
                raise
        except BaseException as exc:
            logger.debug('Encountered exception in call %r', self, exc_info=True)
            self.future.set_exception(exc)
            del self
        else:
            self.future.set_result(result)
            logger.debug('Finished call %r', self)

    async def _run_async(self, coro):
        cancel_scope = None
        try:
            with set_current_call(self):
                with self.future.enforce_async_deadline() as cancel_scope:
                    try:
                        result = await coro
                    finally:
                        self.args = None
                        self.kwargs = None
        except CancelledError:
            if cancel_scope.timedout():
                self.future._timed_out = True
                self.future.cancel()
            elif cancel_scope.cancelled():
                self.future.cancel()
            else:
                raise
        except BaseException as exc:
            logger.debug('Encountered exception in async call %r', self, exc_info=True)
            self.future.set_exception(exc)
            del self
        else:
            self.future.set_result(result)
            logger.debug('Finished async call %r', self)

    def __call__(self) -> T:
        if False:
            print('Hello World!')
        '\n        Execute the call and return its result.\n\n        All executions during execution of the call are re-raised.\n        '
        coro = self.run()
        if coro is not None:

            async def run_and_return_result():
                await coro
                return self.result()
            return run_and_return_result()
        else:
            return self.result()

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        name = getattr(self.fn, '__name__', str(self.fn))
        (args, kwargs) = (self.args, self.kwargs)
        if args is None or kwargs is None:
            call_args = '<dropped>'
        else:
            call_args = ', '.join([repr(arg) for arg in args] + [f'{key}={repr(val)}' for (key, val) in kwargs.items()])
        if len(call_args) > 100:
            call_args = call_args[:100] + '...'
        return f'{name}({call_args})'

class Portal(abc.ABC):
    """
    Allows submission of calls to execute elsewhere.
    """

    @abc.abstractmethod
    def submit(self, call: 'Call') -> 'Call':
        if False:
            return 10
        "\n        Submit a call to execute elsewhere.\n\n        The call's result can be retrieved with `call.result()`.\n\n        Returns the call for convenience.\n        "