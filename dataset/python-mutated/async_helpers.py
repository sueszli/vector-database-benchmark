import abc
import asyncio
import collections
import inspect
import itertools
import logging
import typing
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, AsyncIterator, Awaitable, Callable, Collection, Coroutine, Dict, Generator, Generic, Hashable, Iterable, List, Optional, Set, Tuple, TypeVar, Union, cast, overload
import attr
from typing_extensions import Concatenate, Literal, ParamSpec
from twisted.internet import defer
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import IReactorTime
from twisted.python.failure import Failure
from synapse.logging.context import PreserveLoggingContext, make_deferred_yieldable, run_in_background
from synapse.util import Clock
logger = logging.getLogger(__name__)
_T = TypeVar('_T')

class AbstractObservableDeferred(Generic[_T], metaclass=abc.ABCMeta):
    """Abstract base class defining the consumer interface of ObservableDeferred"""
    __slots__ = ()

    @abc.abstractmethod
    def observe(self) -> 'defer.Deferred[_T]':
        if False:
            return 10
        "Add a new observer for this ObservableDeferred\n\n        This returns a brand new deferred that is resolved when the underlying\n        deferred is resolved. Interacting with the returned deferred does not\n        effect the underlying deferred.\n\n        Note that the returned Deferred doesn't follow the Synapse logcontext rules -\n        you will probably want to `make_deferred_yieldable` it.\n        "
        ...

class ObservableDeferred(Generic[_T], AbstractObservableDeferred[_T]):
    """Wraps a deferred object so that we can add observer deferreds. These
    observer deferreds do not affect the callback chain of the original
    deferred.

    If consumeErrors is true errors will be captured from the origin deferred.

    Cancelling or otherwise resolving an observer will not affect the original
    ObservableDeferred.

    NB that it does not attempt to do anything with logcontexts; in general
    you should probably make_deferred_yieldable the deferreds
    returned by `observe`, and ensure that the original deferred runs its
    callbacks in the sentinel logcontext.
    """
    __slots__ = ['_deferred', '_observers', '_result']
    _deferred: 'defer.Deferred[_T]'
    _observers: Union[List['defer.Deferred[_T]'], Tuple[()]]
    _result: Union[None, Tuple[Literal[True], _T], Tuple[Literal[False], Failure]]

    def __init__(self, deferred: 'defer.Deferred[_T]', consumeErrors: bool=False):
        if False:
            for i in range(10):
                print('nop')
        object.__setattr__(self, '_deferred', deferred)
        object.__setattr__(self, '_result', None)
        object.__setattr__(self, '_observers', [])

        def callback(r: _T) -> _T:
            if False:
                while True:
                    i = 10
            object.__setattr__(self, '_result', (True, r))
            observers = self._observers
            object.__setattr__(self, '_observers', ())
            for observer in observers:
                try:
                    observer.callback(r)
                except Exception as e:
                    logger.exception('%r threw an exception on .callback(%r), ignoring...', observer, r, exc_info=e)
            return r

        def errback(f: Failure) -> Optional[Failure]:
            if False:
                while True:
                    i = 10
            object.__setattr__(self, '_result', (False, f))
            observers = self._observers
            object.__setattr__(self, '_observers', ())
            for observer in observers:
                f.value.__failure__ = f
                try:
                    observer.errback(f)
                except Exception as e:
                    logger.exception('%r threw an exception on .errback(%r), ignoring...', observer, f, exc_info=e)
            if consumeErrors:
                return None
            else:
                return f
        deferred.addCallbacks(callback, errback)

    def observe(self) -> 'defer.Deferred[_T]':
        if False:
            print('Hello World!')
        'Observe the underlying deferred.\n\n        This returns a brand new deferred that is resolved when the underlying\n        deferred is resolved. Interacting with the returned deferred does not\n        effect the underlying deferred.\n        '
        if not self._result:
            assert isinstance(self._observers, list)
            d: 'defer.Deferred[_T]' = defer.Deferred()
            self._observers.append(d)
            return d
        elif self._result[0]:
            return defer.succeed(self._result[1])
        else:
            return defer.fail(self._result[1])

    def observers(self) -> 'Collection[defer.Deferred[_T]]':
        if False:
            i = 10
            return i + 15
        return self._observers

    def has_called(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._result is not None

    def has_succeeded(self) -> bool:
        if False:
            return 10
        return self._result is not None and self._result[0] is True

    def get_result(self) -> Union[_T, Failure]:
        if False:
            while True:
                i = 10
        if self._result is None:
            raise ValueError(f'{self!r} has no result yet')
        return self._result[1]

    def __getattr__(self, name: str) -> Any:
        if False:
            while True:
                i = 10
        return getattr(self._deferred, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        setattr(self._deferred, name, value)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '<ObservableDeferred object at %s, result=%r, _deferred=%r>' % (id(self), self._result, self._deferred)
T = TypeVar('T')

async def concurrently_execute(func: Callable[[T], Any], args: Iterable[T], limit: int, delay_cancellation: bool=False) -> None:
    """Executes the function with each argument concurrently while limiting
    the number of concurrent executions.

    Args:
        func: Function to execute, should return a deferred or coroutine.
        args: List of arguments to pass to func, each invocation of func
            gets a single argument.
        limit: Maximum number of conccurent executions.
        delay_cancellation: Whether to delay cancellation until after the invocations
            have finished.

    Returns:
        None, when all function invocations have finished. The return values
        from those functions are discarded.
    """
    it = iter(args)

    async def _concurrently_execute_inner(value: T) -> None:
        try:
            while True:
                await maybe_awaitable(func(value))
                value = next(it)
        except StopIteration:
            pass
    if delay_cancellation:
        await yieldable_gather_results_delaying_cancellation(_concurrently_execute_inner, (value for value in itertools.islice(it, limit)))
    else:
        await yieldable_gather_results(_concurrently_execute_inner, (value for value in itertools.islice(it, limit)))
P = ParamSpec('P')
R = TypeVar('R')

async def yieldable_gather_results(func: Callable[Concatenate[T, P], Awaitable[R]], iter: Iterable[T], *args: P.args, **kwargs: P.kwargs) -> List[R]:
    """Executes the function with each argument concurrently.

    Args:
        func: Function to execute that returns a Deferred
        iter: An iterable that yields items that get passed as the first
            argument to the function
        *args: Arguments to be passed to each call to func
        **kwargs: Keyword arguments to be passed to each call to func

    Returns
        A list containing the results of the function
    """
    try:
        return await make_deferred_yieldable(defer.gatherResults([run_in_background(func, item, *args, **kwargs) for item in iter], consumeErrors=True))
    except defer.FirstError as dfe:
        assert isinstance(dfe.subFailure.value, BaseException)
        raise dfe.subFailure.value from None

async def yieldable_gather_results_delaying_cancellation(func: Callable[Concatenate[T, P], Awaitable[R]], iter: Iterable[T], *args: P.args, **kwargs: P.kwargs) -> List[R]:
    """Executes the function with each argument concurrently.
    Cancellation is delayed until after all the results have been gathered.

    See `yieldable_gather_results`.

    Args:
        func: Function to execute that returns a Deferred
        iter: An iterable that yields items that get passed as the first
            argument to the function
        *args: Arguments to be passed to each call to func
        **kwargs: Keyword arguments to be passed to each call to func

    Returns
        A list containing the results of the function
    """
    try:
        return await make_deferred_yieldable(delay_cancellation(defer.gatherResults([run_in_background(func, item, *args, **kwargs) for item in iter], consumeErrors=True)))
    except defer.FirstError as dfe:
        assert isinstance(dfe.subFailure.value, BaseException)
        raise dfe.subFailure.value from None
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
T4 = TypeVar('T4')

@overload
def gather_results(deferredList: Tuple[()], consumeErrors: bool=...) -> 'defer.Deferred[Tuple[()]]':
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def gather_results(deferredList: Tuple['defer.Deferred[T1]'], consumeErrors: bool=...) -> 'defer.Deferred[Tuple[T1]]':
    if False:
        while True:
            i = 10
    ...

@overload
def gather_results(deferredList: Tuple['defer.Deferred[T1]', 'defer.Deferred[T2]'], consumeErrors: bool=...) -> 'defer.Deferred[Tuple[T1, T2]]':
    if False:
        i = 10
        return i + 15
    ...

@overload
def gather_results(deferredList: Tuple['defer.Deferred[T1]', 'defer.Deferred[T2]', 'defer.Deferred[T3]'], consumeErrors: bool=...) -> 'defer.Deferred[Tuple[T1, T2, T3]]':
    if False:
        i = 10
        return i + 15
    ...

@overload
def gather_results(deferredList: Tuple['defer.Deferred[T1]', 'defer.Deferred[T2]', 'defer.Deferred[T3]', 'defer.Deferred[T4]'], consumeErrors: bool=...) -> 'defer.Deferred[Tuple[T1, T2, T3, T4]]':
    if False:
        for i in range(10):
            print('nop')
    ...

def gather_results(deferredList: Tuple['defer.Deferred[T1]', ...], consumeErrors: bool=False) -> 'defer.Deferred[Tuple[T1, ...]]':
    if False:
        i = 10
        return i + 15
    'Combines a tuple of `Deferred`s into a single `Deferred`.\n\n    Wraps `defer.gatherResults` to provide type annotations that support heterogenous\n    lists of `Deferred`s.\n    '
    deferred = defer.gatherResults(deferredList, consumeErrors=consumeErrors)
    return deferred.addCallback(tuple)

@attr.s(slots=True, auto_attribs=True)
class _LinearizerEntry:
    count: int
    deferreds: typing.OrderedDict['defer.Deferred[None]', Literal[1]]

class Linearizer:
    """Limits concurrent access to resources based on a key. Useful to ensure
    only a few things happen at a time on a given resource.

    Example:

        async with limiter.queue("test_key"):
            # do some work.

    """

    def __init__(self, name: Optional[str]=None, max_count: int=1, clock: Optional[Clock]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            max_count: The maximum number of concurrent accesses\n        '
        if name is None:
            self.name: Union[str, int] = id(self)
        else:
            self.name = name
        if not clock:
            from twisted.internet import reactor
            clock = Clock(cast(IReactorTime, reactor))
        self._clock = clock
        self.max_count = max_count
        self.key_to_defer: Dict[Hashable, _LinearizerEntry] = {}

    def is_queued(self, key: Hashable) -> bool:
        if False:
            print('Hello World!')
        'Checks whether there is a process queued up waiting'
        entry = self.key_to_defer.get(key)
        if not entry:
            return False
        return bool(entry.deferreds)

    def queue(self, key: Hashable) -> AsyncContextManager[None]:
        if False:
            return 10

        @asynccontextmanager
        async def _ctx_manager() -> AsyncIterator[None]:
            entry = await self._acquire_lock(key)
            try:
                yield
            finally:
                self._release_lock(key, entry)
        return _ctx_manager()

    async def _acquire_lock(self, key: Hashable) -> _LinearizerEntry:
        """Acquires a linearizer lock, waiting if necessary.

        Returns once we have secured the lock.
        """
        entry = self.key_to_defer.setdefault(key, _LinearizerEntry(0, collections.OrderedDict()))
        if entry.count < self.max_count:
            logger.debug('Acquired uncontended linearizer lock %r for key %r', self.name, key)
            entry.count += 1
            return entry
        logger.debug('Waiting to acquire linearizer lock %r for key %r', self.name, key)
        new_defer: 'defer.Deferred[None]' = make_deferred_yieldable(defer.Deferred())
        entry.deferreds[new_defer] = 1
        try:
            await new_defer
        except Exception as e:
            logger.info('defer %r got err %r', new_defer, e)
            if isinstance(e, CancelledError):
                logger.debug('Cancelling wait for linearizer lock %r for key %r', self.name, key)
            else:
                logger.warning('Unexpected exception waiting for linearizer lock %r for key %r', self.name, key)
            del entry.deferreds[new_defer]
            raise
        logger.debug('Acquired linearizer lock %r for key %r', self.name, key)
        entry.count += 1
        try:
            await self._clock.sleep(0)
        except CancelledError:
            self._release_lock(key, entry)
            raise
        return entry

    def _release_lock(self, key: Hashable, entry: _LinearizerEntry) -> None:
        if False:
            i = 10
            return i + 15
        'Releases a held linearizer lock.'
        logger.debug('Releasing linearizer lock %r for key %r', self.name, key)
        entry.count -= 1
        if entry.deferreds:
            (next_def, _) = entry.deferreds.popitem(last=False)
            with PreserveLoggingContext():
                next_def.callback(None)
        elif entry.count == 0:
            del self.key_to_defer[key]

class ReadWriteLock:
    """An async read write lock.

    Example:

        async with read_write_lock.read("test_key"):
            # do some work
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.key_to_current_readers: Dict[str, Set[defer.Deferred]] = {}
        self.key_to_current_writer: Dict[str, defer.Deferred] = {}

    def read(self, key: str) -> AsyncContextManager:
        if False:
            i = 10
            return i + 15

        @asynccontextmanager
        async def _ctx_manager() -> AsyncIterator[None]:
            new_defer: 'defer.Deferred[None]' = defer.Deferred()
            curr_readers = self.key_to_current_readers.setdefault(key, set())
            curr_writer = self.key_to_current_writer.get(key, None)
            curr_readers.add(new_defer)
            try:
                if curr_writer:
                    await make_deferred_yieldable(stop_cancellation(curr_writer))
                yield
            finally:
                with PreserveLoggingContext():
                    new_defer.callback(None)
                self.key_to_current_readers.get(key, set()).discard(new_defer)
        return _ctx_manager()

    def write(self, key: str) -> AsyncContextManager:
        if False:
            i = 10
            return i + 15

        @asynccontextmanager
        async def _ctx_manager() -> AsyncIterator[None]:
            new_defer: 'defer.Deferred[None]' = defer.Deferred()
            curr_readers = self.key_to_current_readers.get(key, set())
            curr_writer = self.key_to_current_writer.get(key, None)
            to_wait_on = list(curr_readers)
            if curr_writer:
                to_wait_on.append(curr_writer)
            curr_readers.clear()
            self.key_to_current_writer[key] = new_defer
            to_wait_on_defer = defer.gatherResults(to_wait_on)
            try:
                await make_deferred_yieldable(delay_cancellation(to_wait_on_defer))
                yield
            finally:
                with PreserveLoggingContext():
                    new_defer.callback(None)
                if self.key_to_current_writer.get(key) == new_defer:
                    self.key_to_current_writer.pop(key)
        return _ctx_manager()

def timeout_deferred(deferred: 'defer.Deferred[_T]', timeout: float, reactor: IReactorTime) -> 'defer.Deferred[_T]':
    if False:
        i = 10
        return i + 15
    "The in built twisted `Deferred.addTimeout` fails to time out deferreds\n    that have a canceller that throws exceptions. This method creates a new\n    deferred that wraps and times out the given deferred, correctly handling\n    the case where the given deferred's canceller throws.\n\n    (See https://twistedmatrix.com/trac/ticket/9534)\n\n    NOTE: Unlike `Deferred.addTimeout`, this function returns a new deferred.\n\n    NOTE: the TimeoutError raised by the resultant deferred is\n    twisted.internet.defer.TimeoutError, which is *different* to the built-in\n    TimeoutError, as well as various other TimeoutErrors you might have imported.\n\n    Args:\n        deferred: The Deferred to potentially timeout.\n        timeout: Timeout in seconds\n        reactor: The twisted reactor to use\n\n\n    Returns:\n        A new Deferred, which will errback with defer.TimeoutError on timeout.\n    "
    new_d: 'defer.Deferred[_T]' = defer.Deferred()
    timed_out = [False]

    def time_it_out() -> None:
        if False:
            return 10
        timed_out[0] = True
        try:
            deferred.cancel()
        except Exception:
            logger.exception('Canceller failed during timeout')
        if not new_d.called:
            new_d.errback(defer.TimeoutError('Timed out after %gs' % (timeout,)))
    delayed_call = reactor.callLater(timeout, time_it_out)

    def convert_cancelled(value: Failure) -> Failure:
        if False:
            while True:
                i = 10
        if timed_out[0] and value.check(CancelledError):
            raise defer.TimeoutError('Timed out after %gs' % (timeout,))
        return value
    deferred.addErrback(convert_cancelled)

    def cancel_timeout(result: _T) -> _T:
        if False:
            while True:
                i = 10
        if delayed_call.active():
            delayed_call.cancel()
        return result
    deferred.addBoth(cancel_timeout)

    def success_cb(val: _T) -> None:
        if False:
            i = 10
            return i + 15
        if not new_d.called:
            new_d.callback(val)

    def failure_cb(val: Failure) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not new_d.called:
            new_d.errback(val)
    deferred.addCallbacks(success_cb, failure_cb)
    return new_d

@attr.s(slots=True, frozen=True, auto_attribs=True)
class DoneAwaitable(Awaitable[R]):
    """Simple awaitable that returns the provided value."""
    value: R

    def __await__(self) -> Generator[Any, None, R]:
        if False:
            i = 10
            return i + 15
        yield None
        return self.value

def maybe_awaitable(value: Union[Awaitable[R], R]) -> Awaitable[R]:
    if False:
        return 10
    'Convert a value to an awaitable if not already an awaitable.'
    if inspect.isawaitable(value):
        return value
    assert not isinstance(value, Awaitable)
    return DoneAwaitable(value)

def stop_cancellation(deferred: 'defer.Deferred[T]') -> 'defer.Deferred[T]':
    if False:
        print('Hello World!')
    'Prevent a `Deferred` from being cancelled by wrapping it in another `Deferred`.\n\n    Args:\n        deferred: The `Deferred` to protect against cancellation. Must not follow the\n            Synapse logcontext rules.\n\n    Returns:\n        A new `Deferred`, which will contain the result of the original `Deferred`.\n        The new `Deferred` will not propagate cancellation through to the original.\n        When cancelled, the new `Deferred` will fail with a `CancelledError`.\n\n        The new `Deferred` will not follow the Synapse logcontext rules and should be\n        wrapped with `make_deferred_yieldable`.\n    '
    new_deferred: 'defer.Deferred[T]' = defer.Deferred()
    deferred.chainDeferred(new_deferred)
    return new_deferred

@overload
def delay_cancellation(awaitable: 'defer.Deferred[T]') -> 'defer.Deferred[T]':
    if False:
        print('Hello World!')
    ...

@overload
def delay_cancellation(awaitable: Coroutine[Any, Any, T]) -> 'defer.Deferred[T]':
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def delay_cancellation(awaitable: Awaitable[T]) -> Awaitable[T]:
    if False:
        print('Hello World!')
    ...

def delay_cancellation(awaitable: Awaitable[T]) -> Awaitable[T]:
    if False:
        print('Hello World!')
    'Delay cancellation of a coroutine or `Deferred` awaitable until it resolves.\n\n    Has the same effect as `stop_cancellation`, but the returned `Deferred` will not\n    resolve with a `CancelledError` until the original awaitable resolves.\n\n    Args:\n        deferred: The coroutine or `Deferred` to protect against cancellation. May\n            optionally follow the Synapse logcontext rules.\n\n    Returns:\n        A new `Deferred`, which will contain the result of the original coroutine or\n        `Deferred`. The new `Deferred` will not propagate cancellation through to the\n        original coroutine or `Deferred`.\n\n        When cancelled, the new `Deferred` will wait until the original coroutine or\n        `Deferred` resolves before failing with a `CancelledError`.\n\n        The new `Deferred` will follow the Synapse logcontext rules if `awaitable`\n        follows the Synapse logcontext rules. Otherwise the new `Deferred` should be\n        wrapped with `make_deferred_yieldable`.\n    '
    if isinstance(awaitable, defer.Deferred):
        deferred = awaitable
    elif asyncio.iscoroutine(awaitable):
        deferred = defer.ensureDeferred(awaitable)
    else:
        return awaitable

    def handle_cancel(new_deferred: 'defer.Deferred[T]') -> None:
        if False:
            return 10
        new_deferred.pause()
        new_deferred.errback(Failure(CancelledError()))
        deferred.addBoth(lambda _: new_deferred.unpause())
    new_deferred: 'defer.Deferred[T]' = defer.Deferred(handle_cancel)
    deferred.chainDeferred(new_deferred)
    return new_deferred

class AwakenableSleeper:
    """Allows explicitly waking up deferreds related to an entity that are
    currently sleeping.
    """

    def __init__(self, reactor: IReactorTime) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._streams: Dict[str, Set[defer.Deferred[None]]] = {}
        self._reactor = reactor

    def wake(self, name: str) -> None:
        if False:
            print('Hello World!')
        'Wake everything related to `name` that is currently sleeping.'
        stream_set = self._streams.pop(name, set())
        for deferred in stream_set:
            try:
                with PreserveLoggingContext():
                    deferred.callback(None)
            except Exception:
                pass

    async def sleep(self, name: str, delay_ms: int) -> None:
        """Sleep for the given number of milliseconds, or return if the given
        `name` is explicitly woken up.
        """
        sleep_deferred: 'defer.Deferred[None]' = defer.Deferred()
        call = self._reactor.callLater(delay_ms / 1000, sleep_deferred.callback, None)
        stream_set = self._streams.setdefault(name, set())
        notify_deferred: 'defer.Deferred[None]' = defer.Deferred()
        stream_set.add(notify_deferred)
        try:
            await make_deferred_yieldable(defer.DeferredList([sleep_deferred, notify_deferred], fireOnOneCallback=True, fireOnOneErrback=True, consumeErrors=True))
        finally:
            curr_stream_set = self._streams.get(name)
            if curr_stream_set is not None:
                curr_stream_set.discard(notify_deferred)
                if len(curr_stream_set) == 0:
                    self._streams.pop(name)
            if call.active():
                call.cancel()