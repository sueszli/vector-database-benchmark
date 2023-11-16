import asyncio
import enum
import sys
import warnings
from types import TracebackType
from typing import Optional, Type
if sys.version_info >= (3, 11):
    from typing import final
else:

    def final(f):
        if False:
            while True:
                i = 10
        'This decorator can be used to indicate to type checkers that\n        the decorated method cannot be overridden, and decorated class\n        cannot be subclassed. For example:\n\n            class Base:\n                @final\n                def done(self) -> None:\n                    ...\n            class Sub(Base):\n                def done(self) -> None:  # Error reported by type checker\n                    ...\n            @final\n            class Leaf:\n                ...\n            class Other(Leaf):  # Error reported by type checker\n                ...\n\n        There is no runtime checking of these properties. The decorator\n        sets the ``__final__`` attribute to ``True`` on the decorated object\n        to allow runtime introspection.\n        '
        try:
            f.__final__ = True
        except (AttributeError, TypeError):
            pass
        return f
__version__ = '4.0.2'
__all__ = ('timeout', 'timeout_at', 'Timeout')

def timeout(delay: Optional[float]) -> 'Timeout':
    if False:
        for i in range(10):
            print('nop')
    "timeout context manager.\n\n    Useful in cases when you want to apply timeout logic around block\n    of code or in cases when asyncio.wait_for is not suitable. For example:\n\n    >>> async with timeout(0.001):\n    ...     async with aiohttp.get('https://github.com') as r:\n    ...         await r.text()\n\n\n    delay - value in seconds or None to disable timeout logic\n    "
    loop = asyncio.get_running_loop()
    if delay is not None:
        deadline = loop.time() + delay
    else:
        deadline = None
    return Timeout(deadline, loop)

def timeout_at(deadline: Optional[float]) -> 'Timeout':
    if False:
        i = 10
        return i + 15
    "Schedule the timeout at absolute time.\n\n    deadline argument points on the time in the same clock system\n    as loop.time().\n\n    Please note: it is not POSIX time but a time with\n    undefined starting base, e.g. the time of the system power on.\n\n    >>> async with timeout_at(loop.time() + 10):\n    ...     async with aiohttp.get('https://github.com') as r:\n    ...         await r.text()\n\n\n    "
    loop = asyncio.get_running_loop()
    return Timeout(deadline, loop)

class _State(enum.Enum):
    INIT = 'INIT'
    ENTER = 'ENTER'
    TIMEOUT = 'TIMEOUT'
    EXIT = 'EXIT'

@final
class Timeout:
    __slots__ = ('_deadline', '_loop', '_state', '_timeout_handler')

    def __init__(self, deadline: Optional[float], loop: asyncio.AbstractEventLoop) -> None:
        if False:
            return 10
        self._loop = loop
        self._state = _State.INIT
        self._timeout_handler = None
        if deadline is None:
            self._deadline = None
        else:
            self.update(deadline)

    def __enter__(self) -> 'Timeout':
        if False:
            while True:
                i = 10
        warnings.warn('with timeout() is deprecated, use async with timeout() instead', DeprecationWarning, stacklevel=2)
        self._do_enter()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        if False:
            while True:
                i = 10
        self._do_exit(exc_type)
        return None

    async def __aenter__(self) -> 'Timeout':
        self._do_enter()
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        self._do_exit(exc_type)
        return None

    @property
    def expired(self) -> bool:
        if False:
            while True:
                i = 10
        'Is timeout expired during execution?'
        return self._state == _State.TIMEOUT

    @property
    def deadline(self) -> Optional[float]:
        if False:
            i = 10
            return i + 15
        return self._deadline

    def reject(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Reject scheduled timeout if any.'
        if self._state not in (_State.INIT, _State.ENTER):
            raise RuntimeError(f'invalid state {self._state.value}')
        self._reject()

    def _reject(self) -> None:
        if False:
            return 10
        if self._timeout_handler is not None:
            self._timeout_handler.cancel()
            self._timeout_handler = None

    def shift(self, delay: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Advance timeout on delay seconds.\n\n        The delay can be negative.\n\n        Raise RuntimeError if shift is called when deadline is not scheduled\n        '
        deadline = self._deadline
        if deadline is None:
            raise RuntimeError('cannot shift timeout if deadline is not scheduled')
        self.update(deadline + delay)

    def update(self, deadline: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set deadline to absolute value.\n\n        deadline argument points on the time in the same clock system\n        as loop.time().\n\n        If new deadline is in the past the timeout is raised immediately.\n\n        Please note: it is not POSIX time but a time with\n        undefined starting base, e.g. the time of the system power on.\n        '
        if self._state == _State.EXIT:
            raise RuntimeError('cannot reschedule after exit from context manager')
        if self._state == _State.TIMEOUT:
            raise RuntimeError('cannot reschedule expired timeout')
        if self._timeout_handler is not None:
            self._timeout_handler.cancel()
        self._deadline = deadline
        if self._state != _State.INIT:
            self._reschedule()

    def _reschedule(self) -> None:
        if False:
            while True:
                i = 10
        assert self._state == _State.ENTER
        deadline = self._deadline
        if deadline is None:
            return
        now = self._loop.time()
        if self._timeout_handler is not None:
            self._timeout_handler.cancel()
        task = asyncio.current_task()
        if deadline <= now:
            self._timeout_handler = self._loop.call_soon(self._on_timeout, task)
        else:
            self._timeout_handler = self._loop.call_at(deadline, self._on_timeout, task)

    def _do_enter(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._state != _State.INIT:
            raise RuntimeError(f'invalid state {self._state.value}')
        self._state = _State.ENTER
        self._reschedule()

    def _do_exit(self, exc_type: Optional[Type[BaseException]]) -> None:
        if False:
            while True:
                i = 10
        if exc_type is asyncio.CancelledError and self._state == _State.TIMEOUT:
            self._timeout_handler = None
            raise asyncio.TimeoutError
        self._state = _State.EXIT
        self._reject()
        return None

    def _on_timeout(self, task: 'asyncio.Task[None]') -> None:
        if False:
            while True:
                i = 10
        task.cancel()
        self._state = _State.TIMEOUT
        self._timeout_handler = None