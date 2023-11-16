from timeit import default_timer
from types import TracebackType
from typing import Any, Callable, Literal, Optional, Tuple, Type, TYPE_CHECKING, TypeVar, Union
from .decorator import decorate
if TYPE_CHECKING:
    from . import Counter
    F = TypeVar('F', bound=Callable[..., Any])

class ExceptionCounter:

    def __init__(self, counter: 'Counter', exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]]) -> None:
        if False:
            return 10
        self._counter = counter
        self._exception = exception

    def __enter__(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def __exit__(self, typ: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> Literal[False]:
        if False:
            print('Hello World!')
        if isinstance(value, self._exception):
            self._counter.inc()
        return False

    def __call__(self, f: 'F') -> 'F':
        if False:
            print('Hello World!')

        def wrapped(func, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            with self:
                return func(*args, **kwargs)
        return decorate(f, wrapped)

class InprogressTracker:

    def __init__(self, gauge):
        if False:
            for i in range(10):
                print('nop')
        self._gauge = gauge

    def __enter__(self):
        if False:
            return 10
        self._gauge.inc()

    def __exit__(self, typ, value, traceback):
        if False:
            return 10
        self._gauge.dec()

    def __call__(self, f: 'F') -> 'F':
        if False:
            for i in range(10):
                print('nop')

        def wrapped(func, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            with self:
                return func(*args, **kwargs)
        return decorate(f, wrapped)

class Timer:

    def __init__(self, metric, callback_name):
        if False:
            while True:
                i = 10
        self._metric = metric
        self._callback_name = callback_name

    def _new_timer(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__(self._metric, self._callback_name)

    def __enter__(self):
        if False:
            return 10
        self._start = default_timer()
        return self

    def __exit__(self, typ, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        duration = max(default_timer() - self._start, 0)
        callback = getattr(self._metric, self._callback_name)
        callback(duration)

    def labels(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        self._metric = self._metric.labels(*args, **kw)

    def __call__(self, f: 'F') -> 'F':
        if False:
            print('Hello World!')

        def wrapped(func, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            with self._new_timer():
                return func(*args, **kwargs)
        return decorate(f, wrapped)