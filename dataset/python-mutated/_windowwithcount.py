import logging
from typing import Callable, List, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.disposable import RefCountDisposable, SingleAssignmentDisposable
from reactivex.internal import ArgumentOutOfRangeException, add_ref
from reactivex.subject import Subject
log = logging.getLogger('Rx')
_T = TypeVar('_T')

def window_with_count_(count: int, skip: Optional[int]=None) -> Callable[[Observable[_T]], Observable[Observable[_T]]]:
    if False:
        while True:
            i = 10
    'Projects each element of an observable sequence into zero or more\n    windows which are produced based on element count information.\n\n    Examples:\n        >>> window_with_count(10)\n        >>> window_with_count(10, 1)\n\n    Args:\n        count: Length of each window.\n        skip: [Optional] Number of elements to skip between creation of\n            consecutive windows. If not specified, defaults to the\n            count.\n\n    Returns:\n        An observable sequence of windows.\n    '
    if count <= 0:
        raise ArgumentOutOfRangeException()
    skip_ = skip if skip is not None else count
    if skip_ <= 0:
        raise ArgumentOutOfRangeException()

    def window_with_count(source: Observable[_T]) -> Observable[Observable[_T]]:
        if False:
            print('Hello World!')

        def subscribe(observer: abc.ObserverBase[Observable[_T]], scheduler: Optional[abc.SchedulerBase]=None):
            if False:
                i = 10
                return i + 15
            m = SingleAssignmentDisposable()
            refCountDisposable = RefCountDisposable(m)
            n = [0]
            q: List[Subject[_T]] = []

            def create_window():
                if False:
                    print('Hello World!')
                s: Subject[_T] = Subject()
                q.append(s)
                observer.on_next(add_ref(s, refCountDisposable))
            create_window()

            def on_next(x: _T) -> None:
                if False:
                    while True:
                        i = 10
                for item in q:
                    item.on_next(x)
                c = n[0] - count + 1
                if c >= 0 and c % skip_ == 0:
                    s = q.pop(0)
                    s.on_completed()
                n[0] += 1
                if n[0] % skip_ == 0:
                    create_window()

            def on_error(exception: Exception) -> None:
                if False:
                    i = 10
                    return i + 15
                while q:
                    q.pop(0).on_error(exception)
                observer.on_error(exception)

            def on_completed() -> None:
                if False:
                    return 10
                while q:
                    q.pop(0).on_completed()
                observer.on_completed()
            m.disposable = source.subscribe(on_next, on_error, on_completed, scheduler=scheduler)
            return refCountDisposable
        return Observable(subscribe)
    return window_with_count
__all__ = ['window_with_count_']