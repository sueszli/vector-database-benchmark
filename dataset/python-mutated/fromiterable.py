from typing import Any, Iterable, Optional, TypeVar
from reactivex import Observable, abc
from reactivex.disposable import CompositeDisposable, Disposable
from reactivex.scheduler import CurrentThreadScheduler
_T = TypeVar('_T')

def from_iterable_(iterable: Iterable[_T], scheduler: Optional[abc.SchedulerBase]=None) -> Observable[_T]:
    if False:
        print('Hello World!')
    'Converts an iterable to an observable sequence.\n\n    Example:\n        >>> from_iterable([1,2,3])\n\n    Args:\n        iterable: A Python iterable\n        scheduler: An optional scheduler to schedule the values on.\n\n    Returns:\n        The observable sequence whose elements are pulled from the\n        given iterable sequence.\n    '

    def subscribe(observer: abc.ObserverBase[_T], scheduler_: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15
        _scheduler = scheduler or scheduler_ or CurrentThreadScheduler.singleton()
        iterator = iter(iterable)
        disposed = False

        def action(_: abc.SchedulerBase, __: Any=None) -> None:
            if False:
                return 10
            nonlocal disposed
            try:
                while not disposed:
                    value = next(iterator)
                    observer.on_next(value)
            except StopIteration:
                observer.on_completed()
            except Exception as error:
                observer.on_error(error)

        def dispose() -> None:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal disposed
            disposed = True
        disp = Disposable(dispose)
        return CompositeDisposable(_scheduler.schedule(action), disp)
    return Observable(subscribe)
__all__ = ['from_iterable_']