from typing import Any, Callable, List, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.disposable import CompositeDisposable, SerialDisposable, SingleAssignmentDisposable
from reactivex.scheduler import ImmediateScheduler
_T = TypeVar('_T')

def expand_(mapper: typing.Mapper[_T, Observable[_T]]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10

    def expand(source: Observable[_T]) -> Observable[_T]:
        if False:
            for i in range(10):
                print('nop')
        'Expands an observable sequence by recursively invoking\n        mapper.\n\n        Args:\n            source: Source obserable to expand.\n\n        Returns:\n            An observable sequence containing all the elements produced\n            by the recursive expansion.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            scheduler = scheduler or ImmediateScheduler.singleton()
            queue: List[Observable[_T]] = []
            m = SerialDisposable()
            d = CompositeDisposable(m)
            active_count = 0
            is_acquired = False

            def ensure_active():
                if False:
                    return 10
                nonlocal is_acquired
                is_owner = False
                if queue:
                    is_owner = not is_acquired
                    is_acquired = True

                def action(scheduler: abc.SchedulerBase, state: Any=None):
                    if False:
                        print('Hello World!')
                    nonlocal is_acquired, active_count
                    if queue:
                        work = queue.pop(0)
                    else:
                        is_acquired = False
                        return
                    sad = SingleAssignmentDisposable()
                    d.add(sad)

                    def on_next(value: _T) -> None:
                        if False:
                            print('Hello World!')
                        nonlocal active_count
                        observer.on_next(value)
                        result = None
                        try:
                            result = mapper(value)
                        except Exception as ex:
                            observer.on_error(ex)
                            return
                        queue.append(result)
                        active_count += 1
                        ensure_active()

                    def on_complete() -> None:
                        if False:
                            print('Hello World!')
                        nonlocal active_count
                        d.remove(sad)
                        active_count -= 1
                        if active_count == 0:
                            observer.on_completed()
                    sad.disposable = work.subscribe(on_next, observer.on_error, on_complete, scheduler=scheduler)
                    m.disposable = scheduler.schedule(action)
                if is_owner:
                    m.disposable = scheduler.schedule(action)
            queue.append(source)
            active_count += 1
            ensure_active()
            return d
        return Observable(subscribe)
    return expand
__all__ = ['expand_']