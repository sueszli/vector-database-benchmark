from typing import Any, Callable, Optional
from reactivex import Observable, abc, typing
from reactivex.disposable import Disposable

def from_callback_(func: Callable[..., Callable[..., None]], mapper: Optional[typing.Mapper[Any, Any]]=None) -> Callable[[], Observable[Any]]:
    if False:
        i = 10
        return i + 15
    'Converts a callback function to an observable sequence.\n\n    Args:\n        func: Function with a callback as the last argument to\n            convert to an Observable sequence.\n        mapper: [Optional] A mapper which takes the arguments\n            from the callback to produce a single item to yield on next.\n\n    Returns:\n        A function, when executed with the required arguments minus\n        the callback, produces an Observable sequence with a single value of\n        the arguments to the callback as a list.\n    '

    def function(*args: Any) -> Observable[Any]:
        if False:
            print('Hello World!')
        arguments = list(args)

        def subscribe(observer: abc.ObserverBase[Any], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                print('Hello World!')

            def handler(*args: Any) -> None:
                if False:
                    i = 10
                    return i + 15
                results = list(args)
                if mapper:
                    try:
                        results = mapper(args)
                    except Exception as err:
                        observer.on_error(err)
                        return
                    observer.on_next(results)
                else:
                    if len(results) <= 1:
                        observer.on_next(*results)
                    else:
                        observer.on_next(results)
                    observer.on_completed()
            arguments.append(handler)
            func(*arguments)
            return Disposable()
        return Observable(subscribe)
    return function
__all__ = ['from_callback_']