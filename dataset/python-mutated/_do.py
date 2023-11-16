from typing import Any, Callable, List, Optional, TypeVar
from reactivex import Observable, abc, typing
from reactivex.disposable import CompositeDisposable
_T = TypeVar('_T')

def do_action_(on_next: Optional[typing.OnNext[_T]]=None, on_error: Optional[typing.OnError]=None, on_completed: Optional[typing.OnCompleted]=None) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        print('Hello World!')

    def do_action(source: Observable[_T]) -> Observable[_T]:
        if False:
            return 10
        'Invokes an action for each element in the observable\n        sequence and invokes an action on graceful or exceptional\n        termination of the observable sequence. This method can be used\n        for debugging, logging, etc. of query behavior by intercepting\n        the message stream to run arbitrary actions for messages on the\n        pipeline.\n\n        Examples:\n            >>> do_action(send)(observable)\n            >>> do_action(on_next, on_error)(observable)\n            >>> do_action(on_next, on_error, on_completed)(observable)\n\n        Args:\n            on_next: [Optional] Action to invoke for each element in\n                the observable sequence.\n            on_error: [Optional] Action to invoke on exceptional\n                termination of the observable sequence.\n            on_completed: [Optional] Action to invoke on graceful\n                termination of the observable sequence.\n\n        Returns:\n            An observable source sequence with the side-effecting\n            behavior applied.\n        '

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')

            def _on_next(x: _T) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                if not on_next:
                    observer.on_next(x)
                else:
                    try:
                        on_next(x)
                    except Exception as e:
                        observer.on_error(e)
                    observer.on_next(x)

            def _on_error(exception: Exception) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                if not on_error:
                    observer.on_error(exception)
                else:
                    try:
                        on_error(exception)
                    except Exception as e:
                        observer.on_error(e)
                    observer.on_error(exception)

            def _on_completed() -> None:
                if False:
                    for i in range(10):
                        print('nop')
                if not on_completed:
                    observer.on_completed()
                else:
                    try:
                        on_completed()
                    except Exception as e:
                        observer.on_error(e)
                    observer.on_completed()
            return source.subscribe(_on_next, _on_error, _on_completed, scheduler=scheduler)
        return Observable(subscribe)
    return do_action

def do_(observer: abc.ObserverBase[_T]) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        while True:
            i = 10
    'Invokes an action for each element in the observable sequence and\n    invokes an action on graceful or exceptional termination of the\n    observable sequence. This method can be used for debugging, logging,\n    etc. of query behavior by intercepting the message stream to run\n    arbitrary actions for messages on the pipeline.\n\n    >>> do(observer)\n\n    Args:\n        observer: Observer\n\n    Returns:\n        An operator function that takes the source observable and\n        returns the source sequence with the side-effecting behavior\n        applied.\n    '
    return do_action_(observer.on_next, observer.on_error, observer.on_completed)

def do_after_next(source: Observable[_T], after_next: typing.OnNext[_T]):
    if False:
        return 10
    'Invokes an action with each element after it has been emitted downstream.\n    This can be helpful for debugging, logging, and other side effects.\n\n    after_next -- Action to invoke on each element after it has been emitted\n    '

    def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')

        def on_next(value: _T):
            if False:
                print('Hello World!')
            try:
                observer.on_next(value)
                after_next(value)
            except Exception as e:
                observer.on_error(e)
        return source.subscribe(on_next, observer.on_error, observer.on_completed)
    return Observable(subscribe)

def do_on_subscribe(source: Observable[Any], on_subscribe: typing.Action):
    if False:
        print('Hello World!')
    'Invokes an action on subscription.\n\n    This can be helpful for debugging, logging, and other side effects\n    on the start of an operation.\n\n    Args:\n        on_subscribe: Action to invoke on subscription\n    '

    def subscribe(observer: abc.ObserverBase[Any], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            return 10
        on_subscribe()
        return source.subscribe(observer.on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
    return Observable(subscribe)

def do_on_dispose(source: Observable[Any], on_dispose: typing.Action):
    if False:
        print('Hello World!')
    'Invokes an action on disposal.\n\n     This can be helpful for debugging, logging, and other side effects\n     on the disposal of an operation.\n\n    Args:\n        on_dispose: Action to invoke on disposal\n    '

    class OnDispose(abc.DisposableBase):

        def dispose(self) -> None:
            if False:
                while True:
                    i = 10
            on_dispose()

    def subscribe(observer: abc.ObserverBase[Any], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            return 10
        composite_disposable = CompositeDisposable()
        composite_disposable.add(OnDispose())
        subscription = source.subscribe(observer.on_next, observer.on_error, observer.on_completed, scheduler=scheduler)
        composite_disposable.add(subscription)
        return composite_disposable
    return Observable(subscribe)

def do_on_terminate(source: Observable[Any], on_terminate: typing.Action):
    if False:
        for i in range(10):
            print('nop')
    'Invokes an action on an on_complete() or on_error() event.\n     This can be helpful for debugging, logging, and other side effects\n     when completion or an error terminates an operation.\n\n\n    on_terminate -- Action to invoke when on_complete or throw is called\n    '

    def subscribe(observer: abc.ObserverBase[Any], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15

        def on_completed():
            if False:
                print('Hello World!')
            try:
                on_terminate()
            except Exception as err:
                observer.on_error(err)
            else:
                observer.on_completed()

        def on_error(exception: Exception):
            if False:
                while True:
                    i = 10
            try:
                on_terminate()
            except Exception as err:
                observer.on_error(err)
            else:
                observer.on_error(exception)
        return source.subscribe(observer.on_next, on_error, on_completed, scheduler=scheduler)
    return Observable(subscribe)

def do_after_terminate(source: Observable[Any], after_terminate: typing.Action):
    if False:
        for i in range(10):
            print('nop')
    'Invokes an action after an on_complete() or on_error() event.\n     This can be helpful for debugging, logging, and other side effects\n     when completion or an error terminates an operation\n\n\n    on_terminate -- Action to invoke after on_complete or throw is called\n    '

    def subscribe(observer: abc.ObserverBase[Any], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            return 10

        def on_completed():
            if False:
                i = 10
                return i + 15
            observer.on_completed()
            try:
                after_terminate()
            except Exception as err:
                observer.on_error(err)

        def on_error(exception: Exception) -> None:
            if False:
                return 10
            observer.on_error(exception)
            try:
                after_terminate()
            except Exception as err:
                observer.on_error(err)
        return source.subscribe(observer.on_next, on_error, on_completed, scheduler=scheduler)
    return Observable(subscribe)

def do_finally(finally_action: typing.Action) -> Callable[[Observable[_T]], Observable[_T]]:
    if False:
        return 10
    'Invokes an action after an on_complete(), on_error(), or disposal\n    event occurs.\n\n    This can be helpful for debugging, logging, and other side effects\n    when completion, an error, or disposal terminates an operation.\n\n    Note this operator will strive to execute the finally_action once,\n    and prevent any redudant calls\n\n    Args:\n        finally_action -- Action to invoke after on_complete, on_error,\n        or disposal is called\n    '

    class OnDispose(abc.DisposableBase):

        def __init__(self, was_invoked: List[bool]):
            if False:
                return 10
            self.was_invoked = was_invoked

        def dispose(self) -> None:
            if False:
                print('Hello World!')
            if not self.was_invoked[0]:
                finally_action()
                self.was_invoked[0] = True

    def partial(source: Observable[_T]) -> Observable[_T]:
        if False:
            while True:
                i = 10

        def subscribe(observer: abc.ObserverBase[_T], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
            if False:
                print('Hello World!')
            was_invoked = [False]

            def on_completed():
                if False:
                    i = 10
                    return i + 15
                observer.on_completed()
                try:
                    if not was_invoked[0]:
                        finally_action()
                        was_invoked[0] = True
                except Exception as err:
                    observer.on_error(err)

            def on_error(exception: Exception):
                if False:
                    while True:
                        i = 10
                observer.on_error(exception)
                try:
                    if not was_invoked[0]:
                        finally_action()
                        was_invoked[0] = True
                except Exception as err:
                    observer.on_error(err)
            composite_disposable = CompositeDisposable()
            composite_disposable.add(OnDispose(was_invoked))
            subscription = source.subscribe(observer.on_next, on_error, on_completed, scheduler=scheduler)
            composite_disposable.add(subscription)
            return composite_disposable
        return Observable(subscribe)
    return partial
__all__ = ['do_', 'do_action_', 'do_after_next', 'do_finally', 'do_on_dispose', 'do_on_subscribe', 'do_on_terminate', 'do_after_terminate']