from typing import Any, Callable, List, Optional, TypeVar, Union, cast
import reactivex
from reactivex import Observable, abc, typing
from reactivex.disposable import Disposable
from reactivex.scheduler import VirtualTimeScheduler
from reactivex.testing.recorded import Recorded
from .coldobservable import ColdObservable
from .hotobservable import HotObservable
from .mockobserver import MockObserver
from .reactivetest import ReactiveTest
_T = TypeVar('_T')
_TState = TypeVar('_TState')

class TestScheduler(VirtualTimeScheduler):
    """Test time scheduler used for testing applications and libraries
    built using Reactive Extensions. All time, both absolute and relative is
    specified as integer ticks"""
    __test__ = False

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: _TState=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed at the specified virtual\n        time.\n\n        Args:\n            duetime: Absolute virtual time at which to execute the\n                action.\n            action: Action to be executed.\n            state: State passed to the action to be executed.\n\n        Returns:\n            Disposable object used to cancel the scheduled action\n            (best effort).\n        '
        duetime = duetime if isinstance(duetime, float) else self.to_seconds(duetime)
        return super().schedule_absolute(duetime, action, state)

    def start(self, create: Optional[Callable[[], Observable[_T]]]=None, created: Optional[float]=None, subscribed: Optional[float]=None, disposed: Optional[float]=None) -> MockObserver[_T]:
        if False:
            for i in range(10):
                print('nop')
        'Starts the test scheduler and uses the specified virtual\n        times to invoke the factory function, subscribe to the\n        resulting sequence, and dispose the subscription.\n\n        Args:\n            create: Factory method to create an observable sequence.\n            created: Virtual time at which to invoke the factory to\n                create an observable sequence.\n            subscribed: Virtual time at which to subscribe to the\n                created observable sequence.\n            disposed: Virtual time at which to dispose the\n            subscription.\n\n        Returns:\n            Observer with timestamped recordings of notification\n            messages that were received during the virtual time window\n            when the subscription to the source sequence was active.\n        '
        created = created or ReactiveTest.created
        subscribed = subscribed or ReactiveTest.subscribed
        disposed = disposed or ReactiveTest.disposed
        observer = self.create_observer()
        subscription: Optional[abc.DisposableBase] = None
        source: Optional[abc.ObservableBase[_T]] = None

        def action_create(scheduler: abc.SchedulerBase, state: Any=None) -> abc.DisposableBase:
            if False:
                for i in range(10):
                    print('nop')
            'Called at create time. Defaults to 100'
            nonlocal source
            source = create() if create is not None else reactivex.never()
            return Disposable()
        self.schedule_absolute(created, action_create)

        def action_subscribe(scheduler: abc.SchedulerBase, state: Any=None) -> abc.DisposableBase:
            if False:
                i = 10
                return i + 15
            'Called at subscribe time. Defaults to 200'
            nonlocal subscription
            if source:
                subscription = source.subscribe(observer, scheduler=scheduler)
            return Disposable()
        self.schedule_absolute(subscribed, action_subscribe)

        def action_dispose(scheduler: abc.SchedulerBase, state: Any=None) -> abc.DisposableBase:
            if False:
                print('Hello World!')
            'Called at dispose time. Defaults to 1000'
            if subscription:
                subscription.dispose()
            return Disposable()
        self.schedule_absolute(disposed, action_dispose)
        super().start()
        return observer

    def create_hot_observable(self, *args: Union[Recorded[_T], List[Recorded[_T]]]) -> HotObservable[_T]:
        if False:
            while True:
                i = 10
        'Creates a hot observable using the specified timestamped\n        notification messages either as a list or by multiple arguments.\n\n        Args:\n            messages: Notifications to surface through the created sequence at\n            their specified absolute virtual times.\n\n        Returns hot observable sequence that can be used to assert the timing\n        of subscriptions and notifications.\n        '
        if args and isinstance(args[0], List):
            messages = args[0]
        else:
            messages = cast(List[Recorded[_T]], list(args))
        return HotObservable(self, messages)

    def create_cold_observable(self, *args: Union[Recorded[_T], List[Recorded[_T]]]) -> ColdObservable[_T]:
        if False:
            return 10
        'Creates a cold observable using the specified timestamped\n        notification messages either as an array or arguments.\n\n        Args:\n            args: Notifications to surface through the created sequence\n                at their specified virtual time offsets from the\n                sequence subscription time.\n\n        Returns:\n            Cold observable sequence that can be used to assert the\n            timing of subscriptions and notifications.\n        '
        if args and isinstance(args[0], list):
            messages = args[0]
        else:
            messages = cast(List[Recorded[_T]], list(args))
        return ColdObservable(self, messages)

    def create_observer(self) -> MockObserver[Any]:
        if False:
            print('Hello World!')
        'Creates an observer that records received notification messages and\n        timestamps those. Return an Observer that can be used to assert the\n        timing of received notifications.\n        '
        return MockObserver(self)