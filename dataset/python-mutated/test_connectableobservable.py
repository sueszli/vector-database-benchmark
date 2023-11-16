import unittest
import reactivex
from reactivex import ConnectableObservable, Observable
from reactivex import operators as ops
from reactivex.abc import ObserverBase
from reactivex.subject import Subject
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class MySubject(Observable, ObserverBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(MySubject, self).__init__()
        self.dispose_on_map = {}
        self.subscribe_count = 0
        self.disposed = False

    def _subscribe_core(self, observer, scheduler=None):
        if False:
            i = 10
            return i + 15
        self.subscribe_count += 1
        self.observer = observer

        class Duck:

            def __init__(self, this):
                if False:
                    for i in range(10):
                        print('nop')
                self.this = this

            def dispose(self) -> None:
                if False:
                    return 10
                self.this.disposed = True
        return Duck(self)

    def dispose_on(self, value, disposable):
        if False:
            for i in range(10):
                print('nop')
        self.dispose_on_map[value] = disposable

    def on_next(self, value):
        if False:
            return 10
        self.observer.on_next(value)
        if value in self.dispose_on_map:
            self.dispose_on_map[value].dispose()

    def on_error(self, error):
        if False:
            i = 10
            return i + 15
        self.observer.on_error(error)

    def on_completed(self):
        if False:
            while True:
                i = 10
        self.observer.on_completed()

class TestConnectableObservable(unittest.TestCase):

    def test_connectable_observable_creation(self):
        if False:
            while True:
                i = 10
        y = [0]
        s2 = Subject()
        co2 = ConnectableObservable(reactivex.return_value(1), s2)

        def on_next(x):
            if False:
                while True:
                    i = 10
            y[0] = x
        co2.subscribe(on_next=on_next)
        self.assertNotEqual(1, y[0])
        co2.connect()
        self.assertEqual(1, y[0])

    def test_connectable_observable_connected(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_next(230, 3), on_next(240, 4), on_completed(250))
        subject = MySubject()
        conn = ConnectableObservable(xs, subject)
        disconnect = conn.connect(scheduler)
        res = scheduler.start(lambda : conn)
        assert res.messages == [on_next(210, 1), on_next(220, 2), on_next(230, 3), on_next(240, 4), on_completed(250)]

    def test_connectable_observable_not_connected(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_next(230, 3), on_next(240, 4), on_completed(250))
        subject = MySubject()
        conn = ConnectableObservable(xs, subject)
        res = scheduler.start(lambda : conn)
        assert res.messages == []

    def test_connectable_observable_disconnected(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_next(230, 3), on_next(240, 4), on_completed(250))
        subject = MySubject()
        conn = ConnectableObservable(xs, subject)
        disconnect = conn.connect(scheduler)
        disconnect.dispose()
        res = scheduler.start(lambda : conn)
        assert res.messages == []

    def test_connectable_observable_disconnect_future(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_next(230, 3), on_next(240, 4), on_completed(250))
        subject = MySubject()
        conn = ConnectableObservable(xs, subject)
        subject.dispose_on(3, conn.connect())
        res = scheduler.start(lambda : conn)
        assert res.messages == [on_next(210, 1), on_next(220, 2), on_next(230, 3)]

    def test_connectable_observable_multiple_non_overlapped_connections(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_next(230, 3), on_next(240, 4), on_next(250, 5), on_next(260, 6), on_next(270, 7), on_next(280, 8), on_next(290, 9), on_completed(300))
        subject = Subject()
        conn = xs.pipe(ops.multicast(subject))
        c1 = [None]

        def action10(scheduler, state):
            if False:
                return 10
            c1[0] = conn.connect(scheduler)
        scheduler.schedule_absolute(225, action10)

        def action11(scheduler, state):
            if False:
                return 10
            c1[0].dispose()
        scheduler.schedule_absolute(241, action11)

        def action12(scheduler, state):
            if False:
                return 10
            c1[0].dispose()
        scheduler.schedule_absolute(245, action12)

        def action13(scheduler, state):
            if False:
                i = 10
                return i + 15
            c1[0].dispose()
        scheduler.schedule_absolute(251, action13)

        def action14(scheduler, state):
            if False:
                print('Hello World!')
            c1[0].dispose()
        scheduler.schedule_absolute(260, action14)
        c2 = [None]

        def action20(scheduler, state):
            if False:
                print('Hello World!')
            c2[0] = conn.connect(scheduler)
        scheduler.schedule_absolute(249, action20)

        def action21(scheduler, state):
            if False:
                for i in range(10):
                    print('nop')
            c2[0].dispose()
        scheduler.schedule_absolute(255, action21)

        def action22(scheduler, state):
            if False:
                for i in range(10):
                    print('nop')
            c2[0].dispose()
        scheduler.schedule_absolute(265, action22)

        def action23(scheduler, state):
            if False:
                return 10
            c2[0].dispose()
        scheduler.schedule_absolute(280, action23)
        c3 = [None]

        def action30(scheduler, state):
            if False:
                return 10
            c3[0] = conn.connect(scheduler)
        scheduler.schedule_absolute(275, action30)

        def action31(scheduler, state):
            if False:
                for i in range(10):
                    print('nop')
            c3[0].dispose()
        scheduler.schedule_absolute(295, action31)
        res = scheduler.start(lambda : conn)
        assert res.messages == [on_next(230, 3), on_next(240, 4), on_next(250, 5), on_next(280, 8), on_next(290, 9)]
        assert xs.subscriptions == [subscribe(225, 241), subscribe(249, 255), subscribe(275, 295)]

    def test_connectable_observable_forward_scheduler(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        subscribe_scheduler = 'unknown'

        def subscribe(observer, scheduler=None):
            if False:
                i = 10
                return i + 15
            nonlocal subscribe_scheduler
            subscribe_scheduler = scheduler
        xs = reactivex.create(subscribe)
        subject = MySubject()
        conn = ConnectableObservable(xs, subject)
        conn.connect(scheduler)
        assert subscribe_scheduler is scheduler