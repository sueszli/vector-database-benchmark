import unittest
from reactivex import operators as ops
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TestTakeWithTime(unittest.TestCase):

    def test_take_zero(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_completed(230))

        def create():
            if False:
                return 10
            return xs.pipe(ops.take_with_time(0))
        res = scheduler.start(create)
        assert res.messages == [on_completed(200)]
        assert xs.subscriptions == [subscribe(200, 200)]

    def test_take_some(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_next(230, 3), on_completed(240))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.take_with_time(25))
        res = scheduler.start(create)
        assert res.messages == [on_next(210, 1), on_next(220, 2), on_completed(225)]
        assert xs.subscriptions == [subscribe(200, 225)]

    def test_take_late(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_completed(230))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.take_with_time(50))
        res = scheduler.start(create)
        assert res.messages == [on_next(210, 1), on_next(220, 2), on_completed(230)]
        assert xs.subscriptions == [subscribe(200, 230)]

    def test_take_Error(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        ex = 'ex'
        xs = scheduler.create_hot_observable(on_error(210, ex))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.take_with_time(50))
        res = scheduler.start(create)
        assert res.messages == [on_error(210, ex)]
        assert xs.subscriptions == [subscribe(200, 210)]

    def test_take_never(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.take_with_time(50))
        res = scheduler.start(create)
        assert res.messages == [on_completed(250)]
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_take_twice1(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_next(230, 3), on_next(240, 4), on_next(250, 5), on_next(260, 6), on_completed(270))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.take_with_time(55), ops.take_with_time(35))
        res = scheduler.start(create)
        assert res.messages == [on_next(210, 1), on_next(220, 2), on_next(230, 3), on_completed(235)]
        assert xs.subscriptions == [subscribe(200, 235)]

    def test_take_twice2(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(220, 2), on_next(230, 3), on_next(240, 4), on_next(250, 5), on_next(260, 6), on_completed(270))

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.take_with_time(35), ops.take_with_time(55))
        res = scheduler.start(create)
        assert res.messages == [on_next(210, 1), on_next(220, 2), on_next(230, 3), on_completed(235)]
        assert xs.subscriptions == [subscribe(200, 235)]