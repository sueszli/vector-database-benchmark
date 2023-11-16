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

class TestTakeLast(unittest.TestCase):

    def test_take_last_zero_completed(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9), on_completed(650))

        def create():
            if False:
                return 10
            return xs.pipe(ops.take_last(0))
        results = scheduler.start(create)
        assert results.messages == [on_completed(650)]
        assert xs.subscriptions == [subscribe(200, 650)]

    def test_take_last_zero_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9), on_error(650, ex))

        def create():
            if False:
                return 10
            return xs.pipe(ops.take_last(0))
        results = scheduler.start(create)
        assert results.messages == [on_error(650, ex)]
        assert xs.subscriptions == [subscribe(200, 650)]

    def test_take_last_zero_disposed(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.take_last(0))
        results = scheduler.start(create)
        assert results.messages == []
        assert xs.subscriptions == [subscribe(200, 1000)]

    def test_take_last_one_completed(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9), on_completed(650))

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.take_last(1))
        results = scheduler.start(create)
        assert results.messages == [on_next(650, 9), on_completed(650)]
        assert xs.subscriptions == [subscribe(200, 650)]

    def test_take_last_one_error(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9), on_error(650, ex))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.take_last(1))
        results = scheduler.start(create)
        assert results.messages == [on_error(650, ex)]
        assert xs.subscriptions == [subscribe(200, 650)]

    def test_take_last_One_disposed(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.take_last(1))
        results = scheduler.start(create)
        assert results.messages == []
        assert xs.subscriptions == [subscribe(200, 1000)]

    def test_take_last_three_completed(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9), on_completed(650))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.take_last(3))
        results = scheduler.start(create)
        assert results.messages == [on_next(650, 7), on_next(650, 8), on_next(650, 9), on_completed(650)]
        assert xs.subscriptions == [subscribe(200, 650)]

    def test_Take_last_three_error(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9), on_error(650, ex))

        def create():
            if False:
                return 10
            return xs.pipe(ops.take_last(3))
        results = scheduler.start(create)
        assert results.messages == [on_error(650, ex)]
        assert xs.subscriptions == [subscribe(200, 650)]

    def test_Take_last_three_disposed(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 1), on_next(210, 2), on_next(250, 3), on_next(270, 4), on_next(310, 5), on_next(360, 6), on_next(380, 7), on_next(410, 8), on_next(590, 9))

        def create():
            if False:
                return 10
            return xs.pipe(ops.take_last(3))
        results = scheduler.start(create)
        assert results.messages == []
        assert xs.subscriptions == [subscribe(200, 1000)]