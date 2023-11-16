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

class RxException(Exception):
    pass

def _raise(ex):
    if False:
        return 10
    raise RxException(ex)

class TestThrottleFirst(unittest.TestCase):

    def test_throttle_first_completed(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(250, 3), on_next(310, 4), on_next(350, 5), on_next(410, 6), on_next(450, 7), on_completed(500))

        def create():
            if False:
                return 10
            return xs.pipe(ops.throttle_first(200))
        results = scheduler.start(create=create)
        assert results.messages == [on_next(210, 2), on_next(410, 6), on_completed(500)]
        assert xs.subscriptions == [subscribe(200, 500)]

    def test_throttle_first_never(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.throttle_first(200))
        results = scheduler.start(create=create)
        assert results.messages == []
        assert xs.subscriptions == [subscribe(200, 1000)]

    def test_throttle_first_empty(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_completed(500))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.throttle_first(200))
        results = scheduler.start(create=create)
        assert results.messages == [on_completed(500)]
        assert xs.subscriptions == [subscribe(200, 500)]

    def test_throttle_first_error(self):
        if False:
            return 10
        error = RxException()
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(250, 3), on_next(310, 4), on_next(350, 5), on_error(410, error), on_next(450, 7), on_completed(500))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.throttle_first(200))
        results = scheduler.start(create=create)
        assert results.messages == [on_next(210, 2), on_error(410, error)]
        assert xs.subscriptions == [subscribe(200, 410)]

    def test_throttle_first_no_end(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(250, 3), on_next(310, 4), on_next(350, 5), on_next(410, 6), on_next(450, 7))

        def create():
            if False:
                return 10
            return xs.pipe(ops.throttle_first(200))
        results = scheduler.start(create=create)
        assert results.messages == [on_next(210, 2), on_next(410, 6)]
        assert xs.subscriptions == [subscribe(200, 1000)]