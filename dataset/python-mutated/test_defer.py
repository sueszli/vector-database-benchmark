import unittest
import reactivex
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

class TestDefer(unittest.TestCase):

    def test_defer_complete(self):
        if False:
            i = 10
            return i + 15
        xs = [None]
        invoked = [0]
        scheduler = TestScheduler()

        def create():
            if False:
                print('Hello World!')

            def defer(scheduler):
                if False:
                    return 10
                invoked[0] += 1
                xs[0] = scheduler.create_cold_observable(on_next(100, scheduler.clock), on_completed(200))
                return xs[0]
            return reactivex.defer(defer)
        results = scheduler.start(create)
        assert results.messages == [on_next(300, 200), on_completed(400)]
        assert 1 == invoked[0]
        assert xs[0].subscriptions == [subscribe(200, 400)]

    def test_defer_error(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        invoked = [0]
        xs = [None]
        ex = 'ex'

        def create():
            if False:
                return 10

            def defer(scheduler):
                if False:
                    return 10
                invoked[0] += 1
                xs[0] = scheduler.create_cold_observable(on_next(100, scheduler.clock), on_error(200, ex))
                return xs[0]
            return reactivex.defer(defer)
        results = scheduler.start(create)
        assert results.messages == [on_next(300, 200), on_error(400, ex)]
        assert 1 == invoked[0]
        assert xs[0].subscriptions == [subscribe(200, 400)]

    def test_defer_dispose(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        invoked = [0]
        xs = [None]

        def create():
            if False:
                print('Hello World!')

            def defer(scheduler):
                if False:
                    i = 10
                    return i + 15
                invoked[0] += 1
                xs[0] = scheduler.create_cold_observable(on_next(100, scheduler.clock), on_next(200, invoked[0]), on_next(1100, 1000))
                return xs[0]
            return reactivex.defer(defer)
        results = scheduler.start(create)
        assert results.messages == [on_next(300, 200), on_next(400, 1)]
        assert 1 == invoked[0]
        assert xs[0].subscriptions == [subscribe(200, 1000)]

    def test_defer_on_error(self):
        if False:
            return 10
        scheduler = TestScheduler()
        invoked = [0]
        ex = 'ex'

        def create():
            if False:
                for i in range(10):
                    print('nop')

            def defer(scheduler):
                if False:
                    while True:
                        i = 10
                invoked[0] += 1
                raise Exception(ex)
            return reactivex.defer(defer)
        results = scheduler.start(create)
        assert results.messages == [on_error(200, ex)]
        assert 1 == invoked[0]