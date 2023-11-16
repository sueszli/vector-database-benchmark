import unittest
import reactivex
from reactivex import operators as ops
from reactivex.scheduler import ImmediateScheduler
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TestObserveOn(unittest.TestCase):

    def test_observe_on_normal(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_completed(250))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.observe_on(scheduler))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_completed(250)]
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_observe_on_error(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        ex = 'ex'
        xs = scheduler.create_hot_observable(on_next(150, 1), on_error(210, ex))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.observe_on(scheduler))
        results = scheduler.start(create)
        assert results.messages == [on_error(210, ex)]
        assert xs.subscriptions == [subscribe(200, 210)]

    def test_observe_on_empty(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_completed(250))

        def create():
            if False:
                return 10
            return xs.pipe(ops.observe_on(scheduler))
        results = scheduler.start(create)
        assert results.messages == [on_completed(250)]
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_observe_on_never(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.observe_on(scheduler))
        results = scheduler.start(create)
        assert results.messages == []
        assert xs.subscriptions == [subscribe(200, 1000)]

    def test_observe_on_forward_subscribe_scheduler(self):
        if False:
            while True:
                i = 10
        scheduler = ImmediateScheduler()
        expected_subscribe_scheduler = ImmediateScheduler()
        actual_subscribe_scheduler = None

        def subscribe(observer, scheduler):
            if False:
                print('Hello World!')
            nonlocal actual_subscribe_scheduler
            actual_subscribe_scheduler = scheduler
            observer.on_completed()
        xs = reactivex.create(subscribe)
        xs.pipe(ops.observe_on(scheduler)).subscribe(scheduler=expected_subscribe_scheduler)
        assert expected_subscribe_scheduler == actual_subscribe_scheduler