import unittest
import reactivex
from reactivex.disposable import SerialDisposable
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
        print('Hello World!')
    raise RxException(ex)

class TestReturnValue(unittest.TestCase):

    def test_return_basic(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()

        def factory():
            if False:
                while True:
                    i = 10
            return reactivex.return_value(42)
        results = scheduler.start(factory)
        assert results.messages == [on_next(200, 42), on_completed(200)]

    def test_return_disposed(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()

        def factory():
            if False:
                print('Hello World!')
            return reactivex.return_value(42)
        results = scheduler.start(factory, disposed=200)
        assert results.messages == []

    def test_return_disposed_after_next(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        d = SerialDisposable()
        xs = reactivex.return_value(42)
        results = scheduler.create_observer()

        def action(scheduler, state):
            if False:
                return 10

            def on_next(x):
                if False:
                    for i in range(10):
                        print('nop')
                d.dispose()
                results.on_next(x)

            def on_error(e):
                if False:
                    while True:
                        i = 10
                results.on_error(e)

            def on_completed():
                if False:
                    while True:
                        i = 10
                results.on_completed()
            d.disposable = xs.subscribe(on_next, on_error, on_completed, scheduler=scheduler)
            return d.disposable
        scheduler.schedule_absolute(100, action)
        scheduler.start()
        assert results.messages == [on_next(100, 42)]

    def test_return_observer_throws(self):
        if False:
            i = 10
            return i + 15
        scheduler1 = TestScheduler()
        xs = reactivex.return_value(1)
        xs.subscribe(lambda x: _raise('ex'), scheduler=scheduler1)
        self.assertRaises(RxException, scheduler1.start)
        scheduler2 = TestScheduler()
        ys = reactivex.return_value(1)
        ys.subscribe(lambda x: x, lambda ex: ex, lambda : _raise('ex'), scheduler=scheduler2)
        self.assertRaises(RxException, scheduler2.start)