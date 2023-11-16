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
        while True:
            i = 10
    raise RxException(ex)

class TestTimeInterval(unittest.TestCase):

    def test_interval_timespan_basic(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()

        def create():
            if False:
                while True:
                    i = 10
            return reactivex.interval(100)
        results = scheduler.start(create)
        assert results.messages == [on_next(300, 0), on_next(400, 1), on_next(500, 2), on_next(600, 3), on_next(700, 4), on_next(800, 5), on_next(900, 6)]

    def test_interval_timespan_disposed(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()

        def create():
            if False:
                print('Hello World!')
            return reactivex.interval(1000)
        results = scheduler.start(create)
        assert results.messages == []

    def test_interval_timespan_observer_throws(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = reactivex.interval(1)
        xs.subscribe(lambda x: _raise('ex'), scheduler=scheduler)
        with self.assertRaises(RxException):
            scheduler.start()