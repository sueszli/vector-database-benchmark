import unittest
from datetime import timedelta
import reactivex
from reactivex import operators as ops
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TimeInterval(object):

    def __init__(self, value, interval):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(interval, timedelta):
            interval = int(interval.seconds)
        self.value = value
        self.interval = interval

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s@%s' % (self.value, self.interval)

    def equals(self, other):
        if False:
            print('Hello World!')
        return other.interval == self.interval and other.value == self.value

class TestTimeInterval(unittest.TestCase):

    def test_time_interval_regular(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(230, 3), on_next(260, 4), on_next(300, 5), on_next(350, 6), on_completed(400))

        def create():
            if False:
                return 10

            def mapper(x):
                if False:
                    return 10
                return TimeInterval(x.value, x.interval)
            return xs.pipe(ops.time_interval(), ops.map(mapper))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, TimeInterval(2, 10)), on_next(230, TimeInterval(3, 20)), on_next(260, TimeInterval(4, 30)), on_next(300, TimeInterval(5, 40)), on_next(350, TimeInterval(6, 50)), on_completed(400)]

    def test_time_interval_empty(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()

        def create():
            if False:
                print('Hello World!')
            return reactivex.empty().pipe(ops.time_interval())
        results = scheduler.start(create)
        assert results.messages == [on_completed(200)]

    def test_time_interval_error(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return reactivex.throw(ex).pipe(ops.time_interval())
        results = scheduler.start(create)
        assert results.messages == [on_error(200, ex)]

    def test_time_interval_never(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()

        def create():
            if False:
                return 10
            return reactivex.never().pipe(ops.time_interval())
        results = scheduler.start(create)
        assert results.messages == []

    def test_time_interval_default_scheduler(self):
        if False:
            return 10
        import datetime
        import time
        xs = reactivex.of(1, 2).pipe(ops.time_interval(), ops.pluck_attr('interval'))
        l = []
        d = xs.subscribe(l.append)
        time.sleep(0.1)
        self.assertEqual(len(l), 2)
        [self.assertIsInstance(el, datetime.timedelta) for el in l]