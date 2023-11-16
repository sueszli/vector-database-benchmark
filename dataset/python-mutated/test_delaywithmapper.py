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
        for i in range(10):
            print('nop')
    raise RxException(ex)

class TestDelayWithMapper(unittest.TestCase):

    def test_delay_duration_simple1(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 10), on_next(220, 30), on_next(230, 50), on_next(240, 35), on_next(250, 20), on_completed(260))

        def create():
            if False:
                while True:
                    i = 10

            def mapper(x):
                if False:
                    for i in range(10):
                        print('nop')
                return scheduler.create_cold_observable(on_next(x, '!'))
            return xs.pipe(ops.delay_with_mapper(mapper))
        results = scheduler.start(create)
        assert results.messages == [on_next(210 + 10, 10), on_next(220 + 30, 30), on_next(250 + 20, 20), on_next(240 + 35, 35), on_next(230 + 50, 50), on_completed(280)]
        assert xs.subscriptions == [subscribe(200, 260)]

    def test_delay_duration_simple2(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_next(250, 6), on_completed(300))
        ys = scheduler.create_cold_observable(on_next(10, '!'))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.delay_with_mapper(lambda _: ys))
        results = scheduler.start(create)
        assert results.messages == [on_next(210 + 10, 2), on_next(220 + 10, 3), on_next(230 + 10, 4), on_next(240 + 10, 5), on_next(250 + 10, 6), on_completed(300)]
        assert xs.subscriptions == [subscribe(200, 300)]
        assert ys.subscriptions == [subscribe(210, 220), subscribe(220, 230), subscribe(230, 240), subscribe(240, 250), subscribe(250, 260)]

    def test_delay_duration_simple3(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_next(250, 6), on_completed(300))
        ys = scheduler.create_cold_observable(on_next(100, '!'))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.delay_with_mapper(lambda _: ys))
        results = scheduler.start(create)
        assert results.messages == [on_next(210 + 100, 2), on_next(220 + 100, 3), on_next(230 + 100, 4), on_next(240 + 100, 5), on_next(250 + 100, 6), on_completed(350)]
        assert xs.subscriptions == [subscribe(200, 300)]
        assert ys.subscriptions == [subscribe(210, 310), subscribe(220, 320), subscribe(230, 330), subscribe(240, 340), subscribe(250, 350)]

    def test_delay_duration_simple4_inner_empty(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_next(250, 6), on_completed(300))
        ys = scheduler.create_cold_observable(on_completed(100))

        def create():
            if False:
                return 10
            return xs.pipe(ops.delay_with_mapper(lambda _: ys))
        results = scheduler.start(create)
        assert results.messages == [on_next(210 + 100, 2), on_next(220 + 100, 3), on_next(230 + 100, 4), on_next(240 + 100, 5), on_next(250 + 100, 6), on_completed(350)]
        assert xs.subscriptions == [subscribe(200, 300)]
        assert ys.subscriptions == [subscribe(210, 310), subscribe(220, 320), subscribe(230, 330), subscribe(240, 340), subscribe(250, 350)]

    def test_delay_duration_dispose1(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_next(250, 6), on_completed(300))
        ys = scheduler.create_cold_observable(on_next(200, '!'))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.delay_with_mapper(lambda _: ys))
        results = scheduler.start(create, disposed=425)
        assert results.messages == [on_next(210 + 200, 2), on_next(220 + 200, 3)]
        assert xs.subscriptions == [subscribe(200, 300)]
        assert ys.subscriptions == [subscribe(210, 410), subscribe(220, 420), subscribe(230, 425), subscribe(240, 425), subscribe(250, 425)]

    def test_delay_duration_dispose2(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(400, 3), on_completed(500))
        ys = scheduler.create_cold_observable(on_next(50, '!'))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.delay_with_mapper(lambda _: ys))
        results = scheduler.start(create, disposed=300)
        assert results.messages == [on_next(210 + 50, 2)]
        assert xs.subscriptions == [subscribe(200, 300)]
        assert ys.subscriptions == [subscribe(210, 260)]