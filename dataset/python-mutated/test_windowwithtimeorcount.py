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

class TestWindowWithTime(unittest.TestCase):

    def test_window_with_time_or_count_basic(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(205, 1), on_next(210, 2), on_next(240, 3), on_next(280, 4), on_next(320, 5), on_next(350, 6), on_next(370, 7), on_next(420, 8), on_next(470, 9), on_completed(600))

        def create():
            if False:
                return 10

            def projection(w, i):
                if False:
                    print('Hello World!')

                def inner_proj(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    return '%s %s' % (i, x)
                return w.pipe(ops.map(inner_proj))
            return xs.pipe(ops.window_with_time_or_count(70, 3), ops.map_indexed(projection), ops.merge_all())
        results = scheduler.start(create)
        assert results.messages == [on_next(205, '0 1'), on_next(210, '0 2'), on_next(240, '0 3'), on_next(280, '1 4'), on_next(320, '2 5'), on_next(350, '2 6'), on_next(370, '2 7'), on_next(420, '3 8'), on_next(470, '4 9'), on_completed(600)]
        assert xs.subscriptions == [subscribe(200, 600)]

    def test_window_with_time_or_count_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(205, 1), on_next(210, 2), on_next(240, 3), on_next(280, 4), on_next(320, 5), on_next(350, 6), on_next(370, 7), on_next(420, 8), on_next(470, 9), on_error(600, ex))

        def create():
            if False:
                i = 10
                return i + 15

            def projection(w, i):
                if False:
                    for i in range(10):
                        print('nop')

                def inner_proj(x):
                    if False:
                        return 10
                    return '%s %s' % (i, x)
                return w.pipe(ops.map(inner_proj))
            return xs.pipe(ops.window_with_time_or_count(70, 3), ops.map_indexed(projection), ops.merge_all())
        results = scheduler.start(create)
        assert results.messages == [on_next(205, '0 1'), on_next(210, '0 2'), on_next(240, '0 3'), on_next(280, '1 4'), on_next(320, '2 5'), on_next(350, '2 6'), on_next(370, '2 7'), on_next(420, '3 8'), on_next(470, '4 9'), on_error(600, ex)]
        assert xs.subscriptions == [subscribe(200, 600)]

    def test_window_with_time_or_count_disposed(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(205, 1), on_next(210, 2), on_next(240, 3), on_next(280, 4), on_next(320, 5), on_next(350, 6), on_next(370, 7), on_next(420, 8), on_next(470, 9), on_completed(600))

        def create():
            if False:
                return 10

            def projection(w, i):
                if False:
                    i = 10
                    return i + 15

                def inner_proj(x):
                    if False:
                        print('Hello World!')
                    return '%s %s' % (i, x)
                return w.pipe(ops.map(inner_proj))
            return xs.pipe(ops.window_with_time_or_count(70, 3), ops.map_indexed(projection), ops.merge_all())
        results = scheduler.start(create, disposed=370)
        assert results.messages == [on_next(205, '0 1'), on_next(210, '0 2'), on_next(240, '0 3'), on_next(280, '1 4'), on_next(320, '2 5'), on_next(350, '2 6'), on_next(370, '2 7')]
        assert xs.subscriptions == [subscribe(200, 370)]