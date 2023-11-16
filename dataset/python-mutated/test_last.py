import unittest
from reactivex import operators as _
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

class TestLast(unittest.TestCase):

    def test_last_async_empty(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_completed(250))

        def create():
            if False:
                return 10
            return xs.pipe(_.last())
        res = scheduler.start(create=create)

        def predicate(e):
            if False:
                print('Hello World!')
            return e is not None
        assert [on_error(250, predicate)] == res.messages
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_last_async_one(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_completed(250))

        def create():
            if False:
                return 10
            return xs.pipe(_.last())
        res = scheduler.start(create=create)
        assert res.messages == [on_next(250, 2), on_completed(250)]
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_last_async_many(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_completed(250))

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(_.last())
        res = scheduler.start(create=create)
        assert res.messages == [on_next(250, 3), on_completed(250)]
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_last_async_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_error(210, ex))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(_.last())
        res = scheduler.start(create=create)
        assert res.messages == [on_error(210, ex)]
        assert xs.subscriptions == [subscribe(200, 210)]

    def test_last_async_predicate(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))

        def create():
            if False:
                return 10

            def predicate(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x % 2 == 1
            return xs.pipe(_.last(predicate))
        res = scheduler.start(create=create)
        assert res.messages == [on_next(250, 5), on_completed(250)]
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_last_async_predicate_none(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))

        def create():
            if False:
                while True:
                    i = 10

            def predicate(x):
                if False:
                    return 10
                return x > 10
            return xs.pipe(_.last(predicate))
        res = scheduler.start(create=create)

        def predicate(e):
            if False:
                return 10
            return e is not None
        assert [on_error(250, predicate)] == res.messages
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_last_async_predicate_on_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_error(210, ex))

        def create():
            if False:
                return 10

            def predicate(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x % 2 == 1
            return xs.pipe(_.last(predicate))
        res = scheduler.start(create=create)
        assert res.messages == [on_error(210, ex)]
        assert xs.subscriptions == [subscribe(200, 210)]

    def test_last_async_predicate_throws(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))

        def create():
            if False:
                for i in range(10):
                    print('nop')

            def predicate(x):
                if False:
                    return 10
                if x < 4:
                    return x % 2 == 1
                else:
                    raise Exception(ex)
            return xs.pipe(_.last(predicate))
        res = scheduler.start(create=create)
        assert res.messages == [on_error(230, ex)]
        assert xs.subscriptions == [subscribe(200, 230)]