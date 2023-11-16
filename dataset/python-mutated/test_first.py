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

class TestFirst(unittest.TestCase):

    def test_first_async_empty(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_completed(250))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.first())
        res = scheduler.start(create=create)
        assert [on_error(250, lambda e: e)] == res.messages
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_first_async_one(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_completed(250))
        res = scheduler.start(lambda : xs.pipe(ops.first()))
        assert res.messages == [on_next(210, 2), on_completed(210)]
        assert xs.subscriptions == [subscribe(200, 210)]

    def test_first_async_many(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_completed(250))
        res = scheduler.start(lambda : xs.pipe(ops.first()))
        assert res.messages == [on_next(210, 2), on_completed(210)]
        assert xs.subscriptions == [subscribe(200, 210)]

    def test_first_async_error(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_error(210, ex))
        res = scheduler.start(lambda : xs.pipe(ops.first()))
        assert res.messages == [on_error(210, ex)]
        assert xs.subscriptions == [subscribe(200, 210)]

    def test_first_async_predicate(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.first(lambda x: x % 2 == 1))
        res = scheduler.start(create=create)
        assert res.messages == [on_next(220, 3), on_completed(220)]
        assert xs.subscriptions == [subscribe(200, 220)]

    def test_first_async_predicate_none(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.first(lambda x: x > 10))
        res = scheduler.start(create=create)
        assert [on_error(250, lambda e: e)] == res.messages
        assert xs.subscriptions == [subscribe(200, 250)]

    def test_first_async_predicate_on_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_error(220, ex))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.first(lambda x: x % 2 == 1))
        res = scheduler.start(create=create)
        assert res.messages == [on_error(220, ex)]
        assert xs.subscriptions == [subscribe(200, 220)]

    def test_first_async_predicate_throws(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))

        def create():
            if False:
                print('Hello World!')

            def predicate(x):
                if False:
                    i = 10
                    return i + 15
                if x < 4:
                    return False
                else:
                    raise Exception(ex)
            return xs.pipe(ops.first(predicate))
        res = scheduler.start(create=create)
        assert res.messages == [on_error(230, ex)]
        assert xs.subscriptions == [subscribe(200, 230)]
if __name__ == '__main__':
    unittest.main()