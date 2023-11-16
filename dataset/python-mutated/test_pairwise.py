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

class TestPairwise(unittest.TestCase):

    def test_pairwise_empty(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 5), on_completed(210))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.pairwise())
        results = scheduler.start(create)
        assert results.messages == [on_completed(210)]
        assert xs.subscriptions == [subscribe(200, 210)]

    def test_pairwise_single(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 5), on_next(210, 4), on_completed(220))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.pairwise())
        results = scheduler.start(create)
        assert results.messages == [on_completed(220)]
        assert xs.subscriptions == [subscribe(200, 220)]

    def test_pairwise_completed(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 5), on_next(210, 4), on_next(240, 3), on_next(290, 2), on_next(350, 1), on_completed(360))

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.pairwise())
        results = scheduler.start(create)
        assert results.messages == [on_next(240, (4, 3)), on_next(290, (3, 2)), on_next(350, (2, 1)), on_completed(360)]
        assert xs.subscriptions == [subscribe(200, 360)]

    def test_pairwise_not_completed(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 5), on_next(210, 4), on_next(240, 3), on_next(290, 2), on_next(350, 1))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.pairwise())
        results = scheduler.start(create)
        assert results.messages == [on_next(240, (4, 3)), on_next(290, (3, 2)), on_next(350, (2, 1))]
        assert xs.subscriptions == [subscribe(200, 1000)]

    def test_pairwise_error(self):
        if False:
            while True:
                i = 10
        error = Exception()
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 5), on_next(210, 4), on_next(240, 3), on_error(290, error), on_next(350, 1), on_completed(360))

        def create():
            if False:
                return 10
            return xs.pipe(ops.pairwise())
        results = scheduler.start(create)
        assert results.messages == [on_next(240, (4, 3)), on_error(290, error)]
        assert xs.subscriptions == [subscribe(200, 290)]

    def test_pairwise_disposed(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(180, 5), on_next(210, 4), on_next(240, 3), on_next(290, 2), on_next(350, 1), on_completed(360))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.pairwise())
        results = scheduler.start(create, disposed=280)
        assert results.messages == [on_next(240, (4, 3))]
        assert xs.subscriptions == [subscribe(200, 280)]