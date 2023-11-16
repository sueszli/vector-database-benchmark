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

class TestIf_then(unittest.TestCase):

    def test_if_true(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(250, 2), on_completed(300))
        ys = scheduler.create_hot_observable(on_next(310, 3), on_next(350, 4), on_completed(400))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return reactivex.if_then(lambda : True, xs, ys)
        results = scheduler.start(create=create)
        assert results.messages == [on_next(210, 1), on_next(250, 2), on_completed(300)]
        assert xs.subscriptions == [subscribe(200, 300)]
        assert ys.subscriptions == []

    def test_if_false(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(250, 2), on_completed(300))
        ys = scheduler.create_hot_observable(on_next(310, 3), on_next(350, 4), on_completed(400))

        def create():
            if False:
                while True:
                    i = 10
            return reactivex.if_then(lambda : False, xs, ys)
        results = scheduler.start(create=create)
        assert results.messages == [on_next(310, 3), on_next(350, 4), on_completed(400)]
        assert xs.subscriptions == []
        assert ys.subscriptions == [subscribe(200, 400)]

    def test_if_on_error(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(250, 2), on_completed(300))
        ys = scheduler.create_hot_observable(on_next(310, 3), on_next(350, 4), on_completed(400))

        def create():
            if False:
                return 10

            def condition():
                if False:
                    return 10
                raise Exception(ex)
            return reactivex.if_then(condition, xs, ys)
        results = scheduler.start(create=create)
        assert results.messages == [on_error(200, ex)]
        assert xs.subscriptions == []
        assert ys.subscriptions == []

    def test_if_dispose(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(210, 1), on_next(250, 2))
        ys = scheduler.create_hot_observable(on_next(310, 3), on_next(350, 4), on_completed(400))

        def create():
            if False:
                return 10
            return reactivex.if_then(lambda : True, xs, ys)
        results = scheduler.start(create=create)
        assert results.messages == [on_next(210, 1), on_next(250, 2)]
        assert xs.subscriptions == [subscribe(200, 1000)]
        assert ys.subscriptions == []

    def test_if_default_completed(self):
        if False:
            for i in range(10):
                print('nop')
        b = [False]
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(110, 1), on_next(220, 2), on_next(330, 3), on_completed(440))

        def action(scheduler, state):
            if False:
                print('Hello World!')
            b[0] = True
        scheduler.schedule_absolute(150, action)

        def create():
            if False:
                return 10

            def condition():
                if False:
                    for i in range(10):
                        print('nop')
                return b[0]
            return reactivex.if_then(condition, xs)
        results = scheduler.start(create)
        assert results.messages == [on_next(220, 2), on_next(330, 3), on_completed(440)]
        assert xs.subscriptions == [subscribe(200, 440)]

    def test_if_default_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        b = [False]
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(110, 1), on_next(220, 2), on_next(330, 3), on_error(440, ex))

        def action(scheduler, state):
            if False:
                while True:
                    i = 10
            b[0] = True
        scheduler.schedule_absolute(150, action)

        def create():
            if False:
                for i in range(10):
                    print('nop')

            def condition():
                if False:
                    return 10
                return b[0]
            return reactivex.if_then(condition, xs)
        results = scheduler.start(create)
        assert results.messages == [on_next(220, 2), on_next(330, 3), on_error(440, ex)]
        assert xs.subscriptions == [subscribe(200, 440)]

    def test_if_default_never(self):
        if False:
            print('Hello World!')
        b = [False]
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(110, 1), on_next(220, 2), on_next(330, 3))

        def action(scheduler, state):
            if False:
                i = 10
                return i + 15
            b[0] = True
        scheduler.schedule_absolute(150, action)

        def create():
            if False:
                i = 10
                return i + 15

            def condition():
                if False:
                    i = 10
                    return i + 15
                return b[0]
            return reactivex.if_then(condition, xs)
        results = scheduler.start(create)
        assert results.messages == [on_next(220, 2), on_next(330, 3)]
        assert xs.subscriptions == [subscribe(200, 1000)]

    def test_if_default_other(self):
        if False:
            for i in range(10):
                print('nop')
        b = [True]
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(110, 1), on_next(220, 2), on_next(330, 3), on_error(440, 'ex'))

        def action(scheduler, state):
            if False:
                i = 10
                return i + 15
            b[0] = False
        scheduler.schedule_absolute(150, action)

        def create():
            if False:
                return 10

            def condition():
                if False:
                    for i in range(10):
                        print('nop')
                return b[0]
            return reactivex.if_then(condition, xs)
        results = scheduler.start(create)
        assert results.messages == [on_completed(200)]
        assert xs.subscriptions == []