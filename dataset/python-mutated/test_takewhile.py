import unittest
from reactivex import operators as ops
from reactivex.testing import ReactiveTest, TestScheduler, is_prime
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TestTakeWhile(unittest.TestCase):

    def test_take_while_complete_Before(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_completed(330), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))
        invoked = 0

        def factory():
            if False:
                return 10

            def predicate(x):
                if False:
                    while True:
                        i = 10
                nonlocal invoked
                invoked += 1
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate))
        results = scheduler.start(factory)
        assert results.messages == [on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_completed(330)]
        assert xs.subscriptions == [subscribe(200, 330)]
        assert invoked == 4

    def test_take_while_complete_after(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))
        invoked = 0

        def factory():
            if False:
                while True:
                    i = 10

            def predicate(x):
                if False:
                    i = 10
                    return i + 15
                nonlocal invoked
                invoked += 1
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate))
        results = scheduler.start(factory)
        assert results.messages == [on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_completed(390)]
        assert xs.subscriptions == [subscribe(200, 390)]
        assert invoked == 6

    def test_take_while_error_before(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(210, 2), on_next(260, 5), on_error(270, ex), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23))
        invoked = 0

        def factory():
            if False:
                i = 10
                return i + 15

            def predicate(x):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal invoked
                invoked += 1
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate))
        results = scheduler.start(factory)
        assert results.messages == [on_next(210, 2), on_next(260, 5), on_error(270, ex)]
        assert xs.subscriptions == [subscribe(200, 270)]
        assert invoked == 2

    def test_take_while_error_after(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_error(600, 'ex'))
        invoked = 0

        def factory():
            if False:
                return 10

            def predicate(x):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal invoked
                invoked += 1
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate))
        results = scheduler.start(factory)
        assert results.messages == [on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_completed(390)]
        assert xs.subscriptions == [subscribe(200, 390)]
        assert invoked == 6

    def test_take_while_dispose_before(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))
        invoked = 0

        def create():
            if False:
                for i in range(10):
                    print('nop')

            def predicate(x):
                if False:
                    print('Hello World!')
                nonlocal invoked
                invoked += 1
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate))
        results = scheduler.start(create, disposed=300)
        assert results.messages == [on_next(210, 2), on_next(260, 5), on_next(290, 13)]
        assert xs.subscriptions == [subscribe(200, 300)]
        assert invoked == 3

    def test_take_while_dispose_after(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))
        invoked = 0

        def create():
            if False:
                return 10

            def predicate(x):
                if False:
                    print('Hello World!')
                nonlocal invoked
                invoked += 1
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate))
        results = scheduler.start(create, disposed=400)
        assert results.messages == [on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_completed(390)]
        assert xs.subscriptions == [subscribe(200, 390)]
        assert invoked == 6

    def test_take_while_zero(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(205, 100), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))
        invoked = 0

        def create():
            if False:
                for i in range(10):
                    print('nop')

            def predicate(x):
                if False:
                    print('Hello World!')
                nonlocal invoked
                invoked += 1
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate))
        results = scheduler.start(create, disposed=300)
        assert results.messages == [on_completed(205)]
        assert xs.subscriptions == [subscribe(200, 205)]
        assert invoked == 1

    def test_take_while_on_error(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))
        invoked = 0

        def factory():
            if False:
                while True:
                    i = 10

            def predicate(x):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal invoked
                invoked += 1
                if invoked == 3:
                    raise Exception(ex)
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate))
        results = scheduler.start(factory)
        assert results.messages == [on_next(210, 2), on_next(260, 5), on_error(290, ex)]
        assert xs.subscriptions == [subscribe(200, 290)]
        assert invoked == 3

    def test_take_while_index(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(205, 100), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))

        def factory():
            if False:
                print('Hello World!')
            return xs.pipe(ops.take_while_indexed(lambda x, i: i < 5))
        results = scheduler.start(factory)
        assert results.messages == [on_next(205, 100), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_completed(350)]
        assert xs.subscriptions == [subscribe(200, 350)]

    def test_take_while_index_inclusive_false(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(205, 100), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))

        def factory():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.take_while_indexed(lambda x, i: i < 5, inclusive=False))
        results = scheduler.start(factory)
        assert results.messages == [on_next(205, 100), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_completed(350)]
        assert xs.subscriptions == [subscribe(200, 350)]

    def test_take_while_index_inclusive_true(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(205, 100), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))

        def factory_inclusive():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.take_while_indexed(lambda x, i: i < 4, inclusive=True))
        results_inclusive = scheduler.start(factory_inclusive)
        assert results_inclusive.messages == [on_next(205, 100), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_completed(320)]
        assert xs.subscriptions == [subscribe(200, 320)]

    def test_take_while_complete_after_inclusive_true(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(90, -1), on_next(110, -1), on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_next(410, 17), on_next(450, 8), on_next(500, 23), on_completed(600))
        invoked = 0

        def factory():
            if False:
                for i in range(10):
                    print('nop')

            def predicate(x):
                if False:
                    print('Hello World!')
                nonlocal invoked
                invoked += 1
                return is_prime(x)
            return xs.pipe(ops.take_while(predicate, inclusive=True))
        results = scheduler.start(factory)
        assert results.messages == [on_next(210, 2), on_next(260, 5), on_next(290, 13), on_next(320, 3), on_next(350, 7), on_next(390, 4), on_completed(390)]
        assert xs.subscriptions == [subscribe(200, 390)]
        assert invoked == 6