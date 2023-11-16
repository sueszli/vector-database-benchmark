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

class TestSome(unittest.TestCase):

    def test_some_empty(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.some())
        res = scheduler.start(create=create).messages
        assert res == [on_next(250, False), on_completed(250)]

    def test_some_return(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_next(210, 2), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.some())
        res = scheduler.start(create=create).messages
        assert res == [on_next(210, True), on_completed(210)]

    def test_some_on_error(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_error(210, ex)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.some())
        res = scheduler.start(create=create).messages
        assert res == [on_error(210, ex)]

    def test_some_never(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs = [on_next(150, 1)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.some())
        res = scheduler.start(create=create).messages
        assert res == []

    def test_some_predicate_empty(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.some(lambda x: x > 0))
        res = scheduler.start(create=create).messages
        assert res == [on_next(250, False), on_completed(250)]

    def test_some_predicate_return(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_next(210, 2), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                return 10
            return xs.pipe(ops.some(lambda x: x > 0))
        res = scheduler.start(create=create).messages
        assert res == [on_next(210, True), on_completed(210)]

    def test_some_predicate_return_not_match(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_next(210, -2), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.some(lambda x: x > 0))
        res = scheduler.start(create=create).messages
        assert res == [on_next(250, False), on_completed(250)]

    def test_some_predicate_some_none_match(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_next(210, -2), on_next(220, -3), on_next(230, -4), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.some(lambda x: x > 0))
        res = scheduler.start(create=create).messages
        assert res == [on_next(250, False), on_completed(250)]

    def test_some_predicate_some_match(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_next(210, -2), on_next(220, 3), on_next(230, -4), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.some(lambda x: x > 0))
        res = scheduler.start(create=create).messages
        assert res == [on_next(220, True), on_completed(220)]

    def test_some_predicate_on_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_error(210, ex)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.some(lambda x: x > 0))
        res = scheduler.start(create=create).messages
        assert res == [on_error(210, ex)]

    def test_some_predicate_never(self):
        if False:
            return 10
        scheduler = TestScheduler()
        msgs = [on_next(150, 1)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                return 10
            return xs.pipe(ops.some(lambda x: x > 0))
        res = scheduler.start(create=create).messages
        assert res == []