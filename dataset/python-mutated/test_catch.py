import unittest
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

class TestCatch(unittest.TestCase):

    def test_catch_no_errors(self):
        if False:
            return 10
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_completed(230)]
        msgs2 = [on_next(240, 5), on_completed(250)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return o1.pipe(ops.catch(o2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_completed(230)]

    def test_catch_never(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs2 = [on_next(240, 5), on_completed(250)]
        o1 = reactivex.never()
        o2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                while True:
                    i = 10
            return o1.pipe(ops.catch(o2))
        results = scheduler.start(create)
        assert results.messages == []

    def test_catch_empty(self):
        if False:
            return 10
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(230)]
        msgs2 = [on_next(240, 5), on_completed(250)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return o1.pipe(ops.catch(o2))
        results = scheduler.start(create)
        assert results.messages == [on_completed(230)]

    def test_catch_return(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_completed(230)]
        msgs2 = [on_next(240, 5), on_completed(250)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                while True:
                    i = 10
            return o1.pipe(ops.catch(o2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_completed(230)]

    def test_catch_error(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_error(230, ex)]
        msgs2 = [on_next(240, 5), on_completed(250)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return o1.pipe(ops.catch(o2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_next(240, 5), on_completed(250)]

    def test_catch_error_never(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_error(230, ex)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = reactivex.never()

        def create():
            if False:
                while True:
                    i = 10
            return o1.pipe(ops.catch(o2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3)]

    def test_catch_error_error(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_error(230, 'ex1')]
        msgs2 = [on_next(240, 4), on_error(250, ex)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return o1.pipe(ops.catch(o2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_next(240, 4), on_error(250, ex)]

    def test_catch_multiple(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_error(215, ex)]
        msgs2 = [on_next(220, 3), on_error(225, ex)]
        msgs3 = [on_next(230, 4), on_completed(235)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)
        o3 = scheduler.create_hot_observable(msgs3)

        def create():
            if False:
                return 10
            return reactivex.catch(o1, o2, o3)
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_next(230, 4), on_completed(235)]

    def test_catch_error_specific_caught(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        handler_called = [False]
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_error(230, ex)]
        msgs2 = [on_next(240, 4), on_completed(250)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')

            def handler(e, source):
                if False:
                    while True:
                        i = 10
                handler_called[0] = True
                return o2
            return o1.pipe(ops.catch(handler))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_next(240, 4), on_completed(250)]
        assert handler_called[0]

    def test_catch_error_specific_caught_immediate(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        handler_called = [False]
        scheduler = TestScheduler()
        msgs2 = [on_next(240, 4), on_completed(250)]
        o2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')

            def handler(e, source):
                if False:
                    return 10
                handler_called[0] = True
                return o2
            return reactivex.throw('ex').pipe(ops.catch(handler))
        results = scheduler.start(create)
        assert results.messages == [on_next(240, 4), on_completed(250)]
        assert handler_called[0]

    def test_catch_handler_throws(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        ex2 = 'ex2'
        handler_called = [False]
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_error(230, ex)]
        o1 = scheduler.create_hot_observable(msgs1)

        def create():
            if False:
                return 10

            def handler(e, source):
                if False:
                    return 10
                handler_called[0] = True
                raise Exception(ex2)
            return o1.pipe(ops.catch(handler))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_error(230, ex2)]
        assert handler_called[0]

    def test_catch_nested_outer_catches(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        first_handler_called = [False]
        second_handler_called = [False]
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_error(215, ex)]
        msgs2 = [on_next(220, 3), on_completed(225)]
        msgs3 = [on_next(220, 4), on_completed(225)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)
        o3 = scheduler.create_hot_observable(msgs3)

        def create():
            if False:
                print('Hello World!')

            def handler1(e, source):
                if False:
                    for i in range(10):
                        print('nop')
                first_handler_called[0] = True
                return o2

            def handler2(e, source):
                if False:
                    i = 10
                    return i + 15
                second_handler_called[0] = True
                return o3
            return o1.pipe(ops.catch(handler1), ops.catch(handler2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_completed(225)]
        assert first_handler_called[0]
        assert not second_handler_called[0]

    def test_catch_throw_from_nested_catch(self):
        if False:
            return 10
        ex = 'ex'
        ex2 = 'ex'
        first_handler_called = [False]
        second_handler_called = [False]
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_error(215, ex)]
        msgs2 = [on_next(220, 3), on_error(225, ex2)]
        msgs3 = [on_next(230, 4), on_completed(235)]
        o1 = scheduler.create_hot_observable(msgs1)
        o2 = scheduler.create_hot_observable(msgs2)
        o3 = scheduler.create_hot_observable(msgs3)

        def create():
            if False:
                for i in range(10):
                    print('nop')

            def handler1(e, source):
                if False:
                    for i in range(10):
                        print('nop')
                first_handler_called[0] = True
                assert str(e) == ex
                return o2

            def handler2(e, source):
                if False:
                    while True:
                        i = 10
                second_handler_called[0] = True
                assert str(e) == ex2
                return o3
            return o1.pipe(ops.catch(handler1), ops.catch(handler2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_next(230, 4), on_completed(235)]
        assert first_handler_called[0]
        assert second_handler_called[0]