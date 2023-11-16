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

class RxException(Exception):
    pass

def _raise(ex):
    if False:
        i = 10
        return i + 15
    raise RxException(ex)

class TestCombineLatest(unittest.TestCase):

    def test_combine_latest_never_never(self):
        if False:
            return 10
        scheduler = TestScheduler()
        e1 = reactivex.never()
        e2 = reactivex.never()

        def create():
            if False:
                print('Hello World!')
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == []

    def test_combine_latest_never_empty(self):
        if False:
            return 10
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_completed(210)]
        e1 = reactivex.never()
        e2 = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == []

    def test_combine_latest_empty_never(self):
        if False:
            return 10
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_completed(210)]
        e1 = reactivex.never()
        e2 = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == []

    def test_combine_latest_empty_empty(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(210)]
        msgs2 = [on_next(150, 1), on_completed(210)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_completed(210)]

    def test_combine_latest_empty_return(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(210)]
        msgs2 = [on_next(150, 1), on_next(215, 2), on_completed(220)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_completed(215)]

    def test_combine_latest_return_empty(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(210)]
        msgs2 = [on_next(150, 1), on_next(215, 2), on_completed(220)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_completed(215)]

    def test_combine_latest_never_feturn(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_next(215, 2), on_completed(220)]
        e1 = scheduler.create_hot_observable(msgs)
        e2 = reactivex.never()

        def create():
            if False:
                print('Hello World!')
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == []

    def test_combine_latest_return_never(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_next(215, 2), on_completed(210)]
        e1 = scheduler.create_hot_observable(msgs)
        e2 = reactivex.never()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == []

    def test_combine_latest_return_return(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_next(220, 3), on_completed(240)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_next(220, 2 + 3), on_completed(240)]

    def test_combine_latest_empty_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(230)]
        msgs2 = [on_next(150, 1), on_error(220, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]

    def test_combine_latest_error_empty(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(230)]
        msgs2 = [on_next(150, 1), on_error(220, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]

    def test_combine_latest_return_on_error(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_error(220, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]

    def test_combine_latest_throw_return(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_error(220, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]

    def test_combine_latest_throw_on_error(self):
        if False:
            for i in range(10):
                print('nop')
        ex1 = 'ex1'
        ex2 = 'ex2'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_error(220, ex1)]
        msgs2 = [on_next(150, 1), on_error(230, ex2)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex1)]

    def test_combine_latest_error_on_error(self):
        if False:
            while True:
                i = 10
        ex1 = 'ex1'
        ex2 = 'ex2'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_error(220, ex1)]
        msgs2 = [on_next(150, 1), on_error(230, ex2)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                while True:
                    i = 10
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex1)]

    def test_combine_latest_throw_error(self):
        if False:
            return 10
        ex1 = 'ex1'
        ex2 = 'ex2'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_error(220, ex1)]
        msgs2 = [on_next(150, 1), on_error(230, ex2)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex1)]

    def test_combine_latest_never_on_error(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_error(220, ex)]
        e1 = reactivex.never()
        e2 = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]

    def test_combine_latest_throw_never(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, 1), on_error(220, ex)]
        e1 = reactivex.never()
        e2 = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]

    def test_combine_latest_some_on_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_error(220, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]

    def test_combine_latest_throw_some(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_error(220, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                while True:
                    i = 10
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]

    def test_combine_latest_throw_after_complete_left(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_completed(220)]
        msgs2 = [on_next(150, 1), on_error(230, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(230, ex)]

    def test_combine_latest_throw_after_complete_right(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_completed(220)]
        msgs2 = [on_next(150, 1), on_error(230, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                while True:
                    i = 10
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(230, ex)]

    def test_combine_latest_interleaved_with_tail(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_next(225, 4), on_completed(230)]
        msgs2 = [on_next(150, 1), on_next(220, 3), on_next(230, 5), on_next(235, 6), on_next(240, 7), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_next(220, 2 + 3), on_next(225, 3 + 4), on_next(230, 4 + 5), on_next(235, 4 + 6), on_next(240, 4 + 7), on_completed(250)]

    def test_combine_latest_consecutive(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_next(225, 4), on_completed(230)]
        msgs2 = [on_next(150, 1), on_next(235, 6), on_next(240, 7), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_next(235, 4 + 6), on_next(240, 4 + 7), on_completed(250)]

    def test_combine_latest_consecutive_end_with_error_left(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_next(225, 4), on_error(230, ex)]
        msgs2 = [on_next(150, 1), on_next(235, 6), on_next(240, 7), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return e1.pipe(ops.combine_latest(e2), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_error(230, ex)]

    def test_combine_latest_consecutive_end_with_error_right(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_next(225, 4), on_completed(230)]
        msgs2 = [on_next(150, 1), on_next(235, 6), on_next(240, 7), on_error(245, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')
            return e2.pipe(ops.combine_latest(e1), ops.map(sum))
        results = scheduler.start(create)
        assert results.messages == [on_next(235, 4 + 6), on_next(240, 4 + 7), on_error(245, ex)]

    def test_combine_latest_mapper_throws(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(215, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_next(220, 3), on_completed(240)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                return 10
            return e1.pipe(ops.combine_latest(e2), ops.map(lambda xy: _raise(ex)))
        results = scheduler.start(create)
        assert results.messages == [on_error(220, ex)]
if __name__ == '__main__':
    unittest.main()