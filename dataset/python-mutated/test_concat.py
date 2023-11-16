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

def _raise(error: str):
    if False:
        i = 10
        return i + 15
    raise RxException(error)

class TestConcat(unittest.TestCase):

    def test_concat_empty_empty(self):
        if False:
            return 10
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(230)]
        msgs2 = [on_next(150, 1), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_completed(250)]

    def test_concat_empty_never(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(230)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = reactivex.never()

        def create():
            if False:
                while True:
                    i = 10
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == []

    def test_concat_never_empty(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(230)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = reactivex.never()

        def create():
            if False:
                print('Hello World!')
            return e2.pipe(ops.concat(e1))
        results = scheduler.start(create)
        assert results.messages == []

    def test_concat_never_never(self):
        if False:
            return 10
        scheduler = TestScheduler()
        e1 = reactivex.never()
        e2 = reactivex.never()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == []

    def test_concat_empty_on_error(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(230)]
        msgs2 = [on_next(150, 1), on_error(250, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_error(250, ex)]

    def test_concat_throw_empty(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_error(230, ex)]
        msgs2 = [on_next(150, 1), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_error(230, ex)]

    def test_concat_throw_on_error(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_error(230, ex)]
        msgs2 = [on_next(150, 1), on_error(250, 'ex2')]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_error(230, ex)]

    def test_concat_return_empty(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                while True:
                    i = 10
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_completed(250)]

    def test_concat_empty_return(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_completed(230)]
        msgs2 = [on_next(150, 1), on_next(240, 2), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_next(240, 2), on_completed(250)]

    def test_concat_return_never(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_completed(230)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = reactivex.never()

        def create():
            if False:
                i = 10
                return i + 15
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2)]

    def test_concat_never_return(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_completed(230)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = reactivex.never()

        def create():
            if False:
                i = 10
                return i + 15
            return e2.pipe(ops.concat(e1))
        results = scheduler.start(create)
        assert results.messages == []

    def test_concat_return_return(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(220, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_next(240, 3), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                i = 10
                return i + 15
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_next(220, 2), on_next(240, 3), on_completed(250)]

    def test_concat_throw_return(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_error(230, ex)]
        msgs2 = [on_next(150, 1), on_next(240, 2), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                print('Hello World!')
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_error(230, ex)]

    def test_concat_return_on_error(self):
        if False:
            while True:
                i = 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(220, 2), on_completed(230)]
        msgs2 = [on_next(150, 1), on_error(250, ex)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                while True:
                    i = 10
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_next(220, 2), on_error(250, ex)]

    def test_concat_some_data_some_data(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        msgs1 = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_completed(225)]
        msgs2 = [on_next(150, 1), on_next(230, 4), on_next(240, 5), on_completed(250)]
        e1 = scheduler.create_hot_observable(msgs1)
        e2 = scheduler.create_hot_observable(msgs2)

        def create():
            if False:
                while True:
                    i = 10
            return e1.pipe(ops.concat(e2))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]

    def test_concat_forward_scheduler(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        subscribe_schedulers = {'e1': 'unknown', 'e2': 'unknown'}

        def subscribe_e1(observer, scheduler='not_set'):
            if False:
                i = 10
                return i + 15
            subscribe_schedulers['e1'] = scheduler
            observer.on_completed()

        def subscribe_e2(observer, scheduler='not_set'):
            if False:
                print('Hello World!')
            subscribe_schedulers['e2'] = scheduler
            observer.on_completed()
        e1 = reactivex.create(subscribe_e1)
        e2 = reactivex.create(subscribe_e2)
        stream = e1.pipe(ops.concat(e2))
        stream.subscribe(scheduler=scheduler)
        scheduler.advance_to(1000)
        assert subscribe_schedulers['e1'] is scheduler
        assert subscribe_schedulers['e2'] is scheduler

    def test_concat_forward_none_scheduler(self):
        if False:
            return 10
        subscribe_schedulers = {'e1': 'unknown', 'e2': 'unknown'}

        def subscribe_e1(observer, scheduler='not_set'):
            if False:
                print('Hello World!')
            subscribe_schedulers['e1'] = scheduler
            observer.on_completed()

        def subscribe_e2(observer, scheduler='not_set'):
            if False:
                i = 10
                return i + 15
            subscribe_schedulers['e2'] = scheduler
            observer.on_completed()
        e1 = reactivex.create(subscribe_e1)
        e2 = reactivex.create(subscribe_e2)
        stream = e1.pipe(ops.concat(e2))
        stream.subscribe()
        assert subscribe_schedulers['e1'] is None
        assert subscribe_schedulers['e2'] is None