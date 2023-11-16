import unittest
import reactivex
from reactivex import Observable
from reactivex import operators as ops
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TestSkipUntil(unittest.TestCase):

    def test_skip_until_somedata_next(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        l_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        r_msgs = [on_next(150, 1), on_next(225, 99), on_completed(230)]
        l = scheduler.create_hot_observable(l_msgs)
        r = scheduler.create_hot_observable(r_msgs)

        def create():
            if False:
                print('Hello World!')
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == [on_next(230, 4), on_next(240, 5), on_completed(250)]

    def test_skip_until_somedata_error(self):
        if False:
            return 10
        scheduler = TestScheduler()
        ex = 'ex'
        l_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        r_msgs = [on_next(150, 1), on_error(225, ex)]
        l = scheduler.create_hot_observable(l_msgs)
        r = scheduler.create_hot_observable(r_msgs)

        def create():
            if False:
                while True:
                    i = 10
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == [on_error(225, ex)]

    def test_skip_until_somedata_empty(self):
        if False:
            return 10
        scheduler = TestScheduler()
        l_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        r_msgs = [on_next(150, 1), on_completed(225)]
        l = scheduler.create_hot_observable(l_msgs)
        r = scheduler.create_hot_observable(r_msgs)

        def create():
            if False:
                print('Hello World!')
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == []

    def test_skip_until_never_next(self):
        if False:
            return 10
        scheduler = TestScheduler()
        r_msgs = [on_next(150, 1), on_next(225, 2), on_completed(250)]
        l = reactivex.never()
        r = scheduler.create_hot_observable(r_msgs)

        def create():
            if False:
                while True:
                    i = 10
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == []

    def test_skip_until_never_error(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        r_msgs = [on_next(150, 1), on_error(225, ex)]
        l = reactivex.never()
        r = scheduler.create_hot_observable(r_msgs)

        def create():
            if False:
                print('Hello World!')
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == [on_error(225, ex)]

    def test_skip_until_somedata_never(self):
        if False:
            return 10
        scheduler = TestScheduler()
        l_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        l = scheduler.create_hot_observable(l_msgs)
        r = reactivex.never()

        def create():
            if False:
                while True:
                    i = 10
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == []

    def test_skip_until_never_empty(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        r_msgs = [on_next(150, 1), on_completed(225)]
        l = reactivex.never()
        r = scheduler.create_hot_observable(r_msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == []

    def test_skip_until_never_never(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        l = reactivex.never()
        r = reactivex.never()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == []

    def test_skip_until_has_completed_causes_disposal(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        l_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        disposed = [False]
        l = scheduler.create_hot_observable(l_msgs)

        def subscribe(observer, scheduler=None):
            if False:
                return 10
            disposed[0] = True
        r = Observable(subscribe)

        def create():
            if False:
                while True:
                    i = 10
            return l.pipe(ops.skip_until(r))
        results = scheduler.start(create)
        assert results.messages == []
        assert disposed[0]