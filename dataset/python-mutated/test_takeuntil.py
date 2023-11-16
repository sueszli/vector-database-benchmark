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

class TestTakeUntil(unittest.TestCase):

    def test_take_until_preempt_somedata_next(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        left_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        right_msgs = [on_next(150, 1), on_next(225, 99), on_completed(230)]
        left = scheduler.create_hot_observable(left_msgs)
        right = scheduler.create_hot_observable(right_msgs)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_completed(225)]

    def test_take_until_preempt_somedata_error(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        left_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        right_msgs = [on_next(150, 1), on_error(225, ex)]
        left = scheduler.create_hot_observable(left_msgs)
        right = scheduler.create_hot_observable(right_msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_error(225, ex)]

    def test_take_until_nopreempt_somedata_empty(self):
        if False:
            return 10
        scheduler = TestScheduler()
        left_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        right_msgs = [on_next(150, 1), on_completed(225)]
        left = scheduler.create_hot_observable(left_msgs)
        right = scheduler.create_hot_observable(right_msgs)

        def create():
            if False:
                print('Hello World!')
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]

    def test_take_until_nopreempt_somedata_never(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        left_msgs = [on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]
        left = scheduler.create_hot_observable(left_msgs)
        right = reactivex.never()

        def create():
            if False:
                print('Hello World!')
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250)]

    def test_take_until_preempt_never_next(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        right_msgs = [on_next(150, 1), on_next(225, 2), on_completed(250)]
        left = reactivex.never()
        right = scheduler.create_hot_observable(right_msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == [on_completed(225)]

    def test_take_until_preempt_never_error(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()
        right_msgs = [on_next(150, 1), on_error(225, ex)]
        left = reactivex.never()
        right = scheduler.create_hot_observable(right_msgs)

        def create():
            if False:
                print('Hello World!')
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == [on_error(225, ex)]

    def test_take_until_nopreempt_never_empty(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        right_msgs = [on_next(150, 1), on_completed(225)]
        left = reactivex.never()
        right = scheduler.create_hot_observable(right_msgs)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == []

    def test_take_until_nopreempt_never_never(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        left = reactivex.never()
        right = reactivex.never()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == []

    def test_take_until_preempt_beforefirstproduced(self):
        if False:
            return 10
        scheduler = TestScheduler()
        left_msgs = [on_next(150, 1), on_next(230, 2), on_completed(240)]
        right_msgs = [on_next(150, 1), on_next(210, 2), on_completed(220)]
        l = scheduler.create_hot_observable(left_msgs)
        r = scheduler.create_hot_observable(right_msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return l.pipe(ops.take_until(r))
        results = scheduler.start(create)
        assert results.messages == [on_completed(210)]

    def test_take_until_preempt_beforefirstproduced_remain_silent_and_proper_disposed(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        left_msgs = [on_next(150, 1), on_error(215, 'ex'), on_completed(240)]
        right_msgs = [on_next(150, 1), on_next(210, 2), on_completed(220)]
        source_not_disposed = [False]

        def action():
            if False:
                for i in range(10):
                    print('nop')
            source_not_disposed[0] = True
        left = scheduler.create_hot_observable(left_msgs).pipe(ops.do_action(on_next=action))
        right = scheduler.create_hot_observable(right_msgs)

        def create():
            if False:
                while True:
                    i = 10
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == [on_completed(210)]
        assert not source_not_disposed[0]

    def test_take_until_nopreempt_afterlastproduced_proper_disposed_signal(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        left_msgs = [on_next(150, 1), on_next(230, 2), on_completed(240)]
        right_msgs = [on_next(150, 1), on_next(250, 2), on_completed(260)]
        signal_not_disposed = [False]
        left = scheduler.create_hot_observable(left_msgs)

        def action():
            if False:
                i = 10
                return i + 15
            signal_not_disposed[0] = True
        right = scheduler.create_hot_observable(right_msgs).pipe(ops.do_action(on_next=action))

        def create():
            if False:
                return 10
            return left.pipe(ops.take_until(right))
        results = scheduler.start(create)
        assert results.messages == [on_next(230, 2), on_completed(240)]
        assert not signal_not_disposed[0]