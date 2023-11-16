import unittest
from reactivex import interval
from reactivex import operators as ops
from reactivex.testing import ReactiveTest, TestScheduler
from reactivex.testing.marbles import marbles_testing
from reactivex.testing.subscription import Subscription
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TestSwitchMapIndex(unittest.TestCase):

    def test_switch_map_indexed_uses_index(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(300, 'a'), on_next(400, 'b'), on_next(500, 'c'))

        def create_inner(x: str, i: int):
            if False:
                for i in range(10):
                    print('nop')

            def create_changing(j: int):
                if False:
                    i = 10
                    return i + 15
                return (i, j, x)
            return interval(20).pipe(ops.map(create_changing))

        def create():
            if False:
                return 10
            return xs.pipe(ops.switch_map_indexed(project=create_inner))
        results = scheduler.start(create, disposed=580)
        assert results.messages == [on_next(320, (0, 0, 'a')), on_next(340, (0, 1, 'a')), on_next(360, (0, 2, 'a')), on_next(380, (0, 3, 'a')), on_next(420, (1, 0, 'b')), on_next(440, (1, 1, 'b')), on_next(460, (1, 2, 'b')), on_next(480, (1, 3, 'b')), on_next(520, (2, 0, 'c')), on_next(540, (2, 1, 'c')), on_next(560, (2, 2, 'c'))]
        assert xs.subscriptions == [Subscription(200, 580)]

    def test_switch_map_indexed_inner_throws(self):
        if False:
            while True:
                i = 10
        'Inner throwing causes outer to throw'
        ex = 'ex'
        scheduler = TestScheduler()
        sources = [scheduler.create_cold_observable(on_next(100, 'a'), on_next(300, 'aa')), scheduler.create_cold_observable(on_next(50, 'b'), on_error(120, ex)), scheduler.create_cold_observable(on_next(50, 'wont happen'), on_error(120, 'no'))]
        xs = scheduler.create_hot_observable(on_next(250, 0), on_next(400, 1), on_next(550, 2))

        def create_inner(x: int, _i: int):
            if False:
                for i in range(10):
                    print('nop')
            return sources[x]

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.switch_map_indexed(create_inner))
        results = scheduler.start(create)
        assert results.messages == [on_next(350, 'a'), on_next(450, 'b'), on_error(520, ex)]
        assert sources[0].subscriptions == [Subscription(250, 400)]
        assert sources[1].subscriptions == [Subscription(400, 520)]
        assert sources[2].subscriptions == []

    def test_switch_map_indexed_outer_throws(self):
        if False:
            i = 10
            return i + 15
        'Outer throwing unsubscribes from all'
        ex = 'ABC'
        scheduler = TestScheduler()
        sources = [scheduler.create_cold_observable(on_next(100, 'a'), on_next(300, 'aa')), scheduler.create_cold_observable(on_next(50, 'b'), on_error(120, ex)), scheduler.create_cold_observable(on_next(50, 'wont happen'), on_error(120, 'no'))]
        xs = scheduler.create_hot_observable(on_next(250, 0), on_next(400, 1), on_error(430, ex))

        def create_inner(x: int, _i: int):
            if False:
                print('Hello World!')
            return sources[x]

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.switch_map_indexed(create_inner))
        results = scheduler.start(create)
        assert results.messages == [on_next(350, 'a'), on_error(430, ex)]
        assert sources[0].subscriptions == [Subscription(250, 400)]
        assert sources[1].subscriptions == [Subscription(400, 430)]
        assert sources[2].subscriptions == []

    def test_switch_map_indexed_no_inner(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_completed(500))
        sources = [scheduler.create_cold_observable(on_next(20, 2))]

        def create_inner(_x: int, i: int):
            if False:
                print('Hello World!')
            return sources[i]

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.switch_map_indexed(create_inner))
        results = scheduler.start(create)
        assert results.messages == [on_completed(500)]
        assert xs.subscriptions == [Subscription(200, 500)]
        assert sources[0].subscriptions == []

    def test_switch_map_indexed_inner_completes(self):
        if False:
            while True:
                i = 10
        'Inner completions do not affect outer'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(300, 'd'), on_next(330, 'f'), on_completed(540))

        def create_inner(x: str, i: int):
            if False:
                print('Hello World!')
            'An observable which will complete after 40 ticks'
            return interval(20).pipe(ops.map(lambda j: (i, j, x)), ops.take(2))

        def create():
            if False:
                return 10
            return xs.pipe(ops.switch_map_indexed(create_inner))
        results = scheduler.start(create)
        assert results.messages == [on_next(320, (0, 0, 'd')), on_next(350, (1, 0, 'f')), on_next(370, (1, 1, 'f')), on_completed(540)]

    def test_switch_map_default_mapper(self):
        if False:
            print('Hello World!')
        with marbles_testing(timespan=10) as (start, cold, hot, exp):
            xs = hot('               ---a---b------c-----', {'a': cold('    --1--2', None, None), 'b': cold('        --1-2-3-4-5|', None, None), 'c': cold('               --1--2', None, None)}, None)
            expected = exp('    -----1---1-2-3--1--2', None, None)
            result = start(xs.pipe(ops.switch_map_indexed()))
            assert result == expected