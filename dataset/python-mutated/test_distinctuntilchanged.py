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
        while True:
            i = 10
    raise RxException(ex)

class TestDistinctUntilChanged(unittest.TestCase):

    def test_distinct_until_changed_never(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return reactivex.never().pipe(ops.distinct_until_changed())
        results = scheduler.start(create)
        assert results.messages == []

    def test_distinct_until_changed_empty(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_completed(250))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.distinct_until_changed())
        results = scheduler.start(create).messages
        self.assertEqual(1, len(results))
        assert results[0].value.kind == 'C' and results[0].time == 250

    def test_distinct_until_changed_return(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(220, 2), on_completed(250))

        def create():
            if False:
                return 10
            return xs.pipe(ops.distinct_until_changed())
        results = scheduler.start(create).messages
        self.assertEqual(2, len(results))
        assert results[0].value.kind == 'N' and results[0].time == 220 and (results[0].value.value == 2)
        assert results[1].value.kind == 'C' and results[1].time == 250

    def test_distinct_until_changed_on_error(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_error(250, ex))

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.distinct_until_changed())
        results = scheduler.start(create).messages
        self.assertEqual(1, len(results))
        assert results[0].value.kind == 'E' and results[0].time == 250 and (str(results[0].value.exception) == ex)

    def test_distinct_until_changed_all_changes(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.distinct_until_changed())
        results = scheduler.start(create).messages
        self.assertEqual(5, len(results))
        assert results[0].value.kind == 'N' and results[0].time == 210 and (results[0].value.value == 2)
        assert results[1].value.kind == 'N' and results[1].time == 220 and (results[1].value.value == 3)
        assert results[2].value.kind == 'N' and results[2].time == 230 and (results[2].value.value == 4)
        assert results[3].value.kind == 'N' and results[3].time == 240 and (results[3].value.value == 5)
        assert results[4].value.kind == 'C' and results[4].time == 250

    def test_distinct_until_changed_all_same(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 2), on_next(230, 2), on_next(240, 2), on_completed(250))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.distinct_until_changed())
        results = scheduler.start(create).messages
        self.assertEqual(2, len(results))
        assert results[0].value.kind == 'N' and results[0].time == 210 and (results[0].value.value == 2)
        assert results[1].value.kind == 'C' and results[1].time == 250

    def test_distinct_until_changed_some_changes(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(215, 3), on_next(220, 3), on_next(225, 2), on_next(230, 2), on_next(230, 1), on_next(240, 2), on_completed(250))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.distinct_until_changed())
        results = scheduler.start(create).messages
        self.assertEqual(6, len(results))
        assert results[0].value.kind == 'N' and results[0].time == 210 and (results[0].value.value == 2)
        assert results[1].value.kind == 'N' and results[1].time == 215 and (results[1].value.value == 3)
        assert results[2].value.kind == 'N' and results[2].time == 225 and (results[2].value.value == 2)
        assert results[3].value.kind == 'N' and results[3].time == 230 and (results[3].value.value == 1)
        assert results[4].value.kind == 'N' and results[4].time == 240 and (results[4].value.value == 2)
        assert results[5].value.kind == 'C' and results[5].time == 250

    def test_distinct_until_changed_comparer_all_equal(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.distinct_until_changed(comparer=lambda x, y: True))
        results = scheduler.start(create).messages
        self.assertEqual(2, len(results))
        assert results[0].value.kind == 'N' and results[0].time == 210 and (results[0].value.value == 2)
        assert results[1].value.kind == 'C' and results[1].time == 250

    def test_distinct_until_changed_comparer_all_different(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 2), on_next(230, 2), on_next(240, 2), on_completed(250))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.distinct_until_changed(comparer=lambda x, y: False))
        results = scheduler.start(create).messages
        self.assertEqual(5, len(results))
        assert results[0].value.kind == 'N' and results[0].time == 210 and (results[0].value.value == 2)
        assert results[1].value.kind == 'N' and results[1].time == 220 and (results[1].value.value == 2)
        assert results[2].value.kind == 'N' and results[2].time == 230 and (results[2].value.value == 2)
        assert results[3].value.kind == 'N' and results[3].time == 240 and (results[3].value.value == 2)
        assert results[4].value.kind == 'C' and results[4].time == 250

    def test_distinct_until_changed_key_mapper_div2(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 4), on_next(230, 3), on_next(240, 5), on_completed(250))

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.distinct_until_changed(lambda x: x % 2))
        results = scheduler.start(create).messages
        self.assertEqual(3, len(results))
        assert results[0].value.kind == 'N' and results[0].time == 210 and (results[0].value.value == 2)
        assert results[1].value.kind == 'N' and results[1].time == 230 and (results[1].value.value == 3)
        assert results[2].value.kind == 'C' and results[2].time == 250

    def test_distinct_until_changed_key_mapper_throws(self):
        if False:
            print('Hello World!')
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_completed(250))

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.distinct_until_changed(lambda x: _raise(ex)))
        results = scheduler.start(create)
        assert results.messages == [on_error(210, ex)]

    def test_distinct_until_changed_comparer_throws(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_completed(250))

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.distinct_until_changed(comparer=lambda x, y: _raise(ex)))
        results = scheduler.start(create)
        assert results.messages == [on_next(210, 2), on_error(220, ex)]