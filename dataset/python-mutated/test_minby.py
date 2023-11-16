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

class RxException(Exception):
    pass

def _raise(ex):
    if False:
        for i in range(10):
            print('nop')
    raise RxException(ex)

class TestMinBy(unittest.TestCase):

    def test_min_by_empty(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, {'key': 1, 'value': 'z'}), on_completed(250))

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.min_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        assert 2 == len(res)
        assert 0 == len(res[0].value.value)
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_min_by_return(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 2, 'value': 'a'}), on_completed(250))

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.min_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        assert res[0].value.kind == 'N'
        self.assertEqual(1, len(res[0].value.value))
        self.assertEqual(2, res[0].value.value[0]['key'])
        self.assertEqual('a', res[0].value.value[0]['value'])
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_min_by_some(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 3, 'value': 'b'}), on_next(220, {'key': 2, 'value': 'c'}), on_next(230, {'key': 4, 'value': 'a'}), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.min_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        assert res[0].value.kind == 'N'
        self.assertEqual(1, len(res[0].value.value))
        self.assertEqual(2, res[0].value.value[0]['key'])
        self.assertEqual('c', res[0].value.value[0]['value'])
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_min_by_multiple(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 3, 'value': 'b'}), on_next(215, {'key': 2, 'value': 'd'}), on_next(220, {'key': 3, 'value': 'c'}), on_next(225, {'key': 2, 'value': 'y'}), on_next(230, {'key': 4, 'value': 'a'}), on_next(235, {'key': 4, 'value': 'r'}), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                return 10
            return xs.pipe(ops.min_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        assert res[0].value.kind == 'N'
        self.assertEqual(2, len(res[0].value.value))
        self.assertEqual(2, res[0].value.value[0]['key'])
        self.assertEqual('d', res[0].value.value[0]['value'])
        self.assertEqual(2, res[0].value.value[1]['key'])
        self.assertEqual('y', res[0].value.value[1]['value'])
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_min_by_on_error(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_error(210, ex)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.min_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        assert res == [on_error(210, ex)]

    def test_min_by_never(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'})]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                return 10
            return xs.pipe(ops.min_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        assert res == []

    def test_min_by_comparer_empty(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_completed(250)]

        def reverse_comparer(a, b):
            if False:
                return 10
            if a > b:
                return -1
            if a == b:
                return 0
            return 1
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.min_by(lambda x: x['key'], reverse_comparer))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        self.assertEqual(0, len(res[0].value.value))
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_min_by_comparer_return(self):
        if False:
            return 10
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 2, 'value': 'a'}), on_completed(250)]

        def reverse_comparer(a, b):
            if False:
                for i in range(10):
                    print('nop')
            if a > b:
                return -1
            if a == b:
                return 0
            return 1
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.min_by(lambda x: x['key'], reverse_comparer))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        assert res[0].value.kind == 'N'
        self.assertEqual(1, len(res[0].value.value))
        self.assertEqual(2, res[0].value.value[0]['key'])
        self.assertEqual('a', res[0].value.value[0]['value'])
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_min_by_comparer_some(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 3, 'value': 'b'}), on_next(220, {'key': 20, 'value': 'c'}), on_next(230, {'key': 4, 'value': 'a'}), on_completed(250)]

        def reverse_comparer(a, b):
            if False:
                i = 10
                return i + 15
            if a > b:
                return -1
            if a == b:
                return 0
            return 1
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.min_by(lambda x: x['key'], reverse_comparer))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        assert res[0].value.kind == 'N'
        self.assertEqual(1, len(res[0].value.value))
        self.assertEqual(20, res[0].value.value[0]['key'])
        self.assertEqual('c', res[0].value.value[0]['value'])
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_min_by_comparer_on_error(self):
        if False:
            for i in range(10):
                print('nop')
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_error(210, ex)]

        def reverse_comparer(a, b):
            if False:
                while True:
                    i = 10
            if a > b:
                return -1
            if a == b:
                return 0
            return 1

        def create():
            if False:
                return 10
            return xs.pipe(ops.min_by(lambda x: x['key'], reverse_comparer))
        xs = scheduler.create_hot_observable(msgs)
        res = scheduler.start(create=create).messages
        assert res == [on_error(210, ex)]

    def test_min_by_comparer_never(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'})]

        def reverse_comparer(a, b):
            if False:
                for i in range(10):
                    print('nop')
            if a > b:
                return -1
            if a == b:
                return 0
            return 1
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                i = 10
                return i + 15
            return xs.pipe(ops.min_by(lambda x: x['key'], reverse_comparer))
        res = scheduler.start(create=create).messages
        assert res == []

    def test_min_by_mapper_throws(self):
        if False:
            i = 10
            return i + 15
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 3, 'value': 'b'}), on_next(220, {'key': 2, 'value': 'c'}), on_next(230, {'key': 4, 'value': 'a'}), on_completed(250)]

        def reverse_comparer(a, b):
            if False:
                i = 10
                return i + 15
            if a > b:
                return -1
            if a == b:
                return 0
            return 1
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.min_by(lambda x: _raise(ex), reverse_comparer))
        res = scheduler.start(create=create).messages
        assert res == [on_error(210, ex)]

    def test_min_by_comparer_throws(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 3, 'value': 'b'}), on_next(220, {'key': 2, 'value': 'c'}), on_next(230, {'key': 4, 'value': 'a'}), on_completed(250)]

        def reverse_comparer(a, b):
            if False:
                return 10
            _raise(ex)
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                print('Hello World!')
            return xs.pipe(ops.min_by(lambda x: x['key'], reverse_comparer))
        res = scheduler.start(create=create).messages
        assert res == [on_error(220, ex)]