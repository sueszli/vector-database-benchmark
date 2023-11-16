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

class TestMaxBy(unittest.TestCase):

    def test_maxby_empty(self):
        if False:
            while True:
                i = 10
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10

            def mapper(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x['key']
            return xs.pipe(ops.max_by(mapper))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        self.assertEqual(0, len(res[0].value.value))
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_maxby_return(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 2, 'value': 'a'}), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                print('Hello World!')

            def mapper(x):
                if False:
                    i = 10
                    return i + 15
                return x['key']
            return xs.pipe(ops.max_by(mapper))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        assert res[0].value.kind == 'N'
        self.assertEqual(1, len(res[0].value.value))
        self.assertEqual(2, res[0].value.value[0]['key'])
        self.assertEqual('a', res[0].value.value[0]['value'])
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_maxby_some(self):
        if False:
            return 10
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 3, 'value': 'b'}), on_next(220, {'key': 4, 'value': 'c'}), on_next(230, {'key': 2, 'value': 'a'}), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                i = 10
                return i + 15

            def mapper(x):
                if False:
                    i = 10
                    return i + 15
                return x['key']
            return xs.pipe(ops.max_by(mapper))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        assert res[0].value.kind == 'N'
        self.assertEqual(1, len(res[0].value.value[0]['value']))
        self.assertEqual(4, res[0].value.value[0]['key'])
        self.assertEqual('c', res[0].value.value[0]['value'])
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_maxby_multiple(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_next(210, {'key': 3, 'value': 'b'}), on_next(215, {'key': 2, 'value': 'd'}), on_next(220, {'key': 3, 'value': 'c'}), on_next(225, {'key': 2, 'value': 'y'}), on_next(230, {'key': 4, 'value': 'a'}), on_next(235, {'key': 4, 'value': 'r'}), on_completed(250)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.max_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        self.assertEqual(2, len(res))
        assert res[0].value.kind == 'N'
        self.assertEqual(2, len(res[0].value.value))
        self.assertEqual(4, res[0].value.value[0]['key'])
        self.assertEqual('a', res[0].value.value[0]['value'])
        self.assertEqual(4, res[0].value.value[1]['key'])
        self.assertEqual('r', res[0].value.value[1]['value'])
        assert res[1].value.kind == 'C' and res[1].time == 250

    def test_maxby_on_error(self):
        if False:
            return 10
        ex = 'ex'
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'}), on_error(210, ex)]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                while True:
                    i = 10
            return xs.pipe(ops.max_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        assert res == [on_error(210, ex)]

    def test_maxby_never(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        msgs = [on_next(150, {'key': 1, 'value': 'z'})]
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                for i in range(10):
                    print('nop')
            return xs.pipe(ops.max_by(lambda x: x['key']))
        res = scheduler.start(create=create).messages
        assert res == []

    def test_maxby_comparerthrows(self):
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
            raise Exception(ex)
        xs = scheduler.create_hot_observable(msgs)

        def create():
            if False:
                return 10
            return xs.pipe(ops.max_by(lambda x: x['key'], reverse_comparer))
        res = scheduler.start(create=create).messages
        assert res == [on_error(220, ex)]