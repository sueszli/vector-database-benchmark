import unittest
import reactivex
from reactivex import operators as _
from reactivex.testing import ReactiveTest, TestScheduler
on_next = ReactiveTest.on_next
on_completed = ReactiveTest.on_completed
on_error = ReactiveTest.on_error
subscribe = ReactiveTest.subscribe
subscribed = ReactiveTest.subscribed
disposed = ReactiveTest.disposed
created = ReactiveTest.created

class TestDo(unittest.TestCase):

    def test_do_should_see_all_values(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))
        i = [0]
        sum = [2 + 3 + 4 + 5]

        def create():
            if False:
                while True:
                    i = 10

            def action(x):
                if False:
                    for i in range(10):
                        print('nop')
                i[0] += 1
                sum[0] -= x
                return sum[0]
            return xs.pipe(_.do_action(action))
        scheduler.start(create)
        self.assertEqual(4, i[0])
        self.assertEqual(0, sum[0])

    def test_do_plain_action(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))
        i = [0]

        def create():
            if False:
                return 10

            def action(x):
                if False:
                    i = 10
                    return i + 15
                i[0] += 1
                return i[0]
            return xs.pipe(_.do_action(action))
        scheduler.start(create)
        self.assertEqual(4, i[0])

    def test_do_next_completed(self):
        if False:
            print('Hello World!')
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_next(220, 3), on_next(230, 4), on_next(240, 5), on_completed(250))
        i = [0]
        sum = [2 + 3 + 4 + 5]
        completed = [False]

        def create():
            if False:
                while True:
                    i = 10

            def on_next(x):
                if False:
                    return 10
                i[0] += 1
                sum[0] -= x

            def on_completed():
                if False:
                    print('Hello World!')
                completed[0] = True
            return xs.pipe(_.do_action(on_next=on_next, on_completed=on_completed))
        scheduler.start(create)
        self.assertEqual(4, i[0])
        self.assertEqual(0, sum[0])
        assert completed[0]

    def test_do_next_completed_never(self):
        if False:
            i = 10
            return i + 15
        scheduler = TestScheduler()
        i = [0]
        completed = False

        def create():
            if False:
                return 10
            nonlocal completed

            def on_next(x):
                if False:
                    while True:
                        i = 10
                i[0] += 1

            def on_completed():
                if False:
                    i = 10
                    return i + 15
                nonlocal completed
                completed = True
            return reactivex.never().pipe(_.do_action(on_next=on_next, on_completed=on_completed))
        scheduler.start(create)
        self.assertEqual(0, i[0])
        assert not completed

    def test_do_action_without_next(self):
        if False:
            return 10
        scheduler = TestScheduler()
        xs = scheduler.create_hot_observable(on_next(150, 1), on_next(210, 2), on_completed(250))
        completed = [False]

        def create():
            if False:
                i = 10
                return i + 15

            def on_completed():
                if False:
                    while True:
                        i = 10
                completed[0] = True
            return xs.pipe(_.do_action(on_completed=on_completed))
        scheduler.start(create)
        assert completed[0]