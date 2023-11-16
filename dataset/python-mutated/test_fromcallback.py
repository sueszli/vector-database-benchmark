import unittest
import reactivex
from reactivex.testing import ReactiveTest
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

class TestFromCallback(unittest.TestCase):

    def test_from_callback(self):
        if False:
            for i in range(10):
                print('nop')
        res = reactivex.from_callback(lambda cb: cb(True))()

        def on_next(r):
            if False:
                print('Hello World!')
            self.assertEqual(r, True)

        def on_error(err):
            if False:
                while True:
                    i = 10
            assert False

        def on_completed():
            if False:
                print('Hello World!')
            assert True
        res.subscribe(on_next, on_error, on_completed)

    def test_from_callback_single(self):
        if False:
            return 10
        res = reactivex.from_callback(lambda file, cb: cb(file))('file.txt')

        def on_next(r):
            if False:
                print('Hello World!')
            self.assertEqual(r, 'file.txt')

        def on_error(err):
            if False:
                return 10
            assert False

        def on_completed():
            if False:
                while True:
                    i = 10
            assert True
        res.subscribe(on_next, on_error, on_completed)

    def test_from_node_callback_mapper(self):
        if False:
            while True:
                i = 10
        res = reactivex.from_callback(lambda f, s, t, cb: cb(f, s, t), lambda r: r[0])(1, 2, 3)

        def on_next(r):
            if False:
                return 10
            self.assertEqual(r, 1)

        def on_error(err):
            if False:
                while True:
                    i = 10
            assert False

        def on_completed():
            if False:
                i = 10
                return i + 15
            assert True
        res.subscribe(on_next, on_error, on_completed)