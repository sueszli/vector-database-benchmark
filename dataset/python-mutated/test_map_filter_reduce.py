from numba import njit
from functools import reduce
import unittest

class TestMap(unittest.TestCase):

    def test_basic_map_external_func(self):
        if False:
            i = 10
            return i + 15
        func = njit(lambda x: x + 10)

        def impl():
            if False:
                return 10
            return [y for y in map(func, range(10))]
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())

    def test_basic_map_closure(self):
        if False:
            print('Hello World!')

        def impl():
            if False:
                while True:
                    i = 10
            return [y for y in map(lambda x: x + 10, range(10))]
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())

    def test_basic_map_closure_multiple_iterator(self):
        if False:
            while True:
                i = 10

        def impl():
            if False:
                for i in range(10):
                    print('nop')
            args = (range(10), range(10, 20))
            return [y for y in map(lambda a, b: (a + 10, b + 5), *args)]
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())

class TestFilter(unittest.TestCase):

    def test_basic_filter_external_func(self):
        if False:
            i = 10
            return i + 15
        func = njit(lambda x: x > 0)

        def impl():
            if False:
                for i in range(10):
                    print('nop')
            return [y for y in filter(func, range(-10, 10))]
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())

    def test_basic_filter_closure(self):
        if False:
            while True:
                i = 10

        def impl():
            if False:
                return 10
            return [y for y in filter(lambda x: x > 0, range(-10, 10))]
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())

    def test_basic_filter_none_func(self):
        if False:
            print('Hello World!')

        def impl():
            if False:
                return 10
            return [y for y in filter(None, range(-10, 10))]
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())

class TestReduce(unittest.TestCase):

    def test_basic_reduce_external_func(self):
        if False:
            while True:
                i = 10
        func = njit(lambda x, y: x + y)

        def impl():
            if False:
                i = 10
                return i + 15
            return reduce(func, range(-10, 10))
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())

    def test_basic_reduce_closure(self):
        if False:
            print('Hello World!')

        def impl():
            if False:
                i = 10
                return i + 15

            def func(x, y):
                if False:
                    print('Hello World!')
                return x + y
            return reduce(func, range(-10, 10), 100)
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())
if __name__ == '__main__':
    unittest.main()