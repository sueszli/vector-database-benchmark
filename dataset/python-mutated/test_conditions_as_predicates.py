from numba.tests.support import TestCase
from numba import njit, types
from numba.typed import List, Dict
import numpy as np

class TestConditionsAsPredicates(TestCase):

    def test_scalars(self):
        if False:
            return 10
        dts = [np.int8, np.uint16, np.int64, np.float32, np.float64, np.complex128, int, float, complex, str, bool]
        for dt in dts:
            for c in (1, 0):
                x = dt(c)

                @njit
                def foo():
                    if False:
                        while True:
                            i = 10
                    if x:
                        return 10
                    else:
                        return 20
                self.assertEqual(foo(), foo.py_func())
                self.assertEqual(foo(), 10 if c == 1 or dt is str else 20)

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            if x:
                return 10
            else:
                return 20
        s = ''
        self.assertEqual(foo(s), foo.py_func(s))

    def test_typed_list(self):
        if False:
            return 10

        @njit
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            if x:
                return 10
            else:
                return 20
        z = List.empty_list(types.int64)
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 20)
        z.append(1)
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 10)

    def test_reflected_list(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            if x:
                return 10
            else:
                return 20
        z = [1]
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 10)

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            y = [1, 2]
            if y:
                return 10
            else:
                return 20
        self.assertEqual(foo(), foo.py_func())
        self.assertEqual(foo.py_func(), 10)

        @njit
        def foo():
            if False:
                return 10
            y = [1, 2]
            y.pop()
            y.pop()
            assert len(y) == 0
            if y:
                return 10
            else:
                return 20
        self.assertEqual(foo(), foo.py_func())
        self.assertEqual(foo.py_func(), 20)

    def test_reflected_set(self):
        if False:
            print('Hello World!')

        @njit
        def foo(x):
            if False:
                i = 10
                return i + 15
            if x:
                return 10
            else:
                return 20
        z = {1}
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 10)

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            y = {1, 2}
            if y:
                return 10
            else:
                return 20
        self.assertEqual(foo(), foo.py_func())
        self.assertEqual(foo.py_func(), 10)

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            y = {1, 2}
            y.pop()
            y.pop()
            assert len(y) == 0
            if y:
                return 10
            else:
                return 20
        self.assertEqual(foo(), foo.py_func())
        self.assertEqual(foo.py_func(), 20)

    def test_typed_dict(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            if x:
                return 10
            else:
                return 20
        z = Dict.empty(types.int64, types.int64)
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 20)
        z[2] = 3
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 10)

    def test_arrays(self):
        if False:
            return 10

        @njit
        def foo(x):
            if False:
                i = 10
                return i + 15
            if x:
                return 10
            else:
                return 20
        z = np.array(1)
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 10)
        z = np.array(0)
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 20)
        z = np.array([[[1]]])
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 10)
        z = np.array([[[0]]])
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 20)
        z = np.empty(0)
        self.assertEqual(foo(z), foo.py_func(z))
        self.assertEqual(foo.py_func(z), 20)
        z = np.array([1, 2])
        with self.assertRaises(ValueError) as raises:
            foo(z)
        msg = 'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()'
        self.assertIn(msg, str(raises.exception))