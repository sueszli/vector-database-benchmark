"""Tests for py_builtins module."""
import unittest
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.core import function_wrappers
from nvidia.dali._autograph.operators import py_builtins

class TestBase(object):

    def overridden_method(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x + 20

class PyBuiltinsTest(unittest.TestCase):

    def test_abs(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(py_builtins.abs_(-1), 1)

    def test_float(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(py_builtins.float_(10), 10.0)
        self.assertEqual(py_builtins.float_('10.0'), 10.0)

    def test_int(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(py_builtins.int_(10.0), 10)
        self.assertEqual(py_builtins.int_('11', 2), 3)

    def test_int_unsupported_base(self):
        if False:
            return 10
        t = 1.0
        with self.assertRaises(TypeError):
            py_builtins.int_(t, 2)

    def test_len(self):
        if False:
            return 10
        self.assertEqual(py_builtins.len_([1, 2, 3]), 3)

    def test_len_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            py_builtins.len_(1)

    def test_max(self):
        if False:
            return 10
        self.assertEqual(py_builtins.max_([1, 3, 2]), 3)
        self.assertEqual(py_builtins.max_(0, 2, 1), 2)

    def test_min(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(py_builtins.min_([2, 1, 3]), 1)
        self.assertEqual(py_builtins.min_(2, 0, 1), 0)

    def test_range(self):
        if False:
            while True:
                i = 10
        self.assertListEqual(list(py_builtins.range_(3)), [0, 1, 2])
        self.assertListEqual(list(py_builtins.range_(1, 3)), [1, 2])
        self.assertListEqual(list(py_builtins.range_(2, 0, -1)), [2, 1])

    def test_enumerate(self):
        if False:
            return 10
        self.assertListEqual(list(py_builtins.enumerate_([3, 2, 1])), [(0, 3), (1, 2), (2, 1)])
        self.assertListEqual(list(py_builtins.enumerate_([3, 2, 1], 5)), [(5, 3), (6, 2), (7, 1)])
        self.assertListEqual(list(py_builtins.enumerate_([-8], -3)), [(-3, -8)])

    def test_zip(self):
        if False:
            while True:
                i = 10
        self.assertListEqual(list(py_builtins.zip_([3, 2, 1], [1, 2, 3])), [(3, 1), (2, 2), (1, 3)])
        self.assertListEqual(list(py_builtins.zip_([4, 5, 6], [-1, -2])), [(4, -1), (5, -2)])

    def test_map(self):
        if False:
            for i in range(10):
                print('nop')

        def increment(x):
            if False:
                return 10
            return x + 1
        add_list = lambda x, y: x + y
        self.assertListEqual(list(py_builtins.map_(increment, [4, 5, 6])), [5, 6, 7])
        self.assertListEqual(list(py_builtins.map_(add_list, [3, 2, 1], [-1, -2, -3])), [2, 0, -2])

    def test_next_normal(self):
        if False:
            while True:
                i = 10
        iterator = iter([1, 2, 3])
        self.assertEqual(py_builtins.next_(iterator), 1)
        self.assertEqual(py_builtins.next_(iterator), 2)
        self.assertEqual(py_builtins.next_(iterator), 3)
        with self.assertRaises(StopIteration):
            py_builtins.next_(iterator)
        self.assertEqual(py_builtins.next_(iterator, 4), 4)

    def _basic_function_scope(self):
        if False:
            print('Hello World!')
        return function_wrappers.FunctionScope('test_function_name', 'test_scope', converter.ConversionOptions())

    def test_eval_in_original_context(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn():
            if False:
                while True:
                    i = 10
            l = 1
            with self._basic_function_scope() as test_scope:
                return py_builtins.eval_in_original_context(eval, ('l',), test_scope)
        self.assertEqual(test_fn(), 1)

    def test_eval_in_original_context_inner_function(self):
        if False:
            while True:
                i = 10

        def test_fn():
            if False:
                return 10
            l = 1
            with self._basic_function_scope() as test_scope:

                def inner_fn():
                    if False:
                        while True:
                            i = 10
                    l = 2
                    return py_builtins.eval_in_original_context(eval, ('l',), test_scope)
                return inner_fn()
        self.assertEqual(test_fn(), 2)

    def test_locals_in_original_context(self):
        if False:
            print('Hello World!')

        def test_fn():
            if False:
                while True:
                    i = 10
            l = 1
            with self._basic_function_scope() as test_scope:
                return py_builtins.locals_in_original_context(test_scope)
        locs = test_fn()
        self.assertEqual(locs['l'], 1)

    def test_locals_in_original_context_inner_function(self):
        if False:
            i = 10
            return i + 15

        def test_fn():
            if False:
                return 10
            l = 1
            with self._basic_function_scope() as test_scope:

                def inner_fn():
                    if False:
                        while True:
                            i = 10
                    l = 2
                    return py_builtins.locals_in_original_context(test_scope)
                return inner_fn()
        locs = test_fn()
        self.assertEqual(locs['l'], 2)

    def test_globals_in_original_context(self):
        if False:
            return 10

        def test_fn():
            if False:
                while True:
                    i = 10
            with self._basic_function_scope() as test_scope:
                return py_builtins.globals_in_original_context(test_scope)
        globs = test_fn()
        self.assertIs(globs['TestBase'], TestBase)

    def test_globals_in_original_context_inner_function(self):
        if False:
            print('Hello World!')

        def test_fn():
            if False:
                while True:
                    i = 10
            with self._basic_function_scope() as test_scope:

                def inner_fn():
                    if False:
                        while True:
                            i = 10
                    return py_builtins.globals_in_original_context(test_scope)
                return inner_fn()
        globs = test_fn()
        self.assertIs(globs['TestBase'], TestBase)

    def test_super_in_original_context_unary_call(self):
        if False:
            i = 10
            return i + 15
        test_case_self = self

        class TestSubclass(TestBase):

            def overridden_method(self, x):
                if False:
                    return 10
                test_case_self.fail('This should never be called.')

            def test_method(self):
                if False:
                    for i in range(10):
                        print('nop')
                with test_case_self._basic_function_scope() as test_scope:
                    test_base_unbound = py_builtins.super_in_original_context(super, (TestSubclass,), test_scope)
                    test_base = test_base_unbound.__get__(self, TestSubclass)
                    return test_base.overridden_method(1)
        tc = TestSubclass()
        self.assertEqual(tc.test_method(), 21)

    def test_super_in_original_context_binary_call(self):
        if False:
            i = 10
            return i + 15
        test_case_self = self

        class TestSubclass(TestBase):

            def overridden_method(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                test_case_self.fail('This should never be called.')

            def test_method(self):
                if False:
                    while True:
                        i = 10
                with test_case_self._basic_function_scope() as test_scope:
                    test_base = py_builtins.super_in_original_context(super, (TestSubclass, self), test_scope)
                    return test_base.overridden_method(1)
        tc = TestSubclass()
        self.assertEqual(tc.test_method(), 21)

    def test_super_in_original_context_niladic_call(self):
        if False:
            return 10
        test_case_self = self

        class TestSubclass(TestBase):

            def overridden_method(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                test_case_self.fail('This should never be called.')

            def test_method(self):
                if False:
                    for i in range(10):
                        print('nop')
                with test_case_self._basic_function_scope() as test_scope:
                    b = py_builtins.super_in_original_context(super, (), test_scope)
                    return b.overridden_method(1)
        tc = TestSubclass()
        self.assertEqual(tc.test_method(), 21)

    def test_super_in_original_context_caller_with_locals(self):
        if False:
            while True:
                i = 10
        test_case_self = self

        class TestSubclass(TestBase):

            def overridden_method(self, x):
                if False:
                    print('Hello World!')
                test_case_self.fail('This should never be called.')

            def test_method(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = 7
                with test_case_self._basic_function_scope() as test_scope:
                    z = 7
                    return py_builtins.super_in_original_context(super, (), test_scope).overridden_method(x + y - z)
        tc = TestSubclass()
        self.assertEqual(tc.test_method(1), 21)

    def test_super_in_original_context_inner_function(self):
        if False:
            return 10
        test_case_self = self

        class TestSubclass(TestBase):

            def overridden_method(self, x):
                if False:
                    return 10
                test_case_self.fail('This should never be called.')

            def test_method(self, x):
                if False:
                    print('Hello World!')
                with test_case_self._basic_function_scope() as test_scope:

                    def inner_fn():
                        if False:
                            i = 10
                            return i + 15
                        return py_builtins.super_in_original_context(super, (), test_scope).overridden_method(x)
                    return inner_fn()
        tc = TestSubclass()
        self.assertEqual(tc.test_method(1), 21)

    def test_super_in_original_context_inner_lambda(self):
        if False:
            return 10
        test_case_self = self

        class TestSubclass(TestBase):

            def overridden_method(self, x):
                if False:
                    while True:
                        i = 10
                test_case_self.fail('This should never be called.')

            def test_method(self, x):
                if False:
                    print('Hello World!')
                with test_case_self._basic_function_scope() as test_scope:
                    l = lambda : py_builtins.super_in_original_context(super, (), test_scope).overridden_method(x)
                    return l()
        tc = TestSubclass()
        self.assertEqual(tc.test_method(1), 21)

    def test_filter(self):
        if False:
            while True:
                i = 10
        self.assertListEqual(list(py_builtins.filter_(lambda x: x == 'b', ['a', 'b', 'c'])), ['b'])
        self.assertListEqual(list(py_builtins.filter_(lambda x: x < 3, [3, 2, 1])), [2, 1])

    def test_any(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(py_builtins.any_([False, True, False]), True)
        self.assertEqual(py_builtins.any_([False, False, False]), False)

    def test_all(self):
        if False:
            print('Hello World!')
        self.assertEqual(py_builtins.all_([False, True, False]), False)
        self.assertEqual(py_builtins.all_([True, True, True]), True)

    def test_sorted(self):
        if False:
            return 10
        self.assertListEqual(py_builtins.sorted_([2, 3, 1]), [1, 2, 3])
        self.assertListEqual(py_builtins.sorted_([2, 3, 1], key=lambda x: -x), [3, 2, 1])
        self.assertListEqual(py_builtins.sorted_([2, 3, 1], reverse=True), [3, 2, 1])
        self.assertListEqual(py_builtins.sorted_([2, 3, 1], key=lambda x: -x, reverse=True), [1, 2, 3])
        self.assertEqual(py_builtins.sorted_([[4, 3], [2, 1]], key=lambda x: sum(x)), [[2, 1], [4, 3]])