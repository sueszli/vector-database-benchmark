import unittest
import inspect
from numba import njit
from numba.tests.support import TestCase
from numba.misc.firstlinefinder import get_func_body_first_lineno

class TestFirstLineFinder(TestCase):
    """
    The following methods contains tests that are sensitive to the source
    locations w.r.t. the beginning of each method.
    """

    def _get_grandparent_caller_code(self):
        if False:
            return 10
        frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(frame)
        return caller_frame[2].frame.f_code

    def assert_line_location(self, expected, offset_from_caller):
        if False:
            return 10
        grandparent_co = self._get_grandparent_caller_code()
        lno = grandparent_co.co_firstlineno
        self.assertEqual(expected, lno + offset_from_caller)

    def test_decorated_odd_comment_indent(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                return 10
            return 1
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 4)

    def test_undecorated_odd_comment_indent(self):
        if False:
            for i in range(10):
                print('nop')

        def foo():
            if False:
                return 10
            return 1
        first_def_line = get_func_body_first_lineno(njit(foo))
        self.assert_line_location(first_def_line, 3)

    def test_unnamed_lambda(self):
        if False:
            for i in range(10):
                print('nop')
        foo = lambda : 1
        first_def_line = get_func_body_first_lineno(njit(foo))
        self.assertIsNone(first_def_line)

    def test_nested_function(self):
        if False:
            while True:
                i = 10

        def foo():
            if False:
                print('Hello World!')

            @njit
            def foo():
                if False:
                    for i in range(10):
                        print('nop')
                return 1
            return foo
        inner = foo()
        first_def_line = get_func_body_first_lineno(inner)
        self.assert_line_location(first_def_line, 5)

    def test_pass_statement(self):
        if False:
            while True:
                i = 10

        @njit
        def foo():
            if False:
                print('Hello World!')
            pass
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 3)

    def test_string_eval(self):
        if False:
            i = 10
            return i + 15
        source = 'def foo():\n            pass\n        '
        globalns = {}
        exec(source, globalns)
        foo = globalns['foo']
        first_def_line = get_func_body_first_lineno(foo)
        self.assertIsNone(first_def_line)

    def test_single_line_function(self):
        if False:
            return 10

        @njit
        def foo():
            if False:
                print('Hello World!')
            pass
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 2)

    def test_docstring(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            'Docstring\n            '
            pass
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 5)

    def test_docstring_2(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            'Docstring\n            '
            'Not Docstring, but a bare string literal\n            '
            pass
        first_def_line = get_func_body_first_lineno(foo)
        self.assert_line_location(first_def_line, 5)
if __name__ == '__main__':
    unittest.main()