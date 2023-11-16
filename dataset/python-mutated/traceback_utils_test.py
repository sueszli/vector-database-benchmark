"""Tests for traceback_utils."""
import traceback
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import traceback_utils

class TracebackUtilsTest(test.TestCase):

    def assert_trace_line_count(self, fn, count, filtering_enabled=True):
        if False:
            for i in range(10):
                print('nop')
        trace_line_count = -1
        if filtering_enabled:
            traceback_utils.enable_traceback_filtering()
        else:
            traceback_utils.disable_traceback_filtering()
        self.assertEqual(traceback_utils.is_traceback_filtering_enabled(), filtering_enabled)
        try:
            fn()
        except Exception as e:
            trace = '\n'.join(traceback.format_tb(e.__traceback__))
            trace_line_count = len(trace.split('\n'))
        self.assertGreater(trace_line_count, 0)
        if filtering_enabled:
            self.assertLess(trace_line_count, count)
        else:
            self.assertGreater(trace_line_count, count)

    def test_eager_add(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            x = array_ops.zeros((2, 3))
            y = array_ops.zeros((2, 4))
            _ = x + y
        self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
        self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

    def test_tfn_add(self):
        if False:
            return 10

        @def_function.function
        def fn():
            if False:
                for i in range(10):
                    print('nop')
            x = array_ops.zeros((2, 3))
            y = array_ops.zeros((2, 4))
            return x + y
        self.assert_trace_line_count(fn, count=10, filtering_enabled=True)
        self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

    def test_tfn_div(self):
        if False:
            while True:
                i = 10

        @def_function.function
        def wrapped_fn(x):
            if False:
                i = 10
                return i + 15
            return x / 0.0

        def fn():
            if False:
                return 10
            wrapped_fn(0.5)
        self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
        self.assert_trace_line_count(fn, count=30, filtering_enabled=False)

    def test_eager_argmax(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                while True:
                    i = 10
            _ = math_ops.argmax([0, 1], axis=2)
        self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
        self.assert_trace_line_count(fn, count=30, filtering_enabled=False)

    def test_tfn_argmax(self):
        if False:
            while True:
                i = 10

        @def_function.function
        def wrapped_fn(x):
            if False:
                i = 10
                return i + 15
            return math_ops.argmax(x, axis=2)

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            wrapped_fn([0, 1])
        self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
        self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

    def test_variable_constructor(self):
        if False:
            return 10

        def fn():
            if False:
                while True:
                    i = 10
            _ = variables.Variable()
        self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
        self.assert_trace_line_count(fn, count=30, filtering_enabled=False)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()