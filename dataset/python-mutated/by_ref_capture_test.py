"""Tests for detecting free vars in tf.function."""
import unittest
from absl.testing import parameterized
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test

class ByRefCaptureTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(val_type=[int, constant_op.constant]))
    def test_direct_capture_mutation(self, val_type):
        if False:
            return 10
        x = val_type(1)

        @def_function.function
        def f():
            if False:
                for i in range(10):
                    print('nop')
            graph = ops.get_default_graph()
            cap_x = graph._experimental_capture_side_input_by_ref('x', lambda : x)
            return cap_x + 1
        self.assertEqual(f(), 2)
        x = val_type(2)
        self.assertEqual(f(), 3)

    @unittest.skip('By ref capture API does not work for nested tf.function.')
    def test_capture_in_nested_function(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant(1)

        @def_function.function
        def f():
            if False:
                i = 10
                return i + 15
            graph = ops.get_default_graph()
            graph._experimental_capture_side_input_by_ref('x', lambda : x)

            @def_function.function
            def g():
                if False:
                    while True:
                        i = 10
                graph = ops.get_default_graph()
                cap_x = graph._experimental_capture_side_input_by_ref('xx', lambda : x)
                return cap_x + 100
            return g()
        self.assertEqual(f(), 2)
        x = constant_op.constant(2)
        self.assertEqual(f(), 102)

    def test_capture_in_outer_function(self):
        if False:
            while True:
                i = 10
        x = 1

        def g():
            if False:
                print('Hello World!')
            graph = ops.get_default_graph()
            cap_x = graph._experimental_capture_side_input_by_ref('x', lambda : x)
            return cap_x + 1

        @def_function.function
        def f():
            if False:
                i = 10
                return i + 15
            return g()
        self.assertEqual(f(), 2)
        x = 2
        self.assertEqual(f(), 3)

    @unittest.skip('By ref capture API does not work for nested tf.function.')
    def test_capture_in_outer_tf_function(self):
        if False:
            i = 10
            return i + 15
        x = 1

        @def_function.function
        def g():
            if False:
                return 10
            graph = ops.get_default_graph()
            cap_x = graph._experimental_capture_side_input_by_ref('x', lambda : x)
            return cap_x + 1

        @def_function.function
        def f():
            if False:
                return 10
            graph = ops.get_default_graph()
            graph._experimental_capture_side_input_by_ref('x', lambda : x)
            return g()
        self.assertEqual(f(), 2)
        x = 2
        self.assertEqual(f(), 3)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()