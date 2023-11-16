"""Tests for side inputs in tf.function."""
import unittest
from absl.testing import parameterized
import tensorflow as tf

class SideInputsTest(parameterized.TestCase):

    @parameterized.parameters((1, tf.constant, 2, tf.constant), (1.0, tf.constant, 2.0, tf.constant), (1, int, 2, int), (1.0, float, 2.0, float), (1, int, 2, tf.constant), (1, tf.constant, 2, int))
    @unittest.skip('Feature not implemented')
    def test_direct_capture(self, val_before, type_before, val_after, type_after):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                print('Hello World!')
            return x + 1
        tf_f = tf.function(f)
        x = type_before(val_before)
        self.assertEqual(f(), tf_f())
        x = type_after(val_after)
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    def test_direct_capture_mutation(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                while True:
                    i = 10
            return glob[-1] + tf.constant(0)
        tf_f = tf.function(f)
        glob = [tf.constant(1), tf.constant(2)]
        self.assertEqual(f(), tf_f())
        glob.append(tf.constant(3))
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    @parameterized.parameters(tf.constant, int)
    def test_dict_capture_mutation_with_tensor_and_non_tensor(self, capture_type):
        if False:
            return 10

        def f():
            if False:
                while True:
                    i = 10
            return d['val']
        tf_f = tf.function(f)
        d = {'int': 1, 'tensor': tf.constant(2), 'val': capture_type(3)}
        self.assertEqual(f(), tf_f())
        d['val'] = capture_type(4)
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    @parameterized.parameters(tf.constant, int)
    def test_capture_with_duplicate_usage(self, capture_type):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return x + x
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(), tf_f())
        self.assertLen(tf_f.get_concrete_function().graph.inputs, 1)
        x = capture_type(2)
        self.assertEqual(f(), tf_f())
        self.assertLen(tf_f.get_concrete_function().graph.inputs, 1)

    @unittest.skip('Feature not implemented')
    def test_local_capture(self):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                for i in range(10):
                    print('nop')
            x = tf.constant(0)

            def g():
                if False:
                    return 10
                return x
            return g()
        tf_f = tf.function(f)
        x = tf.constant(100)
        self.assertEqual(f(), tf_f())
        x = tf.constant(200)
        self.assertEqual(f(), tf_f())

    @parameterized.parameters(tf.constant, int)
    @unittest.skip('Feature not implemented')
    def test_capture_by_nested_function(self, capture_type):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                i = 10
                return i + 15

            def g():
                if False:
                    print('Hello World!')
                return x
            return g()
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(), tf_f())
        x = capture_type(2)
        self.assertEqual(f(), tf_f())

    @parameterized.parameters(tf.constant, int)
    @unittest.skip('Feature not implemented')
    def test_outer_capture_with_function_call(self, capture_type):
        if False:
            i = 10
            return i + 15

        def g():
            if False:
                return 10
            return x

        @tf.function
        def f():
            if False:
                while True:
                    i = 10
            return g()
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(), tf_f())
        x = capture_type(2)
        self.assertEqual(f(), tf_f())

    @parameterized.parameters(tf.constant, int)
    @unittest.skip('Feature not implemented')
    def test_outer_capture_with_nested_function_call(self, capture_type):
        if False:
            i = 10
            return i + 15

        def g_factory():
            if False:
                i = 10
                return i + 15

            def g():
                if False:
                    for i in range(10):
                        print('nop')
                return x
            return g()

        def f():
            if False:
                for i in range(10):
                    print('nop')
            h = g_factory()
            return h()
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(), tf_f())
        x = capture_type(2)
        self.assertEqual(f(), tf_f())

    @parameterized.parameters(tf.constant, int)
    @unittest.skip('Feature not implemented')
    def test_capture_within_function_argument(self, capture_type):
        if False:
            return 10

        def g():
            if False:
                while True:
                    i = 10
            return x

        def f(h):
            if False:
                while True:
                    i = 10
            return h()
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(g), tf_f(g))
        x = capture_type(2)
        self.assertEqual(f(g), tf_f(g))

    @parameterized.parameters(tf.constant, int)
    @unittest.skip('Feature not implemented')
    def test_nested_tf_function_with_capture(self, capture_type):
        if False:
            print('Hello World!')

        @tf.function
        def tf_f():
            if False:
                i = 10
                return i + 15

            @tf.function
            def tf_g():
                if False:
                    i = 10
                    return i + 15
                return x
            return tf_g()
        x = capture_type(0)
        self.assertEqual(tf_f(), tf.constant(0))
        x = capture_type(1)
        self.assertEqual(tf_f(), tf.constant(0))
        self.assertLen(tf_f.get_concrete_function().graph.capture, 1)
if __name__ == '__main__':
    unittest.main()