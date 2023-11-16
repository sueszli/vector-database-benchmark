"""Tests for manual API to support side inputs in tf.function."""
import unittest
from absl.testing import parameterized
import tensorflow as tf

class SideInputsTest(parameterized.TestCase):

    @unittest.skip('Feature not implemented')
    @parameterized.parameters((1, tf.constant, 2, tf.constant), (1.0, tf.constant, 2.0, tf.constant), (1, int, 2, int), (1.0, float, 2.0, float), (1, int, 2, tf.constant), (1, tf.constant, 2, int))
    def test_direct_capture(self, val_before, type_before, val_after, type_after):
        if False:
            print('Hello World!')

        def f():
            if False:
                i = 10
                return i + 15
            return tf.func.experimental.capture(lambda : x) + 1
        tf_f = tf.function(f)
        x = type_before(val_before)
        self.assertEqual(f(), tf_f())
        x = type_after(val_after)
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    def test_direct_capture_mutation(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                i = 10
                return i + 15
            cglob = tf.func.experimental.capture(lambda : glob)
            return cglob[-1] + tf.constant(0)
        tf_f = tf.function(f)
        glob = [tf.constant(1), tf.constant(2)]
        self.assertEqual(f(), tf_f())
        glob.append(tf.constant(3))
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    @parameterized.parameters(tf.constant, int)
    def test_dict_capture_mutation_with_tensor_and_non_tensor(self, capture_type):
        if False:
            print('Hello World!')

        def f():
            if False:
                for i in range(10):
                    print('nop')
            cd = tf.func.experimental.capture(lambda : d)
            return cd['val']
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
                while True:
                    i = 10
            cx = tf.func.experimental.capture(lambda : x)
            return cx + cx
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(), tf_f())
        self.assertLen(tf_f._variable_creation_config._captures_container, 1)

    @unittest.skip('Feature not implemented')
    def test_local_capture(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                i = 10
                return i + 15
            x = tf.constant(0)

            def g():
                if False:
                    print('Hello World!')
                return tf.func.experimental.capture(lambda : x)
            return g()
        tf_f = tf.function(f)
        x = tf.constant(100)
        a = 100
        b = tf_f()
        self.assertEqual(a, b)
        x = tf.constant(200)
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    @parameterized.parameters(tf.constant, int)
    def test_capture_by_nested_function(self, capture_type):
        if False:
            while True:
                i = 10

        def f():
            if False:
                return 10

            def g():
                if False:
                    print('Hello World!')
                cx = tf.func.experimental.capture(lambda : x)
                return cx
            return g()
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(), tf_f())
        x = capture_type(2)
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    @parameterized.parameters(tf.constant, int)
    def test_outer_capture_with_function_call(self, capture_type):
        if False:
            for i in range(10):
                print('nop')

        def g():
            if False:
                for i in range(10):
                    print('nop')
            cx = tf.func.experimental.capture(lambda : x)
            return cx

        def f():
            if False:
                i = 10
                return i + 15
            return g()
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(), tf_f())
        x = capture_type(2)
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    @parameterized.parameters(tf.constant, int)
    def test_outer_capture_with_nested_function_call(self, capture_type):
        if False:
            i = 10
            return i + 15
        x = capture_type(1)

        def g_factory():
            if False:
                i = 10
                return i + 15

            def g():
                if False:
                    print('Hello World!')
                cx = tf.func.experimental.capture(lambda : x)
                return cx
            return g()

        def f():
            if False:
                for i in range(10):
                    print('nop')
            h = g_factory
            return h()
        tf_f = tf.function(f)
        self.assertEqual(f(), tf_f())
        x = capture_type(2)
        self.assertEqual(f(), tf_f())

    @unittest.skip('Feature not implemented')
    @parameterized.parameters(tf.constant, int)
    def test_capture_within_function_argument(self, capture_type):
        if False:
            print('Hello World!')

        def g():
            if False:
                i = 10
                return i + 15
            cx = tf.func.experimental.capture(lambda : x)
            return cx

        def f(h):
            if False:
                for i in range(10):
                    print('nop')
            return h()
        tf_f = tf.function(f)
        x = capture_type(1)
        self.assertEqual(f(g), tf_f(g))
        x = capture_type(2)
        self.assertEqual(f(g), tf_f(g))

    @unittest.skip('Feature not implemented')
    def test_inner_nested_tf_function_raise_error(self):
        if False:
            print('Hello World!')

        @tf.function
        def tf_f():
            if False:
                print('Hello World!')

            @tf.function
            def tf_g():
                if False:
                    while True:
                        i = 10
                cx = tf.func.experimental.capture(lambda : x)
                return cx
            return tf_g()
        x = tf.constant(0)
        with self.assertRaisesRegex(NotImplementedError, 'Manual side input usage for inner nested'):
            tf_f()

    @unittest.skip('Feature not implemented')
    @parameterized.parameters(tf.constant, int)
    def test_outer_nested_tf_function_with_global_capture(self, capture_type):
        if False:
            i = 10
            return i + 15

        @tf.function
        def tf_f():
            if False:
                while True:
                    i = 10

            @tf.function
            def tf_g(x):
                if False:
                    print('Hello World!')
                return x
            cx = tf.func.experimental.capture(lambda : x)
            return tf_g(cx)
        x = capture_type(0)
        self.assertEqual(tf_f(), tf.constant(0))
        x = capture_type(1)
        self.assertEqual(tf_f(), tf.constant(1))

    @unittest.skip('Feature not implemented')
    def test_non_callable_function_raise_error(self):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                while True:
                    i = 10
            return tf.func.experimental.capture(x) + 1
        tf_f = tf.function(f)
        x = 1
        with self.assertRaises(TypeError):
            _ = tf_f()
        x = tf.constant(1)
        with self.assertRaises(TypeError):
            _ = tf_f()

    @unittest.skip('Feature not implemented')
    @parameterized.parameters((1, tf.constant, 2, tf.constant), (1, int, 2, int))
    def test_call_by_value(self, val_before, type_before, val_after, type_after):
        if False:
            print('Hello World!')

        def f():
            if False:
                return 10
            return tf.func.experimental.capture(lambda : x, by_ref=False)
        tf_f = tf.function(f)
        x = type_before(val_before)
        self.assertEqual(tf_f(), val_before)
        x = type_after(val_after)
        self.assertEqual(tf_f(), val_before)
if __name__ == '__main__':
    unittest.main()