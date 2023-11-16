"""Unit tests for authoring package."""
import tensorflow as tf
from tensorflow.lite.python.authoring import authoring

class TFLiteAuthoringTest(tf.test.TestCase):

    def test_simple_cosh(self):
        if False:
            print('Hello World!')

        @authoring.compatible
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def f(x):
            if False:
                i = 10
                return i + 15
            return tf.cosh(x)
        result = f(tf.constant([0.0]))
        log_messages = f.get_compatibility_log()
        self.assertEqual(result, tf.constant([1.0]))
        self.assertIn('COMPATIBILITY WARNING: op \'tf.Cosh\' require(s) "Select TF Ops" for model conversion for TensorFlow Lite. https://www.tensorflow.org/lite/guide/ops_select', log_messages)
        self.assertIn('authoring_test.py', log_messages[-1])

    def test_simple_cosh_raises_CompatibilityError(self):
        if False:
            print('Hello World!')

        @authoring.compatible(raise_exception=True)
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def f(x):
            if False:
                i = 10
                return i + 15
            return tf.cosh(x)
        with self.assertRaises(authoring.CompatibilityError):
            result = f(tf.constant([0.0]))
            del result
        log_messages = f.get_compatibility_log()
        self.assertIn('COMPATIBILITY WARNING: op \'tf.Cosh\' require(s) "Select TF Ops" for model conversion for TensorFlow Lite. https://www.tensorflow.org/lite/guide/ops_select', log_messages)

    def test_flex_compatibility(self):
        if False:
            print('Hello World!')

        @authoring.compatible
        @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3, 3, 3, 3], dtype=tf.float32)])
        def f(inp):
            if False:
                return 10
            tanh = tf.math.tanh(inp)
            conv3d = tf.nn.conv3d(tanh, tf.ones([3, 3, 3, 3, 3]), strides=[1, 1, 1, 1, 1], padding='SAME')
            erf = tf.math.erf(conv3d)
            output = tf.math.tanh(erf)
            return output
        f(tf.ones(shape=(3, 3, 3, 3, 3), dtype=tf.float32))
        log_messages = f.get_compatibility_log()
        self.assertIn('COMPATIBILITY WARNING: op \'tf.Erf\' require(s) "Select TF Ops" for model conversion for TensorFlow Lite. https://www.tensorflow.org/lite/guide/ops_select', log_messages)

    def test_compatibility_error_generic(self):
        if False:
            i = 10
            return i + 15

        @authoring.compatible
        @tf.function
        def f():
            if False:
                print('Hello World!')
            dataset = tf.data.Dataset.range(3)
            dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
            return dataset
        f()
        log_messages = f.get_compatibility_log()
        self.assertIn("COMPATIBILITY ERROR: op 'tf.DummySeedGenerator, tf.RangeDataset, tf.ShuffleDatasetV3' is(are) not natively supported by TensorFlow Lite. You need to provide a custom operator. https://www.tensorflow.org/lite/guide/ops_custom", log_messages)

    def test_compatibility_error_custom(self):
        if False:
            for i in range(10):
                print('nop')
        target_spec = tf.lite.TargetSpec()
        target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

        @authoring.compatible(converter_target_spec=target_spec)
        @tf.function
        def f():
            if False:
                print('Hello World!')
            dataset = tf.data.Dataset.range(3)
            dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
            return dataset
        f()
        log_messages = f.get_compatibility_log()
        self.assertIn("COMPATIBILITY ERROR: op 'tf.DummySeedGenerator, tf.RangeDataset, tf.ShuffleDatasetV3' is(are) not natively supported by TensorFlow Lite. You need to provide a custom operator. https://www.tensorflow.org/lite/guide/ops_custom", log_messages)

    def test_simple_variable(self):
        if False:
            while True:
                i = 10
        external_var = tf.Variable(1.0)

        @authoring.compatible
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * external_var
        result = f(tf.constant(2.0, shape=1))
        log_messages = f.get_compatibility_log()
        self.assertEqual(result, tf.constant([2.0]))
        self.assertEmpty(log_messages)

    def test_class_method(self):
        if False:
            return 10

        class Model(tf.Module):

            @authoring.compatible
            @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
            def eval(self, x):
                if False:
                    while True:
                        i = 10
                return tf.cosh(x)
        m = Model()
        result = m.eval(tf.constant([0.0]))
        log_messages = m.eval.get_compatibility_log()
        self.assertEqual(result, tf.constant([1.0]))
        self.assertIn('COMPATIBILITY WARNING: op \'tf.Cosh\' require(s) "Select TF Ops" for model conversion for TensorFlow Lite. https://www.tensorflow.org/lite/guide/ops_select', log_messages)

    def test_decorated_function_type(self):
        if False:
            while True:
                i = 10

        @authoring.compatible
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def func(x):
            if False:
                print('Hello World!')
            return tf.cos(x)
        result = func(tf.constant([0.0]))
        self.assertEqual(result, tf.constant([1.0]))
        self.assertEqual(func.__name__, 'func')
        converter = tf.lite.TFLiteConverter.from_concrete_functions([func.get_concrete_function()], func)
        converter.convert()

    def test_decorated_class_method_type(self):
        if False:
            print('Hello World!')

        class Model(tf.Module):

            @authoring.compatible
            @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
            def eval(self, x):
                if False:
                    i = 10
                    return i + 15
                return tf.cos(x)
        m = Model()
        result = m.eval(tf.constant([0.0]))
        self.assertEqual(result, tf.constant([1.0]))
        self.assertEqual(m.eval.__name__, 'eval')
        converter = tf.lite.TFLiteConverter.from_concrete_functions([m.eval.get_concrete_function()], m)
        converter.convert()

    def test_simple_cosh_multiple(self):
        if False:
            for i in range(10):
                print('nop')

        @authoring.compatible
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return tf.cosh(x)
        f(tf.constant([1.0]))
        f(tf.constant([2.0]))
        f(tf.constant([3.0]))
        warning_messages = f.get_compatibility_log()
        self.assertEqual(2, len(warning_messages))

    def test_user_tf_ops_all_filtered(self):
        if False:
            print('Hello World!')
        target_spec = tf.lite.TargetSpec()
        target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        target_spec.experimental_select_user_tf_ops = ['RangeDataset', 'DummySeedGenerator', 'ShuffleDatasetV3']

        @authoring.compatible(converter_target_spec=target_spec)
        @tf.function
        def f():
            if False:
                i = 10
                return i + 15
            dataset = tf.data.Dataset.range(3)
            dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
            return dataset
        f()
        log_messages = f.get_compatibility_log()
        self.assertEmpty(log_messages)

    def test_user_tf_ops_partial_filtered(self):
        if False:
            i = 10
            return i + 15
        target_spec = tf.lite.TargetSpec()
        target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        target_spec.experimental_select_user_tf_ops = ['DummySeedGenerator']

        @authoring.compatible(converter_target_spec=target_spec)
        @tf.function
        def f():
            if False:
                return 10
            dataset = tf.data.Dataset.range(3)
            dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
            return dataset
        f()
        log_messages = f.get_compatibility_log()
        self.assertIn("COMPATIBILITY ERROR: op 'tf.RangeDataset, tf.ShuffleDatasetV3' is(are) not natively supported by TensorFlow Lite. You need to provide a custom operator. https://www.tensorflow.org/lite/guide/ops_custom", log_messages)

    def test_allow_custom_ops(self):
        if False:
            i = 10
            return i + 15
        target_spec = tf.lite.TargetSpec()
        target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

        @authoring.compatible(converter_allow_custom_ops=True, converter_target_spec=target_spec)
        @tf.function
        def f():
            if False:
                i = 10
                return i + 15
            dataset = tf.data.Dataset.range(3)
            dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
            return dataset
        f()
        log_messages = f.get_compatibility_log()
        self.assertEmpty(log_messages)

    def test_non_gpu_compatible(self):
        if False:
            while True:
                i = 10
        target_spec = tf.lite.TargetSpec()
        target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        target_spec.experimental_supported_backends = ['GPU']

        @authoring.compatible(converter_target_spec=target_spec)
        @tf.function(input_signature=[tf.TensorSpec(shape=[4, 4], dtype=tf.float32)])
        def func(x):
            if False:
                while True:
                    i = 10
            return tf.cosh(x) + tf.slice(x, [1, 1], [1, 1])
        func(tf.ones(shape=(4, 4), dtype=tf.float32))
        log_messages = func.get_compatibility_log()
        self.assertIn("'tfl.slice' op is not GPU compatible: SLICE supports for 3 or 4 dimensional tensors only, but node has 2 dimensional tensors.", log_messages)
        self.assertIn("COMPATIBILITY WARNING: op 'tf.Cosh, tfl.slice' aren't compatible with TensorFlow Lite GPU delegate. https://www.tensorflow.org/lite/performance/gpu", log_messages)

    def test_gpu_compatible(self):
        if False:
            while True:
                i = 10
        target_spec = tf.lite.TargetSpec()
        target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        target_spec.experimental_supported_backends = ['GPU']

        @authoring.compatible(converter_target_spec=target_spec)
        @tf.function(input_signature=[tf.TensorSpec(shape=[4, 4], dtype=tf.float32)])
        def func(x):
            if False:
                i = 10
                return i + 15
            return tf.cos(x)
        func(tf.ones(shape=(4, 4), dtype=tf.float32))
        log_messages = func.get_compatibility_log()
        self.assertEmpty(log_messages)
if __name__ == '__main__':
    tf.test.main()