"""Tests for interop between TF ops, numpy_ops, and numpy methods."""
import numpy as onp
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.numpy_ops import np_math_ops
np = tf.experimental.numpy

class ReadmeTest(tf.test.TestCase):

    def testBroadcastAdd(self):
        if False:
            return 10
        x_np = np.ones([2, 1]) + np.ones([1, 2])
        x_onp = onp.ones([2, 1]) + onp.ones([1, 2])
        self.assertAllClose(x_onp, x_np)

    def testTypePromotion(self):
        if False:
            print('Hello World!')
        x_np = np.ones([1, 2], dtype=np.int16) + np.ones([2, 1], dtype=np.uint8)
        x_onp = np.ones([1, 2], dtype=np.int16) + np.ones([2, 1], dtype=np.uint8)
        self.assertEqual(x_onp.dtype, x_np.dtype)
        self.assertAllClose(x_onp, x_np)

    def testTFInterop(self):
        if False:
            for i in range(10):
                print('nop')
        x_np = np.sum(np.ones([1, 2]) + tf.ones([2, 1]))
        x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
        self.assertAllClose(x_onp, x_np)

    def testOnpInterop(self):
        if False:
            for i in range(10):
                print('nop')
        x_np = onp.sum(np.ones([1, 2]) + onp.ones([2, 1]))
        x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
        self.assertAllClose(x_onp, x_np)

    def testDevice(self):
        if False:
            return 10
        if tf.test.is_gpu_available():
            with tf.device('GPU:0'):
                x = np.ones([1, 2])
            self.assertIn('GPU', tf.convert_to_tensor(x).device)
        with tf.device('CPU:0'):
            x = np.ones([1, 2])
        self.assertIn('CPU', tf.convert_to_tensor(x).device)

    def testFunction(self):
        if False:
            while True:
                i = 10

        @tf.function
        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return np.sum(x + y)
        x_np = f(np.ones([1, 2]), tf.ones([2, 1]))
        x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
        self.assertAllClose(x_onp, x_np)

class InteropTest(tf.test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(InteropTest, self).setUp()
        physical_devices = tf.config.list_physical_devices('CPU')
        configs = tf.config.get_logical_device_configuration(physical_devices[0])
        if configs is None:
            logical_devices = [tf.config.LogicalDeviceConfiguration() for _ in range(3)]
            tf.config.set_logical_device_configuration(physical_devices[0], logical_devices)

    def testGradientTapeInterop(self):
        if False:
            while True:
                i = 10
        with tf.GradientTape() as t:
            x = np.asarray(3.0)
            y = np.asarray(2.0)
            t.watch([x, y])
            xx = 2 * x
            yy = 3 * y
        (dx, dy) = t.gradient([xx, yy], [x, y])
        self.assertIsInstance(dx, np.ndarray)
        self.assertIsInstance(dy, np.ndarray)
        self.assertAllClose(dx, 2.0)
        self.assertAllClose(dy, 3.0)

    def testGradientTapeNoneGradients(self):
        if False:
            return 10
        y = np.asarray(2.0)
        with tf.GradientTape() as t:
            x = np.asarray(3.0)
            t.watch([x])
            z = 2 * x
        dz = t.gradient(z, y)
        self.assertIsNone(dz)

    def testCondInterop(self):
        if False:
            while True:
                i = 10
        x = np.asarray(3.0)

        def fn(x):
            if False:
                while True:
                    i = 10
            x_plus_1 = tf.cond(x > 0, lambda : x + 1, lambda : x + 2)
            x_plus_2 = tf.cond(x < 0, lambda : x + 1, lambda : x + 2)
            return (x_plus_1, x_plus_2)
        (raw_x_plus_1, raw_x_plus_2) = fn(x)
        (fn_x_plus_1, fn_x_plus_2) = tf.function(fn)(x)
        self.assertAllClose(raw_x_plus_1, x + 1)
        self.assertAllClose(raw_x_plus_2, x + 2)
        self.assertAllClose(fn_x_plus_1, x + 1)
        self.assertAllClose(fn_x_plus_2, x + 2)

    def testWhileInterop(self):
        if False:
            return 10

        def fn():
            if False:
                print('Hello World!')
            x = np.asarray(0)
            c = lambda x: x < 10000
            b = lambda x: [x + 1]
            return tf.while_loop(c, b, [x], parallel_iterations=20)
        self.assertEqual(10000, fn()[0])
        self.assertEqual(10000, tf.function(fn)()[0])

    def testTensorTFNPArrayInterop(self):
        if False:
            while True:
                i = 10
        arr = np.asarray(0.0)
        t = tf.constant(10.0)
        arr_plus_t = arr + t
        t_plus_arr = t + arr
        self.assertIsInstance(arr_plus_t, tf.Tensor)
        self.assertIsInstance(t_plus_arr, tf.Tensor)
        self.assertEqual(10.0, arr_plus_t.numpy())
        self.assertEqual(10.0, t_plus_arr.numpy())

    def testTensorTFNPOp(self):
        if False:
            i = 10
            return i + 15
        t = tf.constant(10.0)
        sq = np.square(t)
        self.assertIsInstance(sq, np.ndarray)
        self.assertEqual(100.0, sq)

    def testTFNPArrayTFOpInterop(self):
        if False:
            return 10
        arr = np.asarray(10.0)
        sq = tf.square(arr)
        self.assertIsInstance(sq, tf.Tensor)
        self.assertEqual(100.0, sq.numpy())

    def testTFNPArrayNPOpInterop(self):
        if False:
            return 10
        arr = np.asarray([10.0])
        sq = onp.square(arr)
        self.assertIsInstance(sq, onp.ndarray)
        self.assertEqual(100.0, sq[0])

    def testArrayModule(self):
        if False:
            while True:
                i = 10
        self.skipTest("Tensor doesn't have __array_module__")
        arr = np.asarray([10])
        module = arr.__array_module__((tf.Tensor,))
        self.assertIs(module, tf.experimental.numpy)

        class Dummy:
            pass
        module = arr.__array_module__((tf.Tensor, Dummy))
        self.assertIs(module, NotImplemented)

    def testDistStratInterop(self):
        if False:
            return 10
        strategy = tf.distribute.MirroredStrategy(devices=['CPU:0', 'CPU:1', 'CPU:2'])
        multiplier = np.asarray(5.0)

        @tf.function
        def run():
            if False:
                print('Hello World!')
            ctx = tf.distribute.get_replica_context()
            val = np.asarray(ctx.replica_id_in_sync_group)
            return val * multiplier
        distributed_values = strategy.run(run)
        reduced = strategy.reduce(tf.distribute.ReduceOp.SUM, distributed_values, axis=None)
        values = strategy.experimental_local_results(distributed_values)
        self.assertLen(values, 3)
        self.assertIsInstance(values[0], np.ndarray)
        self.assertIsInstance(values[1], np.ndarray)
        self.assertIsInstance(values[2], np.ndarray)
        self.assertAllClose(values[0], 0)
        self.assertAllClose(values[1], 5)
        self.assertAllClose(values[2], 10)
        self.assertAllClose(reduced, 15)

    @test_util.disable_tfrt('b/180469928')
    def testPyFuncInterop(self):
        if False:
            for i in range(10):
                print('nop')

        def py_func_fn(a, b):
            if False:
                i = 10
                return i + 15
            return a + b

        @tf.function
        def fn(a, b):
            if False:
                return 10
            result = tf.py_function(py_func_fn, [a, b], a.dtype)
            return np.asarray(result)
        a = np.asarray(1.0)
        b = np.asarray(2.0)
        result = fn(a, b)
        self.assertIsInstance(result, np.ndarray)
        self.assertAllClose(result, 3.0)

    def testDatasetInterop(self):
        if False:
            i = 10
            return i + 15
        values = [1, 2, 3, 4, 5, 6]
        values_as_array = np.asarray(values)
        dataset = tf.data.Dataset.from_tensors(values_as_array)
        for (value, value_from_dataset) in zip([values_as_array], dataset):
            self.assertIsInstance(value_from_dataset, np.ndarray)
            self.assertAllEqual(value_from_dataset, value)
        dataset = tf.data.Dataset.from_tensor_slices(values_as_array)
        for (value, value_from_dataset) in zip(values, dataset):
            self.assertIsInstance(value_from_dataset, np.ndarray)
            self.assertAllEqual(value_from_dataset, value)
        dataset = dataset.map(lambda x: np.add(x, 1))
        for (value, value_from_dataset) in zip(values, dataset):
            self.assertIsInstance(value_from_dataset, np.ndarray)
            self.assertAllEqual(value_from_dataset, value + 1)
        dataset = tf.data.Dataset.from_tensor_slices(values_as_array).batch(2)
        for (value, value_from_dataset) in zip([[1, 2], [3, 4], [5, 6]], dataset):
            self.assertIsInstance(value_from_dataset, np.ndarray)
            self.assertAllEqual(value_from_dataset, value)

    def testKerasInterop(self):
        if False:
            print('Hello World!')
        inputs = tf.keras.layers.Input(shape=(10,))
        output_layer = tf.keras.layers.Lambda(np.square)(inputs)
        model = tf.keras.Model([inputs], output_layer)
        values = onp.arange(10, dtype=onp.float32).reshape((1, 10))
        values_as_array = np.asarray(values)
        result = model(values)
        self.assertIsInstance(result, np.ndarray)
        self.assertAllClose(result, onp.square(values))
        result = model(values_as_array)
        self.assertIsInstance(result, np.ndarray)
        self.assertAllClose(result, onp.square(values))

    def testKerasInteropSequential(self):
        if False:
            print('Hello World!')

        class ProjectionLayer(tf.keras.layers.Layer):
            """Linear projection layer using TF NumPy."""

            def __init__(self, units):
                if False:
                    i = 10
                    return i + 15
                super(ProjectionLayer, self).__init__()
                self._units = units

            def build(self, input_shape):
                if False:
                    while True:
                        i = 10
                stddev = np.sqrt(self._units).astype(np.float32)
                initial_value = np.random.randn(input_shape[1], self._units).astype(np.float32) / stddev
                self.w = tf.Variable(initial_value, trainable=True)

            def call(self, inputs):
                if False:
                    i = 10
                    return i + 15
                return np.matmul(inputs, self.w)
        model = tf.keras.Sequential([tf.keras.layers.Dense(100), ProjectionLayer(2)])
        output = model.call(np.random.randn(10, 100).astype(np.float32))
        self.assertIsInstance(output, np.ndarray)
        dense_layer = tf.keras.layers.Dense(100)
        output = dense_layer(np.random.randn(10, 100).astype(np.float32))

    def testPForInterop(self):
        if False:
            i = 10
            return i + 15

        def outer_product(a):
            if False:
                print('Hello World!')
            return np.tensordot(a, a, 0)
        batch_size = 100
        a = np.ones((batch_size, 32, 32))
        c = tf.vectorized_map(outer_product, a)
        self.assertIsInstance(c, np.ndarray)
        self.assertEqual(c.shape, (batch_size, 32, 32, 32, 32))
        c = tf.vectorized_map(lambda x: x.T, a)
        self.assertIsInstance(c, np.ndarray)
        self.assertEqual(c.shape, (batch_size, 32, 32))

    def testJacobian(self):
        if False:
            while True:
                i = 10
        with tf.GradientTape() as g:
            x = np.asarray([1.0, 2.0])
            y = np.asarray([3.0, 4.0])
            g.watch(x)
            g.watch(y)
            z = x * x * y
        jacobian = g.jacobian(z, [x, y])
        answer = [tf.linalg.diag(2 * x * y), tf.linalg.diag(x * x)]
        self.assertIsInstance(jacobian[0], np.ndarray)
        self.assertIsInstance(jacobian[1], np.ndarray)
        self.assertAllClose(jacobian, answer)

    def testBatchJacobian(self):
        if False:
            for i in range(10):
                print('nop')
        with tf.GradientTape() as g:
            x = np.asarray([[1.0, 2.0], [3.0, 4.0]])
            y = np.asarray([[3.0, 4.0], [5.0, 6.0]])
            g.watch(x)
            g.watch(y)
            z = x * x * y
        batch_jacobian = g.batch_jacobian(z, x)
        answer = tf.stack([tf.linalg.diag(2 * x[0] * y[0]), tf.linalg.diag(2 * x[1] * y[1])])
        self.assertIsInstance(batch_jacobian, np.ndarray)
        self.assertAllClose(batch_jacobian, answer)

    def testForwardprop(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.asarray([1.0, 2.0])
        xt = np.asarray([3.0, 4.0])
        with tf.autodiff.ForwardAccumulator(x, xt) as acc:
            y = x * 2.0
        yt = acc.jvp(y)
        self.assertIsInstance(yt, np.ndarray)
        self.assertAllClose([6.0, 8.0], yt)
        z = np.asarray([1.0])
        self.assertIsNone(acc.jvp(z))

    def testMapFn(self):
        if False:
            while True:
                i = 10
        x = np.asarray([1.0, 2.0])
        mapped_x = tf.map_fn(lambda x: (x[0] + 1, x[1] + 1), (x, x))
        self.assertIsInstance(mapped_x[0], np.ndarray)
        self.assertIsInstance(mapped_x[1], np.ndarray)
        self.assertAllClose(mapped_x[0], [2.0, 3.0])
        self.assertAllClose(mapped_x[1], [2.0, 3.0])

class FunctionTest(InteropTest):

    def testFunctionInterop(self):
        if False:
            i = 10
            return i + 15
        x = np.asarray(3.0)
        y = np.asarray(2.0)
        add = lambda x, y: x + y
        add_fn = tf.function(add)
        raw_result = add(x, y)
        fn_result = add_fn(x, y)
        self.assertIsInstance(raw_result, np.ndarray)
        self.assertIsInstance(fn_result, np.ndarray)
        self.assertAllClose(raw_result, fn_result)

    def testLen(self):
        if False:
            i = 10
            return i + 15

        @tf.function(autograph=False)
        def f(x):
            if False:
                while True:
                    i = 10
            return len(np.where(x)[0])
        t = np.asarray([True, False, True])
        with self.assertRaises(TypeError):
            f(t)

    def testIter(self):
        if False:
            i = 10
            return i + 15

        @tf.function
        def f(x):
            if False:
                i = 10
                return i + 15
            (y, z) = x
            return (y, z)
        with self.assertRaises(TypeError):
            f(np.asarray([3, 4]))

    def testIndex(self):
        if False:
            return 10

        @tf.function
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return [0, 1][x]
        with self.assertRaises(TypeError):
            f(np.asarray([1]))

class VariableTest(InteropTest):

    def test(self):
        if False:
            i = 10
            return i + 15
        tf_var = tf.Variable(2.0)
        value = np.square(tf_var)
        self.assertIsInstance(value, np.ndarray)
        self.assertAllClose(4.0, value)
        with tf.control_dependencies([tf_var.assign_add(value)]):
            tf_var_value = tf_var.read_value()
        self.assertAllClose(6.0, tf_var_value)
if __name__ == '__main__':
    ops.set_dtype_conversion_mode('legacy')
    np_math_ops.enable_numpy_methods_on_tensor()
    tf.compat.v1.enable_eager_execution()
    tf.test.main()