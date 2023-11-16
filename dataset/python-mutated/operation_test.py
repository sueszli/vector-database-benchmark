import numpy as np
from keras import backend
from keras import testing
from keras.backend.common import keras_tensor
from keras.ops import numpy as knp
from keras.ops import operation

class OpWithMultipleInputs(operation.Operation):

    def call(self, x, y, z=None):
        if False:
            print('Hello World!')
        return 3 * z + x + 2 * y

    def compute_output_spec(self, x, y, z=None):
        if False:
            i = 10
            return i + 15
        return keras_tensor.KerasTensor(x.shape, x.dtype)

class OpWithMultipleOutputs(operation.Operation):

    def call(self, x):
        if False:
            return 10
        return (x, x + 1)

    def compute_output_spec(self, x):
        if False:
            while True:
                i = 10
        return (keras_tensor.KerasTensor(x.shape, x.dtype), keras_tensor.KerasTensor(x.shape, x.dtype))

class OpWithCustomConstructor(operation.Operation):

    def __init__(self, alpha, mode='foo'):
        if False:
            return 10
        super().__init__()
        self.alpha = alpha
        self.mode = mode

    def call(self, x):
        if False:
            while True:
                i = 10
        if self.mode == 'foo':
            return x
        return self.alpha * x

    def compute_output_spec(self, x):
        if False:
            i = 10
            return i + 15
        return keras_tensor.KerasTensor(x.shape, x.dtype)

class OperationTest(testing.TestCase):

    def test_symbolic_call(self):
        if False:
            print('Hello World!')
        x = keras_tensor.KerasTensor(shape=(2, 3), name='x')
        y = keras_tensor.KerasTensor(shape=(2, 3), name='y')
        z = keras_tensor.KerasTensor(shape=(2, 3), name='z')
        op = OpWithMultipleInputs(name='test_op')
        self.assertEqual(op.name, 'test_op')
        out = op(x, y, z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)
        op = OpWithMultipleInputs(name='test_op')
        out = op(x=x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)
        op = OpWithMultipleInputs(name='test_op')
        out = op(x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.input, [x, y, z])
        self.assertEqual(op.output, out)
        prev_out = out
        out = op(x, y=y, z=z)
        self.assertIsInstance(out, keras_tensor.KerasTensor)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 2)
        self.assertEqual(op.output, prev_out)
        op = OpWithMultipleOutputs()
        out = op(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], keras_tensor.KerasTensor)
        self.assertIsInstance(out[1], keras_tensor.KerasTensor)
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 3))
        self.assertEqual(len(op._inbound_nodes), 1)
        self.assertEqual(op.output, list(out))

    def test_eager_call(self):
        if False:
            for i in range(10):
                print('nop')
        x = knp.ones((2, 3))
        y = knp.ones((2, 3))
        z = knp.ones((2, 3))
        op = OpWithMultipleInputs(name='test_op')
        self.assertEqual(op.name, 'test_op')
        out = op(x, y, z)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))
        out = op(x=x, y=y, z=z)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))
        out = op(x, y=y, z=z)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2, 3)))
        op = OpWithMultipleOutputs()
        out = op(x)
        self.assertEqual(len(out), 2)
        self.assertTrue(backend.is_tensor(out[0]))
        self.assertTrue(backend.is_tensor(out[1]))
        self.assertAllClose(out[0], np.ones((2, 3)))
        self.assertAllClose(out[1], np.ones((2, 3)) + 1)

    def test_serialization(self):
        if False:
            while True:
                i = 10
        op = OpWithMultipleOutputs(name='test_op')
        config = op.get_config()
        self.assertEqual(config, {'name': 'test_op'})
        op = OpWithMultipleOutputs.from_config(config)
        self.assertEqual(op.name, 'test_op')

    def test_autoconfig(self):
        if False:
            for i in range(10):
                print('nop')
        op = OpWithCustomConstructor(alpha=0.2, mode='bar')
        config = op.get_config()
        self.assertEqual(config, {'alpha': 0.2, 'mode': 'bar'})
        revived = OpWithCustomConstructor.from_config(config)
        self.assertEqual(revived.get_config(), config)

    def test_input_conversion(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.ones((2,))
        y = np.ones((2,))
        z = knp.ones((2,))
        if backend.backend() == 'torch':
            z = z.cpu()
        op = OpWithMultipleInputs()
        out = op(x, y, z)
        self.assertTrue(backend.is_tensor(out))
        self.assertAllClose(out, 6 * np.ones((2,)))

    def test_valid_naming(self):
        if False:
            return 10
        OpWithMultipleOutputs(name='test_op')
        with self.assertRaisesRegex(ValueError, 'must be a string and cannot contain character `/`.'):
            OpWithMultipleOutputs(name='test/op')