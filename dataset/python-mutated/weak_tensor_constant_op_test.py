"""Tests for tensorflow.python.framework.constant_op."""
from absl.testing import parameterized
import numpy as np
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import weak_tensor_ops
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.platform import test
_DTYPE_PROMO_RES = flexible_dtypes._BINARY_DTYPE_RES_FULL
_get_test_input_for_op = weak_tensor_test_util.get_test_input_for_op
constant_op_allowed_promos_list = []
for key in _DTYPE_PROMO_RES:
    if key[0] in (dtypes.bool, dtypes.string):
        continue
    for (k, v) in _DTYPE_PROMO_RES[key].items():
        if not k[1] and k == v[0]:
            constant_op_allowed_promos_list.append((key, k[0]))

class WeakTensorConstantOpTest(test.TestCase, parameterized.TestCase):

    @parameterized.parameters(dtypes.bfloat16, dtypes.complex128, dtypes.complex64, dtypes.double, dtypes.float16, dtypes.float32, dtypes.float64, dtypes.half, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.int8, dtypes.qint16, dtypes.qint32, dtypes.qint8, dtypes.quint16, dtypes.quint8, dtypes.uint16, dtypes.uint32, dtypes.uint64, dtypes.uint8)
    def test_convert_string_to_number(self, dtype):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            constant_op.constant('hello', dtype)

    @parameterized.parameters(dtypes.complex128, dtypes.complex64, dtypes.double, dtypes.float16, dtypes.float32, dtypes.float64, dtypes.half, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.int8, dtypes.uint16, dtypes.uint32, dtypes.uint64, dtypes.uint8)
    @test_util.run_in_graph_and_eager_modes
    def test_constant_fixed_dtype_inputs(self, dtype):
        if False:
            i = 10
            return i + 15
        np_output = constant_op.constant(np.array(1, dtype=dtype.as_numpy_dtype))
        self.assertEqual(np_output.dtype, dtype)
        self.assertIsInstance(np_output, tensor.Tensor)
        tensor_output = constant_op.constant(1, dtype=dtype)
        self.assertEqual(tensor_output.dtype, dtype)
        self.assertIsInstance(tensor_output, tensor.Tensor)

    @test_util.run_in_graph_and_eager_modes
    def test_constant_weak_tensor_creation(self):
        if False:
            for i in range(10):
                print('nop')
        a = constant_op.constant(1)
        self.assertIsInstance(a, WeakTensor)
        a = constant_op.constant(1, dtypes.int32)
        self.assertIsInstance(a, tensor.Tensor)
        a = ops.convert_to_tensor(1)
        self.assertIsInstance(a, tensor.Tensor)
        a = ops.convert_to_tensor(WeakTensor.from_tensor(a))
        self.assertIsInstance(a, tensor.Tensor)

    @parameterized.parameters(constant_op_allowed_promos_list)
    def test_constant_allowed_promotion(self, input_dtype, dtype_arg):
        if False:
            for i in range(10):
                print('nop')
        input_list = _get_test_input_for_op(5, input_dtype)
        for test_input in input_list:
            res = constant_op.constant(test_input, dtype_arg)
            self.assertIsInstance(res, tensor.Tensor)
            self.assertEqual(res.dtype, dtype_arg)

    def test_constant_unallowed_promotion(self):
        if False:
            i = 10
            return i + 15
        a = constant_op.constant(5, dtypes.float32)
        with self.assertRaisesRegex(TypeError, 'Expected tensor 5.0 with dtype tf.int32, but got dtype tf.float32.'):
            _ = constant_op.constant(a, dtypes.int32)
        with self.assertRaisesRegex(TypeError, 'Cannot convert 5.0 to EagerTensor of dtype int32'):
            _ = constant_op.constant(5.0, dtypes.int32)
        a = constant_op.constant(5.0)
        with self.assertRaisesRegex(TypeError, 'Expected tensor 5.0 with dtype tf.int32, but got dtype tf.float32.'):
            _ = constant_op.constant(a, dtypes.int32)

    @test_util.run_in_graph_and_eager_modes
    def test_constant_python_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        a = constant_op.constant(1)
        self.assertIsInstance(a, WeakTensor)
        self.assertEqual(a.dtype, dtypes.int32)
        a = constant_op.constant([1, 2, 3])
        self.assertIsInstance(a, WeakTensor)
        self.assertEqual(a.dtype, dtypes.int32)
        a = constant_op.constant([1, 2.2])
        self.assertIsInstance(a, WeakTensor)
        self.assertEqual(a.dtype, dtypes.float32)
        a = constant_op.constant([1.0, 2.0, 3.0])
        self.assertIsInstance(a, WeakTensor)
        self.assertEqual(a.dtype, dtypes.float32)
        a = constant_op.constant([1j, 2j, 3j])
        self.assertIsInstance(a, WeakTensor)
        self.assertEqual(a.dtype, dtypes.complex128)

    @parameterized.parameters(dtypes.complex128, dtypes.complex64, dtypes.double, dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.half, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.int8, dtypes.uint16, dtypes.uint32, dtypes.uint64, dtypes.uint8)
    def test_constant_python_int_with_dtype_arg(self, dtype):
        if False:
            i = 10
            return i + 15
        a = constant_op.constant([1, 2, 3], dtype)
        self.assertIsInstance(a, tensor.Tensor)
        self.assertEqual(a.dtype, dtype)

    @parameterized.parameters(dtypes.complex128, dtypes.complex64, dtypes.double, dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.half)
    def test_constant_python_float_with_dtype_arg(self, dtype):
        if False:
            print('Hello World!')
        a = constant_op.constant([1.0, 2.0, 3.0], dtype)
        self.assertIsInstance(a, tensor.Tensor)
        self.assertEqual(a.dtype, dtype)

    def test_constant_python_complex_with_dtype_arg(self):
        if False:
            return 10
        a = constant_op.constant([1j, 2j, 3j], dtypes.complex64)
        self.assertIsInstance(a, tensor.Tensor)
        self.assertEqual(a.dtype, dtypes.complex64)
        a = constant_op.constant([1j, 2j, 3j], dtypes.complex128)
        self.assertIsInstance(a, tensor.Tensor)
        self.assertEqual(a.dtype, dtypes.complex128)

    def test_constant_value_weak_tensor(self):
        if False:
            i = 10
            return i + 15
        a = constant_op.constant(1)
        self.assertEqual(tensor_util.constant_value(a), 1)
        a = constant_op.constant([1, 2, 3])
        self.assertAllEqual(tensor_util.constant_value(a), [1, 2, 3])

    def _make_graph_def(self, text):
        if False:
            for i in range(10):
                print('nop')
        ret = graph_pb2.GraphDef()
        text_format.Parse(text, ret)
        return ret

    def test_eager_const_xla(self):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function(jit_compile=True)
        def f_using_eagerconst(x):
            if False:
                i = 10
                return i + 15
            graph_def = self._make_graph_def("\n         node { name: 'x' op: 'Const'\n           attr { key: 'dtype' value { type: DT_FLOAT } }\n           attr { key: 'value' value { tensor {\n             dtype: DT_FLOAT tensor_shape {} float_val: NaN } } } }\n         node { name: 'const' op: '_EagerConst' input: 'x:0'\n                attr { key: 'T' value { type: DT_FLOAT } }}")
            x_id = importer.import_graph_def(graph_def, input_map={'x:0': x}, return_elements=['const'], name='import')[0].outputs[0]
            return x_id
        self.assertAllClose(3.14, f_using_eagerconst(constant_op.constant(3.14)))

    def test_np_array_memory_not_shared(self):
        if False:
            print('Hello World!')
        for _ in range(10000):
            x = np.arange(10)
            xt = constant_op.constant(x)
            x[3] = 42
            self.assertEqual(xt.numpy()[3], 3)

    def test_eager_const_grad_error(self):
        if False:
            while True:
                i = 10

        @def_function.function
        def f_using_eagerconst():
            if False:
                while True:
                    i = 10
            x = constant_op.constant(1.0)
            graph_def = self._make_graph_def("\n         node { name: 'x' op: 'Placeholder'\n                attr { key: 'dtype' value { type: DT_FLOAT } }}\n         node { name: 'const' op: '_EagerConst' input: 'x:0'\n                attr { key: 'T' value { type: DT_FLOAT } }}")
            x_id = importer.import_graph_def(graph_def, input_map={'x:0': x}, return_elements=['const'], name='import')[0].outputs[0]
            gradients_impl.gradients(x_id, x)
            return x_id
        with self.assertRaisesRegex(AssertionError, 'Please file a bug'):
            f_using_eagerconst()

    def test_eager_const_pfor(self):
        if False:
            return 10

        @def_function.function
        def f_using_eagerconst():
            if False:
                return 10

            def vec_fn(x):
                if False:
                    while True:
                        i = 10
                graph_def = self._make_graph_def("\n           node { name: 'x' op: 'Const'\n             attr { key: 'dtype' value { type: DT_FLOAT } }\n             attr { key: 'value' value { tensor {\n               dtype: DT_FLOAT tensor_shape {} float_val: 3.14 } } } }\n           node { name: 'const' op: '_EagerConst' input: 'x:0'\n                  attr { key: 'T' value { type: DT_FLOAT } }}")
                return importer.import_graph_def(graph_def, input_map={'x:0': x}, return_elements=['const'], name='import')[0].outputs[0]
            return control_flow_ops.vectorized_map(vec_fn, constant_op.constant([1.0, 2.0]), fallback_to_while_loop=False)
        self.assertAllClose([1.0, 2.0], f_using_eagerconst())
if __name__ == '__main__':
    ops.enable_eager_execution()
    ops.set_dtype_conversion_mode('all')
    test.main()