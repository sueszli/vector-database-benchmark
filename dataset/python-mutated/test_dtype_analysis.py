from itertools import product
from typing import Tuple
from unittest.case import expectedFailure
import torch
from torch import complex32, float32, float64, int32, int64
from torch.jit._passes import _property_propagation
from torch.testing._internal.common_methods_invocations import SampleInput, sample_inputs_adaptive_avg_pool2d, sample_inputs_conv2d
from torch.testing._internal.common_utils import set_default_dtype, first_sample
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.jit_metaprogramming_utils import create_traced_fn
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db
'\nDtype Analysis relies on symbolic shape analysis, which is still in beta\n'
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')
custom_rules_works_list = {'nn.functional.adaptive_avg_pool1d', 'nn.functional.adaptive_avg_pool2d', 'nn.functional.adaptive_avg_pool3d', 'nn.functional.adaptive_max_pool1d', 'nn.functional.adaptive_max_pool2d', 'avg_pool1d', 'avg_pool3d', 'conv_transpose2d', 'conv1d', 'conv2d', 'hardswish', 'avg_pool2d', 'max_pool1d', 'max_pool2d', 'max_pool3d', 'nn.functional.prelu', 'batch_norm'}
custom_rules_expected_failure_list = {'nn.functional.adaptive_max_pool3d'}
custom_rules_not_tested_list = ['conv3d', 'conv_tbc', 'conv_transpose1d', 'conv_transpose3d', 'convolution', '_convolution', 'max_unpool2d', 'max_unpool3d', 'reflection_pad1d', 'reflection_pad2d', 'reflection_pad3d', 'replication_pad1d', 'replication_pad2d', 'replication_pad3d', 'upsample_bilinear2d', 'upsample_linear1d', 'upsample_nearest1d', 'upsample_nearest2d', 'upsample_nearest3d', 'upsample_trilinear3d', 'flatten']

class TestDtypeBase(JitTestCase):
    SCALAR = 'SCALAR'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.prev_symbolic_shapes_test_enabled = torch._C._jit_symbolic_shapes_test_mode_enabled()
        torch._C._jit_set_symbolic_shapes_test_mode(True)

    def tearDown(self):
        if False:
            print('Hello World!')
        torch._C._jit_set_symbolic_shapes_test_mode(self.prev_symbolic_shapes_test_enabled)

    @staticmethod
    def node_output_dtypes(graph):
        if False:
            print('Hello World!')
        dtypes = []
        for out in graph.outputs():
            if isinstance(out.type(), torch._C.TensorType):
                dtypes.append(out.type().dtype())
            else:
                dtypes.append(None)
        return dtypes

    @staticmethod
    def node_output_dtype_single(graph):
        if False:
            while True:
                i = 10
        dtypes = TestDtypeBase.node_output_dtypes(graph)
        assert len(dtypes) == 1
        return dtypes[0]

    def prop_dtype_on_graph(self, graph, example_inputs):
        if False:
            for i in range(10):
                print('nop')
        torch._C._jit_pass_erase_shape_information(graph)
        _property_propagation.apply_input_props_using_example(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        torch._C._jit_pass_propagate_dtype(graph)

    def assert_dtype_equal(self, fn, in_shapes, in_dtypes):
        if False:
            i = 10
            return i + 15
        inputs = [self.get_rand_tensor(s, d) for (s, d) in zip(in_shapes, in_dtypes)]
        try:
            self.assert_dtype_equal_custom_args(fn, inputs)
        except Exception as e:
            fail_text = f'Failed for shapes {in_shapes}, and dtypes {in_dtypes}'
            raise AssertionError(fail_text) from e

    def assert_dtype_equal_custom_args(self, fn, args):
        if False:
            for i in range(10):
                print('nop')
        try:
            expected_res = fn(*args)
        except RuntimeError as e:
            return
        expected_dtype = expected_res.dtype
        graph = torch.jit.script(fn).graph
        self.prop_dtype_on_graph(graph, args)
        actual_dtype = self.node_output_dtype_single(graph)
        self.assertEqual(actual_dtype, expected_dtype, 'Failed Verification')

    def get_rand_tensor(self, shape, dtype):
        if False:
            return 10
        if shape is self.SCALAR:
            if dtype is float32:
                return 1.1
            elif dtype is int64:
                return 2
            else:
                raise RuntimeError('Testing of scalars only supported for fp32 and int64')
        if dtype in (int32, int64):
            rand_tensor = torch.randint(0, 10, shape, dtype=dtype)
        else:
            rand_tensor = torch.rand(shape, dtype=dtype)
        self.assertEqual(rand_tensor.dtype, dtype)
        return rand_tensor

class TestDtypeAnalysis(TestDtypeBase):

    def test_unary(self):
        if False:
            print('Hello World!')

        def relu_inplace(x):
            if False:
                print('Hello World!')
            return x.relu_()

        def log(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.log(x)
        functions = [relu_inplace, log]
        input_shapes = [((2, 2),), ((0, 2),), ((),)]
        input_dtypes = [(float32,), (int64,), (complex32,)]
        for (fn, in_shapes, in_dtypes) in product(functions, input_shapes, input_dtypes):
            self.assert_dtype_equal(fn, in_shapes, in_dtypes)

    def test_binary_tensors(self):
        if False:
            for i in range(10):
                print('nop')

        def add(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y

        def div(x, y):
            if False:
                print('Hello World!')
            return x / y
        functions = [add, div]
        input_shapes = [((1, 1, 2), (1, 2)), ((), (1, 2)), ((1, 2), ()), ((2, 0, 3), (1, 3)), ((), ())]
        input_dtypes = [(float32, float32), (int32, int64), (float32, int32), (int64, float32), (float64, complex32)]
        for (fn, in_shapes, in_dtypes) in product(functions, input_shapes, input_dtypes):
            self.assert_dtype_equal(fn, in_shapes, in_dtypes)

    def test_binary_scalar(self):
        if False:
            print('Hello World!')
        input_shapes = [((2, 2), self.SCALAR), ((), self.SCALAR)]
        input_dtypes = [(float32, float32), (int32, int64), (int32, float32)]
        with set_default_dtype(float32):
            for (in_shapes, in_dtypes) in product(input_shapes, input_dtypes):
                scalar_type = in_dtypes[1]
                if scalar_type == float32:

                    def add(x, y: float):
                        if False:
                            for i in range(10):
                                print('nop')
                        return x + y
                else:

                    def add(x, y: int):
                        if False:
                            for i in range(10):
                                print('nop')
                        return x + y
                self.assert_dtype_equal(add, in_shapes, in_dtypes)

    def test_custom_rules(self):
        if False:
            print('Hello World!')

        def conv2d_fn(input, weight, bias):
            if False:
                print('Hello World!')
            return torch.nn.functional.conv2d(input, weight, bias)

        def adaptive_avg_pool2d_fn(input, output_size: Tuple[int]):
            if False:
                while True:
                    i = 10
            return torch._C._nn.adaptive_avg_pool2d(input, output_size)
        for (fn, inputs_fn) in ((conv2d_fn, sample_inputs_conv2d), (adaptive_avg_pool2d_fn, sample_inputs_adaptive_avg_pool2d)):
            for dtype in (torch.int8, torch.float64):
                sample_input: SampleInput = list(inputs_fn(None, 'cpu', dtype, False))[-1]
                input_args = [sample_input.input, *sample_input.args]
                self.assert_dtype_equal_custom_args(fn, input_args)

    def test_conv_no_mixed_args(self):
        if False:
            i = 10
            return i + 15

        def conv2d_fn(input, weight, bias):
            if False:
                i = 10
                return i + 15
            return torch.nn.functional.conv2d(input, weight, bias)
        conv_ins = sample_inputs_conv2d(None, 'cpu', torch.float, False)
        conv_in = list(conv_ins)[-1]
        (weight, bias) = conv_in.args
        weight = weight.type(torch.long)
        with self.assertRaises(RuntimeError):
            conv2d_fn(conv_in.input, weight, bias)
        graph = torch.jit.script(conv2d_fn).graph
        self.prop_dtype_on_graph(graph, [conv_in.input, weight, bias])
        actual_dtype = self.node_output_dtype_single(graph)
        self.assertEqual(actual_dtype, None)

    def test_combined(self):
        if False:
            return 10

        def func(input, weight, bias, y):
            if False:
                for i in range(10):
                    print('nop')
            conv_out = torch.nn.functional.conv2d(input, weight, bias)
            conv_2 = conv_out + y
            flattened = torch.flatten(conv_2, start_dim=2)
            add_res = flattened + y
            return add_res
        conv_ins = sample_inputs_conv2d(None, 'cpu', torch.int8, False)
        conv_in = list(conv_ins)[-1]
        y_val = torch.rand((1,), dtype=torch.float32)
        input_args = [conv_in.input, *conv_in.args, y_val]
        self.assert_dtype_equal_custom_args(func, input_args)

class TestDtypeCustomRules(TestDtypeBase):

    def assert_output_dtype_equal(self, expected_res, prop_graph):
        if False:
            while True:
                i = 10
        actual_dtype = self.node_output_dtypes(prop_graph)
        if len(actual_dtype) == 1:
            self.assert_tensor_dtype_equal(expected_res, actual_dtype[0])
        else:
            self.assertEqual(len(expected_res), len(actual_dtype))
            for (expected, actual) in zip(expected_res, actual_dtype):
                self.assert_tensor_dtype_equal(expected, actual)

    def assert_tensor_dtype_equal(self, tensor_output, graph_dtype):
        if False:
            while True:
                i = 10
        if not isinstance(tensor_output, torch.Tensor):
            return
        self.assertEqual(tensor_output.dtype, graph_dtype)

    def custom_rules_test_base(self, device, dtype, op, allow_eager_fail=False):
        if False:
            print('Hello World!')
        try:
            samples = op.sample_inputs(device, dtype, requires_grad=False)
            sample_input = first_sample(self, samples)
            input_args = [sample_input.input, *sample_input.args]
            expected_res = op(*input_args, **sample_input.kwargs)
        except Exception as e:
            if allow_eager_fail:
                return
            else:
                raise e
        func = op.get_op()
        traced_fn = create_traced_fn(self, func)
        traced_fn(sample_input.input, *sample_input.args, **sample_input.kwargs)
        graph = traced_fn.graph
        input_tensors = [t for t in input_args if isinstance(t, torch.Tensor)]
        input_tensors += [v for v in sample_input.kwargs.values() if isinstance(v, torch.Tensor)]
        self.prop_dtype_on_graph(graph, input_tensors)
        self.assert_output_dtype_equal(expected_res, graph)

    @ops([op for op in op_db if op.aten_name in custom_rules_works_list])
    def test_custom_rules(self, device, dtype, op):
        if False:
            return 10
        self.custom_rules_test_base(device, dtype, op)

    @ops([op for op in op_db if op.aten_name in custom_rules_works_list])
    def test_custom_rules_ints(self, device, dtype, op):
        if False:
            return 10
        if dtype == torch.float32:
            dtype = torch.int32
        else:
            dtype = torch.int64
        self.custom_rules_test_base(device, dtype, op, allow_eager_fail=True)

    @expectedFailure
    @ops([op for op in op_db if op.aten_name in custom_rules_expected_failure_list])
    def test_custom_rules_expected_failure(self, device, dtype, op):
        if False:
            for i in range(10):
                print('nop')
        self.custom_rules_test_base(device, dtype, op)
TestDtypeCustomRulesCPU = None
instantiate_device_type_tests(TestDtypeCustomRules, globals(), only_for=('cpu',))