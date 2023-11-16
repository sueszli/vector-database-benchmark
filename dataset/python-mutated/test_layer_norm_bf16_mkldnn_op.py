import unittest
from functools import reduce
from operator import mul
import numpy as np
from op_test import _set_use_system_allocator, convert_float_to_uint16
from test_layer_norm_mkldnn_op import TestLayerNormMKLDNNOp, _reference_layer_norm_naive
from paddle import base, enable_static
from paddle.base import core
np.random.random(123)
_set_use_system_allocator(True)

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestLayerNormBF16MKLDNNOp(TestLayerNormMKLDNNOp):

    def __assert_close(self, tensor, np_array, msg, rtol=0.02, atol=2):
        if False:
            i = 10
            return i + 15
        np.testing.assert_allclose(np.array(tensor), np_array, rtol=rtol, atol=atol, err_msg=msg)

    def check_forward(self, shape, begin_norm_axis, with_scale_bias=True, with_is_test=False):
        if False:
            print('Hello World!')
        epsilon = 1e-05
        x_shape = shape
        D = reduce(mul, x_shape[begin_norm_axis:len(x_shape)], 1)
        scale_shape = [D]
        np.random.seed(123)
        x = np.random.random_sample(x_shape).astype(np.float32)
        x_bf16 = convert_float_to_uint16(x)
        if with_scale_bias:
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
        else:
            scale = np.array([])
            bias = np.array([])
        (y, mean, variance) = _reference_layer_norm_naive(x, scale, bias, epsilon, begin_norm_axis)
        y_bf16 = convert_float_to_uint16(y)
        var_dict = locals()
        var_names = ['x_bf16', 'mean', 'variance', 'y_bf16']
        if with_scale_bias:
            var_names.append('scale')
            var_names.append('bias')
        ground_truth = {name: var_dict[name] for name in var_names}
        program = base.Program()
        with base.program_guard(program):
            block = program.global_block()
            for name in ground_truth:
                if name == 'x_bf16' or name == 'y_bf16':
                    block.create_var(name=name, dtype='uint16', shape=ground_truth[name].shape)
                else:
                    block.create_var(name=name, dtype='float32', shape=ground_truth[name].shape)
            inputs = {'X': block.var('x_bf16')}
            if with_scale_bias:
                inputs['Scale'] = block.var('scale')
                inputs['Bias'] = block.var('bias')
            block.append_op(type='layer_norm', inputs=inputs, outputs={'Y': block.var('y_bf16'), 'Mean': block.var('mean'), 'Variance': block.var('variance')}, attrs={'epsilon': epsilon, 'begin_norm_axis': begin_norm_axis, 'use_mkldnn': True, 'is_test': with_is_test})
            exe = base.Executor(core.CPUPlace())
            input_list = ['x_bf16']
            if with_scale_bias:
                input_list.append('scale')
                input_list.append('bias')
            out = exe.run(program, feed={name: var_dict[name] for name in input_list}, fetch_list=['y_bf16', 'mean', 'variance'])
            self.__assert_close(y_bf16, out[0], 'y_bf16', 2)
            if not with_is_test:
                self.__assert_close(mean, out[1], 'mean')
                self.__assert_close(variance, out[2], 'variance', 0.001)

    def test_check_forward_with_is_test(self):
        if False:
            return 10
        self.check_forward(shape=[2, 3, 4, 5], begin_norm_axis=3, with_is_test=True)

    def test_check_forward_with_scale_and_bias(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_forward_without_scale_and_bias(self):
        if False:
            i = 10
            return i + 15
        pass
if __name__ == '__main__':
    enable_static()
    unittest.main()