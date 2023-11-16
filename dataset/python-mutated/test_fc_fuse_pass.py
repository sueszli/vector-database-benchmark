import unittest
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import IgnoreReasons, PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestFcFusePass(PassAutoScanTest):
    """
    x_var   y_var(persistable)
      \\       /
         mul     bias_var(persistable)
          |
      mul_out_var  bias_var(persistable)
            \\        /
          elementwise_add
    """

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        before_num_ops = len(program_config.ops) + 2
        config = self.create_inference_config(use_gpu=False)
        yield (config, ['fc'], (1e-05, 1e-05))
        config = self.create_inference_config(use_gpu=True)
        yield (config, ['fc'], (1e-05, 1e-05))
        config = self.create_trt_inference_config()
        yield (config, ['fc'], (1e-05, 1e-05))

    def add_ignore_pass_case(self):
        if False:
            while True:
                i = 10

        def teller1(program_config, predictor_config):
            if False:
                return 10
            x_shape = list(program_config.inputs['mul_x'].shape)
            y_shape = list(program_config.weights['mul_y'].shape)
            bias_shape = program_config.weights['bias'].shape
            bias_shape = list(program_config.weights['bias'].shape)
            if predictor_config.tensorrt_engine_enabled():
                predictor_config.exp_disable_tensorrt_ops(['elementwise_add'])
            if bias_shape != [y_shape[-1]] and bias_shape != [1, y_shape[-1]]:
                return True
            return False

        def teller2(program_config, predictor_config):
            if False:
                return 10
            axis = program_config.ops[1].attrs['axis']
            if axis != -1 and axis != program_config.ops[0].attrs['x_num_col_dims']:
                return True
            return False
        self.add_ignore_check_case(teller1, IgnoreReasons.PASS_ACCURACY_ERROR, 'The pass output has diff while shape of bias is not [out_size] or [1, out_size].')
        self.add_ignore_check_case(teller2, IgnoreReasons.PASS_ACCURACY_ERROR, 'The pass output has diff while axis of elementwise_add is not -1.')

    def is_program_valid(self, prog_config):
        if False:
            i = 10
            return i + 15
        add_x_rank = prog_config.ops[0].attrs['x_num_col_dims'] + 1
        add_y_rank = len(prog_config.weights['bias'].shape)
        axis = prog_config.ops[1].attrs['axis']
        if add_x_rank == add_y_rank:
            if axis != -1 or axis != 0:
                return False
        return True

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=4))
        x_num_col_dims = draw(st.integers(min_value=1, max_value=len(x_shape) - 1))
        y_num_col_dims = 1
        y_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=2))
        y_shape[0] = int(np.prod(x_shape[x_num_col_dims:]))
        mul_out_shape = x_shape[:x_num_col_dims] + y_shape[1:]
        axis = draw(st.integers(min_value=-1, max_value=x_num_col_dims))
        if axis >= 0:
            max_bias_rank = x_num_col_dims + 1 - axis
            bias_rank = draw(st.integers(min_value=1, max_value=max_bias_rank))
            bias_shape = mul_out_shape[axis:axis + bias_rank]
        else:
            max_bias_rank = 1
            bias_rank = draw(st.integers(min_value=1, max_value=len(mul_out_shape)))
            bias_shape = mul_out_shape[-1 * bias_rank:]
        if draw(st.booleans()):
            broadcast_dims = draw(st.integers(min_value=1, max_value=bias_rank))
            for i in range(0, broadcast_dims):
                bias_shape[i] = 1
        has_relu = draw(st.booleans())
        mul_op = OpConfig('mul', inputs={'X': ['mul_x'], 'Y': ['mul_y']}, outputs={'Out': ['mul_out']}, x_num_col_dims=x_num_col_dims, y_num_col_dims=y_num_col_dims)
        add_op = OpConfig('elementwise_add', inputs={'X': ['mul_out'], 'Y': ['bias']}, outputs={'Out': ['add_out']}, axis=axis)
        ops = [mul_op, add_op]
        if has_relu:
            relu_op = OpConfig('relu', inputs={'X': ['add_out']}, outputs={'Out': ['relu_out']})
            ops.append(relu_op)
        program_config = ProgramConfig(ops=ops, weights={'mul_y': TensorConfig(shape=y_shape), 'bias': TensorConfig(shape=bias_shape)}, inputs={'mul_x': TensorConfig(shape=x_shape)}, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, max_examples=500, passes=['fc_fuse_pass'])
if __name__ == '__main__':
    unittest.main()