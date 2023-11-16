import unittest
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestAddLayernormXPUFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['add_layernorm_xpu'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15
        batch_size = draw(st.integers(min_value=1, max_value=50))
        x_shape = [batch_size, 16, 128]
        y_shape = x_shape
        axis = -1
        epsilon = draw(st.floats(min_value=1e-07, max_value=0.001))
        begin_norm_axis = 2
        elementwise_op = OpConfig(type='elementwise_add', inputs={'X': ['eltwise_X'], 'Y': ['eltwise_Y']}, outputs={'Out': ['eltwise_output']}, axis=axis)
        layer_norm_op = OpConfig('layer_norm', inputs={'X': ['eltwise_output'], 'Scale': ['layer_norm_scale'], 'Bias': ['layer_norm_bias']}, outputs={'Y': ['layer_norm_out'], 'Mean': ['layer_norm_mean'], 'Variance': ['layer_norm_var']}, begin_norm_axis=begin_norm_axis, epsilon=epsilon)
        mini_graph = [elementwise_op, layer_norm_op]
        program_config = ProgramConfig(ops=mini_graph, weights={'layer_norm_scale': TensorConfig(shape=[x_shape[2]]), 'layer_norm_bias': TensorConfig(shape=[x_shape[2]])}, inputs={'eltwise_X': TensorConfig(shape=x_shape), 'eltwise_Y': TensorConfig(shape=y_shape)}, outputs=mini_graph[-1].outputs['Y'])
        return program_config

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, max_examples=25, passes=['add_layernorm_xpu_fuse_pass'])
if __name__ == '__main__':
    np.random.seed(200)
    unittest.main()