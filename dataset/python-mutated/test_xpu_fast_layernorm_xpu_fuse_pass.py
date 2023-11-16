import unittest
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestFastLayernormXPUFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['fast_layernorm_xpu'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        batch_size = draw(st.integers(min_value=1, max_value=50))
        x_shape = [batch_size, 16, 128]
        y_shape = x_shape
        axis = -1
        epsilon = draw(st.floats(min_value=1e-07, max_value=0.001))
        begin_norm_axis = 2
        layer_norm_op = OpConfig('layer_norm', inputs={'X': ['x'], 'Scale': ['layer_norm_scale'], 'Bias': ['layer_norm_bias']}, outputs={'Y': ['layer_norm_out'], 'Mean': ['layer_norm_mean'], 'Variance': ['layer_norm_var']}, begin_norm_axis=begin_norm_axis, epsilon=epsilon)
        mini_graph = [layer_norm_op]
        program_config = ProgramConfig(ops=mini_graph, weights={'layer_norm_scale': TensorConfig(shape=[x_shape[2]]), 'layer_norm_bias': TensorConfig(shape=[x_shape[2]])}, inputs={'x': TensorConfig(shape=x_shape)}, outputs=mini_graph[-1].outputs['Y'])
        return program_config

    def test(self):
        if False:
            return 10
        self.run_and_statis(quant=False, max_examples=25, passes=['fast_layernorm_xpu_fuse_pass'])
if __name__ == '__main__':
    np.random.seed(200)
    unittest.main()