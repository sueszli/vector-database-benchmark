import unittest
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestXpuLayerNormActFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['layer_norm_act_xpu'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        batch_size = draw(st.integers(min_value=1, max_value=50))
        x_shape = [batch_size, 16, 128]
        y_shape = x_shape
        epsilon = draw(st.floats(min_value=1e-07, max_value=0.001))
        begin_norm_axis = 2
        layer_norm_op = OpConfig('layer_norm', inputs={'X': ['x'], 'Scale': ['layer_norm_scale'], 'Bias': ['layer_norm_bias']}, outputs={'Y': ['layer_norm_out'], 'Mean': ['layer_norm_mean'], 'Variance': ['layer_norm_var']}, begin_norm_axis=begin_norm_axis, epsilon=epsilon)
        alpha = draw(st.floats(min_value=1e-07, max_value=0.001))
        relu_op = OpConfig('leaky_relu', inputs={'X': ['layer_norm_out']}, outputs={'Out': ['relu_out']}, alpha=alpha)
        sub_graph = [layer_norm_op, relu_op]
        program_config = ProgramConfig(ops=sub_graph, weights={'layer_norm_scale': TensorConfig(shape=[x_shape[2]]), 'layer_norm_bias': TensorConfig(shape=[x_shape[2]])}, inputs={'x': TensorConfig(shape=x_shape)}, outputs=['relu_out'])
        return program_config

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, max_examples=25, passes=['layer_norm_act_xpu_fuse_pass'])
if __name__ == '__main__':
    np.random.seed(200)
    unittest.main()