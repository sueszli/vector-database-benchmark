import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMatmulV2ElementwiseAddMkldnnFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15
        axis = draw(st.sampled_from([-1, 0, 1]))
        matmul_as_x = draw(st.booleans())
        batch_size = draw(st.integers(min_value=2, max_value=4))
        channel = draw(st.sampled_from([16, 32, 64]))
        input_dim_shared = draw(st.sampled_from([16, 32, 64]))
        input_dim_X = draw(st.sampled_from([16, 32, 64]))
        input_dim_Y = draw(st.sampled_from([16, 32, 64]))

        def generate_input(type):
            if False:
                i = 10
                return i + 15
            broadcast_X = st.booleans()
            channel_X = 1 if broadcast_X else channel
            channel_Y = channel if broadcast_X else 1
            batch_size_X = 1 if broadcast_X else batch_size
            batch_size_Y = batch_size if broadcast_X else 1
            shape_x = [batch_size_X, channel_X, input_dim_X, input_dim_shared]
            shape_y = [batch_size_Y, channel_Y, input_dim_shared, input_dim_Y]
            if type == 'X':
                return np.random.random(shape_x).astype(np.float32)
            elif type == 'Y':
                return np.random.random(shape_y).astype(np.float32)
            else:
                shape_out = [batch_size, channel, input_dim_X, input_dim_Y]
                return np.random.random(shape_out).astype(np.float32)
        matmul_op = OpConfig(type='matmul_v2', inputs={'X': ['matmul_X'], 'Y': ['matmul_Y']}, outputs={'Out': ['matmul_output']}, attrs={'use_mkldnn': True})
        if matmul_as_x:
            inputs = {'X': ['matmul_output'], 'Y': ['elementwise_addend']}
        else:
            inputs = {'X': ['elementwise_addend'], 'Y': ['matmul_output']}
        elt_add_op = OpConfig(type='elementwise_add', inputs=inputs, outputs={'Out': ['elementwise_add_output']}, attrs={'axis': axis, 'use_mkldnn': True})
        model_net = [matmul_op, elt_add_op]
        program_config = ProgramConfig(ops=model_net, weights={}, inputs={'matmul_X': TensorConfig(data_gen=partial(generate_input, 'X')), 'matmul_Y': TensorConfig(data_gen=partial(generate_input, 'Y')), 'elementwise_addend': TensorConfig(data_gen=partial(generate_input, 'ElAdd'))}, outputs=['elementwise_add_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, ['fused_matmul'], (1e-05, 1e-05))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, max_examples=30, passes=['matmul_elementwise_add_mkldnn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()