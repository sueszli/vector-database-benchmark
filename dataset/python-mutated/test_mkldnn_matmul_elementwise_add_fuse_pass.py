import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMatmulElementwiseAddMkldnnFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        axis = draw(st.sampled_from([-1, 0, 1]))
        matmul_as_x = draw(st.booleans())
        batch_size = draw(st.integers(min_value=2, max_value=4))
        channel = draw(st.sampled_from([16, 32, 64]))
        input_dim = draw(st.sampled_from([16, 32, 64]))

        def generate_input():
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random([batch_size, channel, input_dim, input_dim]).astype(np.float32)
        matmul_op = OpConfig(type='matmul', inputs={'X': ['matmul_x'], 'Y': ['matmul_y']}, outputs={'Out': ['matmul_output']}, attrs={'use_mkldnn': True})
        if matmul_as_x:
            inputs = {'X': ['matmul_output'], 'Y': ['elementwise_addend']}
        else:
            inputs = {'X': ['elementwise_addend'], 'Y': ['matmul_output']}
        elt_add_op = OpConfig(type='elementwise_add', inputs=inputs, outputs={'Out': ['elementwise_add_output']}, attrs={'axis': axis, 'use_mkldnn': True})
        model_net = [matmul_op, elt_add_op]
        program_config = ProgramConfig(ops=model_net, weights={}, inputs={'matmul_x': TensorConfig(data_gen=partial(generate_input)), 'matmul_y': TensorConfig(data_gen=partial(generate_input)), 'elementwise_addend': TensorConfig(data_gen=partial(generate_input))}, outputs=['elementwise_add_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            while True:
                i = 10
        config = self.create_inference_config(use_mkldnn=True, passes=['matmul_elementwise_add_mkldnn_fuse_pass'])
        yield (config, ['fused_matmul'], (1e-05, 1e-05))

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, passes=['matmul_elementwise_add_mkldnn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()