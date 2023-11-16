import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestElementwiseAddActivationOneDNNFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        batch_size = draw(st.sampled_from([1, 32]))
        activation_type = draw(st.sampled_from(['relu', 'gelu', 'swish', 'mish', 'sqrt', 'hard_swish', 'sigmoid', 'abs', 'relu6', 'clip', 'tanh', 'hard_sigmoid', 'leaky_relu', 'scale']))

        def generate_input():
            if False:
                return 10
            return np.random.random([batch_size, 3, 100, 100]).astype(np.float32)
        elementwise_op = OpConfig(type='elementwise_add', inputs={'X': ['eltwise_X'], 'Y': ['eltwise_Y']}, outputs={'Out': ['eltwise_output']}, attrs={'use_mkldnn': True})
        if activation_type == 'relu6':
            activation_op = OpConfig(activation_type, inputs={'X': ['eltwise_output']}, outputs={'Out': ['activation_output']}, threshold=6.0)
        elif activation_type == 'leaky_relu':
            activation_op = OpConfig(activation_type, inputs={'X': ['eltwise_output']}, outputs={'Out': ['activation_output']}, alpha=draw(st.floats(min_value=0.1, max_value=1.0)))
        elif activation_type == 'scale':
            activation_op = OpConfig(activation_type, inputs={'X': ['eltwise_output']}, outputs={'Out': ['activation_output']}, scale=draw(st.sampled_from([0.125, 0.4, 0.875, 2])))
        elif activation_type == 'swish':
            activation_op = OpConfig(activation_type, inputs={'X': ['eltwise_output']}, outputs={'Out': ['activation_output']}, beta=1.0)
        elif activation_type == 'clip':
            activation_op = OpConfig(activation_type, inputs={'X': ['eltwise_output']}, outputs={'Out': ['activation_output']}, min=draw(st.floats(min_value=0.1, max_value=0.49)), max=draw(st.floats(min_value=0.5, max_value=1.0)))
        else:
            activation_op = OpConfig(activation_type, inputs={'X': ['eltwise_output']}, outputs={'Out': ['activation_output']})
        mini_graph = [elementwise_op, activation_op]
        program_config = ProgramConfig(ops=mini_graph, weights={}, inputs={'eltwise_X': TensorConfig(data_gen=partial(generate_input)), 'eltwise_Y': TensorConfig(data_gen=partial(generate_input))}, outputs=['activation_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            while True:
                i = 10
        config = self.create_inference_config(use_mkldnn=True, passes=['elementwise_act_onednn_fuse_pass', 'operator_scale_onednn_fuse_pass'])
        yield (config, ['fused_elementwise_add'], (1e-05, 1e-05))

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, passes=['elementwise_act_onednn_fuse_pass', 'operator_scale_onednn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()