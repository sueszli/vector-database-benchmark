import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestAddActXPUFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['add_act_xpu'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10
        batch_size = draw(st.integers(min_value=1, max_value=50))

        def generate_input():
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random([batch_size, 3, 100, 100]).astype(np.float32)
        axis = -1
        elementwise_op = OpConfig(type='elementwise_add', inputs={'X': ['eltwise_X'], 'Y': ['eltwise_Y']}, outputs={'Out': ['eltwise_output']}, axis=axis)
        relu_op = OpConfig('relu', inputs={'X': ['eltwise_output']}, outputs={'Out': ['relu_out']})
        mini_graph = [elementwise_op, relu_op]
        program_config = ProgramConfig(ops=mini_graph, weights={}, inputs={'eltwise_X': TensorConfig(data_gen=partial(generate_input)), 'eltwise_Y': TensorConfig(data_gen=partial(generate_input))}, outputs=mini_graph[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, max_examples=25, passes=['add_activation_xpu_fuse_pass'])
if __name__ == '__main__':
    unittest.main()