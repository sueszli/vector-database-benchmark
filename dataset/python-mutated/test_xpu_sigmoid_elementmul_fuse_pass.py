import unittest
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestSigmoidElementmulFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['swish'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            return 10
        sigmoid_x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=4))
        sigmoid_op = OpConfig('sigmoid', inputs={'X': ['sigmoid_x']}, outputs={'Out': ['sigmoid_out']}, trans_x=False, trans_y=False)
        mul_op = OpConfig('elementwise_mul', inputs={'X': ['sigmoid_x'], 'Y': ['sigmoid_out']}, outputs={'Out': ['out']}, axis=-1)
        ops = [sigmoid_op, mul_op]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'sigmoid_x': TensorConfig(shape=sigmoid_x_shape)}, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, max_examples=25, passes=['sigmoid_elementmul_fuse_pass'])
if __name__ == '__main__':
    np.random.seed(200)
    unittest.main()