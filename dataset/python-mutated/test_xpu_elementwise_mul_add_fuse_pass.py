import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestGatherAddTransposePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['addcmul_xpu'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=3, max_size=4))

        def generate_data(shape):
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random(shape).astype(np.float32)
        mul_op = OpConfig('elementwise_mul', inputs={'X': ['mul_x'], 'Y': ['mul_y']}, outputs={'Out': ['mul_out']})
        add_op = OpConfig('elementwise_add', inputs={'X': ['mul_out'], 'Y': ['add_w']}, outputs={'Out': ['add_out']})
        ops = [mul_op, add_op]
        program_config = ProgramConfig(ops=ops, inputs={'mul_x': TensorConfig(data_gen=partial(generate_data, x_shape)), 'mul_y': TensorConfig(data_gen=partial(generate_data, x_shape)), 'add_w': TensorConfig(data_gen=partial(generate_data, x_shape))}, weights={}, outputs=['add_out'])
        return program_config

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, max_examples=25, passes=['elementwise_mul_add_fuse_pass'])
if __name__ == '__main__':
    unittest.main()