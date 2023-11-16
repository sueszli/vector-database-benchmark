import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import MkldnnAutoScanTest
from hypothesis import given
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestOneDNNPad3DOp(MkldnnAutoScanTest):

    def sample_program_configs(self, *args, **kwargs):
        if False:
            while True:
                i = 10

        def generate_input(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        def generate_paddings():
            if False:
                while True:
                    i = 10
            return np.random.randint(0, 4, size=6).astype(np.int32)
        pad3d_op = OpConfig(type='pad3d', inputs={'X': ['input_data'], 'Paddings': ['paddings_data']}, outputs={'Out': ['output_data']}, attrs={'mode': 'constant', 'data_format': kwargs['data_format'], 'paddings': kwargs['paddings']})
        program_config = ProgramConfig(ops=[pad3d_op], weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, *args, **kwargs)), 'paddings_data': TensorConfig(data_gen=generate_paddings)}, outputs=['output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config):
        if False:
            while True:
                i = 10
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, (1e-05, 1e-05))

    @given(data_format=st.sampled_from(['NCDHW', 'NDHWC']), use_paddings_tensor=st.sampled_from([True, False]), in_shape=st.sampled_from([[2, 3, 4, 5, 6], [1, 4, 1, 3, 2], [4, 3, 2, 1, 1], [1, 1, 1, 1, 1]]), paddings=st.sampled_from([[0, 0, 0, 0, 0, 0], [1, 2, 0, 1, 2, 1], [2, 5, 11, 3, 4, 3], [0, 5, 0, 1, 0, 2]]))
    def test(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(*args, quant=False, **kwargs)
if __name__ == '__main__':
    unittest.main()