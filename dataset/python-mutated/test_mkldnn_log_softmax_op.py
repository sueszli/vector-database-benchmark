import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import MkldnnAutoScanTest
from hypothesis import given
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMKLDNNLogSoftmaxOp(MkldnnAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def sample_program_configs(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def generate_input(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return np.random.random(kwargs['in_shape']).astype(np.float32)
        logsoftmax_op = OpConfig(type='log_softmax', inputs={'X': ['input_data']}, outputs={'Out': ['output_data']}, attrs={'axis': kwargs['axis']})
        program_config = ProgramConfig(ops=[logsoftmax_op], weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, *args, **kwargs))}, outputs=['output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config):
        if False:
            return 10
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, (1e-05, 1e-05))

    @given(axis=st.sampled_from([-2, -1, 0, 1]), in_shape=st.lists(st.integers(min_value=2, max_value=5), min_size=3, max_size=5))
    def test(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.run_test(*args, quant=False, **kwargs)
if __name__ == '__main__':
    unittest.main()