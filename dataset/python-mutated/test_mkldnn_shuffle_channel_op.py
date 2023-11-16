import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import MkldnnAutoScanTest
from hypothesis import given
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMKLDNNShuffleChannelOp(MkldnnAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_configs(self, *args, **kwargs):
        if False:
            return 10

        def generate_input(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random(kwargs['in_shape']).astype(np.float32)
        shuffle_channel_op = OpConfig(type='shuffle_channel', inputs={'X': ['input_data']}, outputs={'Out': ['output_data']}, attrs={'group': kwargs['group']})
        program_config = ProgramConfig(ops=[shuffle_channel_op], weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, *args, **kwargs))}, outputs=['output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config):
        if False:
            return 10
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, (1e-05, 1e-05))

    @given(group=st.sampled_from([1, 2, 8, 32, 128]), in_shape=st.sampled_from([[5, 512, 2, 3], [2, 256, 5, 4]]))
    def test(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.run_test(*args, quant=False, **kwargs)
if __name__ == '__main__':
    unittest.main()