import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import MkldnnAutoScanTest
from hypothesis import given
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMkldnnMishOp(MkldnnAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        if len(program_config.inputs['input_data'].shape) == 1 and program_config.ops[0].attrs['mode'] == 'channel':
            return False
        return True

    def sample_program_configs(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15

        def generate_input(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return np.random.random(kwargs['in_shape']).astype(np.float32)
        mish_op = OpConfig(type='mish', inputs={'X': ['input_data']}, outputs={'Out': ['output_data']}, attrs={'mode': kwargs['mode'], 'data_format': kwargs['data_format']})
        program_config = ProgramConfig(ops=[mish_op], weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, *args, **kwargs))}, outputs=['output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, (1e-05, 1e-05))

    @given(mode=st.sampled_from(['all', 'channel', 'element']), data_format=st.sampled_from(['NCHW', 'NHWC']), in_shape=st.lists(st.integers(min_value=1, max_value=32), min_size=1, max_size=4))
    def test(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(*args, quant=False, **kwargs)
if __name__ == '__main__':
    unittest.main()