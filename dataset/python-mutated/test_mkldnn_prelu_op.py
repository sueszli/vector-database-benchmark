import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import MkldnnAutoScanTest
from hypothesis import given
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMkldnnPreluOp(MkldnnAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if len(program_config.inputs['input_data'].shape) == 1 and program_config.ops[0].attrs['mode'] == 'channel':
            return False
        return True

    def sample_program_configs(self, *args, **kwargs):
        if False:
            while True:
                i = 10

        def generate_input(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        def generate_alpha(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if kwargs['mode'] == 'all':
                return np.random.random(size=1).astype(np.float32)
            elif kwargs['mode'] == 'channel':
                if len(kwargs['in_shape']) <= 1:
                    return np.zeros(1).astype(np.float32)
                if kwargs['data_format'] == 'NCHW':
                    return np.random.random(kwargs['in_shape'][1]).astype(np.float32)
                else:
                    return np.random.random(kwargs['in_shape'][-1]).astype(np.float32)
            else:
                if len(kwargs['in_shape']) <= 1:
                    return np.zeros(1).astype(np.float32)
                return np.random.random(kwargs['in_shape']).astype(np.float32)
        prelu_op = OpConfig(type='prelu', inputs={'X': ['input_data'], 'Alpha': ['alpha_weight']}, outputs={'Out': ['output_data']}, attrs={'mode': kwargs['mode'], 'data_format': kwargs['data_format']})
        program_config = ProgramConfig(ops=[prelu_op], weights={'alpha_weight': TensorConfig(data_gen=partial(generate_alpha, *args, **kwargs))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, *args, **kwargs))}, outputs=['output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, (1e-05, 1e-05))

    def add_skip_pass_case(self):
        if False:
            print('Hello World!')
        pass

    @given(mode=st.sampled_from(['all', 'channel', 'element']), data_format=st.sampled_from(['NCHW', 'NHWC']), in_shape=st.lists(st.integers(min_value=1, max_value=32), min_size=1, max_size=4))
    def test(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.add_skip_pass_case()
        self.run_test(*args, quant=False, **kwargs)
if __name__ == '__main__':
    unittest.main()