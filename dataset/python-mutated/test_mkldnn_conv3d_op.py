import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import MkldnnAutoScanTest
from hypothesis import given
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMkldnnConv3dOp(MkldnnAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_configs(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def generate_input(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if kwargs['data_format'] == 'NCDHW':
                return np.random.random([kwargs['batch_size'], 48, 64, 32, 64]).astype(np.float32)
            else:
                return np.random.random([kwargs['batch_size'], 64, 32, 64, 48]).astype(np.float32)

        def generate_weight(*args, **kwargs):
            if False:
                return 10
            return np.random.random([16, int(48 / kwargs['groups']), 3, 3, 3]).astype(np.float32)
        conv3d_op = OpConfig(type='conv3d', inputs={'Input': ['input_data'], 'Filter': ['conv_weight']}, outputs={'Output': ['conv_output']}, attrs={'data_format': kwargs['data_format'], 'dilations': kwargs['dilations'], 'padding_algorithm': kwargs['padding_algorithm'], 'groups': kwargs['groups'], 'paddings': kwargs['paddings'], 'strides': kwargs['strides'], 'is_test': True})
        program_config = ProgramConfig(ops=[conv3d_op], weights={'conv_weight': TensorConfig(data_gen=partial(generate_weight, *args, **kwargs))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, *args, **kwargs))}, outputs=['conv_output'])
        yield program_config

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, (1e-05, 1e-05))

    @given(data_format=st.sampled_from(['NCDHW', 'NDHWC']), dilations=st.sampled_from([[1, 2, 1]]), padding_algorithm=st.sampled_from(['EXPLICIT']), groups=st.sampled_from([2]), paddings=st.sampled_from([[0, 3, 2]]), strides=st.sampled_from([[1, 2, 1]]), batch_size=st.integers(min_value=1, max_value=4))
    def test(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.run_test(*args, **kwargs)
if __name__ == '__main__':
    unittest.main()