import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import MkldnnAutoScanTest
from hypothesis import given
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMkldnnShapeOp(MkldnnAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def sample_program_configs(self, *args, **kwargs):
        if False:
            print('Hello World!')

        def generate_input(*args, **kwargs):
            if False:
                return 10
            return np.random.random(kwargs['in_shape']).astype(kwargs['in_dtype'])
        shape_op = OpConfig(type='shape', inputs={'Input': ['input_data']}, outputs={'Out': ['output_data']})
        program_config = ProgramConfig(ops=[shape_op], weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, *args, **kwargs))}, outputs=['output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, (1e-05, 1e-05))

    @given(in_shape=st.lists(st.integers(min_value=1, max_value=3), min_size=1, max_size=6), in_dtype=st.sampled_from([np.float32, np.uint16, np.int8, np.uint8]))
    def test(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.run_test(*args, quant=False, **kwargs)
if __name__ == '__main__':
    unittest.main()