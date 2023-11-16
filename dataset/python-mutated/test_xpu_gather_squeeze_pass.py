import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestGatherAddTransposePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            return 10
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['transpose2', 'gather', 'transpose2', 'gather', 'squeeze2', 'squeeze2'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            return 10
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=3, max_size=3))

        def generate_data(shape):
            if False:
                return 10
            return np.random.random(shape).astype(np.float32)

        def generate_index(*args, **kwargs):
            if False:
                print('Hello World!')
            return np.array([0]).astype(np.int64)
        axis = 2
        axes = [2]
        gather_op0 = OpConfig('gather', inputs={'X': ['gather_in'], 'Index': ['gather_index0']}, outputs={'Out': ['gather_out0']}, axis=axis)
        gather_op1 = OpConfig('gather', inputs={'X': ['gather_in'], 'Index': ['gather_index1']}, outputs={'Out': ['gather_out1']}, axis=axis)
        squeeze_op0 = OpConfig('squeeze2', inputs={'X': ['gather_out0']}, outputs={'Out': ['squeeze_out0']}, axes=axes)
        squeeze_op1 = OpConfig('squeeze2', inputs={'X': ['gather_out1']}, outputs={'Out': ['squeeze_out1']}, axes=axes)
        ops = [gather_op0, gather_op1, squeeze_op0, squeeze_op1]
        program_config = ProgramConfig(ops=ops, inputs={'gather_in': TensorConfig(data_gen=partial(generate_data, x_shape)), 'gather_index0': TensorConfig(data_gen=partial(generate_index)), 'gather_index1': TensorConfig(data_gen=partial(generate_index))}, weights={}, outputs=['squeeze_out0', 'squeeze_out1'])
        return program_config

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, max_examples=25, passes=['gather_squeeze_pass'])
if __name__ == '__main__':
    unittest.main()