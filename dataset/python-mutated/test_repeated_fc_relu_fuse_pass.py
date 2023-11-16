import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestRepeatedFcReluFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15
        x_col = draw(st.sampled_from([1]))
        y_col = draw(st.sampled_from([1]))
        axis = draw(st.sampled_from([-1, 1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        dim = draw(st.sampled_from([32, 64, 128]))

        def generate_input():
            if False:
                i = 10
                return i + 15
            return np.random.random([batch_size, dim]).astype(np.float32)

        def generate_weight(shape):
            if False:
                i = 10
                return i + 15
            return np.random.random(shape).astype(np.float32)
        attrs = [{'x_col': x_col, 'y_col': y_col}, {'axis': axis}, {'batch_size': batch_size, 'dim': dim}]
        mul_op1 = OpConfig(type='mul', inputs={'X': ['input_data'], 'Y': ['mul1_weight']}, outputs={'Out': ['mul1_output']}, attrs={'x_num_col_dims': x_col, 'y_num_col_dims': y_col})
        elt_op1 = OpConfig(type='elementwise_add', inputs={'X': ['mul1_output'], 'Y': ['elementwise1_weight']}, outputs={'Out': ['elementwise1_output']}, attrs={'axis': axis})
        relu_op1 = OpConfig(type='relu', inputs={'X': ['elementwise1_output']}, outputs={'Out': ['relu1_output']}, attrs={})
        mul_op2 = OpConfig(type='mul', inputs={'X': ['relu1_output'], 'Y': ['mul2_weight']}, outputs={'Out': ['mul2_output']}, attrs={'x_num_col_dims': x_col, 'y_num_col_dims': y_col})
        elt_op2 = OpConfig(type='elementwise_add', inputs={'X': ['mul2_output'], 'Y': ['elementwise2_weight']}, outputs={'Out': ['elementwise2_output']}, attrs={'axis': axis})
        relu_op2 = OpConfig(type='relu', inputs={'X': ['elementwise2_output']}, outputs={'Out': ['relu2_output']}, attrs={})
        model_net = [mul_op1, elt_op1, relu_op1, mul_op2, elt_op2, relu_op2]
        program_config = ProgramConfig(ops=model_net, weights={'mul1_weight': TensorConfig(data_gen=partial(generate_weight, [dim, 32])), 'mul2_weight': TensorConfig(data_gen=partial(generate_weight, [32, 128])), 'elementwise1_weight': TensorConfig(data_gen=partial(generate_weight, [32])), 'elementwise2_weight': TensorConfig(data_gen=partial(generate_weight, [128]))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['relu2_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            return 10
        config = self.create_inference_config()
        yield (config, ['fusion_repeated_fc_relu'], (1e-05, 1e-05))

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(passes=['repeated_fc_relu_fuse_pass'])
if __name__ == '__main__':
    unittest.main()