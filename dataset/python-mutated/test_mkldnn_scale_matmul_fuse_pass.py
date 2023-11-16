import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import ProgramConfig, TensorConfig

class TestScaleMatmulMkldnnFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        return True

    def sample_program_config(self, draw):
        if False:
            return 10
        scale = draw(st.floats(min_value=0.01, max_value=2))
        bias = 0.0
        bias_after_scale = draw(st.booleans())
        transpose_X = draw(st.booleans())
        transpose_Y = draw(st.booleans())
        alpha = draw(st.floats(min_value=0.01, max_value=2))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        channel = draw(st.integers(min_value=1, max_value=64))
        input_dim = draw(st.sampled_from([1, 32, 64]))

        def generate_input(attrs, type):
            if False:
                i = 10
                return i + 15
            is_transpose_X = attrs[1]['transpose_X']
            is_transpose_Y = attrs[1]['transpose_Y']
            if is_transpose_X:
                shape_x_3 = attrs[2]['input_dim']
                shape_x_4 = 32
            else:
                shape_x_3 = 32
                shape_x_4 = attrs[2]['input_dim']
            if is_transpose_X and is_transpose_Y:
                shape_y_3 = 64
                shape_y_4 = attrs[2]['input_dim']
            elif is_transpose_X:
                shape_y_3 = attrs[2]['input_dim']
                shape_y_4 = 64
            elif is_transpose_Y:
                shape_y_3 = 8
                shape_y_4 = attrs[2]['input_dim']
            else:
                shape_y_3 = attrs[2]['input_dim']
                shape_y_4 = 16
            shape_x = [attrs[2]['batch_size'], attrs[2]['channel'], shape_x_3, shape_x_4]
            shape_y = [attrs[2]['batch_size'], attrs[2]['channel'], shape_y_3, shape_y_4]
            shape = shape_x if type == 'x' else shape_y
            return np.random.random(shape).astype(np.float32)
        attrs = [{'scale': scale, 'bias': bias, 'bias_after_scale': bias_after_scale}, {'transpose_X': transpose_X, 'transpose_Y': transpose_Y, 'alpha': alpha}, {'batch_size': batch_size, 'channel': channel, 'input_dim': input_dim}]
        ops_config = [{'op_type': 'scale', 'op_inputs': {'X': ['input_data1']}, 'op_outputs': {'Out': ['scale_output']}, 'op_attrs': {'scale': attrs[0]['scale'], 'bias': attrs[0]['bias'], 'bias_after_scale': attrs[0]['bias_after_scale']}}, {'op_type': 'matmul', 'op_inputs': {'X': ['scale_output'], 'Y': ['input_data2']}, 'op_outputs': {'Out': ['matmul_output']}, 'op_attrs': {'transpose_X': attrs[1]['transpose_X'], 'transpose_Y': attrs[1]['transpose_Y'], 'alpha': attrs[1]['alpha']}}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data1': TensorConfig(data_gen=partial(generate_input, attrs, 'x')), 'input_data2': TensorConfig(data_gen=partial(generate_input, attrs, 'y'))}, outputs=['matmul_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            while True:
                i = 10
        config = self.create_inference_config(use_mkldnn=True, passes=['scale_matmul_fuse_pass'])
        yield (config, ['matmul'], (1e-05, 1e-05))

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_and_statis(quant=False, passes=['scale_matmul_fuse_pass'])
if __name__ == '__main__':
    unittest.main()