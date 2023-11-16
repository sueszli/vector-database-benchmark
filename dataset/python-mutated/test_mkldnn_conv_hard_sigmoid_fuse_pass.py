import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import ProgramConfig, TensorConfig

class TestConvHardSigmoidMkldnnFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        return True

    def sample_program_config(self, draw):
        if False:
            return 10
        data_format = draw(st.sampled_from(['NCHW', 'NHWC']))
        dilations = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        padding_algorithm = draw(st.sampled_from(['EXPLICIT', 'SAME', 'VALID']))
        groups = draw(st.sampled_from([1, 2, 4]))
        paddings = draw(st.sampled_from([[0, 3], [1, 2, 3, 4]]))
        strides = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        slope = draw(st.floats(min_value=0, max_value=10))
        offset = draw(st.floats(min_value=0, max_value=10))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input():
            if False:
                print('Hello World!')
            if data_format == 'NCHW':
                return np.random.random([batch_size, 48, 64, 64]).astype(np.float32)
            else:
                return np.random.random([batch_size, 64, 64, 48]).astype(np.float32)

        def generate_weight():
            if False:
                i = 10
                return i + 15
            return np.random.random([16, int(48 / groups), 3, 3]).astype(np.float32)
        ops_config = [{'op_type': 'conv2d', 'op_inputs': {'Input': ['input_data'], 'Filter': ['input_weight']}, 'op_outputs': {'Output': ['conv_output']}, 'op_attrs': {'data_format': data_format, 'dilations': dilations, 'padding_algorithm': padding_algorithm, 'groups': groups, 'paddings': paddings, 'strides': strides}}, {'op_type': 'hard_sigmoid', 'op_inputs': {'X': ['conv_output']}, 'op_outputs': {'Out': ['sigmoid_output']}, 'op_attrs': {'slope': slope, 'offset': offset}}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={'input_weight': TensorConfig(data_gen=partial(generate_weight))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['sigmoid_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, ['fused_conv2d'], (1e-05, 1e-05))

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, passes=['conv_activation_mkldnn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()