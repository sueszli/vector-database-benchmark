import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestConvTransposeMkldnnFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if attrs[0]['data_format'] == 'NCHW' and attrs[1]['axis'] == 3:
            return False
        if attrs[0]['data_format'] == 'NHWC' and attrs[1]['axis'] == 1:
            return False
        return True

    def sample_program_config(self, draw):
        if False:
            return 10
        data_format = draw(st.sampled_from(['NCHW', 'NHWC']))
        dilations = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        padding_algorithm = draw(st.sampled_from(['EXPLICIT', 'SAME', 'VALID']))
        groups = draw(st.sampled_from([1, 2, 4, 8]))
        paddings = draw(st.sampled_from([[0, 3], [1, 2, 3, 4]]))
        strides = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        axis = draw(st.sampled_from([1, 3]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input():
            if False:
                while True:
                    i = 10
            if data_format == 'NCHW':
                return np.random.random([batch_size, 16, 64, 64]).astype(np.float32)
            else:
                return np.random.random([batch_size, 64, 64, 16]).astype(np.float32)

        def generate_weight1():
            if False:
                return 10
            return np.random.random([16, 16, 3, 3]).astype(np.float32)

        def generate_weight2():
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random([16 * groups]).astype(np.float32)
        conv2d_op = OpConfig(type='conv2d_transpose', inputs={'Input': ['input_data'], 'Filter': ['conv2d_weight']}, outputs={'Output': ['conv_output']}, attrs={'data_format': data_format, 'dilations': dilations, 'padding_algorithm': padding_algorithm, 'groups': groups, 'paddings': paddings, 'strides': strides, 'output_size': [], 'output_padding': [], 'is_test': True})
        elt_op = OpConfig(type='elementwise_add', inputs={'X': ['conv_output'], 'Y': ['elementwise_weight']}, outputs={'Out': ['elementwise_output']}, attrs={'axis': axis})
        model_net = [conv2d_op, elt_op]
        program_config = ProgramConfig(ops=model_net, weights={'conv2d_weight': TensorConfig(data_gen=partial(generate_weight1)), 'elementwise_weight': TensorConfig(data_gen=partial(generate_weight2))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['elementwise_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, ['conv2d_transpose'], (1e-05, 1e-05))

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, max_duration=300, passes=['conv_transpose_bias_mkldnn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()