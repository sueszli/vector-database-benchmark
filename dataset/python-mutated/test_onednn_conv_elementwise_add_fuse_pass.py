import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestOneDNNConvElementwiseAddFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if attrs[1]['data_format'] == 'NHWC' and attrs[3]['axis'] == 0:
            return False
        if attrs[1]['data_format'] == 'NCHW' and attrs[3]['axis'] == -1:
            return False
        return True

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15
        data_format = draw(st.sampled_from(['NCHW', 'NHWC']))
        dilations = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        padding_algorithm = draw(st.sampled_from(['EXPLICIT', 'SAME', 'VALID']))
        groups = draw(st.sampled_from([1, 2, 4]))
        paddings = draw(st.sampled_from([[0, 3], [1, 1], [1, 2, 3, 4]]))
        strides = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        axis = draw(st.sampled_from([-1, 0]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input():
            if False:
                i = 10
                return i + 15
            if data_format == 'NCHW':
                return np.random.random([batch_size, 48, 64, 64]).astype(np.float32)
            else:
                return np.random.random([batch_size, 64, 64, 48]).astype(np.float32)

        def generate_weight():
            if False:
                while True:
                    i = 10
            return np.random.random([48, int(48 / groups), 3, 3]).astype(np.float32)
        relu_op = OpConfig(type='relu', inputs={'X': ['input_data']}, outputs={'Out': ['relu_out']}, attrs={})
        conv2d_op1 = OpConfig(type='conv2d', inputs={'Input': ['relu_out'], 'Filter': ['conv_weight1']}, outputs={'Output': ['conv_output1']}, attrs={'data_format': data_format, 'dilations': dilations, 'padding_algorithm': padding_algorithm, 'groups': groups, 'paddings': paddings, 'strides': strides})
        conv2d_op2 = OpConfig(type='conv2d', inputs={'Input': ['input_data'], 'Filter': ['conv_weight2']}, outputs={'Output': ['conv_output2']}, attrs={'data_format': data_format, 'dilations': dilations, 'padding_algorithm': padding_algorithm, 'groups': groups, 'paddings': paddings, 'strides': strides})
        elt_op = OpConfig(type='elementwise_add', inputs={'X': ['conv_output1'], 'Y': ['conv_output2']}, outputs={'Out': ['elementwise_output']}, attrs={'axis': axis})
        model_net = [relu_op, conv2d_op1, conv2d_op2, elt_op]
        program_config = ProgramConfig(ops=model_net, weights={'conv_weight1': TensorConfig(data_gen=partial(generate_weight)), 'conv_weight2': TensorConfig(data_gen=partial(generate_weight))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['elementwise_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, ['relu', 'conv2d', 'fused_conv2d'], (1e-05, 1e-05))

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, passes=['conv_elementwise_add_mkldnn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()