import copy as cp
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import IgnoreReasons, PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class DepthwiseConvMKLDNNPass(PassAutoScanTest):
    """
    conv_input   conv_weight_var(persistable)
      \\       /
         conv_op
          |
      conv_out_var
    """

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_and_statis(quant=False, passes=['depthwise_conv_mkldnn_pass'])

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        random_batch_size = draw(st.integers(min_value=1, max_value=4))
        random_channel = draw(st.integers(min_value=2, max_value=10))
        random_input_dim1 = draw(st.integers(min_value=20, max_value=50))
        random_input_dim2 = draw(st.integers(min_value=20, max_value=50))
        random_out_channel = draw(st.integers(min_value=20, max_value=25))
        random_groups = draw(st.integers(min_value=1, max_value=3))
        random_dilations = draw(st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=2))
        random_strides = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=2))
        random_paddings = draw(st.lists(st.integers(min_value=0, max_value=4), min_size=2, max_size=2))
        random_padding_algorithm = draw(st.sampled_from(['EXPLICIT', 'SAME', 'VALID']))
        random_data_layout = draw(st.sampled_from(['NCHW', 'NHWC']))
        random_filter = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=2))

        def generate_conv2d_Input():
            if False:
                print('Hello World!')
            shape = [random_input_dim1, random_input_dim2]
            if random_data_layout == 'NCHW':
                shape.insert(0, random_channel * random_groups)
                shape.insert(0, random_batch_size)
            else:
                shape.append(random_channel)
                shape.insert(0, random_batch_size)
            return np.random.random(shape).astype(np.float32)

        def generate_conv2d_Filter():
            if False:
                i = 10
                return i + 15
            shape = cp.copy(random_filter)
            shape.insert(0, random_channel)
            shape.insert(0, random_out_channel * random_groups)
            return np.random.random(shape).astype(np.float32)
        conv2d_op = OpConfig(type='depthwise_conv2d', inputs={'Input': ['conv2d_Input'], 'Filter': ['conv2d_Filter']}, outputs={'Output': ['conv2d_Out']}, attrs={'groups': random_groups, 'dilations': random_dilations, 'strides': random_strides, 'paddings': random_paddings, 'padding_algorithm': random_padding_algorithm, 'data_format': random_data_layout, 'use_mkldnn': True})
        model_net = [conv2d_op]
        program_config = ProgramConfig(ops=model_net, inputs={'conv2d_Input': TensorConfig(data_gen=generate_conv2d_Input)}, weights={'conv2d_Filter': TensorConfig(data_gen=generate_conv2d_Filter)}, outputs=['conv2d_Out'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, ['conv2d'], (1e-05, 1e-05))

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if attrs[0]['data_format'] == 'NHWC':
            return False
        return True

    def add_ignore_pass_case(self):
        if False:
            return 10

        def teller1(program_config, predictor_config):
            if False:
                i = 10
                return i + 15
            if program_config.ops[0].attrs['data_format'] == 'NHWC':
                return True
            return False
        self.add_ignore_check_case(teller1, IgnoreReasons.PASS_ACCURACY_ERROR, 'The output format of depthwise_conv2d is wrong when data_format attribute is NHWC')