import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestSeqconvEltaddReluFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_config(self, draw):
        if False:
            return 10
        contextLength = draw(st.sampled_from([1, 2, 3, 4]))
        contextStart = draw(st.sampled_from([1, 2, 3]))
        contextStride = draw(st.sampled_from([1]))
        paddingTrainable = False
        axis = draw(st.sampled_from([1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input():
            if False:
                return 10
            shape = [batch_size, 128, 6, 120]
            return np.random.random(shape).astype(np.float32)

        def generate_weight(shape):
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random(shape).astype(np.float32)
        im2sequence_op = OpConfig(type='im2sequence', inputs={'X': ['input_data']}, outputs={'Out': ['seq_out']}, attrs={'kernels': [6, 1], 'out_stride': [1, 1], 'paddings': [0, 0, 0, 0], 'strides': [1, 1]})
        sequence_conv_op = OpConfig(type='sequence_conv', inputs={'X': ['seq_out'], 'Filter': ['conv_weight']}, outputs={'Out': ['conv_out']}, attrs={'contextLength': contextLength, 'contextStart': contextStart, 'contextStride': contextStride, 'paddingTrainable': paddingTrainable})
        elementwise_add_op = OpConfig(type='elementwise_add', inputs={'X': ['conv_out'], 'Y': ['elt_weight']}, outputs={'Out': ['elt_output']}, attrs={'axis': axis})
        relu_op = OpConfig(type='relu', inputs={'X': ['elt_output']}, outputs={'Out': ['relu_output']}, attrs={})
        model_net = [im2sequence_op, sequence_conv_op, elementwise_add_op, relu_op]
        program_config = ProgramConfig(ops=model_net, weights={'conv_weight': TensorConfig(data_gen=partial(generate_weight, [768 * contextLength, 16])), 'elt_weight': TensorConfig(data_gen=partial(generate_weight, [16]))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['relu_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config()
        yield (config, ['im2sequence', 'fusion_seqconv_eltadd_relu'], (1e-05, 1e-05))

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, passes=['seqconv_eltadd_relu_fuse_pass'])
if __name__ == '__main__':
    unittest.main()