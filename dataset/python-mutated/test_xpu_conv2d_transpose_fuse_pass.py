import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestConvTransposeXPUFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['conv2d_transpose_xpu'], (0.003, 0.003))

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15
        x_shape = draw(st.lists(st.integers(min_value=4, max_value=16), min_size=4, max_size=4))
        oc = draw(st.integers(min_value=2, max_value=16))
        weight_shape = [x_shape[1], oc, 4, 4]
        y_shape = [oc]
        has_bn = draw(st.booleans())
        has_add = draw(st.booleans())
        has_relu = draw(st.booleans())

        def generate_data(shape):
            if False:
                while True:
                    i = 10
            return 0.1 * np.random.random(shape).astype(np.float32)
        deconv_op = OpConfig('conv2d_transpose', inputs={'Input': ['input_x'], 'Filter': ['weight_x']}, outputs={'Output': ['output_x']}, data_format='NCHW', dilations=[1, 1], groups=1, paddings=[0, 0], padding_algorithm='EXPLICIT', strides=[4, 4], fuse_relu=False)
        input_name_op = 'output_x'
        ops = [deconv_op]
        if has_add:
            add_op = OpConfig('elementwise_add', inputs={'X': [input_name_op], 'Y': ['bias']}, outputs={'Out': ['add_out']}, axis=1)
            input_name_op = 'add_out'
            ops.append(add_op)
        if has_bn:
            bn_op = OpConfig('batch_norm', inputs={'X': [input_name_op], 'Bias': ['bn_bias'], 'Mean': ['bn_mean'], 'Scale': ['bn_scale'], 'Variance': ['bn_var']}, outputs={'Y': ['bn_y'], 'MeanOut': ['bn_mean'], 'SavedMean': ['bn_mean_save'], 'SavedVariance': ['bn_save_var'], 'VarianceOut': ['bn_var']}, data_layout='NCHW', epsilon=9.999999747378752e-06, momentum=0.89999, is_test=True, use_global_stats=True)
            input_name_op = 'bn_y'
            ops.append(bn_op)
        if has_relu:
            relu_op = OpConfig('relu', inputs={'X': [input_name_op]}, outputs={'Out': ['relu_out']})
            input_name_op = 'relu_out'
            ops.append(relu_op)
        program_config = ProgramConfig(ops=ops, weights={'weight_x': TensorConfig(data_gen=partial(generate_data, weight_shape)), 'bias': TensorConfig(data_gen=partial(generate_data, y_shape)), 'bn_bias': TensorConfig(data_gen=partial(generate_data, y_shape)), 'bn_mean': TensorConfig(data_gen=partial(generate_data, y_shape)), 'bn_scale': TensorConfig(data_gen=partial(generate_data, y_shape)), 'bn_var': TensorConfig(data_gen=partial(generate_data, y_shape))}, inputs={'input_x': TensorConfig(data_gen=partial(generate_data, x_shape))}, outputs=[input_name_op])
        return program_config

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, max_examples=100, passes=['conv2d_transpose_xpu_fuse_pass'])
if __name__ == '__main__':
    unittest.main()