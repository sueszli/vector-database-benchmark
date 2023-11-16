import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestScaleOneDNNFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        data_layout = draw(st.sampled_from(['NCHW', 'NHWC']))
        epsilon = draw(st.floats(min_value=0.0, max_value=0.001))
        fuse_with_relu = draw(st.booleans())
        is_test = draw(st.sampled_from([True]))
        momentum = draw(st.floats(min_value=0.0, max_value=5))
        trainable_statistics = False
        use_global_stats = draw(st.booleans())
        use_mkldnn1 = draw(st.sampled_from([True]))
        use_cudnn = draw(st.booleans())
        use_mkldnn2 = draw(st.sampled_from([True]))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        channel = draw(st.integers(min_value=1, max_value=64))
        input_dim1 = draw(st.integers(min_value=1, max_value=512))
        input_dim2 = draw(st.integers(min_value=1, max_value=512))

        def generate_input():
            if False:
                for i in range(10):
                    print('nop')
            shape = [input_dim1, input_dim2]
            if data_layout == 'NCHW':
                shape.insert(0, channel)
                shape.insert(0, batch_size)
            else:
                shape.append(channel)
                shape.insert(0, batch_size)
            return np.random.random(shape).astype(np.float32)

        def generate_weight():
            if False:
                i = 10
                return i + 15
            return np.random.random(channel).astype(np.float32)
        batch_norm_op = OpConfig(type='batch_norm', inputs={'X': ['input_data'], 'Bias': ['Bias'], 'Mean': ['Mean'], 'Scale': ['Scale'], 'Variance': ['Variance']}, outputs={'Y': ['norm_output'], 'MeanOut': ['Mean'], 'VarianceOut': ['Variance'], 'SavedMean': ['SavedMean'], 'SavedVariance': ['SavedVariance']}, attrs={'data_layout': data_layout, 'epsilon': epsilon, 'fuse_with_relu': fuse_with_relu, 'is_test': is_test, 'momentum': momentum, 'trainable_statistics': trainable_statistics, 'use_global_stats': use_global_stats, 'use_mkldnn': use_mkldnn1})
        relu_op = OpConfig(type='relu', inputs={'X': ['norm_output']}, outputs={'Out': ['relu_output']}, attrs={'use_cudnn': use_cudnn, 'use_mkldnn': use_mkldnn2})
        model_net = [batch_norm_op, relu_op]
        program_config = ProgramConfig(ops=model_net, weights={'Bias': TensorConfig(data_gen=partial(generate_weight)), 'Mean': TensorConfig(data_gen=partial(generate_weight)), 'Scale': TensorConfig(data_gen=partial(generate_weight)), 'Variance': TensorConfig(data_gen=partial(generate_weight))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['relu_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            return 10
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, ['batch_norm'], (1e-05, 1e-05))

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_and_statis(quant=False, passes=['batch_norm_act_fuse_pass'])
if __name__ == '__main__':
    unittest.main()