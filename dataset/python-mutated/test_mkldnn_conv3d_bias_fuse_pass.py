import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import ProgramConfig, TensorConfig

class TestConv3dBiasMkldnnFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15
        data_format = draw(st.sampled_from(['NCDHW', 'NDHWC']))
        dilations = draw(st.sampled_from([[1, 1, 1], [2, 2, 2], [1, 2, 1]]))
        padding_algorithm = draw(st.sampled_from(['EXPLICIT', 'SAME', 'VALID']))
        groups = draw(st.sampled_from([1, 2, 4]))
        paddings = draw(st.sampled_from([[0, 3, 2], [1, 2, 3, 4, 3, 1]]))
        strides = draw(st.sampled_from([[1, 1, 1], [2, 2, 2], [1, 2, 1]]))
        axis = draw(st.sampled_from([1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input1(attrs):
            if False:
                for i in range(10):
                    print('nop')
            if attrs[0]['data_format'] == 'NCDHW':
                return np.random.random([attrs[2]['batch_size'], 48, 64, 32, 64]).astype(np.float32)
            else:
                return np.random.random([attrs[2]['batch_size'], 64, 32, 64, 48]).astype(np.float32)

        def generate_weight1():
            if False:
                i = 10
                return i + 15
            return np.random.random([16, int(48 / groups), 3, 3, 3]).astype(np.float32)

        def generate_weight2():
            if False:
                i = 10
                return i + 15
            return np.random.random([16]).astype(np.float32)
        attrs = [{'data_format': data_format, 'dilations': dilations, 'padding_algorithm': padding_algorithm, 'groups': groups, 'paddings': paddings, 'strides': strides}, {'axis': axis}, {'batch_size': batch_size}]
        ops_config = [{'op_type': 'conv3d', 'op_inputs': {'Input': ['input_data1'], 'Filter': ['conv_weight']}, 'op_outputs': {'Output': ['conv_output']}, 'op_attrs': {'data_format': attrs[0]['data_format'], 'dilations': attrs[0]['dilations'], 'padding_algorithm': attrs[0]['padding_algorithm'], 'groups': attrs[0]['groups'], 'paddings': attrs[0]['paddings'], 'strides': attrs[0]['strides'], 'is_test': True}}, {'op_type': 'elementwise_add', 'op_inputs': {'X': ['conv_output'], 'Y': ['elementwise_weight']}, 'op_outputs': {'Out': ['elementwise_output']}, 'op_attrs': {'axis': attrs[1]['axis']}}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={'conv_weight': TensorConfig(data_gen=partial(generate_weight1)), 'elementwise_weight': TensorConfig(data_gen=partial(generate_weight2))}, inputs={'input_data1': TensorConfig(data_gen=partial(generate_input1, attrs))}, outputs=['elementwise_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            while True:
                i = 10
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, ['conv3d'], (1e-05, 1e-05))

    def test(self):
        if False:
            i = 10
            return i + 15
        pass
if __name__ == '__main__':
    unittest.main()