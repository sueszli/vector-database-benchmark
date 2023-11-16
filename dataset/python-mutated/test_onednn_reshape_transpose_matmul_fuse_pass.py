import unittest
from functools import partial, reduce
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import ProgramConfig, TensorConfig

class TestOneDNNReshapeTransposeMatmulFusePass(PassAutoScanTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.num = 32 * 64

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        transpose_X = draw(st.booleans())
        transpose_Y = draw(st.booleans())
        alpha = draw(st.floats(min_value=0.01, max_value=2))
        axis = draw(st.sampled_from([[0, 2, 1, 3]]))
        shape = draw(st.sampled_from([[0, 64, -1, 32], [0, 32, -1, 64], [-1, 32, 1, 64]]))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        channel = draw(st.integers(min_value=1, max_value=64))
        input_dim = draw(st.sampled_from([32, 64]))

        def generate_input1(attrs):
            if False:
                for i in range(10):
                    print('nop')
            shape_x = [attrs[3]['batch_size'], attrs[3]['channel'], self.num]
            return np.random.random(shape_x).astype(np.float32)

        def generate_input2(attrs):
            if False:
                i = 10
                return i + 15
            shape_x = [attrs[3]['batch_size'], attrs[3]['channel'], self.num]
            input_volume = reduce(lambda x, y: x * y, shape_x, 1)
            matmul_shape = list(attrs[0]['shape'])
            if 0 in matmul_shape:
                for i in range(len(matmul_shape)):
                    if matmul_shape[i] == 0:
                        matmul_shape[i] = shape_x[i]
            shape_volume = reduce(lambda x, y: x * y, matmul_shape, 1)
            if -1 in matmul_shape:
                for i in range(len(matmul_shape)):
                    if matmul_shape[i] == -1:
                        matmul_shape[i] = int(abs(input_volume / shape_volume))
            (matmul_shape[1], matmul_shape[2]) = (matmul_shape[2], matmul_shape[1])
            if attrs[2]['transpose_X'] and attrs[2]['transpose_Y']:
                shape_y = [matmul_shape[0], matmul_shape[1], matmul_shape[-1], int(self.num / matmul_shape[-1])]
            elif attrs[2]['transpose_X']:
                shape_y = matmul_shape
            elif attrs[2]['transpose_Y']:
                shape_y = matmul_shape
            else:
                shape_y = [matmul_shape[0], matmul_shape[1], matmul_shape[-1], int(self.num / matmul_shape[-1])]
            return np.random.random(shape_y).astype(np.float32)
        attrs = [{'shape': shape}, {'axis': axis}, {'transpose_X': transpose_X, 'transpose_Y': transpose_Y, 'alpha': alpha}, {'batch_size': batch_size, 'channel': channel, 'input_dim': input_dim}]
        ops_config = [{'op_type': 'reshape2', 'op_inputs': {'X': ['input_data1']}, 'op_outputs': {'Out': ['reshape2_output'], 'XShape': ['reshape2_xshape']}, 'op_attrs': {'shape': attrs[0]['shape']}}, {'op_type': 'transpose2', 'op_inputs': {'X': ['reshape2_output']}, 'op_outputs': {'Out': ['transpose2_output'], 'XShape': ['transpose2_xshape']}, 'op_attrs': {'axis': attrs[1]['axis']}}, {'op_type': 'matmul', 'op_inputs': {'X': ['transpose2_output'], 'Y': ['input_data2']}, 'op_outputs': {'Out': ['matmul_output']}, 'op_attrs': {'transpose_X': attrs[2]['transpose_X'], 'transpose_Y': attrs[2]['transpose_Y'], 'alpha': attrs[2]['alpha']}}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data1': TensorConfig(data_gen=partial(generate_input1, attrs)), 'input_data2': TensorConfig(data_gen=partial(generate_input2, attrs))}, outputs=['matmul_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_mkldnn=True)
        yield (config, ['fused_matmul'], (1e-05, 1e-05))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, passes=['reshape_transpose_matmul_mkldnn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()