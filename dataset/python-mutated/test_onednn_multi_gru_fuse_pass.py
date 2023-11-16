import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestOneDNNMultiGruFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        if False:
            return 10
        input_dim_1 = draw(st.integers(min_value=1, max_value=128)) * 3
        input_dim_2 = input_dim_1 // 3
        frame_size = draw(st.integers(min_value=1, max_value=128))
        weight_x_shape = [input_dim_2, frame_size * 3]
        weight_h_shape = [frame_size, frame_size * 3]
        with_bias = draw(st.booleans())
        bias_shape = [1, frame_size * 3]
        lod = [[0, input_dim_1]]

        def generate_data(shape):
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random(shape).astype(np.float32)

        def generate_bias(shape):
            if False:
                return 10
            if with_bias:
                return np.random.random(shape).astype(np.float32)
            else:
                return np.zeros(shape).astype(np.float32)
        fusion_gru_op_1 = OpConfig(type='fusion_gru', inputs={'X': ['input_data'], 'WeightX': ['weight_x_1'], 'WeightH': ['weight_h_1'], 'Bias': ['bias_1']}, outputs={'Hidden': ['fusion_gru_output_hidden_1'], 'XX': ['fusion_gru_output_xx_1']}, attrs={'origin_mode': False, 'is_reverse': False})
        fusion_gru_op_2 = OpConfig(type='fusion_gru', inputs={'X': ['input_data'], 'WeightX': ['weight_x_2'], 'WeightH': ['weight_h_2'], 'Bias': ['bias_2']}, outputs={'Hidden': ['fusion_gru_output_hidden_2'], 'XX': ['fusion_gru_output_xx_2']}, attrs={'origin_mode': False, 'is_reverse': True})
        concat_op = OpConfig(type='concat', inputs={'X': ['fusion_gru_output_hidden_1', 'fusion_gru_output_hidden_2']}, outputs={'Out': ['concat_output']}, attrs={'axis': 1})
        program_config = ProgramConfig(ops=[fusion_gru_op_1, fusion_gru_op_2, concat_op], weights={'weight_x_1': TensorConfig(data_gen=partial(generate_data, weight_x_shape)), 'weight_h_1': TensorConfig(data_gen=partial(generate_data, weight_h_shape)), 'bias_1': TensorConfig(data_gen=partial(generate_bias, bias_shape)), 'weight_x_2': TensorConfig(data_gen=partial(generate_data, weight_x_shape)), 'weight_h_2': TensorConfig(data_gen=partial(generate_data, weight_h_shape)), 'bias_2': TensorConfig(data_gen=partial(generate_data, bias_shape))}, inputs={'input_data': TensorConfig(lod=lod, data_gen=partial(generate_data, [input_dim_1, input_dim_2]))}, outputs=['concat_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_mkldnn=True, passes=['multi_gru_fuse_pass'])
        yield (config, ['multi_gru'], (1e-05, 1e-05))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, passes=['multi_gru_fuse_pass'])
if __name__ == '__main__':
    unittest.main()