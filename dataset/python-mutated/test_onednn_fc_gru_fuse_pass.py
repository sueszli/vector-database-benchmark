import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestOneDNNFCGruFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10

        def generate_input(shape):
            if False:
                return 10
            return np.random.random(shape).astype(np.float32)
        batch_size = draw(st.integers(min_value=1, max_value=16))
        fc_input_shape = [batch_size, 64]
        fc_weight_shape = [64, 192]
        fc_bias_shape = [1, 192]
        lod = [[0, batch_size]]
        gru_weight_shape = [64, 192]
        gru_bias_shape = [1, 192]
        activation = draw(st.sampled_from(['tanh']))
        is_reverse = draw(st.booleans())
        gate_activation = draw(st.sampled_from(['sigmoid']))
        mul_op = OpConfig(type='mul', inputs={'X': ['fc_input'], 'Y': ['fc_weight']}, outputs={'Out': ['mul_out']}, attrs={'x_num_col_dims': 1, 'y_num_col_dims': 1})
        elt_op = OpConfig(type='elementwise_add', inputs={'X': ['mul_out'], 'Y': ['fc_bias']}, outputs={'Out': ['fc_output']}, attrs={'axis': -1})
        gru_op = OpConfig(type='gru', inputs={'Input': ['fc_output'], 'Weight': ['gru_weight'], 'Bias': ['gru_bias']}, outputs={'BatchGate': ['batch_gate'], 'BatchHidden': ['batch_hidden'], 'BatchResetHiddenPrev': ['batch_reset'], 'Hidden': ['gru_hidden']}, attrs={'activation': activation, 'is_reverse': is_reverse, 'gate_activation': gate_activation, 'is_test': True})
        model_net = [mul_op, elt_op, gru_op]
        program_config = ProgramConfig(ops=model_net, inputs={'fc_input': TensorConfig(lod=lod, data_gen=partial(generate_input, fc_input_shape))}, weights={'fc_weight': TensorConfig(data_gen=partial(generate_input, fc_weight_shape)), 'fc_bias': TensorConfig(data_gen=partial(generate_input, fc_bias_shape)), 'gru_weight': TensorConfig(data_gen=partial(generate_input, gru_weight_shape)), 'gru_bias': TensorConfig(data_gen=partial(generate_input, gru_bias_shape))}, outputs=['gru_hidden'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_mkldnn=True, passes=['mkldnn_placement_pass', 'fc_gru_fuse_pass'])
        yield (config, ['fusion_gru'], (1e-05, 1e-05))

    def test(self):
        if False:
            return 10
        self.run_and_statis(quant=False, passes=['mkldnn_placement_pass', 'fc_gru_fuse_pass'], max_examples=100)
if __name__ == '__main__':
    unittest.main()