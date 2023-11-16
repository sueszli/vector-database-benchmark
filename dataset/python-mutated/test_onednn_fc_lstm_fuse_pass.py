import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestOneDNNFCLstmFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15
        batch_size = draw(st.integers(min_value=1, max_value=16))
        fc_input_shape = [batch_size, 64]
        fc_weight_shape = [64, 256]
        fc_bias_shape = [1, 256]
        lod = [[0, batch_size]]
        use_peepholes = draw(st.booleans())
        is_reverse = draw(st.booleans())
        gate_activation = draw(st.sampled_from(['sigmoid']))
        cell_activation = draw(st.sampled_from(['tanh']))
        candidate_activation = draw(st.sampled_from(['tanh']))
        lstm_weight_shape = [64, 256]
        lstm_bias_shape = [1, 448] if use_peepholes else [1, 256]
        mul_op = OpConfig(type='mul', inputs={'X': ['fc_input'], 'Y': ['fc_weight']}, outputs={'Out': ['mul_out']}, attrs={'x_num_col_dims': 1, 'y_num_col_dims': 1})
        elt_op = OpConfig(type='elementwise_add', inputs={'X': ['mul_out'], 'Y': ['fc_bias']}, outputs={'Out': ['fc_output']}, attrs={'axis': -1})
        lstm_op = OpConfig(type='lstm', inputs={'Input': ['fc_output'], 'Weight': ['lstm_weight'], 'Bias': ['lstm_bias']}, outputs={'Hidden': ['lstm_hidden'], 'Cell': ['lstm_cell'], 'BatchGate': ['lstm_gate'], 'BatchCellPreAct': ['lstm_batch_cell']}, attrs={'use_peepholes': use_peepholes, 'is_reverse': is_reverse, 'gate_activation': gate_activation, 'cell_activation': cell_activation, 'candidate_activation': candidate_activation, 'is_test': True})
        model_net = [mul_op, elt_op, lstm_op]

        def generate_data(shape):
            if False:
                print('Hello World!')
            return np.random.random(shape).astype(np.float32)
        program_config = ProgramConfig(ops=model_net, inputs={'fc_input': TensorConfig(lod=lod, data_gen=partial(generate_data, fc_input_shape))}, weights={'fc_weight': TensorConfig(data_gen=partial(generate_data, fc_weight_shape)), 'fc_bias': TensorConfig(data_gen=partial(generate_data, fc_bias_shape)), 'lstm_weight': TensorConfig(data_gen=partial(generate_data, lstm_weight_shape)), 'lstm_bias': TensorConfig(data_gen=partial(generate_data, lstm_bias_shape))}, outputs=['lstm_hidden'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            return 10
        config = self.create_inference_config(use_mkldnn=True, passes=['mkldnn_placement_pass', 'fc_lstm_fuse_pass'])
        yield (config, ['fusion_lstm'], (1e-05, 1e-05))

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_and_statis(quant=False, passes=['mkldnn_placement_pass', 'fc_lstm_fuse_pass'], max_examples=50)
if __name__ == '__main__':
    unittest.main()