import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMulLstmFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        x_col = draw(st.sampled_from([1]))
        y_col = draw(st.sampled_from([1]))
        use_peepholes = draw(st.booleans())
        is_reverse = draw(st.booleans())
        gate_activation = draw(st.sampled_from(['sigmoid']))
        cell_activation = draw(st.sampled_from(['tanh', 'relu', 'identity']))
        candidate_activation = draw(st.sampled_from(['tanh', 'relu', 'identity']))
        batch_size = draw(st.integers(min_value=1, max_value=40))

        def generate_input():
            if False:
                while True:
                    i = 10
            shape = [batch_size, 128, 6, 120]
            return np.full(shape, 0.01).astype(np.float32)

        def generate_weight(shape):
            if False:
                for i in range(10):
                    print('nop')
            return np.full(shape, 0.0001).astype(np.float32)
        im2sequence_op = OpConfig(type='im2sequence', inputs={'X': ['input_data']}, outputs={'Out': ['seq_out']}, attrs={'kernels': [6, 1], 'out_stride': [1, 1], 'paddings': [0, 0, 0, 0], 'strides': [1, 1]})
        mul_op = OpConfig(type='mul', inputs={'X': ['seq_out'], 'Y': ['mul_weight']}, outputs={'Out': ['mul_out']}, attrs={'x_num_col_dims': x_col, 'y_num_col_dims': y_col})
        lstm_op = OpConfig(type='lstm', inputs={'Input': ['mul_out'], 'Weight': ['lstm_weight'], 'Bias': ['lstm_bias']}, outputs={'Hidden': ['lstm_hidden'], 'Cell': ['lstm_cell'], 'BatchGate': ['lstm_gate'], 'BatchCellPreAct': ['lstm_batch_cell']}, attrs={'use_peepholes': use_peepholes, 'is_reverse': is_reverse, 'gate_activation': gate_activation, 'cell_activation': cell_activation, 'candidate_activation': candidate_activation, 'is_test': True})
        model_net = [im2sequence_op, mul_op, lstm_op]
        if use_peepholes:
            lstm_bias_shape = [1, 1050]
        else:
            lstm_bias_shape = [1, 600]
        program_config = ProgramConfig(ops=model_net, weights={'mul_weight': TensorConfig(data_gen=partial(generate_weight, [768, 600])), 'lstm_weight': TensorConfig(data_gen=partial(generate_weight, [150, 600])), 'lstm_bias': TensorConfig(data_gen=partial(generate_weight, lstm_bias_shape))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['lstm_hidden'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config()
        yield (config, ['im2sequence', 'fusion_lstm'], (1e-05, 1e-05))

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, max_duration=300, passes=['mul_lstm_fuse_pass'])
if __name__ == '__main__':
    unittest.main()