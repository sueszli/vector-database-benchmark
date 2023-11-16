import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMulGruFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        x_col = draw(st.sampled_from([1]))
        y_col = draw(st.sampled_from([1]))
        activation = draw(st.sampled_from(['sigmoid', 'tanh']))
        is_reverse = draw(st.booleans())
        has_origin_mode = draw(st.booleans())
        origin_mode = False
        gate_activation = draw(st.sampled_from(['sigmoid', 'tanh']))
        batch_size = draw(st.integers(min_value=1, max_value=40))

        def generate_input():
            if False:
                for i in range(10):
                    print('nop')
            shape = [batch_size, 128, 6, 120]
            return np.full(shape, 0.001).astype(np.float32)

        def generate_weight(shape):
            if False:
                for i in range(10):
                    print('nop')
            return np.full(shape, 0.0001).astype(np.float32)
        im2sequence_op = OpConfig(type='im2sequence', inputs={'X': ['input_data']}, outputs={'Out': ['seq_out']}, attrs={'kernels': [6, 1], 'out_stride': [1, 1], 'paddings': [0, 0, 0, 0], 'strides': [1, 1]})
        mul_op = OpConfig(type='mul', inputs={'X': ['seq_out'], 'Y': ['mul_weight']}, outputs={'Out': ['mul_out']}, attrs={'x_num_col_dims': x_col, 'y_num_col_dims': y_col})
        if has_origin_mode:
            gru_op = OpConfig(type='gru', inputs={'Input': ['mul_out'], 'Weight': ['gru_weight'], 'Bias': ['gru_bias']}, outputs={'BatchGate': ['batch_gate'], 'BatchHidden': ['batch_hidden'], 'BatchResetHiddenPrev': ['batch_reset'], 'Hidden': ['hidden']}, attrs={'activation': activation, 'is_reverse': is_reverse, 'gate_activation': gate_activation, 'is_test': True, 'origin_mode': origin_mode})
        else:
            gru_op = OpConfig(type='gru', inputs={'Input': ['mul_out'], 'Weight': ['gru_weight'], 'Bias': ['gru_bias']}, outputs={'BatchGate': ['batch_gate'], 'BatchHidden': ['batch_hidden'], 'BatchResetHiddenPrev': ['batch_reset'], 'Hidden': ['hidden']}, attrs={'activation': activation, 'is_reverse': is_reverse, 'gate_activation': gate_activation, 'is_test': True})
        model_net = [im2sequence_op, mul_op, gru_op]
        program_config = ProgramConfig(ops=model_net, weights={'mul_weight': TensorConfig(data_gen=partial(generate_weight, [768, 600])), 'gru_weight': TensorConfig(data_gen=partial(generate_weight, [200, 600])), 'gru_bias': TensorConfig(data_gen=partial(generate_weight, [1, 600]))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['hidden'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            return 10
        config = self.create_inference_config()
        yield (config, ['im2sequence', 'fusion_gru'], (1e-05, 1e-05))

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, max_duration=300, passes=['mul_gru_fuse_pass'])
if __name__ == '__main__':
    unittest.main()