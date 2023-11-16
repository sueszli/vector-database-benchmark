import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestDeleteDropoutOpPass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['elementwise_add', 'relu', 'relu6'], (1e-05, 1e-05))

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        add_op = OpConfig('elementwise_add', inputs={'X': ['add_x'], 'Y': ['add_y']}, outputs={'Out': ['add_out']}, axis=-1)
        dropout_op = OpConfig('dropout', inputs={'X': ['add_out']}, outputs={'Out': ['dropout_out'], 'Mask': ['dropout_mask']}, dropout_implementation='upscale_in_train', dropout_prob=0.1, fix_seed=False, is_test=True, seed=0)
        relu_op = OpConfig('relu', inputs={'X': ['dropout_out']}, outputs={'Out': ['relu_out']})
        relu6_op = OpConfig('relu6', inputs={'X': ['dropout_out']}, outputs={'Out': ['relu6_out']})
        ops = [add_op, dropout_op, relu_op, relu6_op]
        add_x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=4))
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'add_x': TensorConfig(shape=add_x_shape), 'add_y': TensorConfig(shape=add_x_shape)}, outputs=['relu_out', 'relu6_out'])
        return program_config

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, max_examples=1, min_success_num=1, passes=['delete_dropout_op_pass'])
if __name__ == '__main__':
    unittest.main()