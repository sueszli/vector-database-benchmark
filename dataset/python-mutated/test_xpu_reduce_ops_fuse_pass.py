import unittest
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestReduceMaxFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['reduce_max'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        s_axes = [2]
        batch_size = draw(st.integers(min_value=1, max_value=4))
        H = draw(st.integers(min_value=1, max_value=64))
        W = draw(st.integers(min_value=1, max_value=64))
        in_shape = [batch_size, H, W]
        transpose_op1 = OpConfig(type='transpose2', inputs={'X': ['transpose_in']}, outputs={'Out': ['transpose_out1']}, attrs={'axis': [0, 2, 1]})
        unsqueeze2_op = OpConfig(type='unsqueeze2', inputs={'X': ['transpose_out1']}, outputs={'Out': ['unsqueeze_out']}, attrs={'axes': s_axes})
        pool_op = OpConfig('pool2d', inputs={'X': ['unsqueeze_out']}, outputs={'Out': ['pool_out']}, ksize=[1, H], adaptive=False, pooling_type='max', data_format='NCHW', strides=[1, H], paddings=[0, 0], ceil_mode=False, global_pooling=False, padding_algorithm='EXPLICIT', exclusive=True)
        squeeze2_op = OpConfig('squeeze2', inputs={'X': ['pool_out']}, axes=s_axes, outputs={'Out': ['squeeze2_out'], 'XShape': ['xshape']})
        transpose_op2 = OpConfig(type='transpose2', inputs={'X': ['squeeze2_out']}, outputs={'Out': ['transpose_out2']}, attrs={'axis': [0, 2, 1]})
        ops = [transpose_op1, unsqueeze2_op, pool_op, squeeze2_op, transpose_op2]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'transpose_in': TensorConfig(shape=in_shape)}, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, max_examples=25, passes=['reduce_ops_fuse_pass'])
if __name__ == '__main__':
    np.random.seed(200)
    unittest.main()