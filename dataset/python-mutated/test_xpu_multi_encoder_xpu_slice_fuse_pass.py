import unittest
import numpy as np
from program_config import OpConfig
from test_xpu_multi_encoder_xpu_fuse_pass import TestMultiEncoderXPUFusePass

class TestMultiEncoderXPUFusePass(TestMultiEncoderXPUFusePass):

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        slice_op = OpConfig('slice', inputs={'Input': ['ln_2_out']}, outputs={'Out': ['slice_out']}, axes=[1], decrease_axis=[1], starts=[0], ends=[1])
        program_config = self.multi_encoder_xpu_program_config(draw)
        program_config.ops.append(slice_op)
        program_config.outputs = ['slice_out']
        return program_config

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_and_statis(quant=False, max_examples=2, min_success_num=2, passes=['multi_encoder_xpu_fuse_pass', 'multi_encoder_xpu_slice_fuse_pass'])
if __name__ == '__main__':
    np.random.seed(200)
    unittest.main()