import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestTranspose2Reshape2OneDNNFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15

        def generate_input(shape):
            if False:
                i = 10
                return i + 15
            return np.random.random(shape).astype(np.float32)
        channel = draw(st.sampled_from([1, 2, 4]))
        axis = draw(st.sampled_from([[0, 1, 2, 3], [2, 1, 3, 0], [3, 2, 1, 0]]))
        shape = draw(st.sampled_from([[channel, 512, 64], [256, 128, channel], [channel, 1024, 32]]))
        transpose2_op = OpConfig(type='transpose2', inputs={'X': ['transpose_x']}, outputs={'Out': ['transpose_out'], 'XShape': ['transpose2_xshape']}, attrs={'axis': axis, 'use_mkldnn': True})
        reshape2_op = OpConfig(type='reshape2', inputs={'X': ['transpose_out']}, outputs={'Out': ['reshape_out']}, attrs={'shape': shape})
        model_net = [transpose2_op, reshape2_op]
        program_config = ProgramConfig(ops=model_net, weights={}, inputs={'transpose_x': TensorConfig(data_gen=partial(generate_input, [channel, 16, 64, 32]))}, outputs=['reshape_out'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            while True:
                i = 10
        config = self.create_inference_config(use_mkldnn=True, passes=['operator_reshape2_onednn_fuse_pass'])
        yield (config, ['fused_transpose'], (1e-05, 1e-05))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, passes=['operator_reshape2_onednn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()