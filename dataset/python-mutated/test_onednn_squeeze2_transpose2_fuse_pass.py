import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestSqueeze2Transpose2OneDNNFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15

        def generate_input(shape):
            if False:
                i = 10
                return i + 15
            return np.random.random(shape).astype(np.float32)
        channel = draw(st.sampled_from([1, 2, 4, 8, 16]))
        transpose_axis = draw(st.sampled_from([[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0], [2, 1, 0]]))
        squeeze2_op = OpConfig(type='squeeze2', inputs={'X': ['squeeze_x']}, outputs={'Out': ['squeeze_out'], 'XShape': ['squeeze2_xshape']}, attrs={'axes': [2], 'use_mkldnn': True})
        transpose2_op = OpConfig(type='transpose2', inputs={'X': ['squeeze_out']}, outputs={'Out': ['trans_out'], 'XShape': ['transpose2_xshape']}, attrs={'axis': transpose_axis, 'use_mkldnn': True})
        model_net = [squeeze2_op, transpose2_op]
        program_config = ProgramConfig(ops=model_net, weights={}, inputs={'squeeze_x': TensorConfig(data_gen=partial(generate_input, [channel, 16, 1, 32]))}, outputs=['trans_out'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_mkldnn=True, passes=['squeeze2_transpose2_onednn_fuse_pass'])
        yield (config, ['fused_transpose'], (1e-05, 1e-05))

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, passes=['squeeze2_transpose2_onednn_fuse_pass'])
if __name__ == '__main__':
    unittest.main()