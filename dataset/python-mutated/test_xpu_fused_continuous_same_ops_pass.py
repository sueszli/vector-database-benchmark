import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestFusedSameUnSqueezePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['scale', 'unsqueeze2'], (1e-05, 1e-05))

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15
        scale_x = draw(st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=3))
        first_unsqueeze_axis = 0
        second_unsqueeze_axis = 1
        third_unsqueeze_axis = 2
        scale_op0 = OpConfig('scale', inputs={'X': ['scale_x']}, scale=2.0, bias=1.0, bias_after_scale=True, outputs={'Out': ['scale0_out']})
        unsqueeze_op0 = OpConfig('unsqueeze2', inputs={'X': ['scale0_out']}, axes=[first_unsqueeze_axis], outputs={'Out': ['unsqueeze0_out']})
        unsqueeze_op1 = OpConfig('unsqueeze2', inputs={'X': ['unsqueeze0_out']}, axes=[second_unsqueeze_axis], outputs={'Out': ['unsqueeze1_out']})
        unsqueeze_op2 = OpConfig('unsqueeze2', inputs={'X': ['unsqueeze1_out']}, axes=[third_unsqueeze_axis], outputs={'Out': ['unsqueeze2_out']})
        ops = [scale_op0, unsqueeze_op0, unsqueeze_op1, unsqueeze_op2]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'scale_x': TensorConfig(shape=scale_x)}, outputs=['unsqueeze2_out'])
        return program_config

class TestFusedSameReshapePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['scale', 'reshape2'], (1e-05, 1e-05))

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10
        scale_x = draw(st.sampled_from([[8, 16], [16, 32], [64, 16], [16, 8], [16, 16]]))
        first_reshape_shape = [-1, 16, 4]
        second_reshape_shape = [-1, 8]
        scale_op0 = OpConfig('scale', inputs={'X': ['scale_x']}, scale=2.0, bias=1.0, bias_after_scale=True, outputs={'Out': ['scale0_out']})
        reshape_op0 = OpConfig('reshape2', inputs={'X': ['scale0_out']}, shape=first_reshape_shape, outputs={'Out': ['reshape0_out']})
        reshape_op1 = OpConfig('reshape2', inputs={'X': ['reshape0_out']}, shape=second_reshape_shape, outputs={'Out': ['reshape1_out']})
        ops = [scale_op0, reshape_op0, reshape_op1]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'scale_x': TensorConfig(shape=scale_x)}, outputs=['reshape1_out'])
        return program_config

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, max_examples=25, min_success_num=5, passes=['fused_continuous_same_ops_pass'])
if __name__ == '__main__':
    unittest.main()