import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestXpuMatmulV2WeightTransPass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['matmul_v2'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=3, max_size=3))
        transpose_shape = x_shape
        transpose_op = OpConfig('transpose2', inputs={'X': ['transpose_input']}, outputs={'Out': ['transpose_out']}, axis=[0, 2, 1])
        matmul_op = OpConfig('matmul_v2', inputs={'X': ['matmul_x'], 'Y': ['transpose_out']}, outputs={'Out': ['matmul_out']}, transpose_X=False, transpose_Y=False)
        ops = [transpose_op, matmul_op]
        weights = {}
        inputs = {'matmul_x': TensorConfig(shape=x_shape), 'transpose_input': TensorConfig(shape=transpose_shape)}
        program_config = ProgramConfig(ops=ops, weights=weights, inputs=inputs, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, max_examples=25, min_success_num=5, passes=['matmul_weight_trans_pass'])
if __name__ == '__main__':
    unittest.main()