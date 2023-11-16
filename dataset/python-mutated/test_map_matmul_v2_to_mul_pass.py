import unittest
import hypothesis.strategies as st
from auto_scan_test import IgnoreReasons, PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestMapMatmulToMulPass(PassAutoScanTest):
    """
    x_var    y_var(persistable)
      \\       /
      matmul_v2
    """

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_gpu=False)
        yield (config, ['mul'], (1e-05, 1e-05))
        config = self.create_inference_config(use_gpu=True)
        yield (config, ['mul'], (1e-05, 1e-05))

    def add_ignore_pass_case(self):
        if False:
            for i in range(10):
                print('nop')

        def teller1(program_config, predictor_config):
            if False:
                while True:
                    i = 10
            if predictor_config.tensorrt_engine_enabled():
                return True
                x_shape = list(program_config.inputs['matmul_x'].shape)
                if len(x_shape) > 5:
                    return True
            return False
        self.add_ignore_check_case(teller1, IgnoreReasons.PASS_ACCURACY_ERROR, 'The pass error on TRT while shape of mul_x > 5.')

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=5))
        y_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=2))
        y_shape[0] = x_shape[-1]
        alpha = 1.0
        transpose_X = False
        transpose_Y = False
        matmul_op = OpConfig('matmul_v2', inputs={'X': ['matmul_x'], 'Y': ['matmul_y']}, outputs={'Out': ['matmul_out']}, alpha=alpha, trans_x=transpose_X, trans_y=transpose_Y)
        ops = [matmul_op]
        weights = {'matmul_y': TensorConfig(shape=y_shape)}
        inputs = {'matmul_x': TensorConfig(shape=x_shape)}
        program_config = ProgramConfig(ops=ops, weights=weights, inputs=inputs, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, max_examples=100, passes=['gpu_cpu_map_matmul_v2_to_mul_pass'])
if __name__ == '__main__':
    unittest.main()