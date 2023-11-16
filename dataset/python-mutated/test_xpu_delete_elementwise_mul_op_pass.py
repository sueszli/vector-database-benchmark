import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestDeleteElementwiseMulOpPass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['relu'], (1e-05, 1e-05))

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=20), min_size=2, max_size=2))
        fill_op = OpConfig('fill_constant_batch_size_like', inputs={'Input': ['fill_x']}, shape=[-1, 1], input_dim_idx=0, output_dim_idx=0, dtype=5, value=1.0, str_value='1', force_cpu=False, outputs={'Out': ['fill_out']})
        mul_op = OpConfig('elementwise_mul', inputs={'X': ['fill_out'], 'Y': ['mul_in']}, axis=0, outputs={'Out': ['mul_out']})
        relu_op = OpConfig('relu', inputs={'X': ['mul_out']}, outputs={'Out': ['relu_out']})
        ops = [fill_op, mul_op, relu_op]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'fill_x': TensorConfig(shape=x_shape), 'mul_in': TensorConfig(shape=x_shape)}, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_and_statis(quant=False, max_examples=25, passes=['delete_elementwise_mul_op_pass'])
if __name__ == '__main__':
    unittest.main()