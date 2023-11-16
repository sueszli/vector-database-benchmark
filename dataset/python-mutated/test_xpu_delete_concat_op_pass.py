import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestDeleteConcatOpPass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['relu'], (1e-05, 1e-05))

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=4))
        relu_op = OpConfig('relu', inputs={'X': ['relu_x']}, outputs={'Out': ['relu_out']})
        concat_op = OpConfig('concat', inputs={'X': ['relu_out']}, axis=0, outputs={'Out': ['concat_out']})
        ops = [relu_op, concat_op]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'relu_x': TensorConfig(shape=x_shape)}, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_and_statis(quant=False, max_examples=25, passes=['delete_concat_op_pass'])
if __name__ == '__main__':
    unittest.main()