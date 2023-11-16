import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestFcXPUFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['fc_xpu'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            return 10
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=4))
        trans_x = False
        trans_y = draw(st.booleans())
        y_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=2))
        if trans_y:
            y_shape[1] = x_shape[-1]
        else:
            y_shape[0] = x_shape[-1]
        axis = -1
        bias_shape = [y_shape[0]] if trans_y else [y_shape[1]]
        has_relu = draw(st.booleans())
        matmul_v2_op = OpConfig('matmul_v2', inputs={'X': ['matmul_v2_x'], 'Y': ['matmul_v2_y']}, outputs={'Out': ['matmul_v2_out']}, trans_x=trans_x, trans_y=trans_y)
        add_op = OpConfig('elementwise_add', inputs={'X': ['matmul_v2_out'], 'Y': ['bias']}, outputs={'Out': ['add_out']}, axis=axis)
        ops = [matmul_v2_op, add_op]
        if has_relu:
            relu_op = OpConfig('relu', inputs={'X': ['add_out']}, outputs={'Out': ['relu_out']})
            ops.append(relu_op)
        program_config = ProgramConfig(ops=ops, weights={'matmul_v2_y': TensorConfig(shape=y_shape), 'bias': TensorConfig(shape=bias_shape)}, inputs={'matmul_v2_x': TensorConfig(shape=x_shape)}, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            return 10
        self.run_and_statis(quant=False, max_examples=25, passes=['fc_xpu_fuse_pass'])
if __name__ == '__main__':
    unittest.main()