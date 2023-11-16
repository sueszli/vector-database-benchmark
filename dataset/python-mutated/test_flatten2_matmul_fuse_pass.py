import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestFlatten2MatmulFusePass(PassAutoScanTest):
    """
        x_var
          |
       flatten2
          \\
    flatten2_out_var    y_var
             \\           /
                 matmul      bias_var
                    \\          /
                   elementwise_add
    """

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_gpu=False)
        yield (config, ['mul', 'elementwise_add'], (1e-05, 1e-05))
        config = self.create_inference_config(use_gpu=True)
        yield (config, ['mul', 'elementwise_add'], (1e-05, 1e-05))

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
        flatten_axis = 1
        flatten_shape = [x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]]
        alpha = 1.0
        transpose_X = False
        transpose_Y = False
        y_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=2))
        y_shape[0] = flatten_shape[1]
        axis = draw(st.integers(min_value=-1, max_value=1))
        if axis == 0:
            bias_shape = [flatten_shape[0]]
        elif axis == 1:
            bias_shape = [y_shape[1]]
        else:
            bias_shape = [flatten_shape[0], y_shape[1]]
            if draw(st.booleans()):
                bias_shape[1] = 1
        flatten2_op = OpConfig('flatten2', inputs={'X': ['flatten2_x']}, axis=flatten_axis, outputs={'Out': ['flatten2_out'], 'XShape': ['xshape']})
        matmul_op = OpConfig('matmul', inputs={'X': ['flatten2_out'], 'Y': ['matmul_y']}, outputs={'Out': ['matmul_out']}, alpha=alpha, transpose_X=transpose_X, transpose_Y=transpose_Y)
        add_op = OpConfig('elementwise_add', inputs={'X': ['matmul_out'], 'Y': ['bias']}, outputs={'Out': ['add_out']}, axis=axis)
        ops = [flatten2_op, matmul_op, add_op]
        if draw(st.integers(min_value=1, max_value=10)) <= 8:
            program_config = ProgramConfig(ops=ops, weights={'matmul_y': TensorConfig(shape=y_shape), 'bias': TensorConfig(shape=bias_shape)}, inputs={'flatten2_x': TensorConfig(shape=x_shape)}, outputs=ops[-1].outputs['Out'])
        else:
            program_config = ProgramConfig(ops=ops, weights={}, inputs={'flatten2_x': TensorConfig(shape=x_shape), 'matmul_y': TensorConfig(shape=y_shape), 'bias': TensorConfig(shape=bias_shape)}, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, max_examples=50, max_duration=1000, passes=['gpu_cpu_flatten2_matmul_fuse_pass'])
if __name__ == '__main__':
    unittest.main()