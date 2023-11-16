import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig
from paddle.base import core

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestInplaceOpPass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_config(self, draw):
        if False:
            i = 10
            return i + 15

        def generate_input():
            if False:
                print('Hello World!')
            return np.random.random(x_shape).astype(np.float32)

        def generate_tmp1(val):
            if False:
                for i in range(10):
                    print('nop')
            return np.array([val]).astype(np.int32)

        def generate_tmp2(val):
            if False:
                return 10
            return np.array([val]).astype(np.int32)

        def generate_tmp3(val):
            if False:
                return 10
            return np.array([val]).astype(np.int32)

        def generate_shape(val):
            if False:
                while True:
                    i = 10
            return np.array(val).astype(np.int32)
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
        shape = [0, -1, x_shape[-1]]
        scale_op = OpConfig('scale', inputs={'X': ['scale_in']}, outputs={'Out': ['scale_out']}, scale=1.3, bias=0.1, bias_after_scale=False)
        test_case = draw(st.sampled_from(['simple_reshape', 'shape_tensor1', 'shape_tensor2']))
        if test_case == 'simple_reshape':
            reshape_op = OpConfig('reshape2', inputs={'X': ['scale_out']}, outputs={'Out': ['reshape_out'], 'XShape': ['reshape_xshape_out']}, shape=shape)
            ops = [scale_op, reshape_op]
            program_config = ProgramConfig(ops=ops, inputs={'scale_in': TensorConfig(data_gen=partial(generate_input))}, weights={}, outputs=['reshape_out'])
            return program_config
        elif test_case == 'shape_tensor1':
            shape = [-1, -1, x_shape[-1]]
            reshape_op = OpConfig('reshape2', inputs={'X': ['scale_out'], 'ShapeTensor': ['tmp1', 'tmp2', 'tmp3']}, outputs={'Out': ['reshape_out'], 'XShape': ['reshape_xshape_out']}, shape=shape)
            ops = [scale_op, reshape_op]
            program_config = ProgramConfig(ops=ops, inputs={'scale_in': TensorConfig(data_gen=partial(generate_input)), 'tmp1': TensorConfig(data_gen=partial(generate_tmp1, x_shape[0])), 'tmp2': TensorConfig(data_gen=partial(generate_tmp2, x_shape[1] * x_shape[2])), 'tmp3': TensorConfig(data_gen=partial(generate_tmp3, x_shape[-1]))}, weights={}, outputs=['reshape_out'])
            return program_config
        else:
            shape = [0, -1, x_shape[-1]]
            reshape_op = OpConfig('reshape2', inputs={'X': ['scale_out'], 'Shape': ['shape']}, outputs={'Out': ['reshape_out'], 'XShape': ['reshape_xshape_out']}, shape=shape)
            ops = [scale_op, reshape_op]
            program_config = ProgramConfig(ops=ops, inputs={'scale_in': TensorConfig(data_gen=partial(generate_input)), 'shape': TensorConfig(data_gen=partial(generate_shape, [x_shape[0], x_shape[1] * x_shape[2], x_shape[3]]))}, weights={}, outputs=['reshape_out'])
            return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            return 10
        config = self.create_inference_config(use_gpu=True)
        yield (config, ['scale', 'reshape2'], (1e-05, 1e-05))

    def add_ignore_pass_case(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, passes=['inplace_op_var_pass'])
if __name__ == '__main__':
    unittest.main()