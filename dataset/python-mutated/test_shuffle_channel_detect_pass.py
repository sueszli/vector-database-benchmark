import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig
import paddle.inference as paddle_infer

class TestShuffleChannelDetectPass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if attrs[0]['input_shape'] != attrs[2]['shape']:
            return False
        return True

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        batch_size = draw(st.integers(min_value=1, max_value=4))
        out_channel = draw(st.integers(min_value=1, max_value=16))
        group = draw(st.integers(min_value=1, max_value=4))
        in_channel = group * out_channel
        x_shape = [batch_size, in_channel, 64, 64]
        shape = [0, group, out_channel, -1, 64]
        axis_v = [0, 2, 1, 3, 4]

        def generate_reshape2_Input():
            if False:
                print('Hello World!')
            return np.random.random(x_shape).astype(np.float32)
        reshape2_op1 = OpConfig('reshape2', inputs={'X': ['reshape2_input1']}, outputs={'Out': ['reshape2_output1'], 'XShape': ['reshape2_xshape1']}, shape=shape, input_shape=x_shape)
        transpose2_op = OpConfig('transpose2', inputs={'X': ['reshape2_output1']}, outputs={'Out': ['transpose2_output'], 'XShape': ['transpose2_xshape']}, axis=axis_v)
        reshape2_op2 = OpConfig('reshape2', inputs={'X': ['transpose2_output']}, outputs={'Out': ['reshape2_output2'], 'XShape': ['reshape2_xshape2']}, shape=x_shape)
        ops = [reshape2_op1, transpose2_op, reshape2_op2]
        program_config = ProgramConfig(ops=ops, inputs={'reshape2_input1': TensorConfig(data_gen=partial(generate_reshape2_Input))}, weights={}, outputs=['reshape2_output2'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(workspace_size=1 << 20, max_batch_size=4, min_subgraph_size=1, precision_mode=paddle_infer.PrecisionType.Float32, use_static=False, use_calib_mode=False)
        yield (config, ['shuffle_channel'], (1e-05, 1e-05))

    def test(self):
        if False:
            while True:
                i = 10
        self.run_and_statis(quant=False, passes=['shuffle_channel_detect_pass'])
if __name__ == '__main__':
    unittest.main()