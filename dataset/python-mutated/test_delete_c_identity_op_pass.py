import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig
import paddle.inference as paddle_infer

class TestDeleteCIdentityPass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(max_batch_size=8, workspace_size=0, min_subgraph_size=0, precision_mode=paddle_infer.PrecisionType.Float32, use_static=False, use_calib_mode=False)
        yield (config, ['relu'], (1e-05, 1e-05))

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        n = draw(st.integers(min_value=1, max_value=2))
        relu_op = OpConfig('relu', inputs={'X': ['relu_x']}, outputs={'Out': ['relu_out']})
        c_identity_op = OpConfig('c_identity', inputs={'X': ['relu_out']}, outputs={'Out': ['id_out']})
        program_config = ProgramConfig(ops=[relu_op, c_identity_op], weights={}, inputs={'relu_x': TensorConfig(shape=[n])}, outputs=['id_out'])
        return program_config

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(max_examples=2, min_success_num=2, passes=['identity_op_clean_pass'])
if __name__ == '__main__':
    unittest.main()