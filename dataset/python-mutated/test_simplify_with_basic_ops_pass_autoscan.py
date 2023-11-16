import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig
import paddle.inference as paddle_infer

class TestSimplifyWithBasicOpsPassUpscale(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        fix_seed = draw(st.booleans())
        dropout_implementation = 'upscale_in_train'
        dropout_prob = draw(st.floats(min_value=0.0, max_value=1.0))
        seed = draw(st.integers(min_value=0, max_value=512))
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=4))
        is_test = True
        dropout_op = OpConfig('dropout', inputs={'X': ['input_data']}, outputs={'Out': ['dropout_output'], 'Mask': ['mask']}, fix_seed=fix_seed, dropout_implementation=dropout_implementation, dropout_prob=dropout_prob, seed=seed, is_test=is_test)
        relu_op = OpConfig('relu', inputs={'X': ['dropout_output']}, outputs={'Out': ['relu_out']})
        ops = [dropout_op, relu_op]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(shape=x_shape)}, outputs=['relu_out'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            i = 10
            return i + 15
        config = self.create_inference_config(use_gpu=True)
        yield (config, ['relu'], (1e-05, 1e-05))
        config = self.create_inference_config(use_gpu=False)
        yield (config, ['relu'], (1e-05, 1e-05))
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(max_batch_size=4, workspace_size=102400, min_subgraph_size=0, precision_mode=paddle_infer.PrecisionType.Float32, use_static=False, use_calib_mode=False)
        yield (config, ['relu'], (1e-05, 1e-05))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, max_examples=30, passes=['simplify_with_basic_ops_pass'], min_success_num=30)

class TestSimplifyWithBasicOpsPassDowngrade(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_config(self, draw):
        if False:
            return 10
        fix_seed = draw(st.booleans())
        dropout_implementation = 'downgrade_in_infer'
        dropout_prob = draw(st.floats(min_value=0.0, max_value=1.0))
        seed = draw(st.integers(min_value=0, max_value=512))
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=4))
        is_test = True
        dropout_op = OpConfig('dropout', inputs={'X': ['input_data']}, outputs={'Out': ['dropout_output'], 'Mask': ['mask']}, fix_seed=fix_seed, dropout_implementation=dropout_implementation, dropout_prob=dropout_prob, seed=seed, is_test=is_test)
        relu_op = OpConfig('relu', inputs={'X': ['dropout_output']}, outputs={'Out': ['relu_out']})
        ops = [dropout_op, relu_op]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(shape=x_shape)}, outputs=['relu_out'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_gpu=True)
        yield (config, ['scale', 'relu'], (1e-05, 1e-05))
        config = self.create_inference_config(use_gpu=False)
        yield (config, ['scale', 'relu'], (1e-05, 1e-05))
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(max_batch_size=4, workspace_size=102400, min_subgraph_size=0, precision_mode=paddle_infer.PrecisionType.Float32, use_static=False, use_calib_mode=False)
        yield (config, ['scale', 'relu'], (1e-05, 1e-05))

    def test(self):
        if False:
            return 10
        self.run_and_statis(quant=False, max_examples=30, passes=['simplify_with_basic_ops_pass'], min_success_num=30)
if __name__ == '__main__':
    unittest.main()