import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig
import paddle.inference as paddle_infer

class TestAdaptivePool2dConvertGlobalPass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_config(self, draw):
        if False:
            for i in range(10):
                print('nop')
        x_shape = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=4, max_size=4))
        pooling_type = draw(st.sampled_from(['max', 'avg']))
        data_format = 'NCHW'
        strides = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=2))
        paddings = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=2, max_size=2))
        ceil_mode = draw(st.booleans())
        exclusive = draw(st.booleans())
        global_pooling = draw(st.booleans())
        padding_algorithm = draw(st.sampled_from(['EXPLICIT', 'SAME', 'VAILD']))
        pool_op = OpConfig('pool2d', inputs={'X': ['input_data']}, outputs={'Out': ['pool_output']}, ksize=[1, 1], adaptive=True, pooling_type=pooling_type, data_format=data_format, strides=strides, paddings=paddings, ceil_mode=ceil_mode, global_pooling=global_pooling, padding_algorithm=padding_algorithm, exclusive=exclusive)
        ops = [pool_op]
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(shape=x_shape)}, outputs=['pool_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(max_batch_size=4, workspace_size=102400, min_subgraph_size=0, precision_mode=paddle_infer.PrecisionType.Float32, use_static=False, use_calib_mode=False)
        yield (config, ['pool2d'], (1e-05, 1e-05))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, max_examples=300, passes=['adaptive_pool2d_convert_global_pass'], min_success_num=40)
if __name__ == '__main__':
    unittest.main()