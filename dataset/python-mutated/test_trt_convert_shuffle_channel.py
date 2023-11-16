import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertShuffleChannelTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10

        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if False:
                i = 10
                return i + 15
            return np.ones([batch, 6, 24, 24]).astype(np.float32)
        for batch in [1, 2, 4]:
            for group in [1, 2, 3]:
                dics = [{'group': group}, {}]
                ops_config = [{'op_type': 'shuffle_channel', 'op_inputs': {'X': ['shuffle_channel_input']}, 'op_outputs': {'Out': ['shuffle_channel_out']}, 'op_attrs': dics[0]}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={}, inputs={'shuffle_channel_input': TensorConfig(data_gen=partial(generate_input1, dics, batch))}, outputs=['shuffle_channel_out'])
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {'shuffle_channel_input': [1, 6, 24, 24]}
            self.dynamic_shape.max_input_shape = {'shuffle_channel_input': [4, 6, 48, 48]}
            self.dynamic_shape.opt_input_shape = {'shuffle_channel_input': [1, 6, 24, 48]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8000 and dynamic_shape:
                return (0, 3)
            else:
                return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        self.trt_param.max_batch_size = 9
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.001)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.001)

    def add_skip_trt_case(self):
        if False:
            print('Hello World!')
        pass

    def test(self):
        if False:
            while True:
                i = 10
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()