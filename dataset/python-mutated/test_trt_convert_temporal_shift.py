import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertTemporalShiftTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input1(attrs):
            if False:
                while True:
                    i = 10
            T = attrs[0]['seg_num']
            shape = [2 * T, 10, 64, 64]
            return np.random.uniform(low=0.1, high=1.0, size=shape).astype(np.float32)
        for shift_value in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49]:
            for T in range(2, 5):
                for data_format in ['NCHW', 'NHWC']:
                    dics = [{'shift_ratio': shift_value, 'seg_num': T, 'data_format': data_format}, {}]
                    ops_config = [{'op_type': 'temporal_shift', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    for i in range(10):
                        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dics))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            t = attrs[0]['seg_num']
            self.dynamic_shape.min_input_shape = {'input_data': [2 * t, 10, 64, 64]}
            self.dynamic_shape.max_input_shape = {'input_data': [5 * t, 10, 64, 64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [3 * t, 10, 64, 64]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, is_dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
            valid_version = (8, 2, 0)
            compile_version = paddle_infer.get_trt_compile_version()
            runtime_version = paddle_infer.get_trt_runtime_version()
            self.assertTrue(compile_version == runtime_version)
            if compile_version < valid_version:
                return (0, 3)
            if is_dynamic_shape:
                return (1, 2)
            return (0, 3)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
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

    def test(self):
        if False:
            while True:
                i = 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()