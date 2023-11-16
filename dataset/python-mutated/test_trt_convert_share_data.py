import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertShareDataTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        compile_version = paddle_infer.get_trt_compile_version()
        runtime_version = paddle_infer.get_trt_runtime_version()
        if compile_version[0] * 1000 + compile_version[1] * 100 + compile_version[2] * 10 < 8400:
            return False
        if runtime_version[0] * 1000 + runtime_version[1] * 100 + runtime_version[2] * 10 < 8400:
            return False
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input(type):
            if False:
                for i in range(10):
                    print('nop')
            if self.dims == 1:
                return np.ones([1]).astype(type)
            else:
                return np.ones([1, 3, 64, 64]).astype(type)
        for dims in [1, 4]:
            self.dims = dims
            for dtype in [np.int32, np.float32, np.int64]:
                self.has_bool_dtype = dtype == np.bool_
                ops_config = [{'op_type': 'share_data', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data0']}, 'op_attrs': {}}, {'op_type': 'share_data', 'op_inputs': {'X': ['output_data0']}, 'op_outputs': {'Out': ['output_data1']}, 'op_attrs': {}}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, dtype))}, outputs=['output_data1'])
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [1]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1]}
            else:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 64, 64]}
                self.dynamic_shape.max_input_shape = {'input_data': [1, 3, 64, 64]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 64, 64]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
            if not dynamic_shape and self.dims == 1:
                return (0, 4)
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.01)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.01)

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_test()
if __name__ == '__main__':
    unittest.main()