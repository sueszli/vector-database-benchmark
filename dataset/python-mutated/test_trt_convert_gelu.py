import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertGeluTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')

        def generate_input1(dims, attrs: List[Dict[str, Any]]):
            if False:
                while True:
                    i = 10
            if dims == 0:
                return np.ones([]).astype(np.float32)
            elif dims == 1:
                return np.ones([32]).astype(np.float32)
            elif dims == 2:
                return np.ones([3, 32]).astype(np.float32)
            elif dims == 3:
                return np.ones([3, 32, 32]).astype(np.float32)
            else:
                return np.ones([1, 3, 32, 32]).astype(np.float32)
        for dims in [0, 1, 2, 3, 4]:
            for approximate in [True, False]:
                self.dims = dims
                dics = [{'approximate': approximate}]
                ops_config = [{'op_type': 'gelu', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dims, dics))}, outputs=['output_data'])
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                return 10
            if self.dims == 0:
                self.dynamic_shape.min_input_shape = {'input_data': []}
                self.dynamic_shape.max_input_shape = {'input_data': []}
                self.dynamic_shape.opt_input_shape = {'input_data': []}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [64]}
                self.dynamic_shape.opt_input_shape = {'input_data': [32]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 16]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [3, 32]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 16, 16]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [3, 32, 32]}
            else:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 16, 16]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 32, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 32, 32]}

        def clear_dynamic_shape():
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                return 10
            valid_version = (7, 0, 0)
            compile_version = paddle_infer.get_trt_compile_version()
            runtime_version = paddle_infer.get_trt_runtime_version()
            self.assertTrue(compile_version == runtime_version)
            if not dynamic_shape and (self.dims == 1 or self.dims == 0):
                return (0, 3)
            if compile_version >= valid_version:
                return (1, 2)
            elif attrs[0]['approximate']:
                return (0, 3)
            else:
                return (1, 2)
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
            i = 10
            return i + 15
        self.run_test()
if __name__ == '__main__':
    unittest.main()