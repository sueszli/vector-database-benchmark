import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertActivationTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10
        self.trt_param.workspace_size = 1073741824

        def generate_input1(dims, batch, attrs: List[Dict[str, Any]]):
            if False:
                return 10
            if dims == 1:
                return np.random.random([32]).astype(np.float32)
            elif dims == 2:
                return np.random.random([3, 32]).astype(np.float32)
            elif dims == 3:
                return np.random.random([3, 32, 32]).astype(np.float32)
            else:
                return np.random.random([batch, 3, 32, 32]).astype(np.float32)
        for dims in [2, 3, 4, 5]:
            for batch in [1]:
                for k in [1, 3]:
                    self.dims = dims
                    dics = [{'k': k}]
                    ops_config = [{'op_type': 'top_k', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data'], 'Indices': ['indices_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'indices_data': np.int32}}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dims, batch, dics))}, outputs=['output_data', 'indices_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            if self.dims == 1:
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
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
            if not dynamic_shape and self.dims == 1:
                return (0, 4)
            return (1, 3)
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