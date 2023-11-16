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
            print('Hello World!')
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')
        self.trt_param.workspace_size = 1073741824

        def generate_input1(dims, batch, attrs: List[Dict[str, Any]]):
            if False:
                for i in range(10):
                    print('nop')
            if dims == 0:
                return np.random.random([]).astype(np.bool8)
            elif dims == 1:
                return np.random.random([32]).astype(np.bool8)
            elif dims == 2:
                return np.random.random([3, 32]).astype(np.int8)
            elif dims == 3:
                return np.random.random([3, 32, 32]).astype(np.int32)
            else:
                return np.random.random([batch, 3, 32, 32]).astype(np.int64)
        for dims in [0, 1, 2, 3, 4]:
            for batch in [1, 4]:
                self.dims = dims
                dics = [{}]
                ops_config = [{'op_type': 'bitwise_not', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dims, batch, dics))}, outputs=['output_data'])
                program_config.input_type = program_config.inputs['input_data'].dtype
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
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
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            ver = paddle_infer.get_trt_compile_version()
            trt_version = ver[0] * 1000 + ver[1] * 100 + ver[2] * 10
            if not dynamic_shape:
                if self.dims == 1 or self.dims == 0:
                    return (0, 3)
            if program_config.input_type in ['int8', 'uint8']:
                return (0, 3)
            elif program_config.input_type == 'bool':
                if trt_version <= 8600 and self.dims == 0:
                    return (0, 3)
                elif trt_version <= 8400:
                    return (0, 3)
                else:
                    return (1, 2)
            else:
                return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test()
if __name__ == '__main__':
    unittest.main()