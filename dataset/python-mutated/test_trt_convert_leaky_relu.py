import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertLeakyReluTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input1(shape):
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random(shape).astype(np.float32)
        for batch in [1, 2]:
            for shape in [[batch, 64], [batch, 32, 64], [batch, 8, 32, 32]]:
                self.input_dim = len(shape)
                for alpha in [0.02, 1.0, 100.0, -1.0, 0.0]:
                    dics = [{'alpha': alpha}]
                    ops_config = [{'op_type': 'leaky_relu', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['y_data']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, shape))}, outputs=['y_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            if self.input_dim == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 8]}
                self.dynamic_shape.max_input_shape = {'input_data': [64, 128]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 16]}
            elif self.input_dim == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 8, 8]}
                self.dynamic_shape.max_input_shape = {'input_data': [64, 128, 256]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 16, 64]}
            elif self.input_dim == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 8, 8, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [64, 64, 128, 128]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 16, 64, 32]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        program_config.set_input_type(np.int8)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        program_config.set_input_type(np.int8)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def test(self):
        if False:
            while True:
                i = 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()