import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertSumTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10

        def generate_input1(batch):
            if False:
                return 10
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)
        for dims in [1, 2, 3, 4]:
            for batch in [1, 4]:
                self.dims = dims
                ops_config = [{'op_type': 'shape', 'op_inputs': {'Input': ['input1']}, 'op_outputs': {'Out': ['output']}, 'op_attrs': {}}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input1': TensorConfig(data_gen=partial(generate_input1, batch))}, outputs=['output'])
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape():
            if False:
                while True:
                    i = 10
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input1': [1, 3, 24, 24]}
                self.dynamic_shape.max_input_shape = {'input1': [4, 3, 48, 48]}
                self.dynamic_shape.opt_input_shape = {'input1': [1, 3, 24, 24]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input1': [1, 3, 24]}
                self.dynamic_shape.max_input_shape = {'input1': [4, 3, 48]}
                self.dynamic_shape.opt_input_shape = {'input1': [1, 3, 24]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input1': [1, 24]}
                self.dynamic_shape.max_input_shape = {'input1': [4, 48]}
                self.dynamic_shape.opt_input_shape = {'input1': [1, 24]}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input1': [24]}
                self.dynamic_shape.max_input_shape = {'input1': [48]}
                self.dynamic_shape.opt_input_shape = {'input1': [24]}

        def generate_trt_nodes_num(dynamic_shape):
            if False:
                i = 10
                return i + 15
            if not dynamic_shape:
                return (0, 3)
            return (1, 2)

        def clear_dynamic_shape():
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(False), 0.001)
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(True), 0.001)

    def test(self):
        if False:
            return 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()