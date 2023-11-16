import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertTransposeTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if len(inputs['transpose_input'].shape) != len(attrs[0]['axis']):
            return False
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if False:
                while True:
                    i = 10
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
        for dims in [2, 3, 4]:
            for batch in [1, 2, 4]:
                for axis in [[0, 1, 3, 2], [0, 3, 2, 1], [3, 2, 0, 1], [0, 1, 2, 3], [0, 1, 2], [2, 0, 1], [1, 0], [0, 1]]:
                    self.dims = dims
                    dics = [{'axis': axis}, {}]
                    ops_config = [{'op_type': 'transpose', 'op_inputs': {'X': ['transpose_input']}, 'op_outputs': {'Out': ['transpose_out']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'transpose_input': TensorConfig(data_gen=partial(generate_input1, dics, batch))}, outputs=['transpose_out'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {'transpose_input': [1, 3, 24, 24]}
                self.dynamic_shape.max_input_shape = {'transpose_input': [9, 6, 48, 48]}
                self.dynamic_shape.opt_input_shape = {'transpose_input': [1, 3, 48, 24]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'transpose_input': [1, 3, 24]}
                self.dynamic_shape.max_input_shape = {'transpose_input': [9, 6, 48]}
                self.dynamic_shape.opt_input_shape = {'transpose_input': [1, 3, 24]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'transpose_input': [1, 24]}
                self.dynamic_shape.max_input_shape = {'transpose_input': [9, 48]}
                self.dynamic_shape.opt_input_shape = {'transpose_input': [1, 24]}

        def clear_dynamic_shape():
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
            if dynamic_shape:
                return (1, 2)
            elif attrs[0]['axis'][0] == 0:
                return (1, 2)
            else:
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
            i = 10
            return i + 15
        self.run_test()
if __name__ == '__main__':
    unittest.main()