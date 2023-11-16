import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertStackTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if len(inputs['stack_input1'].shape) < attrs[0]['axis']:
            return False
        if -(len(inputs['stack_input1'].shape) + 1) > attrs[0]['axis']:
            return False
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if False:
                i = 10
                return i + 15
            if self.dims == 4:
                return np.random.random([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.random.random([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.random.random([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.random.random([24]).astype(np.float32)
            elif self.dims == 0:
                return np.random.random([]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], batch):
            if False:
                while True:
                    i = 10
            if self.dims == 4:
                return np.random.random([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.random.random([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.random.random([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.random.random([24]).astype(np.float32)
            elif self.dims == 0:
                return np.random.random([]).astype(np.float32)

        def generate_input3(attrs: List[Dict[str, Any]], batch):
            if False:
                print('Hello World!')
            if self.dims == 4:
                return np.random.random([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.random.random([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.random.random([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.random.random([24]).astype(np.float32)
            elif self.dims == 0:
                return np.random.random([]).astype(np.float32)
        for dims in [0, 1, 2, 3, 4]:
            for batch in [1, 4]:
                for axis in [-2, -1, 0, 1, 2, 3]:
                    self.dims = dims
                    dics = [{'axis': axis}, {}]
                    ops_config = [{'op_type': 'stack', 'op_inputs': {'X': ['stack_input1', 'stack_input2', 'stack_input3']}, 'op_outputs': {'Y': ['stack_output']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'stack_input1': TensorConfig(data_gen=partial(generate_input1, dics, batch)), 'stack_input2': TensorConfig(data_gen=partial(generate_input2, dics, batch)), 'stack_input3': TensorConfig(data_gen=partial(generate_input3, dics, batch))}, outputs=['stack_output'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {'stack_input1': [1, 3, 24, 24], 'stack_input2': [1, 3, 24, 24], 'stack_input3': [1, 3, 24, 24]}
                self.dynamic_shape.max_input_shape = {'stack_input1': [4, 3, 48, 48], 'stack_input2': [4, 3, 48, 48], 'stack_input3': [4, 3, 48, 48]}
                self.dynamic_shape.opt_input_shape = {'stack_input1': [1, 3, 24, 24], 'stack_input2': [1, 3, 24, 24], 'stack_input3': [1, 3, 24, 24]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'stack_input1': [1, 3, 24], 'stack_input2': [1, 3, 24], 'stack_input3': [1, 3, 24]}
                self.dynamic_shape.max_input_shape = {'stack_input1': [4, 3, 48], 'stack_input2': [4, 3, 48], 'stack_input3': [4, 3, 48]}
                self.dynamic_shape.opt_input_shape = {'stack_input1': [1, 3, 24], 'stack_input2': [1, 3, 24], 'stack_input3': [1, 3, 24]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'stack_input1': [1, 24], 'stack_input2': [1, 24], 'stack_input3': [1, 24]}
                self.dynamic_shape.max_input_shape = {'stack_input1': [4, 48], 'stack_input2': [4, 48], 'stack_input3': [4, 48]}
                self.dynamic_shape.opt_input_shape = {'stack_input1': [1, 24], 'stack_input2': [1, 24], 'stack_input3': [1, 24]}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {'stack_input1': [24], 'stack_input2': [24], 'stack_input3': [24]}
                self.dynamic_shape.max_input_shape = {'stack_input1': [48], 'stack_input2': [48], 'stack_input3': [48]}
                self.dynamic_shape.opt_input_shape = {'stack_input1': [24], 'stack_input2': [24], 'stack_input3': [24]}
            elif self.dims == 0:
                self.dynamic_shape.min_input_shape = {'stack_input1': [], 'stack_input2': [], 'stack_input3': []}
                self.dynamic_shape.max_input_shape = {'stack_input1': [], 'stack_input2': [], 'stack_input3': []}
                self.dynamic_shape.opt_input_shape = {'stack_input1': [], 'stack_input2': [], 'stack_input3': []}

        def clear_dynamic_shape():
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
            if dynamic_shape:
                return (1, 4)
            else:
                return (0, 5)
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

    def add_skip_trt_case(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test(self):
        if False:
            print('Hello World!')
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()