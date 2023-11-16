import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertTakeAlongAxisTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        inputs = program_config.inputs
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if len(inputs['input_data'].shape) <= attrs[0]['Axis']:
            return False
        if len(inputs['input_data'].shape) != len(inputs['index_data'].shape):
            return False
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input1(shape):
            if False:
                while True:
                    i = 10
            return np.random.random(shape).astype(np.float32)

        def generate_input2(index):
            if False:
                return 10
            return np.zeros(index).astype(np.int32)

        def generate_input3(axis):
            if False:
                while True:
                    i = 10
            return np.array([axis]).astype(np.int32)
        for shape in [[32], [3, 64], [1, 64, 16], [1, 64, 16, 32]]:
            for index in [[1], [1, 1], [1, 1, 2], [1, 1, 1, 1]]:
                for axis in [0, 1, 2, 3]:
                    self.shape = shape
                    self.axis = axis
                    dics = [{'Axis': axis}]
                    ops_config = [{'op_type': 'take_along_axis', 'op_inputs': {'Input': ['input_data'], 'Index': ['index_data']}, 'op_outputs': {'Result': ['output_data']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, shape)), 'index_data': TensorConfig(data_gen=partial(generate_input2, index))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                return 10
            if len(self.shape) == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [4], 'index_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [128], 'index_data': [4]}
                self.dynamic_shape.opt_input_shape = {'input_data': [16], 'index_data': [2]}
            elif len(self.shape) == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [3, 64], 'index_data': [1, 1]}
                self.dynamic_shape.max_input_shape = {'input_data': [3, 64], 'index_data': [1, 1]}
                self.dynamic_shape.opt_input_shape = {'input_data': [3, 64], 'index_data': [1, 1]}
            elif len(self.shape) == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 64, 16], 'index_data': [1, 1, 2]}
                self.dynamic_shape.max_input_shape = {'input_data': [1, 64, 16], 'index_data': [1, 1, 2]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 64, 16], 'index_data': [1, 1, 2]}
            elif len(self.shape) == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 64, 16, 32], 'index_data': [1, 1, 1, 1]}
                self.dynamic_shape.max_input_shape = {'input_data': [1, 64, 16, 32], 'index_data': [1, 1, 1, 1]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 64, 16, 32], 'index_data': [1, 1, 1, 1]}

        def clear_dynamic_shape():
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(dynamic_shape):
            if False:
                while True:
                    i = 10
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 > 8200 and dynamic_shape:
                return (1, 3)
            else:
                return (0, 4)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(False), 1e-05)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(True), 0.001)

    def add_skip_trt_case(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()