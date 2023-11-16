import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertRangeDynamicTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input():
            if False:
                for i in range(10):
                    print('nop')
            return np.array([1]).astype(np.int32)
        for in_dtype in [2]:
            self.in_dtype = in_dtype
            dics = [{}]
            ops_config = [{'op_type': 'fill_constant', 'op_inputs': {}, 'op_outputs': {'Out': ['start_data']}, 'op_attrs': {'dtype': self.in_dtype, 'str_value': '7', 'shape': [1]}}, {'op_type': 'fill_constant', 'op_inputs': {}, 'op_outputs': {'Out': ['end_data']}, 'op_attrs': {'dtype': self.in_dtype, 'str_value': '256', 'shape': [1]}}, {'op_type': 'fill_constant', 'op_inputs': {}, 'op_outputs': {'Out': ['step_data']}, 'op_attrs': {'dtype': self.in_dtype, 'str_value': '1', 'shape': [1]}}, {'op_type': 'range', 'op_inputs': {'Start': ['start_data'], 'End': ['end_data'], 'Step': ['step_data']}, 'op_outputs': {'Out': ['range_output_data1']}, 'op_attrs': dics[0]}, {'op_type': 'cast', 'op_inputs': {'X': ['range_output_data1']}, 'op_outputs': {'Out': ['range_output_data']}, 'op_attrs': {'in_dtype': self.in_dtype, 'out_dtype': 5}}]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(ops=ops, weights={}, inputs={'step_data': TensorConfig(data_gen=partial(generate_input))}, outputs=['range_output_data'])
            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'start_data': [1], 'end_data': [1], 'step_data': [1]}
            self.dynamic_shape.max_input_shape = {'start_data': [1], 'end_data': [1], 'step_data': [1]}
            self.dynamic_shape.opt_input_shape = {'start_data': [1], 'end_data': [1], 'step_data': [1]}

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
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
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

class TrtConvertRangeStaticTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input():
            if False:
                i = 10
                return i + 15
            return np.array([0]).astype(np.int32)

        def generate_input1():
            if False:
                i = 10
                return i + 15
            return np.array([128]).astype(np.int32)

        def generate_input2():
            if False:
                return 10
            return np.array([1]).astype(np.int32)
        for in_dtype in [2]:
            self.in_dtype = in_dtype
            dics = [{}]
            ops_config = [{'op_type': 'range', 'op_inputs': {'Start': ['start_data'], 'End': ['end_data'], 'Step': ['step_data']}, 'op_outputs': {'Out': ['range_output_data1']}, 'op_attrs': dics[0]}, {'op_type': 'cast', 'op_inputs': {'X': ['range_output_data1']}, 'op_outputs': {'Out': ['range_output_data']}, 'op_attrs': {'in_dtype': self.in_dtype, 'out_dtype': 5}}]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(ops=ops, weights={}, inputs={'start_data': TensorConfig(data_gen=partial(generate_input)), 'end_data': TensorConfig(data_gen=partial(generate_input1)), 'step_data': TensorConfig(data_gen=partial(generate_input2))}, outputs=['range_output_data'])
            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def clear_dynamic_shape():
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            return (0, 6)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.01)

    def test(self):
        if False:
            return 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()