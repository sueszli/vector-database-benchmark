import unittest
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertNearestInterpV2Test(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10

        def generate_input():
            if False:
                while True:
                    i = 10
            return np.ones([1, 3, 32, 32]).astype(np.float32)
        ops_config = [{'op_type': 'nearest_interp_v2', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['interp_output_data']}, 'op_attrs': {'data_layout': 'NCHW', 'interp_method': 'nearest', 'align_corners': False, 'align_mode': 1, 'scale': [2.0, 2.0], 'out_d': 0, 'out_h': 0, 'out_w': 0}}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=generate_input)}, outputs=['interp_output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 64, 64]}
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

class TrtConvertNearestInterpV2ShapeTensorTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')

        def generate_input():
            if False:
                while True:
                    i = 10
            return np.ones([1, 3, 32, 32]).astype(np.float32)

        def generate_weight():
            if False:
                while True:
                    i = 10
            return np.array([64]).astype(np.int32)
        ops_config = [{'op_type': 'nearest_interp_v2', 'op_inputs': {'X': ['input_data'], 'SizeTensor': ['size_tensor_data0', 'size_tensor_data1']}, 'op_outputs': {'Out': ['interp_output_data']}, 'op_attrs': {'data_layout': 'NCHW', 'interp_method': 'nearest', 'align_corners': False, 'align_mode': 1, 'scale': [2.0, 2.0], 'out_d': 0, 'out_h': 0, 'out_w': 0}}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={'size_tensor_data0': TensorConfig(data_gen=generate_weight), 'size_tensor_data1': TensorConfig(data_gen=generate_weight)}, inputs={'input_data': TensorConfig(data_gen=generate_input)}, outputs=['interp_output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            print('Hello World!')

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 64, 64]}

        def clear_dynamic_shape():
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
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
            return 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()