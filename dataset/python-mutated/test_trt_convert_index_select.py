import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertIndexSelectTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        inputs = program_config.inputs
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if len(inputs['input_data'].shape) <= attrs[0]['dim']:
            return False
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input1(shape):
            if False:
                print('Hello World!')
            return np.random.random(shape).astype(np.float32)

        def generate_input2(index):
            if False:
                return 10
            return np.array(index).astype(np.int32)

        def generate_input4(index):
            if False:
                print('Hello World!')
            return np.array(index).astype(np.int64)

        def generate_input3(axis):
            if False:
                while True:
                    i = 10
            return np.array([axis]).astype(np.int32)
        for shape in [[32, 64, 16, 32]]:
            for index in [[1, 4], [4, 8]]:
                for axis in [0, 1, 2, 3]:
                    for overwrite in [True, False]:
                        for input in [{'X': ['input_data'], 'Index': ['index_data']}]:
                            for index_type_int32 in [True, False]:
                                self.shape = shape
                                self.axis = axis
                                self.input_num = len(input)
                                self.index_type_int32 = index_type_int32
                                dics = [{'dim': axis}]
                                ops_config = [{'op_type': 'index_select', 'op_inputs': input, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                                ops = self.generate_op_config(ops_config)
                                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, shape)), 'index_data': TensorConfig(data_gen=partial(generate_input2 if index_type_int32 else generate_input4, index))}, outputs=['output_data'])
                                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            if len(self.shape) == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [4], 'index_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [128], 'index_data': [4]}
                self.dynamic_shape.opt_input_shape = {'input_data': [16], 'index_data': [2]}
            elif len(self.shape) == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [2, 4], 'index_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [256, 256], 'index_data': [4]}
                self.dynamic_shape.opt_input_shape = {'input_data': [64, 32], 'index_data': [2]}
            elif len(self.shape) == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [2, 4, 4], 'index_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [128, 256, 256], 'index_data': [4]}
                self.dynamic_shape.opt_input_shape = {'input_data': [16, 64, 32], 'index_data': [2]}
            elif len(self.shape) == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [2, 4, 4, 2], 'index_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [128, 256, 64, 128], 'index_data': [4]}
                self.dynamic_shape.opt_input_shape = {'input_data': [16, 64, 16, 32], 'index_data': [2]}

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
            if dynamic_shape:
                ver = paddle_infer.get_trt_compile_version()
                if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
                    return (0, 4)
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
        yield (self.create_inference_config(), generate_trt_nodes_num(False), 0.001)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(True), 0.001)

    def test(self):
        if False:
            while True:
                i = 10
        self.trt_param.workspace_size = 1 << 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()