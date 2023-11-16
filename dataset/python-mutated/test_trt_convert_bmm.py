import os
import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertBmmTest_dynamic(TrtLayerAutoScanTest):

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input(shape):
            if False:
                while True:
                    i = 10
            return np.random.random(shape).astype(np.float32)
        for batch in [10, 11, 12, 13, 14, 15]:
            for trans_x in [False]:
                for trans_y in [False]:
                    input1_shape = [batch, 350, 75]
                    input2_shape = [batch, 75, 25]
                    dics = [{}]
                    ops_config = [{'op_type': 'bmm', 'op_inputs': {'X': ['input1_data'], 'Y': ['input2_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input1_data': TensorConfig(data_gen=partial(generate_input, input1_shape)), 'input2_data': TensorConfig(data_gen=partial(generate_input, input2_shape))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {'input1_data': [10, 350, 75], 'input2_data': [10, 75, 25]}
            self.dynamic_shape.max_input_shape = {'input1_data': [100, 350, 75], 'input2_data': [100, 75, 25]}
            self.dynamic_shape.opt_input_shape = {'input1_data': [15, 350, 75], 'input2_data': [15, 75, 25]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            if dynamic_shape:
                return (1, 3)
            else:
                return (0, 4)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.001)
        tol_fp32 = 0.0001
        tol_half = 0.0001
        if os.name == 'nt':
            tol_fp32 = 0.01
            tol_half = 0.01
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), tol_fp32)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), tol_half)

    def add_skip_trt_case(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test(self):
        if False:
            while True:
                i = 10
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()