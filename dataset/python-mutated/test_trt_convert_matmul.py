import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertMatmulTest_static(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input(shape):
            if False:
                i = 10
                return i + 15
            return np.random.random(shape).astype(np.float32)
        for batch in [1, 4]:
            for trans_x in [True, False]:
                for trans_y in [True, False]:
                    if trans_x and trans_y:
                        input1_shape = [batch, 6, 11]
                        input2_shape = [batch, 32, 6]
                    if trans_x and (not trans_y):
                        input1_shape = [batch, 6, 11]
                        input2_shape = [batch, 6, 32]
                    if not trans_x and trans_y:
                        input1_shape = [batch, 32, 6]
                        input2_shape = [batch, 11, 6]
                    if not trans_x and (not trans_y):
                        input1_shape = [batch, 32, 6]
                        input2_shape = [batch, 6, 11]
                    for alpha in [0.3, 1.0]:
                        dics = [{'transpose_X': trans_x, 'transpose_Y': trans_y, 'alpha': alpha}]
                        ops_config = [{'op_type': 'matmul', 'op_inputs': {'X': ['input1_data'], 'Y': ['input2_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input1_data': TensorConfig(data_gen=partial(generate_input, input1_shape)), 'input2_data': TensorConfig(data_gen=partial(generate_input, input2_shape))}, outputs=['output_data'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            pass

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 3), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (1, 3), 0.001)

    def test(self):
        if False:
            while True:
                i = 10
        self.run_test()

class TrtConvertMatmulTest_dynamic(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input(shape):
            if False:
                return 10
            return np.random.random(shape).astype(np.float32)
        for trans_x in [True]:
            for trans_y in [True]:
                if trans_x and trans_y:
                    input1_shape = [4, 4, 4]
                    input2_shape = [4, 4, 4]
                for alpha in [0.3, 1.0]:
                    dics = [{'transpose_X': trans_x, 'transpose_Y': trans_y, 'alpha': alpha}]
                    ops_config = [{'op_type': 'matmul', 'op_inputs': {'X': ['input1_data'], 'Y': ['input2_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input1_data': TensorConfig(data_gen=partial(generate_input, input1_shape)), 'input2_data': TensorConfig(data_gen=partial(generate_input, input2_shape))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'input1_data': [1, 4, 4], 'input2_data': [1, 4, 4]}
            self.dynamic_shape.max_input_shape = {'input1_data': [16, 4, 4], 'input2_data': [16, 4, 4]}
            self.dynamic_shape.opt_input_shape = {'input1_data': [8, 4, 4], 'input2_data': [8, 4, 4]}
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 3), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (1, 3), 0.001)

    def add_skip_trt_case(self):
        if False:
            print('Hello World!')
        pass

    def test(self):
        if False:
            print('Hello World!')
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()