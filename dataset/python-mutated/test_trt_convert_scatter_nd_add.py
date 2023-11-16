import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertScatterNd(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')

        def generate_input1():
            if False:
                i = 10
                return i + 15
            return np.random.random([6]).astype(np.float32)

        def generate_input2():
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random([4, 1]).astype(np.int32)

        def generate_input3():
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random([4]).astype(np.float32)
        ops_config = [{'op_type': 'scatter_nd_add', 'op_inputs': {'X': ['input_data'], 'Index': ['index_data'], 'Updates': ['update_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': {}}]
        ops = self.generate_op_config(ops_config)
        for i in range(10):
            program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1)), 'index_data': TensorConfig(data_gen=partial(generate_input2)), 'update_data': TensorConfig(data_gen=partial(generate_input3))}, outputs=['output_data'])
            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'input_data': [1], 'index_data': [2, 1], 'update_data': [1]}
            self.dynamic_shape.max_input_shape = {'input_data': [6], 'index_data': [4, 1], 'update_data': [4]}
            self.dynamic_shape.opt_input_shape = {'input_data': [6], 'index_data': [4, 1], 'update_data': [4]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (0, 5), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (0, 5), 0.001)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 4), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (1, 4), 0.001)

    def test(self):
        if False:
            print('Hello World!')
        self.run_test()
if __name__ == '__main__':
    unittest.main()