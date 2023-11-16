import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertUnbind(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input1():
            if False:
                print('Hello World!')
            self.input_shape = [3, 400, 196, 80]
            return np.random.random([3, 400, 196, 80]).astype(np.float32)
        for dims in [4]:
            for axis in [0]:
                self.dims = dims
                ops_config = [{'op_type': 'unbind', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data0', 'output_data1', 'output_data2']}, 'op_attrs': {'axis': axis}}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1))}, outputs=['output_data0', 'output_data1', 'output_data2'])
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            return (1, 4)

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'input_data': [3, 100, 196, 80]}
            self.dynamic_shape.max_input_shape = {'input_data': [3, 400, 196, 80]}
            self.dynamic_shape.opt_input_shape = {'input_data': [3, 400, 196, 80]}
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.001)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test()
if __name__ == '__main__':
    unittest.main()