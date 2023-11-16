import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertCumsum(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7220:
            return False
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')
        self.trt_param.workspace_size = 1073741824

        def generate_input1():
            if False:
                return 10
            if self.dims == 0:
                self.input_shape = []
                return np.random.random([]).astype(np.float32)
            elif self.dims == 2:
                self.input_shape = [2, 3]
                return np.random.random([2, 3]).astype(np.int32)
            elif self.dims == 3:
                self.input_shape = [2, 3, 4]
                return np.random.random([2, 3, 4]).astype(np.int64)
            elif self.dims == 4:
                self.input_shape = [4, 3, 32, 32]
                return np.random.random([4, 3, 32, 32]).astype(np.float32) - 0.5
        for dims in [0, 2, 3, 4]:
            test_dims = dims
            if dims == 0:
                test_dims = 1
            for axis in range(-1, test_dims):
                for type in ['int32', 'int64', 'float32', 'float64']:
                    self.dims = dims
                    ops_config = [{'op_type': 'cumsum', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': {'axis': axis, 'dtype': type}}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1))}, outputs=['output_data'])
                    yield program_config
        for dims in [0, 2, 3, 4]:
            self.dims = dims
            ops_config = [{'op_type': 'cumsum', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': {}}]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1))}, outputs=['output_data'])
            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape():
            if False:
                return 10
            if self.dims == 0:
                self.dynamic_shape.min_input_shape = {'input_data': []}
                self.dynamic_shape.max_input_shape = {'input_data': []}
                self.dynamic_shape.opt_input_shape = {'input_data': []}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [2, 3]}
                self.dynamic_shape.max_input_shape = {'input_data': [2, 3]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 3]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [2, 3, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [2, 3, 4]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 3, 4]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [4, 3, 32, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 32, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [4, 3, 32, 32]}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7220:
                return (0, 3)
            return (1, 2)

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        generate_dynamic_shape()
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
if __name__ == '__main__':
    unittest.main()