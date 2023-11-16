import unittest
from functools import partial
from typing import List, Tuple
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertArgMinTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        input_shape = program_config.inputs['arg_min_input'].shape
        axis = program_config.ops[0].attrs['axis']
        if axis < 0:
            axis += len(input_shape)
        if len(input_shape) <= axis or axis == 0:
            return False
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input(rank, batch):
            if False:
                i = 10
                return i + 15
            dims = [batch]
            for i in range(rank - 1):
                dims.append((i + 1) * 8)
            size = np.prod(dims)
            return (np.arange(size) % 10 - 5).astype('float32').reshape(dims)
        for rank in [3, 4]:
            for batch in [1, 4]:
                for axis in [-1, 0, 1, 2, 3]:
                    for keepdims in [True, False]:
                        self.rank = rank
                        flatten = False
                        dtype = 2
                        ops_config = [{'op_type': 'arg_min', 'op_inputs': {'X': ['arg_min_input']}, 'op_outputs': {'Out': ['arg_min_out']}, 'op_attrs': {'axis': axis, 'keepdims': keepdims, 'flatten': flatten, 'dtype': dtype}, 'outputs_dtype': {'arg_min_out': np.int32}}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={}, inputs={'arg_min_input': TensorConfig(data_gen=partial(generate_input, rank, batch))}, outputs=['arg_min_out'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> Tuple[paddle_infer.Config, List[int], float]:
        if False:
            print('Hello World!')

        def generate_dynamic_shape(attrs):
            if False:
                return 10
            if self.rank == 3:
                self.dynamic_shape.min_input_shape = {'arg_min_input': [1, 8, 16]}
                self.dynamic_shape.max_input_shape = {'arg_min_input': [4, 8, 16]}
                self.dynamic_shape.opt_input_shape = {'arg_min_input': [3, 8, 16]}
            else:
                self.dynamic_shape.min_input_shape = {'arg_min_input': [1, 8, 16, 24]}
                self.dynamic_shape.max_input_shape = {'arg_min_input': [4, 8, 16, 24]}
                self.dynamic_shape.opt_input_shape = {'arg_min_input': [1, 8, 16, 24]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        self.trt_param.workspace_size = 1024000
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

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_test()
if __name__ == '__main__':
    unittest.main()