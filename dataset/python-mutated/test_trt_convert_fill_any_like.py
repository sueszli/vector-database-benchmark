import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertExpandV2Test(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        if self.dtype in [0, 1, 4]:
            return False
        if self.dims != 4 and self.dtype != 2:
            return False
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input1(attrs: List[Dict[str, Any]]):
            if False:
                print('Hello World!')
            if self.dims == 4:
                self.input_shape = [1, 1, 4, 6]
                if self.dtype == 0:
                    return np.random.random([1, 1, 4, 6]).astype(np.bool_)
                elif self.dtype == 1:
                    return np.random.random([1, 1, 4, 6]).astype(np.int16)
                elif self.dtype == 2:
                    return np.random.random([1, 1, 4, 6]).astype(np.int32)
                elif self.dtype == 3:
                    return np.random.random([1, 1, 4, 6]).astype(np.int64)
                elif self.dtype == 4:
                    return np.random.random([1, 1, 4, 6]).astype(np.float16)
                elif self.dtype == 5:
                    return np.random.random([1, 1, 4, 6]).astype(np.float32)
                elif self.dtype == 6:
                    return np.random.random([1, 1, 4, 6]).astype(np.float64)
                else:
                    return np.random.random([1, 1, 4, 6]).astype(np.int32)
            elif self.dims == 3:
                self.input_shape = [1, 8, 6]
                return np.random.random([1, 8, 6]).astype(np.int32)
            elif self.dims == 2:
                self.input_shape = [1, 48]
                return np.random.random([1, 48]).astype(np.int32)
            elif self.dims == 1:
                self.input_shape = [48]
                return np.random.random([48]).astype(np.int32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            if False:
                for i in range(10):
                    print('nop')
            return np.array([1, 48]).astype(np.int32)

        def generate_shapeT1_data(attrs: List[Dict[str, Any]]):
            if False:
                return 10
            return np.array([2]).astype(np.int32)

        def generate_shapeT2_data(attrs: List[Dict[str, Any]]):
            if False:
                print('Hello World!')
            return np.array([24]).astype(np.int32)
        for dims in [1, 2, 3, 4]:
            for value in [2]:
                for dtype in [-1, 0, 1, 2, 3, 4, 5, 6]:
                    dics = [{'value': value, 'dtype': dtype}]
                    self.dims = dims
                    self.dtype = dtype
                    dics_intput = [{'X': ['fill_any_like_input']}]
                    ops_config = [{'op_type': 'fill_any_like', 'op_inputs': dics_intput[0], 'op_outputs': {'Out': ['fill_any_like_out']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'fill_any_like_input': TensorConfig(data_gen=partial(generate_input1, dics))}, outputs=['fill_any_like_out'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], int):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {'fill_any_like_input': [1, 1, 4, 6]}
                self.dynamic_shape.max_input_shape = {'fill_any_like_input': [10, 1, 4, 6]}
                self.dynamic_shape.opt_input_shape = {'fill_any_like_input': [1, 1, 4, 6]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'fill_any_like_input': [1, 8, 6]}
                self.dynamic_shape.max_input_shape = {'fill_any_like_input': [4, 8, 6]}
                self.dynamic_shape.opt_input_shape = {'fill_any_like_input': [1, 8, 6]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'fill_any_like_input': [1, 48]}
                self.dynamic_shape.max_input_shape = {'fill_any_like_input': [4, 48]}
                self.dynamic_shape.opt_input_shape = {'fill_any_like_input': [1, 48]}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {'fill_any_like_input': [48]}
                self.dynamic_shape.max_input_shape = {'fill_any_like_input': [48]}
                self.dynamic_shape.opt_input_shape = {'fill_any_like_input': [48]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            if not dynamic_shape:
                return (0, 3)
            else:
                return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)

    def add_skip_trt_case(self):
        if False:
            while True:
                i = 10
        pass

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()