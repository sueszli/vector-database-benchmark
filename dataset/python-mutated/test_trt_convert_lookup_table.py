import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertLookupTableV2Test(TrtLayerAutoScanTest):

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15
        self.trt_param.workspace_size = 102400

        def generate_input1(dims, attrs: List[Dict[str, Any]]):
            if False:
                while True:
                    i = 10
            if dims == 1:
                return np.array([[32], [2], [19]]).astype(np.int64)
            elif dims == 2:
                return np.array([[[3], [16], [24]], [[6], [4], [47]]]).astype(np.int64)
            else:
                return np.array([[[[3], [16], [24]], [[30], [16], [14]], [[2], [6], [24]]], [[[3], [26], [34]], [[3], [16], [24]], [[3], [6], [4]]], [[[3], [16], [24]], [[53], [16], [54]], [[30], [1], [24]]]]).astype(np.int64)

        def generate_input2(dims, attrs: List[Dict[str, Any]]):
            if False:
                print('Hello World!')
            return np.random.uniform(-1, 1, [64, 4]).astype('float32')
        for dims in [1, 2, 3]:
            self.dims = dims
            ops_config = [{'op_type': 'lookup_table', 'op_inputs': {'Ids': ['indices'], 'W': ['data']}, 'op_outputs': {'Out': ['out_data']}, 'op_attrs': {}}]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(ops=ops, weights={'data': TensorConfig(data_gen=partial(generate_input2, {}, {}))}, inputs={'indices': TensorConfig(data_gen=partial(generate_input1, dims, {}))}, outputs=['out_data'])
            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            print('Hello World!')

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'indices': [1, 1], 'data': [64, 4]}
                self.dynamic_shape.max_input_shape = {'indices': [16, 1], 'data': [64, 4]}
                self.dynamic_shape.opt_input_shape = {'indices': [8, 1], 'data': [64, 4]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'indices': [1, 1, 1], 'data': [64, 4]}
                self.dynamic_shape.max_input_shape = {'indices': [16, 32, 1], 'data': [64, 4]}
                self.dynamic_shape.opt_input_shape = {'indices': [2, 16, 1], 'data': [64, 4]}
            else:
                self.dynamic_shape.min_input_shape = {'indices': [1, 1, 1, 1], 'data': [64, 4]}
                self.dynamic_shape.max_input_shape = {'indices': [16, 16, 16, 1], 'data': [64, 4]}
                self.dynamic_shape.opt_input_shape = {'indices': [2, 8, 8, 1], 'data': [64, 4]}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test()
if __name__ == '__main__':
    unittest.main()