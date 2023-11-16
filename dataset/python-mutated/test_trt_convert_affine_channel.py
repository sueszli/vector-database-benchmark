import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertAffineChannelTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10

        def generate_input1(batch, dims, attrs: List[Dict[str, Any]]):
            if False:
                return 10
            if dims == 2:
                return np.ones([batch, 64]).astype(np.float32)
            elif attrs[0]['data_layout'] == 'NCHW':
                return np.ones([batch, 3, 64, 64]).astype(np.float32)
            else:
                return np.ones([batch, 64, 64, 3]).astype(np.float32)

        def generate_weight1(dims, attrs: List[Dict[str, Any]]):
            if False:
                i = 10
                return i + 15
            if dims == 2:
                return np.random.random([64]).astype(np.float32)
            else:
                return np.random.random([3]).astype(np.float32)
        for dims in [2, 4]:
            for batch in [1, 2, 4]:
                for data_layout in ['NCHW', 'NHWC']:
                    self.dims = dims
                    dics = [{'data_layout': data_layout}]
                    ops_config = [{'op_type': 'affine_channel', 'op_inputs': {'X': ['input_data'], 'Scale': ['scale'], 'Bias': ['bias']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={'scale': TensorConfig(data_gen=partial(generate_weight1, dims, dics)), 'bias': TensorConfig(data_gen=partial(generate_weight1, dims, dics))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, batch, dims, dics))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            if self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 64]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 64]}
            elif attrs[0]['data_layout'] == 'NCHW':
                self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 64, 64]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 64, 64]}
            else:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 32, 3]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 64, 64, 3]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 64, 64, 3]}

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
            if self.dims == 4 and attrs[0]['data_layout'] == 'NCHW':
                return (1, 2)
            else:
                return (0, 3)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def test(self):
        if False:
            print('Hello World!')
        self.run_test()
if __name__ == '__main__':
    unittest.main()