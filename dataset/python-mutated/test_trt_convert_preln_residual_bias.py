import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertSkipLayernormTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if 'begin_norm_axis' in attrs[0] and attrs[0]['begin_norm_axis'] >= 0:
            if len(inputs['inputX_data'].shape) <= attrs[0]['begin_norm_axis']:
                return False
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if False:
                while True:
                    i = 10
            return np.ones([batch, 128, 768]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], batch):
            if False:
                return 10
            return np.ones([batch, 128, 768]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            if False:
                return 10
            return np.random.random([768]).astype(np.float32)

        def generate_weight2(attrs: List[Dict[str, Any]]):
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random([768]).astype(np.float32)
        for batch in [4]:
            for epsilon in [1e-05]:
                for begin_norm_axis in [2]:
                    for enable_int8 in [False, True]:
                        dics = [{'epsilon': epsilon, 'begin_norm_axis': begin_norm_axis}, {}]
                        ops_config = [{'op_type': 'elementwise_add', 'op_inputs': {'X': ['inputX_data'], 'Y': ['EleBias']}, 'op_outputs': {'Out': ['bias_out']}, 'op_attrs': {'axis': -1}}, {'op_type': 'elementwise_add', 'op_inputs': {'X': ['bias_out'], 'Y': ['inputY_data']}, 'op_outputs': {'Out': ['ele_out']}, 'op_attrs': {'axis': -1}}, {'op_type': 'layer_norm', 'op_inputs': {'X': ['ele_out'], 'Bias': ['Bias'], 'Scale': ['Scale']}, 'op_outputs': {'Y': ['layernorm_out'], 'Mean': ['Mean'], 'Variance': ['Variance']}, 'op_attrs': dics[0]}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={'Bias': TensorConfig(data_gen=partial(generate_weight1, dics)), 'Scale': TensorConfig(data_gen=partial(generate_weight2, dics)), 'EleBias': TensorConfig(data_gen=partial(generate_weight2, dics))}, inputs={'inputX_data': TensorConfig(data_gen=partial(generate_input1, dics, batch)), 'inputY_data': TensorConfig(data_gen=partial(generate_input2, dics, batch))}, outputs=['ele_out', 'layernorm_out'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {'inputX_data': [4, 128, 768], 'inputY_data': [4, 128, 768], 'Bias': [768], 'Scale': [768]}
            self.dynamic_shape.max_input_shape = {'inputX_data': [4, 128, 768], 'inputY_data': [4, 128, 768], 'Bias': [768], 'Scale': [768]}
            self.dynamic_shape.opt_input_shape = {'inputX_data': [4, 128, 768], 'inputY_data': [4, 128, 768], 'Bias': [768], 'Scale': [768]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            if dynamic_shape:
                return (1, 4)
            else:
                return (0, 5)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.01)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.01)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.01)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.01)

    def add_skip_trt_case(self):
        if False:
            while True:
                i = 10
        pass

    def test(self):
        if False:
            i = 10
            return i + 15
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()