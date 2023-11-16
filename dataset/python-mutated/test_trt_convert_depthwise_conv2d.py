import itertools
import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertDepthwiseConv2dTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if inputs['input_data'].shape[1] != weights['conv2d_weight'].shape[1] * attrs[0]['groups']:
            return False
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            if False:
                i = 10
                return i + 15
            groups = attrs[0]['groups']
            return np.ones([batch, groups, 64, 64]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            if False:
                print('Hello World!')
            return np.random.random([24, 1, 3, 3]).astype(np.float32)
        batch_options = [1, 4]
        strides_options = [[1, 2]]
        paddings_options = [[0, 3], [1, 2, 3, 4]]
        groups_options = [1, 3]
        padding_algorithm_options = ['EXPLICIT', 'SAME', 'VAILD']
        dilations_options = [[1, 1], [1, 2]]
        data_format_options = ['NCHW']
        configurations = [batch_options, strides_options, paddings_options, groups_options, padding_algorithm_options, dilations_options, data_format_options]
        for (batch, strides, paddings, groups, padding_algorithm, dilations, data_format) in itertools.product(*configurations):
            attrs = [{'strides': strides, 'paddings': paddings, 'groups': groups, 'padding_algorithm': padding_algorithm, 'dilations': dilations, 'data_fromat': data_format}]
            ops_config = [{'op_type': 'depthwise_conv2d', 'op_inputs': {'Input': ['input_data'], 'Filter': ['conv2d_weight']}, 'op_outputs': {'Output': ['output_data']}, 'op_attrs': attrs[0]}]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(ops=ops, weights={'conv2d_weight': TensorConfig(data_gen=partial(generate_weight1, attrs))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, batch, attrs))}, outputs=['output_data'])
            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            print('Hello World!')

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            groups = attrs[0]['groups']
            self.dynamic_shape.min_input_shape = {'input_data': [1, groups, 32, 32], 'output_data': [1, 24, 32, 32]}
            self.dynamic_shape.max_input_shape = {'input_data': [4, groups, 64, 64], 'output_data': [4, 24, 64, 64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, groups, 64, 64], 'output_data': [1, 24, 64, 64]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num():
            if False:
                while True:
                    i = 10
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(), (0.001, 0.001))
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        program_config.set_input_type(np.int8)
        yield (self.create_inference_config(), generate_trt_nodes_num(), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(), (0.001, 0.001))
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        program_config.set_input_type(np.int8)
        yield (self.create_inference_config(), generate_trt_nodes_num(), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            i = 10
            return i + 15

        def teller1(program_config, predictor_config):
            if False:
                while True:
                    i = 10
            if program_config.ops[0].attrs['padding_algorithm'] == 'SAME' or program_config.ops[0].attrs['padding_algorithm'] == 'VALID':
                return True
            return False
        self.add_skip_case(teller1, SkipReasons.TRT_NOT_IMPLEMENTED, "When padding_algorithm is 'SAME' or 'VALID', Trt dose not support. In this case, trt build error is caused by scale op.")

        def teller2(program_config, predictor_config):
            if False:
                return 10
            if self.trt_param.precision == paddle_infer.PrecisionType.Int8:
                return True
            return False
        self.add_skip_case(teller2, SkipReasons.TRT_NOT_IMPLEMENTED, 'When precisionType is int8 without relu op, output is different between Trt and Paddle.')

    def test(self):
        if False:
            print('Hello World!')
        self.add_skip_trt_case()
        self.run_test()

    def test_quant(self):
        if False:
            return 10
        self.add_skip_trt_case()
        self.run_test(quant=True)
if __name__ == '__main__':
    unittest.main()