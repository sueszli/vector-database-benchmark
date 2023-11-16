import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertDepthwiseConv2dTransposeTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if inputs['input_data'].shape[1] != weights['conv2d_weight'].shape[1] * attrs[0]['groups']:
            return False
        if inputs['input_data'].shape[1] != weights['conv2d_weight'].shape[1]:
            return False
        if inputs['input_data'].shape[1] != attrs[0]['groups']:
            return False
        if attrs[0]['dilations'][0] != 1 or attrs[0]['dilations'][1] != 1:
            return False
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7000:
            return False
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            if False:
                for i in range(10):
                    print('nop')
            return np.ones([batch, attrs[0]['groups'], 64, 64]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            if False:
                print('Hello World!')
            return np.random.random([attrs[0]['groups'], 1, 3, 3]).astype(np.float32)
        for batch in [1, 2, 4]:
            for strides in [[1, 1], [2, 2], [1, 2]]:
                for paddings in [[0, 3], [1, 2, 3, 4]]:
                    for groups in [1, 2, 3]:
                        for padding_algorithm in ['EXPLICIT', 'SAME', 'VALID']:
                            for dilations in [[1, 1], [2, 2], [1, 2]]:
                                for data_format in ['NCHW']:
                                    dics = [{'data_fromat': data_format, 'dilations': dilations, 'padding_algorithm': padding_algorithm, 'groups': groups, 'paddings': paddings, 'strides': strides, 'data_format': data_format, 'output_size': [], 'output_padding': []}]
                                    ops_config = [{'op_type': 'conv2d_transpose', 'op_inputs': {'Input': ['input_data'], 'Filter': ['conv2d_weight']}, 'op_outputs': {'Output': ['output_data']}, 'op_attrs': dics[0]}]
                                    ops = self.generate_op_config(ops_config)
                                    program_config = ProgramConfig(ops=ops, weights={'conv2d_weight': TensorConfig(data_gen=partial(generate_weight1, dics))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, batch, dics))}, outputs=['output_data'])
                                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {'input_data': [1, attrs[0]['groups'], 32, 32], 'output_data': [1, attrs[0]['groups'], 32, 32]}
            self.dynamic_shape.max_input_shape = {'input_data': [4, attrs[0]['groups'], 64, 64], 'output_data': [4, attrs[0]['groups'], 64, 64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, attrs[0]['groups'], 64, 64], 'output_data': [1, attrs[0]['groups'], 64, 64]}

        def clear_dynamic_shape():
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
            return (1, 2)
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

    def add_skip_trt_case(self):
        if False:
            i = 10
            return i + 15

        def teller1(program_config, predictor_config):
            if False:
                for i in range(10):
                    print('nop')
            if self.trt_param.precision == paddle_infer.PrecisionType.Int8:
                return True
            return False
        self.add_skip_case(teller1, SkipReasons.TRT_NOT_IMPLEMENTED, 'When precisionType is int8 without relu op, output is different between Trt and Paddle.')

    def test(self):
        if False:
            while True:
                i = 10
        self.add_skip_trt_case()
        self.run_test()

    def test_quant(self):
        if False:
            print('Hello World!')
        self.add_skip_trt_case()
        self.run_test(quant=True)
if __name__ == '__main__':
    unittest.main()