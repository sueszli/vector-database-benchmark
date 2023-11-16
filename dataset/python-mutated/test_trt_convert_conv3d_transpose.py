import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertConv3dTransposeTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8400:
            return False
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, num_channels, attrs: List[Dict[str, Any]]):
            if False:
                print('Hello World!')
            return np.ones([batch, num_channels, 4, 20, 30]).astype(np.float32)

        def generate_weight1(num_channels, attrs: List[Dict[str, Any]]):
            if False:
                while True:
                    i = 10
            return np.random.random([num_channels, 64, 3, 3, 3]).astype(np.float32)
        num_channels = 128
        batch = 1
        self.num_channels = num_channels
        dics = [{'data_fromat': 'NCHW', 'dilations': [1, 1, 1], 'padding_algorithm': 'EXPLICIT', 'groups': 1, 'paddings': [1, 1, 1], 'strides': [2, 2, 2], 'output_padding': [1, 1, 1], 'output_size': []}]
        ops_config = [{'op_type': 'conv3d_transpose', 'op_inputs': {'Input': ['input_data'], 'Filter': ['conv3d_weight']}, 'op_outputs': {'Output': ['output_data']}, 'op_attrs': dics[0]}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={'conv3d_weight': TensorConfig(data_gen=partial(generate_weight1, num_channels, dics))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, batch, num_channels, dics))}, outputs=['output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {'input_data': [1, 128, 4, 20, 30]}
            self.dynamic_shape.max_input_shape = {'input_data': [1, 128, 4, 20, 30]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, 128, 4, 20, 30]}

        def clear_dynamic_shape():
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.001)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.001)

    def add_skip_trt_case(self):
        if False:
            return 10
        pass

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_skip_trt_case()
        self.run_test()

    def test_quant(self):
        if False:
            return 10
        self.add_skip_trt_case()
        self.run_test(quant=True)
if __name__ == '__main__':
    unittest.main()