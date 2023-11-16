import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertGroupNormTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if attrs[0]['epsilon'] < 0 or attrs[0]['epsilon'] > 0.001:
            return False
        if attrs[0]['groups'] <= 0:
            return False
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input(attrs: List[Dict[str, Any]], batch):
            if False:
                while True:
                    i = 10
            if attrs[0]['data_layout'] == 'NCHW':
                return np.random.random([batch, 32, 64, 64]).astype(np.float32)
            else:
                return np.random.random([batch, 64, 64, 32]).astype(np.float32)

        def generate_scale():
            if False:
                print('Hello World!')
            return np.random.randn(32).astype(np.float32)

        def generate_bias():
            if False:
                i = 10
                return i + 15
            return np.random.randn(32).astype(np.float32)
        for batch in [1, 2, 4]:
            for group in [1, 4, 32, -1]:
                for epsilon in [1e-05, 5e-05]:
                    for data_layout in ['NCHW']:
                        dics = [{'epsilon': epsilon, 'groups': group, 'data_layout': data_layout}, {}]
                        ops_config = [{'op_type': 'group_norm', 'op_inputs': {'X': ['input_data'], 'Scale': ['scale_weight'], 'Bias': ['bias_weight']}, 'op_outputs': {'Y': ['y_output'], 'Mean': ['mean_output'], 'Variance': ['variance_output']}, 'op_attrs': dics[0]}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={'scale_weight': TensorConfig(data_gen=partial(generate_scale)), 'bias_weight': TensorConfig(data_gen=partial(generate_bias))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, dics, batch))}, outputs=['y_output'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {'input_data': [1, 16, 16, 16]}
            self.dynamic_shape.max_input_shape = {'input_data': [4, 64, 128, 128]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, 32, 64, 64]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.workspace_size = 2013265920
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.01)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        generate_dynamic_shape(attrs)
        self.trt_param.workspace_size = 2013265920
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.01)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)

    def test(self):
        if False:
            print('Hello World!')
        self.run_test()
if __name__ == '__main__':
    unittest.main()