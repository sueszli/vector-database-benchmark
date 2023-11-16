import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertNearestInterpTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if attrs[0]['scale'] <= 0 and (attrs[0]['out_h'] <= 0 or attrs[0]['out_w'] <= 0):
            return False
        if (attrs[0]['out_h'] <= 0) ^ (attrs[0]['out_w'] <= 0):
            return False
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input1(attrs: List[Dict[str, Any]]):
            if False:
                for i in range(10):
                    print('nop')
            return np.ones([1, 3, 64, 64]).astype(np.float32)
        for data_layout in ['NCHW', 'NHWC']:
            for interp_method in ['nearest']:
                for align_corners in [True, False]:
                    for scale in [2.0, -1.0, 0.0]:
                        for out_h in [32, 64, 128 - 32]:
                            for out_w in [32, -32]:
                                dics = [{'data_layout': data_layout, 'interp_method': interp_method, 'align_corners': align_corners, 'scale': scale, 'out_h': out_h, 'out_w': out_w}]
                                ops_config = [{'op_type': 'nearest_interp', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['nearest_interp_output_data']}, 'op_attrs': dics[0]}]
                                ops = self.generate_op_config(ops_config)
                                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dics))}, outputs=['nearest_interp_output_data'])
                                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 64, 64]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.01)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.01)

    def add_skip_trt_case(self):
        if False:
            return 10

        def teller1(program_config, predictor_config):
            if False:
                i = 10
                return i + 15
            if program_config.ops[0].attrs['scale'] <= 0 and self.dynamic_shape.min_input_shape:
                return True
            if program_config.ops[0].attrs['align_corners']:
                return True
            return False
        self.add_skip_case(teller1, SkipReasons.TRT_NOT_IMPLEMENTED, 'NOT Implemented: we need to add support scale <= 0 in dynamic shape in the future')

    def test(self):
        if False:
            print('Hello World!')
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()