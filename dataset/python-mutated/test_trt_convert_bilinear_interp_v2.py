import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertBilinearInterpV2Test(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7100:
            return False
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input1(attrs: List[Dict[str, Any]]):
            if False:
                i = 10
                return i + 15
            return np.random.uniform(low=0.0, high=1.0, size=[1, 3, 64, 64]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]]):
            if False:
                return 10
            return np.random.uniform(low=0.5, high=6.0, size=2).astype('float32')
        for data_layout in ['NCHW', 'NHWC']:
            for align_corners in [False, True]:
                for scale_y in [2.0, 1.0]:
                    for scale_x in [2.0]:
                        scale = [scale_y, scale_x]
                        dics = [{'data_layout': data_layout, 'interp_method': 'bilinear', 'align_corners': align_corners, 'align_mode': 0, 'scale': scale, 'out_h': -1, 'out_w': -1}]
                        ops_config = [{'op_type': 'bilinear_interp_v2', 'op_inputs': {'X': ['input_data'], 'Scale': ['input_scale']}, 'op_outputs': {'Out': ['bilinear_interp_v2_output_data']}, 'op_attrs': dics[0]}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={'input_scale': TensorConfig(data_gen=partial(generate_input2, dics))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dics))}, outputs=['bilinear_interp_v2_output_data'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 64, 64]}
            self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 64, 64]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
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
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.01)

    def test(self):
        if False:
            return 10
        self.run_test()

class TrtConvertBilinearInterpV2Test1(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7100:
            return False
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')
        self.workspace_size = 1 << 32

        def generate_input1(attrs: List[Dict[str, Any]]):
            if False:
                print('Hello World!')
            return np.random.uniform(low=0.0, high=1.0, size=[1, 18, 144, 144]).astype(np.float32)
        for data_layout in ['NCHW', 'NHWC']:
            for align_corners in [False, True]:
                for out_h in [128, 288]:
                    for out_w in [288]:
                        dics = [{'data_layout': data_layout, 'interp_method': 'bilinear', 'align_corners': align_corners, 'align_mode': 0, 'scale': [], 'out_h': out_h, 'out_w': out_w}]
                        ops_config = [{'op_type': 'bilinear_interp_v2', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['bilinear_interp_v2_output_data']}, 'op_attrs': dics[0]}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dics))}, outputs=['bilinear_interp_v2_output_data'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'input_data': [1, 18, 144, 144]}
            self.dynamic_shape.max_input_shape = {'input_data': [8, 18, 144, 144]}
            self.dynamic_shape.opt_input_shape = {'input_data': [4, 18, 144, 144]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.01)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.01)

    def test(self):
        if False:
            while True:
                i = 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()