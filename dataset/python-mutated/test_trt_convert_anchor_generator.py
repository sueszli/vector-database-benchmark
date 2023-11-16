import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertAnchorGeneratorTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            if False:
                return 10
            return np.random.random([batch, 3, 64, 64]).astype(np.float32)
        for batch in [1, 2, 4]:
            for anchor_sizes in [[64.0, 128.0, 256.0, 512.0]]:
                for aspect_ratios in [[0.5, 1, 2], [0.4, 1.2, 3]]:
                    for variances in [[1.0, 1.0, 1.0, 1.0], [0.5, 1.0, 0.5, 1.0]]:
                        for stride in [[16.0, 16.0], [16.0, 32.0]]:
                            for offset in [0.5, 0.8]:
                                dics = [{'anchor_sizes': anchor_sizes, 'aspect_ratios': aspect_ratios, 'variances': variances, 'stride': stride, 'offset': offset}]
                                ops_config = [{'op_type': 'anchor_generator', 'op_inputs': {'Input': ['input_data']}, 'op_outputs': {'Anchors': ['output_anchors'], 'Variances': ['output_variances']}, 'op_attrs': dics[0]}]
                                ops = self.generate_op_config(ops_config)
                                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, batch, dics))}, outputs=['output_anchors', 'output_variances'])
                                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 64, 64]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                return 10
            if dynamic_shape:
                return (1, 3)
            else:
                return (0, 4)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.001)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 0.001)

    def test(self):
        if False:
            print('Hello World!')
        self.run_test()
if __name__ == '__main__':
    unittest.main()