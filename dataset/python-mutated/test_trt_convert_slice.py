import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertSliceTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        out_shape = list(inputs['input_data'].shape)
        for x in range(len(attrs[0]['axes'])):
            start = 0
            end = 0
            if attrs[0]['starts'][x] < 0:
                start = attrs[0]['starts'][x] + inputs['input_data'].shape[attrs[0]['axes'][x]]
            else:
                start = attrs[0]['starts'][x]
            if attrs[0]['ends'][x] < 0:
                end = attrs[0]['ends'][x] + inputs['input_data'].shape[attrs[0]['axes'][x]]
            else:
                end = attrs[0]['ends'][x]
            start = max(0, start)
            end = max(0, end)
            out_shape[attrs[0]['axes'][x]] = end - start
            if start >= end:
                return False
        for x in attrs[0]['decrease_axis']:
            if x < 0:
                return False
            if out_shape[x] != 1:
                return False
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input1(attrs: List[Dict[str, Any]]):
            if False:
                print('Hello World!')
            return np.random.random([6, 6, 64, 64]).astype(np.float32)
        for axes in [[0, 1], [1, 3], [2, 3]]:
            for starts in [[0, 1]]:
                for ends in [[2, 2], [5, 5], [1, -1]]:
                    for decrease_axis in [[], [1], [2], [-1], [-100]]:
                        for infer_flags in [[-1]]:
                            dics = [{'axes': axes, 'starts': starts, 'ends': ends, 'decrease_axis': decrease_axis, 'infer_flags': infer_flags}]
                            ops_config = [{'op_type': 'slice', 'op_inputs': {'Input': ['input_data']}, 'op_outputs': {'Out': ['slice_output_data']}, 'op_attrs': dics[0]}]
                            ops = self.generate_op_config(ops_config)
                            program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dics))}, outputs=['slice_output_data'])
                            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {'input_data': [8, 8, 64, 64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [6, 6, 64, 64]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
            if not dynamic_shape:
                for x in attrs[0]['axes']:
                    if x == 0:
                        return (0, 3)
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        self.trt_param.max_batch_size = 9
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
            for i in range(10):
                print('nop')
        self.run_test()
if __name__ == '__main__':
    unittest.main()