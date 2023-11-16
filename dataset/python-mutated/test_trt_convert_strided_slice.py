import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertStridedSliceTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input1(attrs: List[Dict[str, Any]]):
            if False:
                while True:
                    i = 10
            return np.random.random([1, 56, 56, 192]).astype(np.float32)
        for axes in [[1, 2]]:
            for starts in [[1, 1]]:
                for ends in [[10000000, 10000000]]:
                    for decrease_axis in [[]]:
                        for infer_flags in [[1, 1]]:
                            for strides in [[2, 2]]:
                                dics = [{'axes': axes, 'starts': starts, 'ends': ends, 'decrease_axis': decrease_axis, 'infer_flags': infer_flags, 'strides': strides}]
                                ops_config = [{'op_type': 'strided_slice', 'op_inputs': {'Input': ['input_data']}, 'op_outputs': {'Out': ['slice_output_data']}, 'op_attrs': dics[0]}]
                                ops = self.generate_op_config(ops_config)
                                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dics))}, outputs=['slice_output_data'])
                                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'input_data': [1, 56, 56, 192]}
            self.dynamic_shape.max_input_shape = {'input_data': [8, 56, 56, 192]}
            self.dynamic_shape.opt_input_shape = {'input_data': [4, 56, 56, 192]}

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
            inputs = program_config.inputs
            if dynamic_shape:
                for i in range(len(attrs[0]['starts'])):
                    if attrs[0]['starts'][i] < 0 or attrs[0]['ends'][i] < 0:
                        return (0, 3)
            if not dynamic_shape:
                for x in attrs[0]['axes']:
                    if x == 0:
                        return (0, 3)
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7000:
                return (0, 3)
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_test()

class TrtConvertStridedSliceTest2(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input1(attrs: List[Dict[str, Any]]):
            if False:
                return 10
            return np.random.random([1, 56, 56, 192]).astype(np.float32)
        for axes in [[1, 2], [2, 3], [1, 3]]:
            for starts in [[-10, 1], [-10, 20], [-10, 15], [-10, 16], [-10, 20]]:
                for ends in [[-9, 10000], [-9, -1], [-9, 40]]:
                    for decrease_axis in [[]]:
                        for infer_flags in [[1, 1]]:
                            for strides in [[2, 2]]:
                                dics = [{'axes': axes, 'starts': starts, 'ends': ends, 'decrease_axis': [axes[0]], 'infer_flags': infer_flags, 'strides': strides}]
                                ops_config = [{'op_type': 'strided_slice', 'op_inputs': {'Input': ['input_data']}, 'op_outputs': {'Out': ['slice_output_data']}, 'op_attrs': dics[0]}]
                                ops = self.generate_op_config(ops_config)
                                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dics))}, outputs=['slice_output_data'])
                                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {'input_data': [1, 56, 56, 192]}
            self.dynamic_shape.max_input_shape = {'input_data': [8, 100, 100, 200]}
            self.dynamic_shape.opt_input_shape = {'input_data': [4, 56, 56, 192]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 2), 1e-05)
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 2), 1e-05)

    def test(self):
        if False:
            print('Hello World!')
        self.run_test()
if __name__ == '__main__':
    unittest.main()