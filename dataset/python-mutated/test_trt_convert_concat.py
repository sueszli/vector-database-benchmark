import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertConcatTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        if len(inputs['concat_input1'].shape) <= attrs[0]['axis']:
            return False
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if False:
                return 10
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], batch):
            if False:
                for i in range(10):
                    print('nop')
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        def generate_input3(attrs: List[Dict[str, Any]], batch):
            if False:
                for i in range(10):
                    print('nop')
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            if False:
                for i in range(10):
                    print('nop')
            return np.zeros([1]).astype(np.int32)
        for dims in [2, 3, 4]:
            for num_input in [0, 1]:
                for batch in [1, 2, 4]:
                    for axis in [-1, 0, 1, 2, 3]:
                        self.num_input = num_input
                        self.dims = dims
                        dics = [{'axis': axis}, {}]
                        dics_intput = [{'X': ['concat_input1', 'concat_input2', 'concat_input3'], 'AxisTensor': ['AxisTensor']}, {'X': ['concat_input1', 'concat_input2', 'concat_input3']}]
                        dics_inputs = [{'concat_input1': TensorConfig(data_gen=partial(generate_input1, dics, batch)), 'concat_input2': TensorConfig(data_gen=partial(generate_input2, dics, batch)), 'concat_input3': TensorConfig(data_gen=partial(generate_input3, dics, batch)), 'AxisTensor': TensorConfig(data_gen=partial(generate_weight1, dics))}, {'concat_input1': TensorConfig(data_gen=partial(generate_input1, dics, batch)), 'concat_input2': TensorConfig(data_gen=partial(generate_input2, dics, batch)), 'concat_input3': TensorConfig(data_gen=partial(generate_input3, dics, batch))}]
                        ops_config = [{'op_type': 'concat', 'op_inputs': dics_intput[num_input], 'op_outputs': {'Out': ['concat_output']}, 'op_attrs': dics[0]}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={}, inputs=dics_inputs[num_input], outputs=['concat_output'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            if self.num_input == 0:
                if self.dims == 4:
                    self.dynamic_shape.min_input_shape = {'concat_input1': [1, 3, 24, 24], 'concat_input2': [1, 3, 24, 24], 'concat_input3': [1, 3, 24, 24], 'AxisTensor': [1]}
                    self.dynamic_shape.max_input_shape = {'concat_input1': [4, 3, 48, 48], 'concat_input2': [4, 3, 48, 48], 'concat_input3': [4, 3, 48, 48], 'AxisTensor': [1]}
                    self.dynamic_shape.opt_input_shape = {'concat_input1': [1, 3, 24, 24], 'concat_input2': [1, 3, 24, 24], 'concat_input3': [1, 3, 24, 24], 'AxisTensor': [1]}
                elif self.dims == 3:
                    self.dynamic_shape.min_input_shape = {'concat_input1': [1, 3, 24], 'concat_input2': [1, 3, 24], 'concat_input3': [1, 3, 24], 'AxisTensor': [1]}
                    self.dynamic_shape.max_input_shape = {'concat_input1': [4, 12, 48], 'concat_input2': [4, 12, 48], 'concat_input3': [4, 12, 48], 'AxisTensor': [1]}
                    self.dynamic_shape.opt_input_shape = {'concat_input1': [1, 3, 24], 'concat_input2': [1, 3, 24], 'concat_input3': [1, 3, 24], 'AxisTensor': [1]}
                elif self.dims == 2:
                    self.dynamic_shape.min_input_shape = {'concat_input1': [1, 24], 'concat_input2': [1, 24], 'concat_input3': [1, 24], 'AxisTensor': [1]}
                    self.dynamic_shape.max_input_shape = {'concat_input1': [4, 48], 'concat_input2': [4, 48], 'concat_input3': [4, 48], 'AxisTensor': [1]}
                    self.dynamic_shape.opt_input_shape = {'concat_input1': [1, 24], 'concat_input2': [1, 24], 'concat_input3': [1, 24], 'AxisTensor': [1]}
                elif self.dims == 1:
                    self.dynamic_shape.min_input_shape = {'concat_input1': [24], 'concat_input2': [24], 'concat_input3': [24], 'AxisTensor': [0]}
                    self.dynamic_shape.max_input_shape = {'concat_input1': [48], 'concat_input2': [48], 'concat_input3': [48], 'AxisTensor': [0]}
                    self.dynamic_shape.opt_input_shape = {'concat_input1': [24], 'concat_input2': [24], 'concat_input3': [24], 'AxisTensor': [0]}
            elif self.num_input == 1:
                if self.dims == 4:
                    self.dynamic_shape.min_input_shape = {'concat_input1': [1, 3, 24, 24], 'concat_input2': [1, 3, 24, 24], 'concat_input3': [1, 3, 24, 24]}
                    self.dynamic_shape.max_input_shape = {'concat_input1': [4, 3, 48, 48], 'concat_input2': [4, 3, 48, 48], 'concat_input3': [4, 3, 48, 48]}
                    self.dynamic_shape.opt_input_shape = {'concat_input1': [1, 3, 24, 24], 'concat_input2': [1, 3, 24, 24], 'concat_input3': [1, 3, 24, 24]}
                elif self.dims == 3:
                    self.dynamic_shape.min_input_shape = {'concat_input1': [1, 3, 24], 'concat_input2': [1, 3, 24], 'concat_input3': [1, 3, 24]}
                    self.dynamic_shape.max_input_shape = {'concat_input1': [4, 12, 48], 'concat_input2': [4, 12, 48], 'concat_input3': [4, 12, 48]}
                    self.dynamic_shape.opt_input_shape = {'concat_input1': [1, 3, 24], 'concat_input2': [1, 3, 24], 'concat_input3': [1, 3, 24]}
                elif self.dims == 2:
                    self.dynamic_shape.min_input_shape = {'concat_input1': [1, 24], 'concat_input2': [1, 24], 'concat_input3': [1, 24]}
                    self.dynamic_shape.max_input_shape = {'concat_input1': [4, 48], 'concat_input2': [4, 48], 'concat_input3': [4, 48]}
                    self.dynamic_shape.opt_input_shape = {'concat_input1': [1, 24], 'concat_input2': [1, 24], 'concat_input3': [1, 24]}
                elif self.dims == 1:
                    self.dynamic_shape.min_input_shape = {'concat_input1': [24], 'concat_input2': [24], 'concat_input3': [24]}
                    self.dynamic_shape.max_input_shape = {'concat_input1': [48], 'concat_input2': [48], 'concat_input3': [48]}
                    self.dynamic_shape.opt_input_shape = {'concat_input1': [24], 'concat_input2': [24], 'concat_input3': [24]}

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
            if dynamic_shape:
                return (1, 4)
            elif attrs[0]['axis'] != 0:
                return (1, 4)
            else:
                return (0, 5)
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

    def add_skip_trt_case(self):
        if False:
            i = 10
            return i + 15

        def teller1(program_config, predictor_config):
            if False:
                print('Hello World!')
            if len(program_config.inputs) == 4:
                return True
            return False
        self.add_skip_case(teller1, SkipReasons.TRT_NOT_SUPPORT, 'INPUT AxisTensor NOT SUPPORT')

    def test(self):
        if False:
            while True:
                i = 10
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()