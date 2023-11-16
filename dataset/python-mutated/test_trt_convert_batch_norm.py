import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertBatchNormTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10

        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if False:
                i = 10
                return i + 15
            if self.dims == 4:
                if attrs[0]['data_layout'] == 'NCHW':
                    return np.ones([batch, 3, 24, 24]).astype(np.float32)
                elif attrs[0]['data_layout'] == 'NHWC':
                    return np.ones([batch, 24, 24, 3]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 3]).astype(np.float32)

        def generate_bias(attrs: List[Dict[str, Any]], batch):
            if False:
                print('Hello World!')
            return np.full(3, 0.9).astype('float32')

        def generate_mean(attrs: List[Dict[str, Any]], batch):
            if False:
                for i in range(10):
                    print('nop')
            return np.full(3, 0.9).astype('float32')

        def generate_scale(attrs: List[Dict[str, Any]], batch):
            if False:
                i = 10
                return i + 15
            return np.full(3, 1.1).astype('float32')

        def generate_variance(attrs: List[Dict[str, Any]], batch):
            if False:
                for i in range(10):
                    print('nop')
            return np.full(3, 1.2).astype('float32')

        def generate_MomentumTensor(attrs: List[Dict[str, Any]], batch):
            if False:
                i = 10
                return i + 15
            return np.full(3, 0.9).astype('float32')
        for dims in [2, 3, 4]:
            for num_input in [0, 1]:
                for batch in [1, 4]:
                    for epsilon in [1e-06, 1e-05, 0.0001]:
                        for data_layout in ['NCHW']:
                            for momentum in [0.9, 0.8]:
                                self.num_input = num_input
                                self.dims = dims
                                dics = [{'epsilon': epsilon, 'data_layout': data_layout, 'momentum': momentum, 'is_test': True, 'trainable_statistics': False}, {}]
                                dics_intput = [{'X': ['batch_norm_input'], 'Bias': ['Bias'], 'Mean': ['Mean'], 'Scale': ['Scale'], 'Variance': ['Variance'], 'MomentumTensor': ['MomentumTensor']}, {'X': ['batch_norm_input'], 'Bias': ['Bias'], 'Mean': ['Mean'], 'Scale': ['Scale'], 'Variance': ['Variance']}]
                                dics_intputs = [{'Bias': TensorConfig(data_gen=partial(generate_bias, dics, batch)), 'Mean': TensorConfig(data_gen=partial(generate_mean, dics, batch)), 'Scale': TensorConfig(data_gen=partial(generate_scale, dics, batch)), 'Variance': TensorConfig(data_gen=partial(generate_variance, dics, batch)), 'MomentumTensor': TensorConfig(data_gen=partial(generate_MomentumTensor, dics, batch))}, {'Bias': TensorConfig(data_gen=partial(generate_bias, dics, batch)), 'Mean': TensorConfig(data_gen=partial(generate_mean, dics, batch)), 'Scale': TensorConfig(data_gen=partial(generate_scale, dics, batch)), 'Variance': TensorConfig(data_gen=partial(generate_variance, dics, batch))}]
                                ops_config = [{'op_type': 'batch_norm', 'op_inputs': dics_intput[num_input], 'op_outputs': {'Y': ['batch_norm_out'], 'MeanOut': ['Mean'], 'VarianceOut': ['Variance'], 'SavedMean': ['SavedMean'], 'SavedVariance': ['SavedVariance']}, 'op_attrs': dics[0]}]
                                ops = self.generate_op_config(ops_config)
                                program_config = ProgramConfig(ops=ops, weights=dics_intputs[num_input], inputs={'batch_norm_input': TensorConfig(data_gen=partial(generate_input1, dics, batch))}, outputs=['batch_norm_out'])
                                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            if self.dims == 4:
                if attrs[0]['data_layout'] == 'NCHW':
                    self.dynamic_shape.min_input_shape = {'batch_norm_input': [1, 3, 12, 12]}
                    self.dynamic_shape.max_input_shape = {'batch_norm_input': [4, 3, 24, 24]}
                    self.dynamic_shape.opt_input_shape = {'batch_norm_input': [1, 3, 24, 24]}
                elif attrs[0]['data_layout'] == 'NHWC':
                    self.dynamic_shape.min_input_shape = {'batch_norm_input': [1, 12, 12, 3]}
                    self.dynamic_shape.max_input_shape = {'batch_norm_input': [4, 24, 24, 3]}
                    self.dynamic_shape.opt_input_shape = {'batch_norm_input': [1, 24, 24, 3]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'batch_norm_input': [1, 3, 12]}
                self.dynamic_shape.max_input_shape = {'batch_norm_input': [4, 3, 24]}
                self.dynamic_shape.opt_input_shape = {'batch_norm_input': [1, 3, 24]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'batch_norm_input': [1, 3]}
                self.dynamic_shape.max_input_shape = {'batch_norm_input': [4, 3]}
                self.dynamic_shape.opt_input_shape = {'batch_norm_input': [1, 3]}

        def clear_dynamic_shape():
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                return 10
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
            while True:
                i = 10

        def teller1(program_config, predictor_config):
            if False:
                return 10
            if len(program_config.weights) == 5:
                return True
            return False
        self.add_skip_case(teller1, SkipReasons.TRT_NOT_SUPPORT, 'INPUT MomentumTensor NOT SUPPORT')

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()