import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertPNormTest(TrtLayerAutoScanTest):

    def sample_program_configs(self):
        if False:
            while True:
                i = 10

        def generate_input1(dims, attrs: List[Dict[str, Any]]):
            if False:
                return 10
            if dims == 1:
                return np.ones([3]).astype(np.float32)
            elif dims == 2:
                return np.ones([3, 64]).astype(np.float32)
            elif dims == 3:
                return np.ones([3, 64, 64]).astype(np.float32)
            else:
                return np.ones([1, 3, 64, 64]).astype(np.float32)
        for dims in [2, 3, 4]:
            for asvector in [False]:
                for keepdim in [False, True]:
                    for porder in [0, 1, 2, 3]:
                        for axis in [-1]:
                            self.dims = dims
                            dics = [{'asvector': asvector, 'keepdim': keepdim, 'axis': axis, 'porder': porder}]
                            ops_config = [{'op_type': 'p_norm', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0]}]
                            ops = self.generate_op_config(ops_config)
                            program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1, dims, dics))}, outputs=['output_data'])
                            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [128]}
                self.dynamic_shape.opt_input_shape = {'input_data': [64]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 64]}
                self.dynamic_shape.opt_input_shape = {'input_data': [3, 64]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [10, 64, 64]}
                self.dynamic_shape.opt_input_shape = {'input_data': [3, 64, 64]}
            else:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 3, 64, 64]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 64, 64]}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                return 10
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def test(self):
        if False:
            return 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()