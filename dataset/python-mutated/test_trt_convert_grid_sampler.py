import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertGridSampler(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        self.trt_param.workspace_size = 1073741824
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input1():
            if False:
                for i in range(10):
                    print('nop')
            if self.dims == 4:
                self.input_shape = [1, 3, 32, 32]
                return np.random.random([1, 3, 32, 32]).astype(np.float32)
            elif self.dims == 5:
                self.input_shape = [1, 3, 32, 32, 64]
                return np.random.random([1, 3, 32, 32, 64]).astype(np.float32)

        def generate_input2():
            if False:
                return 10
            if self.dims == 4:
                self.input_shape = [1, 3, 3, 2]
                return np.random.random([1, 3, 3, 2]).astype(np.float32)
            elif self.dims == 5:
                self.input_shape = [1, 3, 3, 2, 3]
                return np.random.random([1, 3, 3, 2, 3]).astype(np.float32)
        mode = ['bilinear', 'nearest']
        padding_mode = ['zeros', 'reflection', 'border']
        align_corners = [True, False]
        descs = []
        for m in mode:
            for p in padding_mode:
                for a in align_corners:
                    descs.append({'mode': m, 'padding_mode': p, 'align_corners': a})
        for dims in [4, 5]:
            for desc in descs:
                self.dims = dims
                ops_config = [{'op_type': 'grid_sampler', 'op_inputs': {'X': ['input_data'], 'Grid': ['grid_data']}, 'op_outputs': {'Output': ['output_data']}, 'op_attrs': desc}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1)), 'grid_data': TensorConfig(data_gen=partial(generate_input2))}, outputs=['output_data'])
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape():
            if False:
                i = 10
                return i + 15
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32], 'grid_data': [1, 3, 3, 2]}
                self.dynamic_shape.max_input_shape = {'input_data': [1, 3, 64, 64], 'grid_data': [1, 3, 6, 2]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 32, 32], 'grid_data': [1, 3, 3, 2]}
            elif self.dims == 5:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 3, 32, 32, 64], 'grid_data': [1, 3, 3, 2, 3]}
                self.dynamic_shape.max_input_shape = {'input_data': [1, 3, 64, 64, 128], 'grid_data': [1, 3, 3, 6, 3]}
                self.dynamic_shape.opt_input_shape = {'input_data': [1, 3, 32, 32, 64], 'grid_data': [1, 3, 3, 2, 3]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 3), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (1, 3), 0.001)

    def test(self):
        if False:
            return 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()