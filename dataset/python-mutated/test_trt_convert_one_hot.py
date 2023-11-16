import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertOneHotTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8510:
            return False
        return True

    def sample_program_configs(self):
        if False:
            return 10
        self.trt_param.workspace_size = 1073741824

        def generate_indices(dims, batch):
            if False:
                return 10
            if dims == 2:
                return np.random.randint(0, 10, (batch, 4), dtype=np.int32)
            elif dims == 3:
                return np.random.randint(0, 10, (batch, 4, 6), dtype=np.int32)
            else:
                return np.random.randint(0, 10, (batch, 4, 6, 8), dtype=np.int32)

        def generate_depth(dims, batch):
            if False:
                print('Hello World!')
            return np.ones((1,), dtype=np.int32) * 10
        for dims in [2, 3, 4]:
            for batch in [1, 2]:
                self.dims = dims
                dics = [{'dtype': 5, 'depth': 10}, {}]
                ops_config = [{'op_type': 'one_hot_v2', 'op_inputs': {'X': ['indices_tensor'], 'depth_tensor': ['depth_tensor_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.int_}}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={'depth_tensor_data': TensorConfig(data_gen=partial(generate_depth, dims, batch))}, inputs={'indices_tensor': TensorConfig(data_gen=partial(generate_indices, dims, batch))}, outputs=['output_data'])
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            print('Hello World!')

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'indices_tensor': [1]}
                self.dynamic_shape.max_input_shape = {'indices_tensor': [2]}
                self.dynamic_shape.opt_input_shape = {'indices_tensor': [1]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'indices_tensor': [1, 4]}
                self.dynamic_shape.max_input_shape = {'indices_tensor': [2, 4]}
                self.dynamic_shape.opt_input_shape = {'indices_tensor': [1, 4]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'indices_tensor': [1, 4, 6]}
                self.dynamic_shape.max_input_shape = {'indices_tensor': [2, 4, 6]}
                self.dynamic_shape.opt_input_shape = {'indices_tensor': [1, 4, 6]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {'indices_tensor': [1, 4, 6, 8]}
                self.dynamic_shape.max_input_shape = {'indices_tensor': [2, 4, 6, 8]}
                self.dynamic_shape.opt_input_shape = {'indices_tensor': [1, 4, 6, 8]}

        def clear_dynamic_shape():
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            if not dynamic_shape:
                return (0, 3)
            return (1, 2)
        attrs = [op.attrs for op in program_config.ops]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)

    def test(self):
        if False:
            print('Hello World!')
        self.run_test()
if __name__ == '__main__':
    unittest.main()