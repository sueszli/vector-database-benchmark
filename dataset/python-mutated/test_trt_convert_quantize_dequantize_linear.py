import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertQuantizeDequantizeTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 8517:
            return False
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')
        self.trt_param.workspace_size = 1073741824

        def generate_input1(shape):
            if False:
                i = 10
                return i + 15
            return np.random.random(shape).astype(np.float32)

        def generate_add(shape):
            if False:
                i = 10
                return i + 15
            return np.ones(shape).astype(np.float32)

        def generate_scale():
            if False:
                i = 10
                return i + 15
            return np.ones([1]).astype(np.float32) + 2.521234002

        def generate_zeropoint():
            if False:
                for i in range(10):
                    print('nop')
            return np.zeros([1]).astype(np.float32)
        desc = [{'quant_axis': -1}]
        ops_config = [{'op_type': 'quantize_linear', 'op_inputs': {'X': ['input_data_1'], 'Scale': ['scale_data_1'], 'ZeroPoint': ['zeropoint_data_1']}, 'op_outputs': {'Y': ['y_data_1']}, 'op_attrs': desc[0]}, {'op_type': 'dequantize_linear', 'op_inputs': {'X': ['y_data_1'], 'Scale': ['scale_data_2'], 'ZeroPoint': ['zeropoint_data_2']}, 'op_outputs': {'Y': ['y_data_2']}, 'op_attrs': desc[0]}, {'op_type': 'elementwise_add', 'op_inputs': {'X': ['y_data_2'], 'Y': ['add']}, 'op_outputs': {'Out': ['y_data_3']}, 'op_attrs': {'axis': -1}, 'outputs_dtype': {'output_data': np.float32}}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={'scale_data_1': TensorConfig(data_gen=partial(generate_scale)), 'zeropoint_data_1': TensorConfig(data_gen=partial(generate_zeropoint)), 'scale_data_2': TensorConfig(data_gen=partial(generate_scale)), 'zeropoint_data_2': TensorConfig(data_gen=partial(generate_zeropoint)), 'add': TensorConfig(data_gen=partial(generate_add, [1, 8, 32, 32]))}, inputs={'input_data_1': TensorConfig(data_gen=partial(generate_input1, [1, 8, 32, 32]))}, outputs=['y_data_3'])
        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {'input_data_1': [1, 8, 32, 32], 'add': [1, 8, 32, 32]}
            self.dynamic_shape.max_input_shape = {'input_data_1': [16, 8, 32, 32], 'add': [16, 8, 32, 32]}
            self.dynamic_shape.opt_input_shape = {'input_data_1': [16, 8, 32, 32], 'add': [16, 8, 32, 32]}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.01, 0.01))

    def test(self):
        if False:
            return 10
        self.run_test(quant=False, explicit=True)
if __name__ == '__main__':
    unittest.main()