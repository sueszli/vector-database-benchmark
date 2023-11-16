from functools import partial
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertSetValue(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input1():
            if False:
                for i in range(10):
                    print('nop')
            return np.random.random([1, 6, 20, 50, 10, 3]).astype(np.float32)

        def generate_input2():
            if False:
                while True:
                    i = 10
            return np.random.random([1, 6, 20, 50, 10, 1]).astype(np.float32)
        ops_config = [{'op_type': 'set_value', 'op_inputs': {'Input': ['input_data'], 'ValueTensor': ['update_data']}, 'op_outputs': {'Out': ['set_output_data']}, 'op_attrs': {'axes': [5], 'starts': [0], 'ends': [1], 'steps': [1]}}, {'op_type': 'gelu', 'op_inputs': {'X': ['set_output_data']}, 'op_outputs': {'Out': ['set_tmp_output_data']}, 'op_attrs': {'approximate': True}}, {'op_type': 'slice', 'op_inputs': {'Input': ['set_tmp_output_data']}, 'op_outputs': {'Out': ['slice3_output_data']}, 'op_attrs': {'decrease_axis': [], 'axes': [5], 'starts': [1], 'ends': [2]}}, {'op_type': 'scale', 'op_inputs': {'X': ['slice3_output_data']}, 'op_outputs': {'Out': ['scale5_output_data']}, 'op_attrs': {'scale': 62.1, 'bias': 1, 'bias_after_scale': True}}, {'op_type': 'scale', 'op_inputs': {'X': ['scale5_output_data']}, 'op_outputs': {'Out': ['scale6_output_data']}, 'op_attrs': {'scale': 0.1, 'bias': 0, 'bias_after_scale': True}}, {'op_type': 'set_value', 'op_inputs': {'Input': ['set_tmp_output_data'], 'ValueTensor': ['scale6_output_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': {'axes': [5], 'starts': [1], 'ends': [2], 'steps': [1]}}]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1)), 'update_data': TensorConfig(data_gen=partial(generate_input2))}, outputs=['output_data'])
        yield program_config

    def sample_predictor_configs(self, program_config):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {'input_data': [1, 6, 20, 50, 10, 3], 'update_data': [1, 6, 20, 50, 10, 1], 'output_data': [1, 6, 20, 50, 10, 3], 'set_output_data': [1, 6, 20, 50, 10, 3]}
            self.dynamic_shape.max_input_shape = {'input_data': [1, 6, 20, 50, 10, 3], 'update_data': [1, 6, 20, 50, 10, 1], 'output_data': [1, 6, 20, 50, 10, 3], 'set_output_data': [1, 6, 20, 50, 10, 3]}
            self.dynamic_shape.opt_input_shape = {'input_data': [1, 6, 20, 50, 10, 3], 'update_data': [1, 6, 20, 50, 10, 1], 'output_data': [1, 6, 20, 50, 10, 3], 'set_output_data': [1, 6, 20, 50, 10, 3]}

        def clear_dynamic_shape():
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                return 10
            if dynamic_shape:
                ver = paddle_infer.get_trt_compile_version()
                if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
                    return (1, 5)
                return (1, 3)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        self.trt_param.workspace_size = 2013265920
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (1e-05, 0.0001))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test()