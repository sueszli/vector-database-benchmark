import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertSplitTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')
        for dims in [2, 3, 4]:
            for batch in [3, 4]:
                for axes in [[-2, 3], [1], [2], [2, 3]]:
                    self.batch = batch
                    self.dims = dims
                    self.axes = axes
                    dics = [{'axes': axes}]
                    ops_config = [{'op_type': 'unsqueeze2', 'op_inputs': {'X': ['in_data']}, 'op_outputs': {'Out': ['out_data'], 'XShape': ['XShape_data']}, 'op_attrs': dics[0]}]
                    self.input_shape = [1] * dims
                    for i in range(dims):
                        self.input_shape[i] = np.random.randint(1, 20)

                    def generate_input1(attrs: List[Dict[str, Any]], batch):
                        if False:
                            while True:
                                i = 10
                        self.input_shape[0] = batch
                        return np.random.random(self.input_shape).astype(np.float32)
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'in_data': TensorConfig(data_gen=partial(generate_input1, dics, batch))}, outputs=['out_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            max_shape = list(self.input_shape)
            min_shape = list(self.input_shape)
            opt_shape = list(self.input_shape)
            for i in range(len(self.input_shape)):
                max_shape[i] = max_shape[i] + 1
            self.dynamic_shape.min_input_shape = {'in_data': min_shape}
            self.dynamic_shape.max_input_shape = {'in_data': max_shape}
            self.dynamic_shape.opt_input_shape = {'in_data': opt_shape}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
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

    def add_skip_trt_case(self):
        if False:
            print('Hello World!')
        pass

    def test(self):
        if False:
            print('Hello World!')
        self.add_skip_trt_case()
        self.run_test()
if __name__ == '__main__':
    unittest.main()