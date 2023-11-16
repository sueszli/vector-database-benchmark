import unittest
from functools import partial
from typing import Any, Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertFusedTokenPruneTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10
        self.trt_param.workspace_size = 1073741824

        def generate_attn_or_mask(attrs: List[Dict[str, Any]]):
            if False:
                i = 10
                return i + 15
            return np.ones([4, 12, 64, 64]).astype(np.float32)

        def generate_x(attrs: List[Dict[str, Any]]):
            if False:
                while True:
                    i = 10
            return np.random.random([4, 64, 76]).astype(np.float32)

        def generate_new_mask(attrs: List[Dict[str, Any]]):
            if False:
                while True:
                    i = 10
            return np.random.random([4, 12, 32, 32]).astype(np.float32)
        for keep_first_token in [True, False]:
            for keep_order in [True, False]:
                dics = [{'keep_first_token': keep_first_token, 'keep_order': keep_order}]
                ops_config = [{'op_type': 'fused_token_prune', 'op_inputs': {'Attn': ['attn'], 'X': ['x'], 'Mask': ['mask'], 'NewMask': ['new_mask']}, 'op_outputs': {'SlimmedX': ['slimmed_x'], 'CLSInds': ['cls_inds']}, 'op_attrs': dics[0]}]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(ops=ops, weights={}, inputs={'attn': TensorConfig(data_gen=partial(generate_attn_or_mask, dics)), 'x': TensorConfig(data_gen=partial(generate_x, dics)), 'mask': TensorConfig(data_gen=partial(generate_attn_or_mask, dics)), 'new_mask': TensorConfig(data_gen=partial(generate_new_mask, dics))}, outputs=['slimmed_x', 'cls_inds'])
                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {'attn': [4, 12, 64, 64], 'x': [4, 64, 76], 'mask': [4, 12, 64, 64], 'new_mask': [4, 12, 32, 32]}
            self.dynamic_shape.max_input_shape = {'attn': [4, 12, 64, 64], 'x': [4, 64, 76], 'mask': [4, 12, 64, 64], 'new_mask': [4, 12, 32, 32]}
            self.dynamic_shape.opt_input_shape = {'attn': [4, 12, 64, 64], 'x': [4, 64, 76], 'mask': [4, 12, 64, 64], 'new_mask': [4, 12, 32, 32]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            return (1, 6)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.01, 0.01))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.1, 0.01))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test()
if __name__ == '__main__':
    unittest.main()