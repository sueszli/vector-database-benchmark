import os
import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertSliceTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10
        self.trt_param.workspace_size = 1073741824
        for hidden_size in [30]:
            for input_size in [30]:
                for batch in [2]:
                    for seq_len in [5]:
                        for num_layers in [1, 2]:
                            for is_bidirec in [True, False]:
                                dics = []
                                dics.append({'hidden_size': hidden_size, 'input_size': input_size, 'num_layers': num_layers, 'mode': 'LSTM', 'is_bidirec': is_bidirec, 'is_test': True, 'dropout_prob': 0.0, 'batch': batch, 'seq_len': seq_len})
                                K = 1
                                if dics[0]['is_bidirec']:
                                    K = 2

                                def generate_input1():
                                    if False:
                                        while True:
                                            i = 10
                                    return np.random.random([batch, seq_len, input_size]).astype(np.float32) * 2 - 1

                                def generate_w0():
                                    if False:
                                        print('Hello World!')
                                    return np.random.random([4 * hidden_size, input_size]).astype(np.float32) * 2 - 1

                                def generate_w1():
                                    if False:
                                        print('Hello World!')
                                    return np.random.random([4 * hidden_size, K * hidden_size]).astype(np.float32) * 2 - 1

                                def generate_w2():
                                    if False:
                                        print('Hello World!')
                                    return np.random.random([4 * hidden_size, hidden_size]).astype(np.float32) * 2 - 1

                                def generate_b():
                                    if False:
                                        return 10
                                    return np.random.random([4 * hidden_size]).astype(np.float32) * 2 - 1
                                dics.append({'dtype': 5, 'input_dim_idx': 0, 'str_value': '', 'value': 0.0, 'shape': [K * num_layers, -1, hidden_size], 'output_dim_idx': 1})
                                dics.append({'axis': [1, 0, 2]})
                                WeightList = ['weight' + str(i) for i in range(4 * K * dics[0]['num_layers'])]
                                weights = {}
                                for i in range(int(len(WeightList) / 2)):
                                    if i % 2 == 0:
                                        if i <= K:
                                            weights[WeightList[i]] = TensorConfig(data_gen=partial(generate_w0))
                                        else:
                                            weights[WeightList[i]] = TensorConfig(data_gen=partial(generate_w1))
                                    if i % 2 == 1:
                                        weights[WeightList[i]] = TensorConfig(data_gen=partial(generate_w2))
                                for i in range(int(len(WeightList) / 2), len(WeightList)):
                                    weights[WeightList[i]] = TensorConfig(data_gen=partial(generate_b))
                                ops_config = [{'op_type': 'fill_constant_batch_size_like', 'op_inputs': {'Input': ['input_data']}, 'op_outputs': {'Out': ['prestate1']}, 'op_attrs': dics[1]}, {'op_type': 'fill_constant_batch_size_like', 'op_inputs': {'Input': ['input_data']}, 'op_outputs': {'Out': ['prestate2']}, 'op_attrs': dics[1]}, {'op_type': 'transpose2', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['rnn_input_data']}, 'op_attrs': dics[2]}, {'op_type': 'rnn', 'op_inputs': {'Input': ['rnn_input_data'], 'PreState': ['prestate1', 'prestate2'], 'WeightList': WeightList}, 'op_outputs': {'Out': ['rnn_output_data'], 'State': ['state_output_data0', 'state_output_data1'], 'Reserve': ['reserve_data'], 'DropoutState': ['DropoutState_data']}, 'op_attrs': dics[0]}]
                                ops = self.generate_op_config(ops_config)
                                program_config = ProgramConfig(ops=ops, weights=weights, inputs={'input_data': TensorConfig(data_gen=partial(generate_input1))}, outputs=['rnn_output_data'])
                                yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        num_layers = attrs[3]['num_layers']
        hidden_size = attrs[3]['hidden_size']
        batch = attrs[3]['batch']
        input_size = attrs[3]['input_size']
        seq_len = attrs[3]['seq_len']
        K = 1
        if attrs[3]['is_bidirec']:
            K = 2

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {'input_data': [batch - 1, seq_len, input_size]}
            self.dynamic_shape.max_input_shape = {'input_data': [batch + 1, seq_len, input_size]}
            self.dynamic_shape.opt_input_shape = {'input_data': [batch, seq_len, input_size]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        tol_fp32 = 1e-05
        tol_half = 0.01
        if os.name == 'nt':
            tol_fp32 = 0.01
            tol_half = 0.1
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), tol_fp32)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), tol_half)

    def test(self):
        if False:
            while True:
                i = 10
        self.run_test()
if __name__ == '__main__':
    unittest.main()