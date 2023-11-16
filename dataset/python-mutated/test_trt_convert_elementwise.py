import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertElementwiseTestOneInputSpecialCase0(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10

        def generate_input(shape, op_type):
            if False:
                while True:
                    i = 10
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=shape, dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=shape).astype(np.float32)
            else:
                return np.random.random(shape).astype(np.float32)

        def generate_weight(op_type):
            if False:
                return 10
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=[1, 32, 1, 1], dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=[1, 32, 1, 1]).astype(np.float32)
            else:
                return np.random.randn(1, 32, 1, 1).astype(np.float32)
        for batch in [1, 4]:
            for shape in [[batch, 32, 16, 32]]:
                for op_type in ['elementwise_add', 'elementwise_mul', 'elementwise_sub', 'elementwise_div', 'elementwise_pow', 'elementwise_min', 'elementwise_max', 'elementwise_floordiv', 'elementwise_mod']:
                    for axis in [-1]:
                        self.dims = len(shape)
                        dics = [{'axis': axis}]
                        ops_config = [{'op_type': op_type, 'op_inputs': {'X': ['input_data'], 'Y': ['weight']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.float32 if op_type != 'elementwise_floordiv' else np.int32}}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={'weight': TensorConfig(data_gen=partial(generate_weight, op_type))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, shape, op_type))}, outputs=['output_data'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            print('Hello World!')

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 4, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32, 32, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [4, 32, 16, 32]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            print('Hello World!')
        pass

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_skip_trt_case()
        self.run_test()

class TrtConvertElementwiseTestOneInputSpecialCase1(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')

        def generate_input(shape, op_type):
            if False:
                while True:
                    i = 10
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=shape, dtype=np.int32)
            if op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=shape).astype(np.float32)
            else:
                return np.random.random(shape).astype(np.float32)

        def generate_weight(op_type):
            if False:
                for i in range(10):
                    print('nop')
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=[1], dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=[1]).astype(np.float32)
            else:
                return np.random.randn(1).astype(np.float32)
        for shape in [[32]]:
            for op_type in ['elementwise_add', 'elementwise_mul', 'elementwise_sub', 'elementwise_div', 'elementwise_pow', 'elementwise_min', 'elementwise_max', 'elementwise_floordiv', 'elementwise_mod']:
                for axis in [-1]:
                    self.dims = len(shape)
                    dics = [{'axis': axis}]
                    ops_config = [{'op_type': op_type, 'op_inputs': {'X': ['input_data'], 'Y': ['weight']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.float32 if op_type != 'elementwise_floordiv' else np.int32}}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={'weight': TensorConfig(data_gen=partial(generate_weight, op_type))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, shape, op_type))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {'input_data': [32]}
            self.dynamic_shape.max_input_shape = {'input_data': [64]}
            self.dynamic_shape.opt_input_shape = {'input_data': [32]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            if not dynamic_shape:
                return (0, 3)
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            print('Hello World!')
        pass

    def test(self):
        if False:
            return 10
        self.add_skip_trt_case()
        self.run_test()

class TrtConvertElementwiseTestOneInput(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        return True

    def sample_program_configs(self):
        if False:
            print('Hello World!')

        def generate_input(shape, op_type):
            if False:
                print('Hello World!')
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=shape, dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=shape).astype(np.float32)
            else:
                return np.random.random(shape).astype(np.float32)

        def generate_weight(op_type):
            if False:
                print('Hello World!')
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=[32], dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=[32]).astype(np.float32)
            else:
                return np.random.randn(32).astype(np.float32)
        for batch in [1, 4]:
            for shape in [[32], [batch, 32], [batch, 32, 32], [batch, 32, 16, 32]]:
                for op_type in ['elementwise_add', 'elementwise_mul', 'elementwise_sub', 'elementwise_div', 'elementwise_pow', 'elementwise_min', 'elementwise_max', 'elementwise_floordiv', 'elementwise_mod']:
                    for axis in [-1 if len(shape) == 1 else 1]:
                        self.dims = len(shape)
                        dics = [{'axis': axis}]
                        ops_config = [{'op_type': op_type, 'op_inputs': {'X': ['input_data'], 'Y': ['weight']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.float32 if op_type != 'elementwise_floordiv' else np.int32}}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={'weight': TensorConfig(data_gen=partial(generate_weight, op_type))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, shape, op_type))}, outputs=['output_data'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [4]}
                self.dynamic_shape.max_input_shape = {'input_data': [32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [16]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 32]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 32, 32]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 4, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32, 32, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [4, 32, 16, 32]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                i = 10
                return i + 15
            if self.dims == 1 and (not dynamic_shape):
                return (0, 3)
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            while True:
                i = 10
        pass

    def test(self):
        if False:
            i = 10
            return i + 15
        self.add_skip_trt_case()
        self.run_test()

class TrtConvertElementwiseTestTwoInputWithoutBroadcast(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input(shape, op_type):
            if False:
                return 10
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=shape, dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=shape).astype(np.float32)
            else:
                return np.random.random(shape).astype(np.float32)
        for shape in [[4], [4, 32], [2, 32, 16], [1, 8, 16, 32]]:
            for op_type in ['elementwise_add', 'elementwise_mul', 'elementwise_sub', 'elementwise_div', 'elementwise_pow', 'elementwise_min', 'elementwise_max', 'elementwise_floordiv', 'elementwise_mod']:
                for axis in [0, -1]:
                    self.dims = len(shape)
                    dics = [{'axis': axis}]
                    ops_config = [{'op_type': op_type, 'op_inputs': {'X': ['input_data1'], 'Y': ['input_data2']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.float32 if op_type != 'elementwise_floordiv' else np.int32}}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data1': TensorConfig(data_gen=partial(generate_input, shape, op_type)), 'input_data2': TensorConfig(data_gen=partial(generate_input, shape, op_type))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data1': [1], 'input_data2': [1]}
                self.dynamic_shape.max_input_shape = {'input_data1': [128], 'input_data2': [128]}
                self.dynamic_shape.opt_input_shape = {'input_data1': [32], 'input_data2': [32]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data1': [1, 4], 'input_data2': [1, 4]}
                self.dynamic_shape.max_input_shape = {'input_data1': [128, 256], 'input_data2': [128, 256]}
                self.dynamic_shape.opt_input_shape = {'input_data1': [32, 64], 'input_data2': [32, 64]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data1': [1, 4, 4], 'input_data2': [1, 4, 4]}
                self.dynamic_shape.max_input_shape = {'input_data1': [128, 128, 256], 'input_data2': [128, 128, 256]}
                self.dynamic_shape.opt_input_shape = {'input_data1': [2, 32, 16], 'input_data2': [2, 32, 16]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data1': [1, 4, 4, 4], 'input_data2': [1, 4, 4, 4]}
                self.dynamic_shape.max_input_shape = {'input_data1': [8, 128, 64, 128], 'input_data2': [8, 128, 64, 128]}
                self.dynamic_shape.opt_input_shape = {'input_data1': [2, 64, 32, 32], 'input_data2': [2, 64, 32, 32]}

        def clear_dynamic_shape():
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            if self.dims == 1 and (not dynamic_shape):
                return (0, 4)
            return (1, 3)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 3), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (1, 3), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            while True:
                i = 10
        pass

    def test(self):
        if False:
            print('Hello World!')
        self.add_skip_trt_case()
        self.run_test()

class TrtConvertElementwiseTestTwoInputWithBroadcast(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        inputs = program_config.inputs
        if len(inputs['input_data1'].shape) != len(inputs['input_data2'].shape):
            return False
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input(shape, op_type):
            if False:
                print('Hello World!')
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=shape, dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=shape).astype(np.float32)
            else:
                return np.random.random(shape).astype(np.float32)
        input1_shape_list = [[4, 32], [2, 4, 32], [4, 2, 4, 32]]
        input2_shape1_list = [[32], [4, 32], [2, 4, 32]]
        input2_shape2_list = [[4, 1], [2, 4, 1], [4, 2, 4, 1]]
        input2_shape3_list = [[32], [2, 1, 1], [4, 2, 1, 32]]
        input2_shape4_list = [[32], [4, 32], [4, 1, 4, 32]]
        input2_shape5_list = [[32], [2, 1, 32], [4, 1, 1, 32]]
        input2_shape6_list = [[1, 32], [1, 32], [1, 1, 1, 32]]
        input2_shape_list = [input2_shape1_list, input2_shape2_list, input2_shape3_list, input2_shape4_list, input2_shape5_list, input2_shape6_list]
        axis1_list = [[-1], [1, -1], [1, -1]]
        axis2_list = [[-1], [0], [0]]
        axis3_list = [[-1], [0], [0]]
        axis4_list = [[-1], [-1], [0]]
        axis5_list = [[-1, 1], [-1, 0], [-1, 0]]
        axis6_list = [[-1, 0], [-1, 1], [-1, 0]]
        axis_list = [axis1_list, axis2_list, axis3_list, axis4_list, axis5_list, axis6_list]
        for i in range(3):
            input1_shape = input1_shape_list[i]
            for j in range(6):
                input2_shape = input2_shape_list[j][i]
                for op_type in ['elementwise_add', 'elementwise_mul', 'elementwise_sub', 'elementwise_div', 'elementwise_pow', 'elementwise_min', 'elementwise_max', 'elementwise_floordiv', 'elementwise_mod']:
                    for axis in axis_list[j][i]:
                        self.shape1 = input1_shape
                        self.shape2 = input2_shape
                        dics = [{'axis': axis}]
                        ops_config = [{'op_type': op_type, 'op_inputs': {'X': ['input_data1'], 'Y': ['input_data2']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.float32 if op_type != 'elementwise_floordiv' else np.int32}}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data1': TensorConfig(data_gen=partial(generate_input, input1_shape, op_type)), 'input_data2': TensorConfig(data_gen=partial(generate_input, input2_shape, op_type))}, outputs=['output_data'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            while True:
                i = 10

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            max_shape = [[128], [128, 128], [128, 128, 128], [128, 128, 128, 128]]
            min_shape = [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]
            opt_shape = [[32], [32, 32], [32, 32, 32], [32, 32, 32, 32]]
            self.dynamic_shape.min_input_shape = {'input_data1': min_shape[len(self.shape1) - 1], 'input_data2': min_shape[len(self.shape2) - 1]}
            self.dynamic_shape.max_input_shape = {'input_data1': max_shape[len(self.shape1) - 1], 'input_data2': max_shape[len(self.shape2) - 1]}
            self.dynamic_shape.opt_input_shape = {'input_data1': opt_shape[len(self.shape1) - 1], 'input_data2': opt_shape[len(self.shape2) - 1]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        if self.shape1[0] == self.shape2[0]:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            program_config.set_input_type(np.float32)
            yield (self.create_inference_config(), (1, 3), (1e-05, 1e-05))
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            program_config.set_input_type(np.float16)
            yield (self.create_inference_config(), (1, 3), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 3), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (1, 3), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            print('Hello World!')
        pass

    def test(self):
        if False:
            i = 10
            return i + 15
        self.add_skip_trt_case()
        self.run_test()

class TrtConvertElementwiseTestOneInputCornerCase(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        return True

    def sample_program_configs(self):
        if False:
            return 10

        def generate_input(shape, op_type):
            if False:
                return 10
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=shape, dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=shape).astype(np.float32)
            else:
                return np.random.random(shape).astype(np.float32)

        def generate_weight(op_type):
            if False:
                return 10
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=[32], dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=[32]).astype(np.float32)
            else:
                return np.random.rand(32).astype(np.float32)
        for batch in [1, 2, 4]:
            for shape in [[32], [batch, 32], [batch, 32, 32], [batch, 32, 16, 32]]:
                for op_type in ['elementwise_add', 'elementwise_mul', 'elementwise_sub', 'elementwise_div', 'elementwise_pow', 'elementwise_min', 'elementwise_max', 'elementwise_floordiv', 'elementwise_mod']:
                    self.op_type = op_type
                    for axis in [-1 if len(shape) == 1 else 1]:
                        self.dims = len(shape)
                        dics = [{'axis': axis}]
                        ops_config = [{'op_type': op_type, 'op_inputs': {'X': ['weight'], 'Y': ['input_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.float32 if op_type != 'elementwise_floordiv' else np.int32}}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={'weight': TensorConfig(data_gen=partial(generate_weight, op_type))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, shape, op_type))}, outputs=['output_data'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [4]}
                self.dynamic_shape.max_input_shape = {'input_data': [64]}
                self.dynamic_shape.opt_input_shape = {'input_data': [32]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 32]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32, 256]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 32, 16]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 4, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32, 128, 256]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 32, 32, 16]}

        def clear_dynamic_shape():
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (0, 3), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (0, 3), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (1, 2), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (1, 2), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            i = 10
            return i + 15
        pass

    def test(self):
        if False:
            return 10
        self.add_skip_trt_case()
        self.run_test()

class TrtConvertElementwiseTestTwoInputSkipCase(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input(shape, op_type):
            if False:
                i = 10
                return i + 15
            if op_type == 'elementwise_pow':
                return np.random.randint(low=1, high=10000, size=shape, dtype=np.int32)
            if op_type == 'elementwise_mul':
                return np.random.random(shape).astype(np.bool_)
        for shape in [[4], [4, 32], [2, 32, 16], [1, 8, 16, 32]]:
            for op_type in ['elementwise_pow', 'elementwise_mul']:
                for axis in [0, -1]:
                    self.dims = len(shape)
                    dics = [{'axis': axis}]
                    ops_config = [{'op_type': op_type, 'op_inputs': {'X': ['input_data1'], 'Y': ['input_data2']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.int32 if op_type == 'elementwise_pow' else np.bool_}}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data1': TensorConfig(data_gen=partial(generate_input, shape, op_type)), 'input_data2': TensorConfig(data_gen=partial(generate_input, shape, op_type))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data1': [1], 'input_data2': [1]}
                self.dynamic_shape.max_input_shape = {'input_data1': [128], 'input_data2': [128]}
                self.dynamic_shape.opt_input_shape = {'input_data1': [32], 'input_data2': [32]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data1': [1, 4], 'input_data2': [1, 4]}
                self.dynamic_shape.max_input_shape = {'input_data1': [128, 256], 'input_data2': [128, 256]}
                self.dynamic_shape.opt_input_shape = {'input_data1': [32, 64], 'input_data2': [32, 64]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data1': [1, 4, 4], 'input_data2': [1, 4, 4]}
                self.dynamic_shape.max_input_shape = {'input_data1': [128, 128, 256], 'input_data2': [128, 128, 256]}
                self.dynamic_shape.opt_input_shape = {'input_data1': [2, 32, 16], 'input_data2': [2, 32, 16]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data1': [1, 4, 4, 4], 'input_data2': [1, 4, 4, 4]}
                self.dynamic_shape.max_input_shape = {'input_data1': [8, 128, 64, 128], 'input_data2': [8, 128, 64, 128]}
                self.dynamic_shape.opt_input_shape = {'input_data1': [2, 64, 32, 32], 'input_data2': [2, 64, 32, 32]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                return 10
            return (0, 4)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), (0, 4), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), (0, 4), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test(self):
        if False:
            print('Hello World!')
        self.add_skip_trt_case()
        self.run_test()

class TrtConvertPowOp(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10

        def generate_input(shape):
            if False:
                for i in range(10):
                    print('nop')
            if len(shape) == 0:
                return np.random.random([]).astype(np.float32)
            return np.random.random(shape).astype(np.float32)
        for batch in [1, 4]:
            for shape in [[], [32], [batch, 32], [batch, 32, 32], [batch, 32, 16, 32]]:
                for factor in [1.0, 2.0, -1.0, 0.5, -2]:
                    self.dims = len(shape)
                    dics = [{'factor': factor}]
                    ops_config = [{'op_type': 'pow', 'op_inputs': {'X': ['input_data']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.float32}}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, shape))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            if self.dims == 0:
                self.dynamic_shape.min_input_shape = {'input_data': []}
                self.dynamic_shape.max_input_shape = {'input_data': []}
                self.dynamic_shape.opt_input_shape = {'input_data': []}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [4]}
                self.dynamic_shape.max_input_shape = {'input_data': [32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [16]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 32]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 32, 32]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 32, 4, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 32, 32, 32]}
                self.dynamic_shape.opt_input_shape = {'input_data': [4, 32, 16, 32]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
            if (self.dims == 1 or self.dims == 0) and (not dynamic_shape):
                return (0, 3)
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def add_skip_trt_case(self):
        if False:
            while True:
                i = 10
        pass

    def test(self):
        if False:
            return 10
        self.add_skip_trt_case()
        self.run_test()

class TrtConvertElementwise0D(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            print('Hello World!')
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_input(dims, op_type):
            if False:
                while True:
                    i = 10
            shape = []
            if dims == 0:
                shape = []
            elif dims == 1:
                shape = [8]
            elif dims == 2:
                shape = [1, 8]
            elif dims == 3:
                shape = [1, 8, 8]
            else:
                shape = [1, 8, 8, 8]
            if op_type == 'elementwise_floordiv':
                return np.random.randint(low=1, high=10000, size=shape, dtype=np.int32)
            elif op_type == 'elementwise_mod':
                return np.random.uniform(low=0.1, high=1.0, size=shape).astype(np.float32)
            else:
                return np.random.random(shape).astype(np.float32)
        for dims in [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]:
            for op_type in ['elementwise_add', 'elementwise_mul', 'elementwise_sub', 'elementwise_div', 'elementwise_pow', 'elementwise_min', 'elementwise_max', 'elementwise_floordiv', 'elementwise_mod']:
                for axis in [-1 if dims[0] == 1 or dims[0] == 0 else 1]:
                    self.dims = dims[0]
                    dics = [{'axis': axis}]
                    ops_config = [{'op_type': op_type, 'op_inputs': {'X': ['input_data'], 'Y': ['weight']}, 'op_outputs': {'Out': ['output_data']}, 'op_attrs': dics[0], 'outputs_dtype': {'output_data': np.float32 if op_type != 'elementwise_floordiv' else np.int32}}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={'weight': TensorConfig(data_gen=partial(generate_input, dims[1], op_type))}, inputs={'input_data': TensorConfig(data_gen=partial(generate_input, dims[0], op_type))}, outputs=['output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                for i in range(10):
                    print('nop')
            if self.dims == 0:
                self.dynamic_shape.min_input_shape = {'input_data': []}
                self.dynamic_shape.max_input_shape = {'input_data': []}
                self.dynamic_shape.opt_input_shape = {'input_data': []}
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'input_data': [1]}
                self.dynamic_shape.max_input_shape = {'input_data': [16]}
                self.dynamic_shape.opt_input_shape = {'input_data': [8]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 8]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 8]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 8]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 1, 4]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 16, 16]}
                self.dynamic_shape.opt_input_shape = {'input_data': [2, 8, 8]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {'input_data': [1, 8, 8, 8]}
                self.dynamic_shape.max_input_shape = {'input_data': [4, 8, 8, 8]}
                self.dynamic_shape.opt_input_shape = {'input_data': [4, 8, 8, 8]}

        def clear_dynamic_shape():
            if False:
                print('Hello World!')
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
            if not dynamic_shape and (self.dims == 1 or self.dims == 0):
                return (0, 3)
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), (0.001, 0.001))
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (1e-05, 1e-05))
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), (0.001, 0.001))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test()
if __name__ == '__main__':
    unittest.main()