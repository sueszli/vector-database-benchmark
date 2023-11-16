import unittest
from functools import partial
from typing import List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertEinsumTest_SingleOperand(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            while True:
                i = 10
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
            return False
        return True

    def sample_program_configs(self):
        if False:
            while True:
                i = 10
        self.trt_param.workspace_size = 1073741824

        def generate_input1(dims, batch):
            if False:
                return 10
            if dims == 1:
                return np.ones(shape=[batch]).astype(np.float32)
            elif dims == 2:
                return np.ones(shape=[batch, 3]).astype(np.float32)
            elif dims == 3:
                return np.ones((batch, 2, 3)).astype(np.float32)

        def generate_equation1(dims):
            if False:
                while True:
                    i = 10
            if dims == 1:
                return ['i->']
            elif dims == 2:
                return ['ij->ji', 'ij->i', 'ij->j']
            elif dims == 3:
                return ['ijk->ikj', 'ijk->i', 'ijk->ij', 'ijk->ik', 'ijk->ijk', 'ijk->jk']
        for dims in [1, 2, 3]:
            for batch in [2]:
                equation_list = generate_equation1(dims)
                for equation in equation_list:
                    self.equation = equation
                    self.dims = dims
                    dics = [{'equation': equation}]
                    ops_config = [{'op_type': 'einsum', 'op_inputs': {'Operands': ['operands_data0']}, 'op_outputs': {'Out': ['einsum_output_data']}, 'op_attrs': dics[0]}]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(ops=ops, weights={}, inputs={'operands_data0': TensorConfig(data_gen=partial(generate_input1, dims, batch))}, outputs=['einsum_output_data'])
                    yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                i = 10
                return i + 15
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'operands_data0': [1]}
                self.dynamic_shape.max_input_shape = {'operands_data0': [3]}
                self.dynamic_shape.opt_input_shape = {'operands_data0': [2]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'operands_data0': [1, 3]}
                self.dynamic_shape.max_input_shape = {'operands_data0': [4, 3]}
                self.dynamic_shape.opt_input_shape = {'operands_data0': [2, 3]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'operands_data0': [1, 2, 3]}
                self.dynamic_shape.max_input_shape = {'operands_data0': [4, 2, 3]}
                self.dynamic_shape.opt_input_shape = {'operands_data0': [2, 2, 3]}

        def clear_dynamic_shape():
            if False:
                i = 10
                return i + 15
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                for i in range(10):
                    print('nop')
            if not dynamic_shape or '...' in self.equation:
                return (0, 3)
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
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
            return 10
        self.run_test()

class TrtConvertEinsumTest_DoubuleOperand_Vector_Matrix(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            return 10
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
            return False
        return True

    def sample_program_configs(self):
        if False:
            for i in range(10):
                print('nop')
        self.trt_param.workspace_size = 1073741824

        def generate_input_matrix(dims, batch):
            if False:
                while True:
                    i = 10
            if dims == 1:
                return np.ones(shape=[batch]).astype(np.float32)
            elif dims == 2:
                return np.ones(shape=[batch, 3]).astype(np.float32)
            elif dims == 3:
                return np.ones((batch, 2, 3)).astype(np.float32)
        '\n        genertate_vector\n        '

        def generate_input_vector(vec_shape):
            if False:
                return 10
            return np.ones(vec_shape).astype(np.float32)

        def generate_equation_matrix_vector(dims, vec_shape):
            if False:
                for i in range(10):
                    print('nop')
            if dims == 1:
                return ['i,i->', 'i,i->i', 'i,j->ij']
            elif dims == 2 and vec_shape == [3]:
                return ['ij,j->i', 'ij,j->j', 'ij,j->ij', 'ij,j', 'ij,j->']
            elif dims == 3 and vec_shape == [3]:
                return ['ijk,k->i', 'ijk,k->j', 'ijk,k->k', 'ijk,k->ij', 'ijk,k->ik', 'ijk,k->jk', 'ijk,k->ijk', 'ijk,k', 'ijk,k->']
        for dims in [1]:
            self.dims = dims
            for vec_shape in [[2], [3]]:
                for batch in [2]:
                    equation_list = generate_equation_matrix_vector(dims, vec_shape)
                    for equation in equation_list:
                        if dims == 1 and vec_shape != [2] and (equation != 'i,j->ij') or ((dims == 2 or dims == 3) and vec_shape != [3]):
                            continue
                        self.equation = equation
                        self.dims = dims
                        dics = [{'equation': equation}, {}]
                        ops_config = [{'op_type': 'einsum', 'op_inputs': {'Operands': ['operands_data0', 'operands_data1']}, 'op_outputs': {'Out': ['einsum_output_data']}, 'op_attrs': dics[0]}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={}, inputs={'operands_data0': TensorConfig(data_gen=partial(generate_input_matrix, dims, batch)), 'operands_data1': TensorConfig(data_gen=partial(generate_input_vector, vec_shape))}, outputs=['einsum_output_data'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            return 10

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {'operands_data0': [1], 'operands_data1': [1]}
                self.dynamic_shape.max_input_shape = {'operands_data0': [4], 'operands_data1': [4]}
                self.dynamic_shape.opt_input_shape = {'operands_data0': [2], 'operands_data1': [2]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {'operands_data0': [1, 3], 'operands_data1': [1]}
                self.dynamic_shape.max_input_shape = {'operands_data0': [4, 3], 'operands_data1': [4]}
                self.dynamic_shape.opt_input_shape = {'operands_data0': [2, 3], 'operands_data1': [3]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {'operands_data0': [1, 2, 3], 'operands_data1': [1]}
                self.dynamic_shape.max_input_shape = {'operands_data0': [4, 2, 3], 'operands_data1': [4]}
                self.dynamic_shape.opt_input_shape = {'operands_data0': [2, 2, 3], 'operands_data1': [3]}

        def clear_dynamic_shape():
            if False:
                while True:
                    i = 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                print('Hello World!')
            if not dynamic_shape or '...' in self.equation:
                return (0, 4)
            return (1, 3)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
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
            while True:
                i = 10
        self.run_test()

class TrtConvertEinsumTest_DoubuleOperand_Matrix_Matrix(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            i = 10
            return i + 15
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
            return False
        return True

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15
        self.trt_param.workspace_size = 1073741824

        def generate_input_matrix(input_shape):
            if False:
                for i in range(10):
                    print('nop')
            return np.ones(shape=input_shape).astype(np.float32)
        for item in [[[4, 5], [4, 5], 'ij,ij->ij'], [[4, 5], [2, 5], 'ij,kj->ik'], [[4, 5], [3, 7], 'ij,kl->ijkl'], [[3, 4, 5], [3, 5, 2], 'bij,bjk->bik'], [[3, 4, 5], [4, 5], 'ijk,jk->i'], [[3, 4, 5], [2, 5], 'ijk,lk->ijl'], [[2, 4, 5, 3], [3, 4, 5], 'ijkl,lmn->ijkmn'], [[3, 4, 5], [4, 5], 'ijk,jk->ik'], [[3, 4, 5], [4, 5], 'ijk,jk->ij'], [[4, 5], [4, 2, 5], 'ik,ijk->j'], [[4, 2, 5], [4, 5], 'ijk,ik->jk'], [[2, 4, 5, 3], [3, 2, 4], 'ijkl,lmn->kmn'], [[2, 4, 5, 3], [3, 2, 4], 'ijkl,lmn->ijn'], [[1, 3, 5], [1, 2, 3, 4], 'blq,bhlk->bhlqk']]:
            self.x_shape = item[0]
            self.y_shape = item[1]
            equation = item[2]
            self.equation = equation
            dics = [{'equation': equation}, {}]
            ops_config = [{'op_type': 'einsum', 'op_inputs': {'Operands': ['operands_data0', 'operands_data1']}, 'op_outputs': {'Out': ['einsum_output_data']}, 'op_attrs': dics[0]}]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(ops=ops, weights={}, inputs={'operands_data0': TensorConfig(data_gen=partial(generate_input_matrix, self.x_shape)), 'operands_data1': TensorConfig(data_gen=partial(generate_input_matrix, self.y_shape))}, outputs=['einsum_output_data'])
            yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            for i in range(10):
                print('nop')

        def generate_dynamic_shape(attrs):
            if False:
                while True:
                    i = 10
            min_xshape = self.x_shape[:]
            max_xshape = self.x_shape[:]
            min_yshape = self.y_shape[:]
            max_yshape = self.y_shape[:]
            if 'b' in self.equation:
                min_xshape[0] = 1
                max_xshape[0] = 4
                min_yshape[0] = 1
                max_yshape[0] = 4
            self.dynamic_shape.min_input_shape = {'operands_data0': min_xshape, 'operands_data1': min_yshape}
            self.dynamic_shape.max_input_shape = {'operands_data0': max_xshape, 'operands_data1': max_yshape}
            self.dynamic_shape.opt_input_shape = {'operands_data0': self.x_shape, 'operands_data1': self.y_shape}

        def clear_dynamic_shape():
            if False:
                for i in range(10):
                    print('nop')
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            if not dynamic_shape or '...' in self.equation:
                return (0, 4)
            return (1, 3)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
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
            for i in range(10):
                print('nop')
        self.run_test()
if __name__ == '__main__':
    unittest.main()