import unittest
import paddle
from paddle.base.layer_helper import LayerHelper
from paddle.incubate.autograd.primrules import _jvp, _transpose
paddle.enable_static()

class TestAddPJVPAndTranspose(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()
        self.layer_help = LayerHelper('TestPrim2Orig')
        with paddle.static.program_guard(self.main_program, self.startup_program):
            self.init_data()

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'add_p'
        X = paddle.static.data(name='X', shape=[2, 2], dtype='float')
        Y = paddle.static.data(name='Y', shape=[2, 2], dtype='float')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[2, 2], dtype='float')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[2, 2], dtype='float')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: True
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[2, 2], dtype='float')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}
        self.all_ops = ['add_p', 'add_p']

    def test_op(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle.static.program_guard(self.main_program, self.startup_program):
            op = self.layer_help.append_op(type=self.op_type, inputs=self.prim_input, outputs=self.prim_output, attrs=self.prim_attrs)
            jvp_out = _jvp(op, *self.jvp_args)
            jvp_out = paddle.utils.flatten(jvp_out)
            for (k, v) in self.jvp_out_shape_map.items():
                self.assertEqual(jvp_out[k].shape, v.shape)
            if hasattr(self, 'transpose_args'):
                transpose_out = _transpose(op, *self.transpose_args)
                transpose_out = paddle.utils.flatten(transpose_out)
                for (k, v) in self.transpose_out_shape_map.items():
                    self.assertEqual(transpose_out[k].shape, v.shape)
            all_ops = [op.type for op in self.main_program.block(0).ops]
            self.assertEqual(sorted(all_ops), sorted(self.all_ops))

class TestSubPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'sub_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: True
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[5, 6], dtype='int64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}
        self.all_ops = ['sub_p', 'sub_p', 'fill_constant_p', 'sub_p']

class TestMulPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'mul_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: v is X
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[5, 6], dtype='int64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['mul_p', 'mul_p', 'mul_p', 'add_p', 'mul_p']

class TestDivPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'div_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: v is X
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[5, 6], dtype='int64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['div_p', 'div_p', 'div_p', 'mul_p', 'mul_p', 'sub_p', 'div_p']

class TestSqrtPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'sqrt_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['sqrt_p', 'div_p', 'mul_p', 'fill_constant_p']

class TestRSqrtPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            print('Hello World!')
        self.op_type = 'rsqrt_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['rsqrt_p', 'div_p', 'div_p', 'mul_p', 'fill_constant_p']

class TestTanhPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'tanh_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['tanh_p', 'mul_p', 'sub_p', 'fill_constant_p', 'mul_p']

class TestSinPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.op_type = 'sin_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['sin_p', 'mul_p', 'cos_p']

class TestCosPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.op_type = 'cos_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['cos_p', 'mul_p', 'sin_p', 'fill_constant_p', 'sub_p']

class TestExpPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'exp_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['exp_p', 'mul_p']

class TestErfPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            print('Hello World!')
        self.op_type = 'erf_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['erf_p', 'exp_p', 'fill_constant_p', 'fill_constant_p', 'fill_constant_p', 'mul_p', 'mul_p', 'pow_p', 'sub_p']

class TestAbsPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'abs_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['abs_p', 'select_p', 'ge_p', 'fill_constant_p', 'fill_constant_p', 'sub_p']

class TestCastPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            return 10
        self.op_type = 'cast_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'dtype': paddle.float64}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        check_dot = lambda v: True
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[5, 6], dtype='float')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['cast_p', 'cast_p', 'cast_p']

class TestLogPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'log_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        self.all_ops = ['log_p', 'div_p']

class TestReshapePJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'reshape_p'
        X = paddle.static.data(name='X', shape=[8, 8], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'shape': [2, 32]}
        X_DOT = paddle.static.data(name='X_DOT', shape=[8, 8], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[2, 32], dtype='int64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['reshape_p', 'reshape_p', 'reshape_p']

class TestBroadcastPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'broadcast_p'
        X = paddle.static.data(name='X', shape=[10, 1], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'shape': [2, 10, 7]}
        X_DOT = paddle.static.data(name='X_DOT', shape=[10, 7], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[2, 10, 7], dtype='int64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['broadcast_p', 'broadcast_p', 'reduce_sum_p', 'reshape_p']

class TestTransposePJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.op_type = 'transpose_p'
        X = paddle.static.data(name='X', shape=[2, 3, 4, 5], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': [0, 2, 3, 1]}
        X_DOT = paddle.static.data(name='X_DOT', shape=[2, 3, 4, 5], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[2, 4, 5, 3], dtype='int64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['transpose_p', 'transpose_p', 'transpose_p']

class TestSplitPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.op_type = 'split_p'
        X = paddle.static.data(name='X', shape=[2, 7, 10], dtype='int64')
        self.prim_input = {'X': X}
        self.prim_output = {'YS': [self.layer_help.create_variable_for_type_inference(dtype=X.dtype) for i in range(4)]}
        self.prim_attrs = {'num_or_sections': [2, 3, 4, 1], 'axis': 2}
        X_DOT = paddle.static.data(name='X_DOT', shape=[2, 7, 10], dtype='int64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['YS'][0], 1: self.prim_output['YS'][1], 2: self.prim_output['YS'][2], 3: self.prim_output['YS'][3]}
        check_dot = lambda v: v is X
        YS_BAR = [paddle.static.data(name='Y_BAR1', shape=[2, 7, 2], dtype='int64'), paddle.static.data(name='Y_BAR2', shape=[2, 7, 3], dtype='int64'), paddle.static.data(name='Y_BAR3', shape=[2, 7, 4], dtype='int64'), paddle.static.data(name='Y_BAR4', shape=[2, 7, 1], dtype='int64')]
        self.transpose_args = (check_dot, YS_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['split_p', 'split_p', 'concat_p']

class TestConcatPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'concat_p'
        X = paddle.static.data(name='X', shape=[3, 9, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[3, 2, 5], dtype='float64')
        Z = paddle.static.data(name='Z', shape=[3, 3, 5], dtype='float64')
        self.prim_input = {'XS': [X, Y, Z]}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': 1}
        XS_DOT = [paddle.static.data(name='X_DOT1', shape=[3, 9, 5], dtype='float64'), paddle.static.data(name='X_DOT2', shape=[3, 2, 5], dtype='float64'), paddle.static.data(name='X_DOT3', shape=[3, 3, 5], dtype='float64')]
        self.jvp_args = (XS_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        check_dot = lambda v: v is X or v is Y or v is Z
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[3, 14, 5], dtype='float64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y, 2: Z}
        self.all_ops = ['concat_p', 'concat_p', 'split_p']

class TestReduceSumPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'reduce_sum_p'
        X = paddle.static.data(name='X', shape=[2, 3, 4, 5], dtype='float64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': [2], 'keepdim': False}
        X_DOT = paddle.static.data(name='X_DOT1', shape=[2, 3, 4, 5], dtype='float64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[2, 3, 5], dtype='float64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['reduce_sum_p', 'reduce_sum_p', 'reshape_p', 'broadcast_p']

class TestMatmulPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'matmul_p'
        X = paddle.static.data(name='X', shape=[2, 3], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[3, 4], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[2, 3], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[3, 4], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: v is X
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[2, 4], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['matmul_p', 'matmul_p', 'matmul_p', 'add_p', 'matmul_p', 'transpose_p']

class TestSliceSelectPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            print('Hello World!')
        self.op_type = 'slice_select_p'
        X = paddle.static.data(name='X', shape=[3, 20], dtype='float64')
        self.prim_input = {'X': X}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': [1], 'starts': [0], 'ends': [20], 'strides': [2]}
        X_DOT = paddle.static.data(name='X_DOT', shape=[3, 20], dtype='float64')
        self.jvp_args = (X_DOT,)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[3, 10], dtype='float64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['slice_select_p', 'slice_select_p', 'slice_assign_p', 'fill_constant_p']

class TestSliceAssignPJVPAndTranspose1(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            return 10
        self.op_type = 'slice_assign_p'
        X = paddle.static.data(name='X', shape=[3, 20], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[3, 5], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': [1], 'starts': [0], 'ends': [10], 'strides': [2]}
        X_DOT = paddle.static.data(name='X_DOT', shape=[3, 20], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[3, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: v is X
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[3, 20], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['slice_assign_p', 'slice_assign_p', 'slice_assign_p', 'add_p', 'fill_constant_p', 'fill_constant_p', 'slice_assign_p', 'fill_constant_p']

class TestSliceAssignPJVPAndTranspose2(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.op_type = 'slice_assign_p'
        X = paddle.static.data(name='X', shape=[3, 20], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[3, 5], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': [1], 'starts': [0], 'ends': [10], 'strides': [2]}
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[3, 5], dtype='float64')
        self.jvp_args = (None, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: v is Y
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[3, 20], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {1: Y}
        self.all_ops = ['slice_assign_p', 'slice_assign_p', 'fill_constant_p', 'slice_select_p', 'fill_constant_p']

class TestSliceAssignPJVPAndTranspose3(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'slice_assign_p'
        X = paddle.static.data(name='X', shape=[3, 20], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[3, 5], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': [1], 'starts': [0], 'ends': [10], 'strides': [2]}
        X_DOT = paddle.static.data(name='X_DOT', shape=[3, 20], dtype='float64')
        self.jvp_args = (X_DOT, None)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: v is X
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[3, 20], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['slice_assign_p', 'slice_assign_p', 'fill_constant_p', 'slice_assign_p', 'fill_constant_p']

class TestGatherPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.op_type = 'gather_p'
        X = paddle.static.data(name='X', shape=[9, 5], dtype='float64')
        IndexTensor = paddle.static.data(name='IndexTensor', shape=[3], dtype='int32')
        self.prim_input = {'X': X, 'IndexTensor': IndexTensor}
        self.prim_output = {'Y': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': 1}
        X_DOT = paddle.static.data(name='X_DOT', shape=[9, 5], dtype='float64')
        self.jvp_args = (X_DOT, IndexTensor)
        self.jvp_out_shape_map = {0: self.prim_output['Y']}
        check_dot = lambda v: v is X
        Y_BAR = paddle.static.data(name='Y_BAR', shape=[9, 3], dtype='float64')
        self.transpose_args = (check_dot, Y_BAR)
        self.transpose_out_shape_map = {0: X}
        self.all_ops = ['gather_p', 'gather_p', 'scatter_add_p', 'fill_constant_p']

class TestScatterAddPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'scatter_add_p'
        X = paddle.static.data(name='X', shape=[9, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[9, 3], dtype='float64')
        IndexTensor = paddle.static.data(name='IndexTensor', shape=[3], dtype='int32')
        self.prim_input = {'X': X, 'Y': Y, 'IndexTensor': IndexTensor}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {'axis': 1}
        X_DOT = paddle.static.data(name='X_DOT', shape=[9, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[9, 3], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: v is X or v is Y
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[9, 5], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}
        self.all_ops = ['scatter_add_p', 'scatter_add_p', 'scatter_add_p', 'fill_constant_p', 'gather_p']

class TestSelectPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'select_p'
        Cond = paddle.static.data(name='Condition', shape=[9, 5], dtype='bool')
        X = paddle.static.data(name='X', shape=[9, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[9, 5], dtype='float64')
        self.prim_input = {'Condition': Cond, 'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        Cond_DOT = paddle.static.data(name='Cond_DOT', shape=[9, 5], dtype='float64')
        X_DOT = paddle.static.data(name='X_DOT', shape=[9, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[9, 5], dtype='float64')
        self.jvp_args = (Cond_DOT, X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        check_dot = lambda v: True
        Z_BAR = paddle.static.data(name='Z_BAR', shape=[9, 5], dtype='float64')
        self.transpose_args = (check_dot, Z_BAR)
        self.transpose_out_shape_map = {0: X, 1: Y}
        self.all_ops = ['select_p', 'select_p', 'fill_constant_p', 'fill_constant_p', 'fill_constant_p', 'select_p', 'select_p']

class TestEqPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'eq_p'
        X = paddle.static.data(name='X', shape=[4, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[4, 5], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[4, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[4, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        self.all_ops = ['eq_p', 'fill_constant_p']

class TestGtPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            return 10
        self.op_type = 'gt_p'
        X = paddle.static.data(name='X', shape=[4, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[4, 5], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[4, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[4, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        self.all_ops = ['gt_p', 'fill_constant_p']

class TestGePJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            return 10
        self.op_type = 'ge_p'
        X = paddle.static.data(name='X', shape=[4, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[4, 5], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[4, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[4, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        self.all_ops = ['ge_p', 'fill_constant_p']

class TestNePJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'ne_p'
        X = paddle.static.data(name='X', shape=[4, 5], dtype='float64')
        Y = paddle.static.data(name='Y', shape=[4, 5], dtype='float64')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[4, 5], dtype='float64')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[4, 5], dtype='float64')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        self.all_ops = ['ne_p', 'fill_constant_p']

class TestPowPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            print('Hello World!')
        self.op_type = 'pow_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='float32')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='float32')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='float32')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='float32')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        self.all_ops = ['pow_p', 'fill_constant_p', 'fill_constant_p', 'eq_p', 'select_p', 'sub_p', 'mul_p', 'mul_p', 'pow_p', 'mul_p', 'mul_p', 'log_p', 'add_p']

class TestMaxPJVPAndTranspose(TestAddPJVPAndTranspose):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'max_p'
        X = paddle.static.data(name='X', shape=[5, 6], dtype='float32')
        Y = paddle.static.data(name='Y', shape=[5, 6], dtype='float32')
        self.prim_input = {'X': X, 'Y': Y}
        self.prim_output = {'Z': self.layer_help.create_variable_for_type_inference(dtype=X.dtype)}
        self.prim_attrs = {}
        X_DOT = paddle.static.data(name='X_DOT', shape=[5, 6], dtype='float32')
        Y_DOT = paddle.static.data(name='Y_DOT', shape=[5, 6], dtype='float32')
        self.jvp_args = (X_DOT, Y_DOT)
        self.jvp_out_shape_map = {0: self.prim_output['Z']}
        self.all_ops = ['max_p', 'fill_constant_p', 'eq_p', 'select_p']
if __name__ == '__main__':
    unittest.main()