import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op import Operator
from op_test_xpu import XPUOpTest
import paddle
from paddle.base import core

class XPUTestAdamOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'adam'
        self.use_dynamic_create_class = False

    class TestAdamOp(XPUOpTest):
        """Test Adam Op with supplied attributes"""

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'adam'
            self.place = paddle.XPUPlace(0)
            self.set_data()
            self.set_attrs()
            self.set_shape()
            self.set_inputs()
            self.set_steps()
            (param_out, moment1_out, moment2_out) = adam_step(self.inputs, self.attrs)
            self.outputs = {'Moment1Out': moment1_out, 'Moment2Out': moment2_out, 'ParamOut': param_out, 'Beta1PowOut': np.array([self.beta1_pow]).astype('float32') * self.beta1, 'Beta2PowOut': np.array([self.beta2_pow]).astype('float32') * self.beta2}

        def set_xpu(self):
            if False:
                for i in range(10):
                    print('nop')
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type

        def set_attrs(self):
            if False:
                while True:
                    i = 10
            self.attrs = {'epsilon': self.epsilon, 'beta1': self.beta1, 'beta2': self.beta2}

        def set_data(self):
            if False:
                i = 10
                return i + 15
            self.beta1 = 0.78
            self.beta2 = 0.836
            self.learning_rate = 0.004
            self.epsilon = 0.0001

        def set_steps(self):
            if False:
                print('Hello World!')
            self.num_steps = 1

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = (102, 105)

        def set_inputs(self):
            if False:
                print('Hello World!')
            param = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            grad = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            moment1 = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            moment2 = np.random.random(self.shape).astype(self.dtype)
            self.beta1_pow = self.beta1 ** 10
            self.beta2_pow = self.beta2 ** 10
            self.inputs = {'Param': param, 'Grad': grad, 'Moment1': moment1, 'Moment2': moment2, 'LearningRate': np.array([self.learning_rate]).astype('float32'), 'Beta1Pow': np.array([self.beta1_pow]).astype('float32'), 'Beta2Pow': np.array([self.beta2_pow]).astype('float32')}

        def test_check_output(self):
            if False:
                print('Hello World!')
            self.check_output_with_place(place=paddle.XPUPlace(0), atol=0.01)

    class TestAdamOp2(TestAdamOp):
        """Test Adam Op with supplied attributes"""

        def set_data(self):
            if False:
                print('Hello World!')
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.learning_rate = 0.001
            self.epsilon = 1e-08

    class TestAdamOp3(TestAdamOp2):
        """Test Adam Op with supplied attributes"""

        def set_shape(self):
            if False:
                return 10
            self.shape = (101, 47)

    class TestAdamOp4(TestAdamOp2):
        """Test Adam Op with supplied attributes"""

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = (512, 26)

    class TestAdamOp5(TestAdamOp2):
        """Test Adam Op with supplied attributes"""

        def set_shape(self):
            if False:
                while True:
                    i = 10
            self.shape = (11, 1)

    class TestAdamOp6(TestAdamOp2):
        """Test Adam Op with beta as Variable"""

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = (10, 10)

        def set_data(self):
            if False:
                print('Hello World!')
            self.beta1 = 0.85
            self.beta2 = 0.95
            self.learning_rate = 0.001
            self.epsilon = 1e-08

    class TestAdamOp7(TestAdamOp):
        """Test Adam Op with float16 accuracy"""

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'adam'
            self.place = paddle.XPUPlace(0)
            self.set_data()
            self.set_attrs()
            self.set_shape()
            self.set_inputs()
            self.set_steps()
            (param_out, moment1_out, moment2_out) = adam_step(self.inputs, self.attrs)
            self.outputs = {'Moment1Out': moment1_out, 'Moment2Out': moment2_out, 'ParamOut': param_out, 'Beta1PowOut': np.array([self.beta1_pow]).astype('float16') * self.beta1, 'Beta2PowOut': np.array([self.beta2_pow]).astype('float16') * self.beta2}

        def set_inputs(self):
            if False:
                return 10
            param = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            grad = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            moment1 = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            moment2 = np.random.random(self.shape).astype(self.dtype)
            self.beta1_pow = self.beta1 ** 10
            self.beta2_pow = self.beta2 ** 10
            self.inputs = {'Param': param, 'Grad': grad, 'Moment1': moment1, 'Moment2': moment2, 'LearningRate': np.array([self.learning_rate]).astype('float16'), 'Beta1Pow': np.array([self.beta1_pow]).astype('float16'), 'Beta2Pow': np.array([self.beta2_pow]).astype('float16')}

    class TestAdamOpMultipleSteps(TestAdamOp2):
        """Test Adam Operator with supplied attributes"""

        def set_steps(self):
            if False:
                return 10
            self.num_steps = 10

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            for _ in range(self.num_steps):
                (param_out, moment1_out, moment2_out) = adam_step(self.inputs, self.attrs)
                beta1_pow_out = self.inputs['Beta1Pow'] * self.beta1
                beta2_pow_out = self.inputs['Beta2Pow'] * self.beta2
                self.outputs = {'Moment1Out': moment1_out, 'Moment2Out': moment2_out, 'ParamOut': param_out, 'Beta1PowOut': beta1_pow_out, 'Beta2PowOut': beta2_pow_out}
                self.check_output_with_place(place=paddle.XPUPlace(0), atol=0.01)
                self.inputs['Param'] = param_out
                self.inputs['Moment1'] = moment1_out
                self.inputs['Moment2'] = moment2_out
                self.inputs['Beta1Pow'] = beta1_pow_out
                self.inputs['Beta2Pow'] = beta2_pow_out
                self.inputs['Grad'] = np.random.uniform(-1, 1, (102, 105)).astype('float32')

def adam_step(inputs, attributes):
    if False:
        while True:
            i = 10
    '\n    Simulate one step of the adam optimizer\n    :param inputs: dict of inputs\n    :param attributes: dict of attributes\n    :return tuple: tuple of output param, moment1, moment2,\n    beta1 power accumulator and beta2 power accumulator\n    '
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']
    epsilon = attributes['epsilon']
    if 'beta1' in attributes:
        beta1 = attributes['beta1']
    else:
        beta1 = inputs['Beta1Tensor'][0]
    if 'beta2' in attributes:
        beta2 = attributes['beta2']
    else:
        beta2 = inputs['Beta2Tensor'][0]
    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))
    return (param_out, moment1_out, moment2_out)

def adam_step_sparse(inputs, attributes, height, rows, row_numel, np_grad, lazy_mode):
    if False:
        for i in range(10):
            print('nop')
    '\n    Simulate one step of the adam optimizer\n    :param inputs: dict of inputs\n    :param attributes: dict of attributes\n    :return tuple: tuple of output param, moment1, moment2,\n    beta1 power accumulator and beta2 power accumulator\n    '
    param = inputs['Param']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']
    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']
    moment1_out = np.zeros(shape=[height, row_numel])
    moment2_out = np.zeros(shape=[height, row_numel])
    param_out = np.zeros(shape=[height, row_numel])

    def update_row(row_id, update_value):
        if False:
            for i in range(10):
                print('nop')
        moment1_out[row_id] = beta1 * moment1[row_id] + (1 - beta1) * update_value
        moment2_out[row_id] = beta2 * moment2[row_id] + (1 - beta2) * np.square(update_value)
        lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
        param_out[row_id] = param[row_id] - lr_t * (moment1_out[row_id] / (np.sqrt(moment2_out[row_id]) + epsilon))
    if lazy_mode:
        for (idx, row_id) in enumerate(rows):
            update_row(row_id, np_grad[idx])
    else:
        for row_id in range(param_out.shape[0]):
            update_value = np.zeros(np_grad[0].shape).astype('float32')
            if row_id in rows:
                update_value = np_grad[rows.index(row_id)]
            update_row(row_id, update_value)
    return (param_out, moment1_out, moment2_out)

class TestSparseAdamOp(unittest.TestCase):

    def setup(self, scope, place, lazy_mode):
        if False:
            for i in range(10):
                print('nop')
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 0.0001
        beta1_pow = np.array([beta1 ** 10]).astype('float32')
        beta2_pow = np.array([beta2 ** 10]).astype('float32')
        height = 10
        rows = [0, 4, 7]
        self.rows = rows
        row_numel = 12
        self.row_numel = row_numel
        self.dense_inputs = {'Param': np.full((height, row_numel), 5.0).astype('float32'), 'Moment1': np.full((height, row_numel), 5.0).astype('float32'), 'Moment2': np.full((height, row_numel), 5.0).astype('float32'), 'Beta1Pow': beta1_pow, 'Beta2Pow': beta2_pow, 'LearningRate': np.full(1, 2.0).astype('float32')}
        self.init_output = np.full((height, row_numel), 0.0).astype('float32')
        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2, 'min_row_size_to_use_multithread': 2}
        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype('float32')
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0
        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)
        self.sparse_inputs = ['Grad']
        (param_out, mom1, mom2) = adam_step_sparse(self.dense_inputs, self.attrs, height, rows, row_numel, np_array, lazy_mode)
        self.outputs = {'ParamOut': param_out, 'Moment1Out': mom1, 'Moment2Out': mom2, 'Beta1PowOut': beta1_pow * beta1, 'Beta2PowOut': beta2_pow * beta2}

    def check_with_place(self, place, lazy_mode):
        if False:
            print('Hello World!')
        scope = core.Scope()
        self.setup(scope, place, lazy_mode)
        op_args = {}
        op_args['lazy_mode'] = lazy_mode
        for (key, np_array) in self.dense_inputs.items():
            var = scope.var(key).get_tensor()
            var.set(np_array, place)
            op_args[key] = key
        for s in self.sparse_inputs:
            op_args[s] = s
        for s in self.outputs:
            var = scope.var(s).get_tensor()
            var.set(self.init_output, place)
            op_args[s] = s
        for k in self.attrs:
            op_args[k] = self.attrs[k]
        adam_op = Operator('adam', **op_args)
        adam_op.run(scope, place)
        for (key, np_array) in self.outputs.items():
            out_var = scope.var(key).get_tensor()
            actual = np.array(out_var)
            actual = actual.reshape([actual.size])
            np_array = np_array.reshape([np_array.size])
            for i in range(np_array.size):
                self.assertLess(actual[i] - np_array[i], 1e-05)

    def test_sparse_adam(self):
        if False:
            return 10
        xpu_version = core.get_xpu_device_version(0)
        version_str = 'xpu2' if xpu_version == core.XPUVersion.XPU2 else 'xpu1'
        if 'xpu2' == version_str:
            self.check_with_place(paddle.XPUPlace(0), False)

class TestSparseAdamOp1(TestSparseAdamOp):

    def setup(self, scope, place, lazy_mode):
        if False:
            i = 10
            return i + 15
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 0.0001
        beta1_pow = np.array([beta1 ** 10]).astype('float16')
        beta2_pow = np.array([beta2 ** 10]).astype('float16')
        height = 10
        rows = [0, 4, 7]
        self.rows = rows
        row_numel = 12
        self.row_numel = row_numel
        self.dense_inputs = {'Param': np.full((height, row_numel), 5.0).astype('float16'), 'Moment1': np.full((height, row_numel), 5.0).astype('float16'), 'Moment2': np.full((height, row_numel), 5.0).astype('float16'), 'Beta1Pow': beta1_pow, 'Beta2Pow': beta2_pow, 'LearningRate': np.full(1, 2.0).astype('float16')}
        self.init_output = np.full((height, row_numel), 0.0).astype('float16')
        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2, 'min_row_size_to_use_multithread': 2}
        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype('float16')
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0
        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)
        self.sparse_inputs = ['Grad']
        (param_out, mom1, mom2) = adam_step_sparse(self.dense_inputs, self.attrs, height, rows, row_numel, np_array, lazy_mode)
        self.outputs = {'ParamOut': param_out, 'Moment1Out': mom1, 'Moment2Out': mom2, 'Beta1PowOut': beta1_pow * beta1, 'Beta2PowOut': beta2_pow * beta2}
support_types = get_xpu_op_support_types('adam')
for stype in support_types:
    create_test_class(globals(), XPUTestAdamOp, stype)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()