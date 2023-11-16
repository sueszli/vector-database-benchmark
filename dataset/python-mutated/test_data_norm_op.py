"""This is unit test of Test data_norm Op."""
import unittest
import numpy as np
from op import Operator
from op_test import OpTest
import paddle
from paddle import base
from paddle.base import Program, core, program_guard

def _reference_testing(x, batch_size, batch_sum, batch_square_sum, slot_dim=-1):
    if False:
        return 10
    x_shape = x.shape
    means_arr = batch_sum / batch_size
    scales_arr = np.sqrt(batch_size / batch_square_sum)
    min_precision = 1e-07
    if slot_dim <= 0:
        for i in range(x_shape[0]):
            x[i] -= means_arr
            x[i] *= scales_arr
        y = np.array(x)
    else:
        y = np.zeros(x_shape).astype(np.float32)
        for i in range(x_shape[0]):
            for j in range(0, x_shape[1], slot_dim):
                if x[i][j] <= -min_precision or x[i][j] >= min_precision:
                    for k in range(0, slot_dim):
                        y[i][j + k] = (x[i][j + k] - means_arr[j + k]) * scales_arr[j + k]
    return y

def create_or_get_tensor(scope, var_name, var, place):
    if False:
        return 10
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_recursive_sequence_lengths([])
        tensor.set(var, place)
    return tensor

class TestDataNormOpInference(unittest.TestCase):
    """
    test class for data norm op
    test forward
    """

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        init members of this class\n        '
        self.dtype = np.float32
        self.use_mkldnn = False

    def __assert_close(self, tensor, np_array, msg, atol=0.0001):
        if False:
            i = 10
            return i + 15
        np.testing.assert_allclose(np.array(tensor), np_array, rtol=1e-05, atol=atol, err_msg=msg)

    def check_with_place(self, place, data_layout, dtype, shape, slot_dim=-1, enable_scale_and_shift=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        do forward and check\n\n        Args:\n            place(Place): CPUPlace\n            data_layout(str): NCHW or NWHC\n            dtype(dtype): np.float32\n            shape(list): input shape\n            slot_dim(int): dimension of one slot. Refer to data_norm api.\n            enable_scale_and_shift(bool): if enable scale and shift after normalization.\n\n        '
        epsilon = 1e-05
        if len(shape) == 2:
            x_shape = shape
            c = x_shape[1]
        else:
            ValueError('len(shape) should be equal to 2')
        scale_shape = [c]
        x_val = np.random.random_sample(x_shape).astype(dtype)
        x_val = x_val - 0.5
        x_val[0][1] = 0.0
        x_val[1][1] = 0.0
        batch_size = np.ones(scale_shape).astype(np.float32)
        batch_size *= 10000.0
        batch_sum = np.zeros(scale_shape).astype(np.float32)
        batch_square_sum = np.ones(scale_shape).astype(np.float32)
        batch_square_sum *= 10000.0
        y_out = _reference_testing(x_val, batch_size, batch_sum, batch_square_sum, slot_dim).astype(dtype)
        scope = core.Scope()
        x_tensor = create_or_get_tensor(scope, 'x_val', OpTest.np_dtype_to_base_dtype(x_val), place)
        batch_size_tensor = create_or_get_tensor(scope, 'batch_size', OpTest.np_dtype_to_base_dtype(batch_size), place)
        batch_sum_tensor = create_or_get_tensor(scope, 'batch_sum', OpTest.np_dtype_to_base_dtype(batch_sum), place)
        batch_square_sum_tensor = create_or_get_tensor(scope, 'batch_square_sum', OpTest.np_dtype_to_base_dtype(batch_square_sum), place)
        y_tensor = create_or_get_tensor(scope, 'y_out', None, place)
        mean_tensor = create_or_get_tensor(scope, 'mean', None, place)
        scales_tensor = create_or_get_tensor(scope, 'scales', None, place)
        if not enable_scale_and_shift:
            data_norm_op = Operator('data_norm', X='x_val', BatchSize='batch_size', BatchSum='batch_sum', BatchSquareSum='batch_square_sum', Y='y_out', Means='mean', Scales='scales', epsilon=epsilon, use_mkldnn=self.use_mkldnn, slot_dim=slot_dim, enable_scale_and_shift=False)
        else:
            scale_w = np.ones(scale_shape).astype(np.float32)
            bias = np.zeros(scale_shape).astype(np.float32)
            scale_w_tensor = create_or_get_tensor(scope, 'scale_w', OpTest.np_dtype_to_base_dtype(scale_w), place)
            bias_tensor = create_or_get_tensor(scope, 'bias', OpTest.np_dtype_to_base_dtype(bias), place)
            data_norm_op = Operator('data_norm', X='x_val', BatchSize='batch_size', BatchSum='batch_sum', BatchSquareSum='batch_square_sum', scale_w='scale_w', bias='bias', Y='y_out', Means='mean', Scales='scales', epsilon=epsilon, use_mkldnn=self.use_mkldnn, slot_dim=slot_dim, enable_scale_and_shift=True)
        data_norm_op.run(scope, place)
        self.__assert_close(y_tensor, y_out, 'inference output are different at ' + str(place) + ', ' + data_layout + ', ' + str(np.dtype(dtype)) + str(np.array(y_tensor)) + str(y_out), atol=0.001)

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        '\n        test check forward, check output\n        '
        places = [core.CPUPlace()]
        for place in places:
            for data_format in ['NCHW', 'NHWC']:
                for slot_dim in [-1, 1]:
                    for enable_scale_and_shift in [False, True]:
                        self.check_with_place(place, data_format, self.dtype, [2, 3], slot_dim=slot_dim, enable_scale_and_shift=enable_scale_and_shift)

class TestDataNormOp(OpTest):
    """
    test class for data norm op
    test forward and backward
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        init data norm op test env\n        '
        self.op_type = 'data_norm'
        self.use_mkldnn = False
        epsilon = 1e-05
        x_shape = [10, 12]
        scale_shape = [12]
        tp = np.float32
        x_val = np.random.random(x_shape).astype(tp)
        batch_size = np.ones(scale_shape).astype(tp)
        batch_size *= 10000.0
        batch_sum = np.zeros(scale_shape).astype(tp)
        batch_square_sum = np.ones(scale_shape).astype(tp)
        batch_square_sum *= 10000.0
        y = np.array(x_val)
        mean = np.zeros(x_shape[1]).astype(tp)
        scale = np.ones(x_shape[1]).astype(tp)
        self.inputs = {'X': x_val, 'BatchSize': batch_size, 'BatchSum': batch_sum, 'BatchSquareSum': batch_square_sum}
        self.outputs = {'Y': y, 'Means': mean, 'Scales': scale}
        self.attrs = {'epsilon': epsilon, 'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        '\n        test check forward, check output\n        '
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test check backward, check grad\n        '
        self.check_grad(['X'], 'Y', no_grad_set=set(), check_dygraph=False)

class TestDataNormOpWithEnableScaleAndShift(OpTest):
    """
    test class for data norm op
    test forward and backward
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        init data norm op test env\n        '
        self.op_type = 'data_norm'
        self.use_mkldnn = False
        epsilon = 1e-05
        slot_dim = -1
        enable_scale_and_shift = True
        x_shape = [2, 50]
        scale_shape = [50]
        tp = np.float32
        x_val = np.random.uniform(-1, 1, x_shape).astype(tp)
        batch_size = np.ones(scale_shape).astype(tp)
        batch_size *= 10000.0
        batch_sum = np.zeros(scale_shape).astype(tp)
        batch_square_sum = np.ones(scale_shape).astype(tp)
        batch_square_sum *= 10000.0
        scale_w = np.ones(scale_shape).astype(tp)
        bias = np.zeros(scale_shape).astype(tp)
        y = np.array(x_val)
        mean = np.zeros(x_shape[1]).astype(tp)
        scale = np.ones(x_shape[1]).astype(tp)
        self.inputs = {'X': x_val, 'BatchSize': batch_size, 'BatchSum': batch_sum, 'BatchSquareSum': batch_square_sum, 'scale_w': scale_w, 'bias': bias}
        self.outputs = {'Y': y, 'Means': mean, 'Scales': scale}
        self.attrs = {'epsilon': epsilon, 'use_mkldnn': self.use_mkldnn, 'slot_dim': slot_dim, 'enable_scale_and_shift': True}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        '\n        test check forward, check output\n        '
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        '\n        test check backward, check grad\n        '
        self.check_grad(['X'], 'Y', no_grad_set=set(), check_dygraph=False)

class TestDataNormOpWithoutEnableScaleAndShift(OpTest):
    """
    test class for data norm op
    test forward and backward
    """

    def setUp(self):
        if False:
            return 10
        '\n        init data norm op test env\n        '
        self.op_type = 'data_norm'
        self.use_mkldnn = False
        epsilon = 1e-05
        slot_dim = -1
        enable_scale_and_shift = True
        x_shape = [2, 50]
        scale_shape = [50]
        tp = np.float32
        x_val = np.random.uniform(-1, 1, x_shape).astype(tp)
        batch_size = np.ones(scale_shape).astype(tp)
        batch_size *= 10000.0
        batch_sum = np.zeros(scale_shape).astype(tp)
        batch_square_sum = np.ones(scale_shape).astype(tp)
        batch_square_sum *= 10000.0
        scale_w = np.ones(scale_shape).astype(tp)
        bias = np.zeros(scale_shape).astype(tp)
        y = np.array(x_val)
        mean = np.zeros(x_shape[1]).astype(tp)
        scale = np.ones(x_shape[1]).astype(tp)
        self.inputs = {'X': x_val, 'BatchSize': batch_size, 'BatchSum': batch_sum, 'BatchSquareSum': batch_square_sum, 'scale_w': scale_w, 'bias': bias}
        self.outputs = {'Y': y, 'Means': mean, 'Scales': scale}
        self.attrs = {'epsilon': epsilon, 'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        if False:
            return 10
        '\n        test check forward, check output\n        '
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        '\n        test check backward, check grad\n        '
        self.check_grad(['X'], 'Y', no_grad_set=set(), check_dygraph=False)

class TestDataNormOpWithEnableScaleAndShift_1(OpTest):
    """
    test class for data norm op
    test forward and backward
    """

    def setUp(self):
        if False:
            return 10
        '\n        init data norm op test env\n        '
        self.op_type = 'data_norm'
        self.use_mkldnn = False
        epsilon = 1e-05
        slot_dim = 1
        enable_scale_and_shift = True
        x_shape = [2, 50]
        scale_shape = [50]
        tp = np.float32
        x_val = np.random.uniform(-1, 1, x_shape).astype(tp)
        batch_size = np.ones(scale_shape).astype(tp)
        batch_size *= 10000.0
        batch_sum = np.zeros(scale_shape).astype(tp)
        batch_square_sum = np.ones(scale_shape).astype(tp)
        batch_square_sum *= 10000.0
        scale_w = np.ones(scale_shape).astype(tp)
        bias = np.zeros(scale_shape).astype(tp)
        y = np.array(x_val)
        mean = np.zeros(x_shape[1]).astype(tp)
        scale = np.ones(x_shape[1]).astype(tp)
        self.inputs = {'X': x_val, 'BatchSize': batch_size, 'BatchSum': batch_sum, 'BatchSquareSum': batch_square_sum, 'scale_w': scale_w, 'bias': bias}
        self.outputs = {'Y': y, 'Means': mean, 'Scales': scale}
        self.attrs = {'epsilon': epsilon, 'use_mkldnn': self.use_mkldnn, 'slot_dim': slot_dim, 'enable_scale_and_shift': True}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        '\n        test check forward, check output\n        '
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        '\n        test check backward, check grad\n        '
        self.check_grad(['X'], 'Y', no_grad_set=set(), check_dygraph=False)

class TestDataNormOpWithSlotDim(OpTest):
    """
    test class for data norm op
    test forward and backward
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        init data norm op test env\n        '
        self.op_type = 'data_norm'
        self.use_mkldnn = False
        epsilon = 1e-05
        slot_dim = 1
        x_shape = [2, 50]
        scale_shape = [50]
        tp = np.float32
        x_val = np.random.uniform(-1, 1, x_shape).astype(tp)
        batch_size = np.ones(scale_shape).astype(tp)
        batch_size *= 10000.0
        batch_sum = np.zeros(scale_shape).astype(tp)
        batch_square_sum = np.ones(scale_shape).astype(tp)
        batch_square_sum *= 10000.0
        y = np.array(x_val)
        mean = np.zeros(x_shape[1]).astype(tp)
        scale = np.ones(x_shape[1]).astype(tp)
        self.inputs = {'X': x_val, 'BatchSize': batch_size, 'BatchSum': batch_sum, 'BatchSquareSum': batch_square_sum}
        self.outputs = {'Y': y, 'Means': mean, 'Scales': scale}
        self.attrs = {'epsilon': epsilon, 'use_mkldnn': self.use_mkldnn, 'slot_dim': slot_dim}

    def test_check_output(self):
        if False:
            print('Hello World!')
        '\n        test check forward, check output\n        '
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test check backward, check grad\n        '
        self.check_grad(['X'], 'Y', no_grad_set=set(), check_dygraph=False)

class TestDataNormOpErrorr(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')
        with program_guard(Program(), Program()):
            x2 = paddle.static.data(name='x2', shape=[-1, 3, 4], dtype='int32')
            paddle.static.nn.data_norm(input=x2, param_attr={}, enable_scale_and_shift=True)
            paddle.enable_static()
            x3 = paddle.static.data('', shape=[0], dtype='float32')
            self.assertRaises(ValueError, paddle.static.nn.data_norm, x3)

            def test_0_size():
                if False:
                    print('Hello World!')
                paddle.enable_static()
                x = paddle.static.data(name='x', shape=[0, 3], dtype='float32')
                out = paddle.static.nn.data_norm(x, slot_dim=1)
                cpu = base.core.CPUPlace()
                exe = base.Executor(cpu)
                exe.run(base.default_startup_program())
                test_program = base.default_main_program().clone(for_test=True)
                exe.run(test_program, fetch_list=out, feed={'x': np.ones([0, 3]).astype('float32')})
            self.assertRaises(ValueError, test_0_size)
if __name__ == '__main__':
    unittest.main()