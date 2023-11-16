import re
import sys
import numpy as np
from op_test import OpTest
from spectral_op_np import fft_c2c, fft_c2r, fft_r2c
import paddle
from paddle import _C_ops
paddle.enable_static()
TEST_CASE_NAME = 'test_case'

def parameterize(attrs, input_values=None):
    if False:
        print('Hello World!')
    if isinstance(attrs, str):
        attrs = [attrs]
    input_dicts = attrs if input_values is None else [dict(zip(attrs, vals)) for vals in input_values]

    def decorator(base_class):
        if False:
            i = 10
            return i + 15
        test_class_module = sys.modules[base_class.__module__].__dict__
        for (idx, input_dict) in enumerate(input_dicts):
            test_class_dict = dict(base_class.__dict__)
            test_class_dict.update(input_dict)
            name = class_name(base_class, idx, input_dict)
            test_class_module[name] = type(name, (base_class,), test_class_dict)
        for method_name in list(base_class.__dict__):
            if method_name.startswith('test'):
                delattr(base_class, method_name)
        return base_class
    return decorator

def to_safe_name(s):
    if False:
        while True:
            i = 10
    return str(re.sub('[^a-zA-Z0-9_]+', '_', s))

def class_name(cls, num, params_dict):
    if False:
        while True:
            i = 10
    suffix = to_safe_name(next((v for v in params_dict.values() if isinstance(v, str)), ''))
    if TEST_CASE_NAME in params_dict:
        suffix = to_safe_name(params_dict['test_case'])
    return '{}_{}{}'.format(cls.__name__, num, suffix and '_' + suffix)

def fft_c2c_python_api(x, axes, norm, forward):
    if False:
        i = 10
        return i + 15
    return _C_ops.fft_c2c(x, axes, norm, forward)

def fft_r2c_python_api(x, axes, norm, forward, onesided):
    if False:
        while True:
            i = 10
    return _C_ops.fft_r2c(x, axes, norm, forward, onesided)

def fft_c2r_python_api(x, axes, norm, forward, last_dim_size=0):
    if False:
        print('Hello World!')
    return _C_ops.fft_c2r(x, axes, norm, forward, last_dim_size)

@parameterize((TEST_CASE_NAME, 'x', 'axes', 'norm', 'forward'), [('test_axes_is_sqe_type', (np.random.random((12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128), [0, 1], 'forward', True), ('test_axis_not_last', (np.random.random((4, 8, 4)) + 1j * np.random.random((4, 8, 4))).astype(np.complex128), (0, 1), 'backward', False), ('test_norm_forward', (np.random.random((12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128), (0,), 'forward', False), ('test_norm_backward', (np.random.random((12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128), (0,), 'backward', True), ('test_norm_ortho', (np.random.random((12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128), (1,), 'ortho', True)])
class TestFFTC2COp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'fft_c2c'
        self.dtype = self.x.dtype
        self.python_api = fft_c2c_python_api
        out = fft_c2c(self.x, self.axes, self.norm, self.forward)
        self.inputs = {'X': self.x}
        self.attrs = {'axes': self.axes, 'normalization': self.norm, 'forward': self.forward}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out')

@parameterize((TEST_CASE_NAME, 'x', 'axes', 'norm', 'forward', 'last_dim_size'), [('test_axes_is_sqe_type', (np.random.random((12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128), [0, 1], 'forward', True, 26), ('test_axis_not_last', (np.random.random((4, 7, 4)) + 1j * np.random.random((4, 7, 4))).astype(np.complex128), (0, 1), 'backward', False, None), ('test_norm_forward', (np.random.random((12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128), (0,), 'forward', False, 22), ('test_norm_backward', (np.random.random((12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128), (0,), 'backward', True, 22), ('test_norm_ortho', (np.random.random((12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128), (1,), 'ortho', True, 26)])
class TestFFTC2ROp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'fft_c2r'
        self.dtype = self.x.dtype
        self.python_api = fft_c2r_python_api
        out = fft_c2r(self.x, self.axes, self.norm, self.forward, self.last_dim_size)
        self.inputs = {'X': self.x}
        self.attrs = {'axes': self.axes, 'normalization': self.norm, 'forward': self.forward, 'last_dim_size': self.last_dim_size}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out')

@parameterize((TEST_CASE_NAME, 'x', 'axes', 'norm', 'forward', 'onesided'), [('test_axes_is_sqe_type', np.random.randn(12, 18).astype(np.float64), (0, 1), 'forward', True, True), ('test_axis_not_last', np.random.randn(4, 8, 4).astype(np.float64), (0, 1), 'backward', False, True), ('test_norm_forward', np.random.randn(12, 18).astype(np.float64), (0, 1), 'forward', False, False), ('test_norm_backward', np.random.randn(12, 18).astype(np.float64), (0,), 'backward', True, False), ('test_norm_ortho', np.random.randn(12, 18).astype(np.float64), (1,), 'ortho', True, False)])
class TestFFTR2COp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'fft_r2c'
        self.dtype = self.x.dtype
        self.python_api = fft_r2c_python_api
        out = fft_r2c(self.x, self.axes, self.norm, self.forward, self.onesided)
        self.inputs = {'X': self.x}
        self.attrs = {'axes': self.axes, 'normalization': self.norm, 'forward': self.forward, 'onesided': self.onesided}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out')