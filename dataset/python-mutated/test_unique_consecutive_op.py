import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

def reference_unique_consecutive(X, return_inverse=False, return_counts=False, axis=None):
    if False:
        i = 10
        return i + 15
    "\n    Reference unique_consecutive implementation using python.\n    Args:\n        x(Tensor): the input tensor, it's data type should be float32, float64, int32, int64.\n        return_inverse(bool, optional): If True, also return the indices for where elements in\n            the original input ended up in the returned unique consecutive tensor. Default is False.\n        return_counts(bool, optional): If True, also return the counts for each unique consecutive element.\n    "
    X = list(X)
    is_empty = len(X) == 0
    counts_vec = [1] * len(X)
    i = 0
    counts = 1
    last = 0
    inverse_vec = [0] * len(X)
    if not is_empty:
        inverse_vec[last] = i
    cnt = 0
    while i < len(X) - 1:
        if X[i] == X[i + 1]:
            if return_counts:
                counts_vec[cnt] += 1
            del X[i]
        else:
            i += 1
            cnt += 1
        if return_inverse:
            last += 1
            inverse_vec[last] = i
    if return_counts:
        counts_vec = counts_vec[:len(X)]
    if return_inverse and return_counts:
        return (X, np.array(inverse_vec), np.array(counts_vec))
    elif return_counts:
        return (X, np.array(counts_vec))
    elif return_inverse:
        return (X, np.array(inverse_vec))
    else:
        return X

class TestUniqueConsecutiveOp(OpTest):
    """case 1"""

    def config(self):
        if False:
            i = 10
            return i + 15
        self.x_size = 100
        self.x_range = 20
        self.return_inverse = False
        self.return_counts = False
        self.python_api = paddle.unique_consecutive

    def init_kernel_type(self):
        if False:
            while True:
                i = 10
        self.dtype = 'float32' if core.is_compiled_with_rocm() else 'float64'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_kernel_type()
        self.config()
        self.op_type = 'unique_consecutive'
        x = np.random.randint(self.x_range, size=self.x_size).astype(self.dtype)
        result = reference_unique_consecutive(x, self.return_inverse, self.return_counts)
        out = reference_unique_consecutive(x)
        out = np.array(out).astype(self.dtype)
        self.inputs = {'X': x}
        self.python_out_sig = ['Out']
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output()

class TestUniqueConsecutiveOp2(TestUniqueConsecutiveOp):
    """case 2"""

    def config(self):
        if False:
            print('Hello World!')
        self.x_size = 100
        self.x_range = 20
        self.return_inverse = True
        self.return_counts = False
        self.python_api = paddle.unique_consecutive

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_kernel_type()
        self.config()
        self.op_type = 'unique_consecutive'
        x = np.random.randint(self.x_range, size=self.x_size).astype(self.dtype)
        (result, inverse) = reference_unique_consecutive(x, self.return_inverse, self.return_counts)
        result = np.array(result).astype(self.dtype)
        inverse = inverse.astype(self.dtype)
        self.inputs = {'X': x}
        self.attrs = {'return_inverse': self.return_inverse, 'dtype': int(core.VarDesc.VarType.INT32)}
        self.python_out_sig = ['Out']
        self.outputs = {'Out': result, 'Index': inverse}

class TestUniqueConsecutiveOp3(TestUniqueConsecutiveOp):
    """case 3"""

    def config(self):
        if False:
            return 10
        self.x_size = 100
        self.x_range = 20
        self.return_inverse = False
        self.return_counts = True
        self.python_api = paddle.unique_consecutive

    def setUp(self):
        if False:
            while True:
                i = 10
        self.init_kernel_type()
        self.config()
        self.op_type = 'unique_consecutive'
        x = np.random.randint(self.x_range, size=self.x_size).astype(self.dtype)
        (result, counts) = reference_unique_consecutive(x, self.return_inverse, self.return_counts)
        result = np.array(result).astype(self.dtype)
        counts = counts.astype(self.dtype)
        self.inputs = {'X': x}
        self.attrs = {'return_counts': self.return_counts, 'dtype': int(core.VarDesc.VarType.INT32)}
        self.python_out_sig = ['Out']
        self.outputs = {'Out': result, 'Counts': counts}

class TestUniqueConsecutiveOp4(TestUniqueConsecutiveOp):
    """case 4"""

    def config(self):
        if False:
            while True:
                i = 10
        self.x_size = 100
        self.x_range = 20
        self.return_inverse = True
        self.return_counts = True
        self.python_api = paddle.unique_consecutive

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_kernel_type()
        self.config()
        self.op_type = 'unique_consecutive'
        x = np.random.randint(self.x_range, size=self.x_size).astype(self.dtype)
        (result, inverse, counts) = reference_unique_consecutive(x, self.return_inverse, self.return_counts)
        result = np.array(result).astype(self.dtype)
        inverse = inverse.astype(self.dtype)
        counts = counts.astype(self.dtype)
        self.inputs = {'X': x}
        self.attrs = {'return_inverse': self.return_inverse, 'return_counts': self.return_counts, 'dtype': int(core.VarDesc.VarType.INT32)}
        self.python_out_sig = ['Out']
        self.outputs = {'Out': result, 'Index': inverse, 'Counts': counts}

class TestUniqueConsecutiveAPI(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    @test_with_pir_api
    def check_static_result(self, place):
        if False:
            print('Hello World!')
        with base.program_guard(base.Program(), base.Program()):
            paddle.enable_static()
            input_x = paddle.static.data(name='input_x', shape=[100], dtype='float32')
            result = paddle.unique_consecutive(input_x)
            x_np = np.random.randint(20, size=100).astype('float32')
            exe = base.Executor(place)
            fetches = exe.run(feed={'input_x': x_np}, fetch_list=[result])

    def test_static(self):
        if False:
            while True:
                i = 10
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        for place in self.places:
            with base.dygraph.guard(place):
                input_x = np.random.randint(20, size=100).astype('float64')
                x = paddle.to_tensor(input_x)
                result = paddle.unique_consecutive(x)

class TestUniqueConsecutiveCase2API(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    @test_with_pir_api
    def check_static_result(self, place):
        if False:
            print('Hello World!')
        with base.program_guard(base.Program(), base.Program()):
            paddle.enable_static()
            input_x = paddle.static.data(name='input_x', shape=[100], dtype='float32')
            (result, inverse, counts) = paddle.unique_consecutive(input_x, return_inverse=True, return_counts=True)
            x_np = np.random.randint(20, size=100).astype('float32')
            exe = base.Executor(place)
            fetches = exe.run(feed={'input_x': x_np}, fetch_list=[result])

    def test_static(self):
        if False:
            while True:
                i = 10
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            i = 10
            return i + 15
        for place in self.places:
            with base.dygraph.guard(place):
                input_x = np.random.randint(20, size=100).astype('float64')
                x = paddle.to_tensor(input_x)
                (result, inverse, counts) = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)

class TestUniqueConsecutiveCase3API(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    @test_with_pir_api
    def check_static_result(self, place):
        if False:
            print('Hello World!')
        with base.program_guard(base.Program(), base.Program()):
            paddle.enable_static()
            input_x = paddle.static.data(name='input_x', shape=[100], dtype='float32')
            (result, inverse, counts) = paddle.unique_consecutive(input_x, return_inverse=True, return_counts=True, axis=-1)
            x_np = np.random.randint(20, size=100).astype('float32')
            exe = base.Executor(place)
            fetches = exe.run(feed={'input_x': x_np}, fetch_list=[result])

    def test_static(self):
        if False:
            print('Hello World!')
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        for place in self.places:
            with base.dygraph.guard(place):
                input_x = np.random.randint(20, size=100).astype('float64')
                x = paddle.to_tensor(input_x)
                (result, inverse, counts) = paddle.unique_consecutive(x, return_inverse=True, return_counts=True, axis=-1)

class TestUniqueConsecutiveEmptyInput(OpTest):
    """empty input"""

    def config(self):
        if False:
            i = 10
            return i + 15
        self.return_inverse = True
        self.return_counts = True
        self.python_api = paddle.unique_consecutive

    def init_kernel_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = 'float32' if core.is_compiled_with_rocm() else 'float64'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.init_kernel_type()
        self.config()
        self.op_type = 'unique_consecutive'
        x = np.array([]).astype(self.dtype)
        result = reference_unique_consecutive(x, self.return_inverse, self.return_counts)
        out = reference_unique_consecutive(x)
        out = np.array(out).astype(self.dtype)
        self.inputs = {'X': x}
        self.python_out_sig = ['Out']
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_pir=True)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()