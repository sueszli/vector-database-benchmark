import os
import tempfile
import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase
import paddle
from paddle import base
from paddle.static import InputSpec

@paddle.jit.to_static
def for_in_range(x):
    if False:
        print('Hello World!')
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x = base.dygraph.to_variable(x)
    for i in range(x.numpy().item()):
        z = z + i
    return z

@paddle.jit.to_static
def for_iter_list(x_array):
    if False:
        print('Hello World!')
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    for x in x_array:
        z = z + x
    return z

@paddle.jit.to_static
def for_enumerate_list(x_array):
    if False:
        for i in range(10):
            print('nop')
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    for (i, x) in enumerate(x_array):
        z = z + x + i
    return z

@paddle.jit.to_static
def for_iter_var_numpy(x_array):
    if False:
        return 10
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for x in x_array.numpy():
        z = z + x
    return z

@paddle.jit.to_static
def for_enumerate_var_numpy(x_array):
    if False:
        print('Hello World!')
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for (i, x) in enumerate(x_array.numpy()):
        y = y + i
        z = z + x
    return (y, z)

@paddle.jit.to_static
def for_enumerate_var_numpy_with_start(x_array):
    if False:
        return 10
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for (i, x) in enumerate(x_array.numpy(), 1):
        y = y + i
        z = z + x
    return (y, z)

@paddle.jit.to_static
def for_in_range_with_break(x):
    if False:
        print('Hello World!')
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x = base.dygraph.to_variable(x)
    for i in range(x.numpy()[0]):
        z = z + i
        if i > 2:
            break
    return z

@paddle.jit.to_static
def for_enumerate_var_numpy_with_break(x_array):
    if False:
        i = 10
        return i + 15
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for (i, x) in enumerate(x_array.numpy()):
        y = y + i
        z = z + x
        if i > 2:
            break
    return (y, z)

@paddle.jit.to_static
def for_enumerate_var_numpy_with_continue(x_array):
    if False:
        return 10
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for (i, x) in enumerate(x_array.numpy()):
        y = y + i
        if i > 2:
            continue
        z = z + x
    return (y, z)

@paddle.jit.to_static
def for_enumerate_var_numpy_with_start_break(x_array):
    if False:
        print('Hello World!')
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for (i, x) in enumerate(x_array.numpy(), 1):
        y = y + i
        z = z + x
        if i > 2:
            break
    return (y, z)

@paddle.jit.to_static
def for_enumerate_var_numpy_with_start_continue(x_array):
    if False:
        while True:
            i = 10
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for (i, x) in enumerate(x_array.numpy(), 1):
        y = y + i
        if i > 2:
            continue
        z = z + x
    return (y, z)

@paddle.jit.to_static
def for_iter_var(x_array):
    if False:
        i = 10
        return i + 15
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for x in x_array:
        z = z + x
    return z

@paddle.jit.to_static
def for_enumerate_var(x_array):
    if False:
        while True:
            i = 10
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for (i, x) in enumerate(x_array):
        y = y + i
        z = z + x
    return (y, z)

@paddle.jit.to_static
def for_iter_var_list(x):
    if False:
        i = 10
        return i + 15
    x = base.dygraph.to_variable(x)
    iter_num = paddle.tensor.fill_constant(shape=[1], value=5, dtype='int32')
    a = []
    for i in range(iter_num):
        a.append(x + i)
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    for x in a:
        y = y + x
    return y

@paddle.jit.to_static
def for_enumerate_var_list(x):
    if False:
        for i in range(10):
            print('nop')
    x = base.dygraph.to_variable(x)
    iter_num = paddle.tensor.fill_constant(shape=[1], value=5, dtype='int32')
    a = []
    for i in range(iter_num):
        a.append(x + i)
    y = paddle.tensor.fill_constant([1], 'int32', 0)
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    for (i, x) in enumerate(a):
        y = y + i
        z = z + x
    return (y, z)

@paddle.jit.to_static
def for_enumerate_var_with_nested_range(x_array):
    if False:
        return 10
    x = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for (i, num) in enumerate(x_array):
        for idx in range(num):
            x = x + num
    return x

@paddle.jit.to_static
def for_iter_var_idx(x_array):
    if False:
        return 10
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = base.dygraph.to_variable(x_array)
    for x in x_array[0:]:
        z = z + x
    return z

@paddle.jit.to_static
def for_tuple_as_iter_var(x_array):
    if False:
        print('Hello World!')
    x = paddle.to_tensor(x_array)
    z = paddle.to_tensor(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
    a_result = paddle.zeros([3])
    b_result = paddle.zeros([3])
    c_result = paddle.zeros([3])
    for (a, b, c) in z:
        a_result += a
        b_result += b
        c_result += c
    return (a_result, b_result, c_result)

@paddle.jit.to_static
def for_tuple_as_enumerate_iter(x_array):
    if False:
        print('Hello World!')
    x = paddle.to_tensor(x_array)
    x_list = [x, x, x]
    a_result = paddle.zeros([5])
    for t in enumerate(x_list):
        a_result += t[1]
    return a_result

@paddle.jit.to_static
def for_tuple_as_enumerate_value(x_array):
    if False:
        return 10
    x = paddle.to_tensor(x_array)
    x_list = [x, x, x]
    a_result = paddle.zeros([1])
    b_result = paddle.zeros([1])
    c_result = paddle.zeros([1])
    d_result = paddle.zeros([1])
    e_result = paddle.zeros([1])
    for (i, (a, b, c, d, e)) in enumerate(x_list):
        a_result += a
        b_result += b
        c_result += c
        d_result += d
        e_result += e
    return a_result

class ForwardContainsForLayer(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.high = 5
        self.low = 3

    @paddle.jit.to_static
    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        y = paddle.zeros([10, 2, 3])
        z = []
        for i in range(self.high - self.low):
            z.append(y[i].clone())
        return z

@paddle.jit.to_static
def for_original_list():
    if False:
        i = 10
        return i + 15
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    for x in [1, 2, 3]:
        z = z + x
    return z

@paddle.jit.to_static
def for_original_tuple():
    if False:
        for i in range(10):
            print('nop')
    z = paddle.tensor.fill_constant([1], 'int32', 0)
    for x in (1, 2, 3):
        z = z + x
    return z

@paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 10]), InputSpec(shape=[None, 10])])
def for_zip_error(x, y):
    if False:
        print('Hello World!')
    for (i, j) in zip(x, y):
        a = i + j
    return x + y

@paddle.jit.to_static(input_spec=[InputSpec(shape=[2, 10]), InputSpec(shape=[2, 10])])
def for_zip(x, y):
    if False:
        for i in range(10):
            print('nop')
    for (i, j) in zip(x, y):
        a = i + j
    return x + y

@paddle.jit.to_static
def tensor_array_slice_in_enumerate():
    if False:
        while True:
            i = 10
    feats = {}
    feats['key'] = []
    feats_idx = paddle.arange(0, 10)
    for (i, idx) in enumerate(feats_idx):
        if i > 1:
            feat_n2 = feats['key'][-2]
        feats['key'].append(idx)
    return feat_n2

class TestTransformBase(Dy2StTestBase):

    def setUp(self):
        if False:
            return 10
        self.place = base.CUDAPlace(0) if base.is_compiled_with_cuda() else base.CPUPlace()
        self.set_input()
        self.set_test_func()

    def set_input(self):
        if False:
            for i in range(10):
                print('nop')
        self.input = [1, 2, 3]

    def set_test_func(self):
        if False:
            return 10
        raise NotImplementedError('For Enumerate test should implement set_test_func')

    def _run(self, to_static):
        if False:
            for i in range(10):
                print('nop')
        paddle.jit.enable_to_static(to_static)
        with base.dygraph.guard():
            return self.dygraph_func(self.input)

    def get_dygraph_output(self):
        if False:
            i = 10
            return i + 15
        return self._run(to_static=False)

    def get_static_output(self):
        if False:
            i = 10
            return i + 15
        return self._run(to_static=True)

class TestTransform(TestTransformBase):

    def transformed_result_compare(self):
        if False:
            while True:
                i = 10
        dy_outs = self.get_dygraph_output()
        if not isinstance(dy_outs, (tuple, list)):
            dy_outs = (dy_outs,)
        self.dygraph_func.eval()
        st_outs = self.get_static_output()
        if not isinstance(st_outs, (tuple, list)):
            st_outs = (st_outs,)
        for (x, y) in zip(dy_outs, st_outs):
            np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=1e-05)

class TestTransformForOriginalList(TestTransform):

    def _run(self, to_static):
        if False:
            i = 10
            return i + 15
        paddle.jit.enable_to_static(to_static)
        with base.dygraph.guard():
            return self.dygraph_func()

class TestTransformError(TestTransformBase):

    def transformed_error(self, etype):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(etype):
            dy_out = self.get_dygraph_output()
            st_out = self.get_static_output()

class TestForInRange(TestTransform):

    def set_input(self):
        if False:
            for i in range(10):
                print('nop')
        self.input = np.array([5])

    def set_test_func(self):
        if False:
            while True:
                i = 10
        self.dygraph_func = for_in_range

    def test_transformed_result_compare(self):
        if False:
            for i in range(10):
                print('nop')
        self.transformed_result_compare()

class TestForIterList(TestTransform):

    def set_test_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.dygraph_func = for_iter_list

    def test_transformed_result_compare(self):
        if False:
            return 10
        self.transformed_result_compare()

class TestForEnumerateSimple(TestForIterList):

    def set_test_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = for_enumerate_list

class TestForInRangeWithBreak(TestForInRange):

    def set_test_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = for_in_range_with_break

class TestForIterVarNumpy(TestTransform):

    def set_input(self):
        if False:
            return 10
        self.input = np.array([1, 2, 3, 4, 5])

    def set_test_func(self):
        if False:
            while True:
                i = 10
        self.dygraph_func = for_iter_var_numpy

    def test_transformed_result_compare(self):
        if False:
            while True:
                i = 10
        self.transformed_result_compare()

class TestForEnumerateVarNumpy(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = for_enumerate_var_numpy

class TestForEnumerateVarNumpyWithStart(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.dygraph_func = for_enumerate_var_numpy_with_start

class TestForEnumerateVarNumpyWithBreak(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            i = 10
            return i + 15
        self.dygraph_func = for_enumerate_var_numpy_with_break

class TestForEnumerateVarNumpyWithContinue(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = for_enumerate_var_numpy_with_continue

class TestForEnumerateVarNumpyWithStartAndBreak(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            while True:
                i = 10
        self.dygraph_func = for_enumerate_var_numpy_with_start_break

class TestForEnumerateVarNumpyWithStartAndContinue(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = for_enumerate_var_numpy_with_start_continue

class TestForIterVar(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            while True:
                i = 10
        self.dygraph_func = for_iter_var

class TestForIterVarIdx(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = for_iter_var_idx

class TestForEnumerateVar(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            return 10
        self.dygraph_func = for_enumerate_var

class TestForEnumerateVarWithNestedRange(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = for_enumerate_var_with_nested_range

class TestForIterVarList(TestForInRange):

    def set_test_func(self):
        if False:
            return 10
        self.dygraph_func = for_iter_var_list

class TestForEnumerateVarList(TestForInRange):

    def set_test_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.dygraph_func = for_enumerate_var_list

class TestForTupleAsIterVar(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            return 10
        self.dygraph_func = for_tuple_as_iter_var

class TestForTupleAsEnumerateIter(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            i = 10
            return i + 15
        self.dygraph_func = for_tuple_as_enumerate_iter

class TestForTupleAsEnumerateValue(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            i = 10
            return i + 15
        self.dygraph_func = for_tuple_as_enumerate_value

class TestForwardContainsForLayer(TestForIterVarNumpy):

    def set_test_func(self):
        if False:
            i = 10
            return i + 15
        self.dygraph_func = ForwardContainsForLayer()

class TestForOriginalList(TestTransformForOriginalList):

    def set_test_func(self):
        if False:
            i = 10
            return i + 15
        self.dygraph_func = for_original_list

    def test_transformed_result_compare(self):
        if False:
            while True:
                i = 10
        self.transformed_result_compare()

class TestForOriginalTuple(TestTransformForOriginalList):

    def set_test_func(self):
        if False:
            while True:
                i = 10
        self.dygraph_func = for_original_tuple

    def test_transformed_result_compare(self):
        if False:
            while True:
                i = 10
        self.transformed_result_compare()

class TestSliceTensorArrayInEnumerate(TestTransformForOriginalList):

    def set_test_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = tensor_array_slice_in_enumerate

    def test_transformed_result_compare(self):
        if False:
            while True:
                i = 10
        self.transformed_result_compare()

class TestForZip(Dy2StTestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def test_for_zip_error(self):
        if False:
            return 10
        with self.assertRaises(RuntimeError):
            model_path = os.path.join(self.temp_dir.name, 'for_zip_error')
            paddle.jit.save(for_zip_error, model_path)

    def test_for_zip(self):
        if False:
            while True:
                i = 10
        model_path = os.path.join(self.temp_dir.name, 'for_zip')
        paddle.jit.save(for_zip, model_path)
if __name__ == '__main__':
    unittest.main()