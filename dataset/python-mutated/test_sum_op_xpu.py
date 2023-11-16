import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class XPUTestSumOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'sum'
        self.use_dynamic_create_class = False

    class TestSumOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'sum'
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            x0 = np.random.random(self.shape).astype(self.dtype)
            x1 = np.random.random(self.shape).astype(self.dtype)
            x2 = np.random.random(self.shape).astype(self.dtype)
            self.inputs = {'X': [('x0', x0), ('x1', x1), ('x2', x2)]}
            y = x0 + x1 + x2
            self.outputs = {'Out': y}

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type

        def set_xpu(self):
            if False:
                while True:
                    i = 10
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.dtype

        def set_shape(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shape = (3, 10)

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_grad_with_place(self.place, ['x0'], 'Out')

    class TestSumOp1(TestSumOp):

        def set_shape(self):
            if False:
                print('Hello World!')
            self.shape = 5

    class TestSumOp2(TestSumOp):

        def set_shape(self):
            if False:
                return 10
            self.shape = (1, 1, 1, 1, 1)

    class TestSumOp3(TestSumOp):

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = (10, 5, 7)

    class TestSumOp4(TestSumOp):

        def set_shape(self):
            if False:
                i = 10
                return i + 15
            self.shape = (2, 2, 3, 3)

def create_test_sum_fp16_class(parent):
    if False:
        while True:
            i = 10

    class TestSumFp16Case(parent):

        def init_kernel_type(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = np.float16

        def test_w_is_selected_rows(self):
            if False:
                while True:
                    i = 10
            place = core.XPUPlace(0)
            for inplace in [True, False]:
                self.check_with_place(place, inplace)
    cls_name = '{}_{}'.format(parent.__name__, 'SumFp16Test')
    TestSumFp16Case.__name__ = cls_name
    globals()[cls_name] = TestSumFp16Case

class API_Test_Add_n(unittest.TestCase):

    def test_api(self):
        if False:
            print('Hello World!')
        with base.program_guard(base.Program(), base.Program()):
            input0 = paddle.tensor.fill_constant(shape=[2, 3], dtype='int64', value=5)
            input1 = paddle.tensor.fill_constant(shape=[2, 3], dtype='int64', value=3)
            expected_result = np.empty((2, 3))
            expected_result.fill(8)
            sum_value = paddle.add_n([input0, input1])
            exe = base.Executor(base.XPUPlace(0))
            result = exe.run(fetch_list=[sum_value])
            self.assertEqual((result == expected_result).all(), True)
        with base.dygraph.guard():
            input0 = paddle.ones(shape=[2, 3], dtype='float32')
            expected_result = np.empty((2, 3))
            expected_result.fill(2)
            sum_value = paddle.add_n([input0, input0])
            self.assertEqual((sum_value.numpy() == expected_result).all(), True)

class TestRaiseSumError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15

        def test_type():
            if False:
                while True:
                    i = 10
            paddle.add_n([11, 22])
        self.assertRaises(TypeError, test_type)

        def test_dtype():
            if False:
                return 10
            data1 = paddle.static.data(name='input1', shape=[10], dtype='int8')
            data2 = paddle.static.data(name='input2', shape=[10], dtype='int8')
            paddle.add_n([data1, data2])
        self.assertRaises(TypeError, test_dtype)

        def test_dtype1():
            if False:
                i = 10
                return i + 15
            data1 = paddle.static.data(name='input1', shape=[10], dtype='int8')
            paddle.add_n(data1)
        self.assertRaises(TypeError, test_dtype1)

class TestRaiseSumsError(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')

        def test_type():
            if False:
                i = 10
                return i + 15
            paddle.add_n([11, 22])
        self.assertRaises(TypeError, test_type)

        def test_dtype():
            if False:
                for i in range(10):
                    print('nop')
            data1 = paddle.static.data(name='input1', shape=[10], dtype='int8')
            data2 = paddle.static.data(name='input2', shape=[10], dtype='int8')
            paddle.add_n([data1, data2])
        self.assertRaises(TypeError, test_dtype)

        def test_dtype1():
            if False:
                return 10
            data1 = paddle.static.data(name='input1', shape=[10], dtype='int8')
            paddle.add_n(data1)
        self.assertRaises(TypeError, test_dtype1)

        def test_out_type():
            if False:
                for i in range(10):
                    print('nop')
            data1 = paddle.static.data(name='input1', shape=[10], dtype='flaot32')
            data2 = paddle.static.data(name='input2', shape=[10], dtype='float32')
            out = [10]
            out = paddle.add_n([data1, data2])
        self.assertRaises(TypeError, test_out_type)

        def test_out_dtype():
            if False:
                while True:
                    i = 10
            data1 = paddle.static.data(name='input1', shape=[10], dtype='flaot32')
            data2 = paddle.static.data(name='input2', shape=[10], dtype='float32')
            out = paddle.static.data(name='out', shape=[10], dtype='int8')
            out = paddle.add_n([data1, data2])
        self.assertRaises(TypeError, test_out_dtype)

class TestSumOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')

        def test_empty_list_input():
            if False:
                return 10
            with base.dygraph.guard():
                base._legacy_C_ops.sum([])

        def test_list_of_none_input():
            if False:
                while True:
                    i = 10
            with base.dygraph.guard():
                base._legacy_C_ops.sum([None])
        self.assertRaises(Exception, test_empty_list_input)
        self.assertRaises(Exception, test_list_of_none_input)

class TestLoDTensorAndSelectedRowsOp(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.height = 10
        self.row_numel = 12
        self.rows = [0, 1, 2, 3, 4, 5, 6]
        self.dtype = np.float32
        self.init_kernel_type()

    def check_with_place(self, place, inplace):
        if False:
            while True:
                i = 10
        self.check_input_and_optput(place, inplace, True, True, True)

    def init_kernel_type(self):
        if False:
            print('Hello World!')
        pass

    def _get_array(self, rows, row_numel):
        if False:
            for i in range(10):
                print('nop')
        array = np.ones((len(rows), row_numel)).astype(self.dtype)
        for i in range(len(rows)):
            array[i] *= rows[i]
        return array

    def check_input_and_optput(self, place, inplace, w1_has_data=False, w2_has_data=False, w3_has_data=False):
        if False:
            print('Hello World!')
        paddle.disable_static()
        w1 = self.create_lod_tensor(place)
        w2 = self.create_selected_rows(place, w2_has_data)
        x = [w1, w2]
        out = paddle.add_n(x)
        result = np.ones((1, self.height)).astype(np.int32).tolist()[0]
        for ele in self.rows:
            result[ele] += 1
        out_t = np.array(out)
        self.assertEqual(out_t.shape[0], self.height)
        np.testing.assert_array_equal(out_t, self._get_array(list(range(self.height)), self.row_numel) * np.tile(np.array(result).reshape(self.height, 1), self.row_numel))
        paddle.enable_static()

    def create_selected_rows(self, place, has_data):
        if False:
            return 10
        if has_data:
            rows = self.rows
        else:
            rows = []
        w_array = self._get_array(self.rows, self.row_numel)
        var = core.eager.Tensor(core.VarDesc.VarType.FP32, w_array.shape, 'selected_rows', core.VarDesc.VarType.SELECTED_ROWS, True)
        w_selected_rows = var.value().get_selected_rows()
        w_selected_rows.set_height(self.height)
        w_selected_rows.set_rows(rows)
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)
        return var

    def create_lod_tensor(self, place):
        if False:
            print('Hello World!')
        w_array = self._get_array(list(range(self.height)), self.row_numel)
        return paddle.to_tensor(w_array)

    def test_w_is_selected_rows(self):
        if False:
            for i in range(10):
                print('nop')
        places = [core.XPUPlace(0)]
        for place in places:
            self.check_with_place(place, True)
support_types = get_xpu_op_support_types('sum')
for stype in support_types:
    create_test_class(globals(), XPUTestSumOp, stype)
if __name__ == '__main__':
    unittest.main()