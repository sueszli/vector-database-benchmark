import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestFlattenOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'flatten_contiguous_range'
        self.use_dynamic_create_class = False

    class TestFlattenOp(XPUOpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            self.set_xpu()
            self.op_type = 'flatten_contiguous_range'
            self.place = paddle.XPUPlace(0)
            self.use_xpu = True
            self.use_mkldnn = False
            self.start_axis = 0
            self.stop_axis = -1
            self.dtype = self.in_type
            self.init_test_case()
            self.inputs = {'X': np.random.random(self.in_shape).astype(self.dtype)}
            self.init_attrs()
            self.outputs = {'Out': self.inputs['X'].reshape(self.new_shape), 'XShape': np.random.random(self.in_shape).astype(self.dtype)}

        def set_xpu(self):
            if False:
                while True:
                    i = 10
            self.__class__.use_xpu = True

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(self.place, no_check_set=['XShape'])

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = -1
            self.new_shape = 120

        def init_attrs(self):
            if False:
                print('Hello World!')
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis, 'use_xpu': True}

    class TestFlattenOp_1(TestFlattenOp):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 1
            self.stop_axis = 2
            self.new_shape = (3, 10, 4)

        def init_attrs(self):
            if False:
                print('Hello World!')
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

    class TestFlattenOp_2(TestFlattenOp):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)

        def init_attrs(self):
            if False:
                print('Hello World!')
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

    class TestFlattenOp_3(TestFlattenOp):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 2
            self.new_shape = (30, 4)

        def init_attrs(self):
            if False:
                while True:
                    i = 10
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

    class TestFlattenOp_4(TestFlattenOp):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = -2
            self.stop_axis = -1
            self.new_shape = (3, 2, 20)

        def init_attrs(self):
            if False:
                while True:
                    i = 10
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

    class TestFlattenOp_5(TestFlattenOp):

        def init_test_case(self):
            if False:
                return 10
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 2
            self.stop_axis = 2
            self.new_shape = (3, 2, 5, 4)

        def init_attrs(self):
            if False:
                i = 10
                return i + 15
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

    class TestFlattenOpSixDims(TestFlattenOp):

        def init_test_case(self):
            if False:
                print('Hello World!')
            self.in_shape = (3, 2, 3, 2, 4, 4)
            self.start_axis = 3
            self.stop_axis = 5
            self.new_shape = (3, 2, 3, 32)

        def init_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

    class TestFlattenOp_Float32(TestFlattenOp):

        def init_test_case(self):
            if False:
                return 10
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)
            self.dtype = np.float32

        def init_attrs(self):
            if False:
                while True:
                    i = 10
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

    class TestFlattenOp_int32(TestFlattenOp):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)
            self.dtype = np.int32

        def init_attrs(self):
            if False:
                return 10
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis, 'use_xpu': True}

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            pass

    class TestFlattenOp_int8(TestFlattenOp):

        def init_test_case(self):
            if False:
                for i in range(10):
                    print('nop')
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)
            self.dtype = np.int8

        def init_attrs(self):
            if False:
                for i in range(10):
                    print('nop')
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            pass

    class TestFlattenOp_int64(TestFlattenOp):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.in_shape = (3, 2, 5, 4)
            self.start_axis = 0
            self.stop_axis = 1
            self.new_shape = (6, 5, 4)
            self.dtype = np.int64

        def init_attrs(self):
            if False:
                return 10
            self.attrs = {'start_axis': self.start_axis, 'stop_axis': self.stop_axis}

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            pass

class TestFlatten2OpError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        image_shape = (2, 3, 4, 4)
        x = np.arange(image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]).reshape(image_shape) / 100.0
        x = x.astype('float32')

        def test_ValueError1():
            if False:
                while True:
                    i = 10
            x_var = paddle.static.data(name='x', shape=image_shape, dtype='float32')
            out = paddle.flatten(x_var, start_axis=2, stop_axis=1)
        self.assertRaises(ValueError, test_ValueError1)

        def test_ValueError2():
            if False:
                i = 10
                return i + 15
            x_var = paddle.static.data(name='x', shape=image_shape, dtype='float32')
            paddle.flatten(x_var, start_axis=10, stop_axis=1)
        self.assertRaises(ValueError, test_ValueError2)

        def test_ValueError3():
            if False:
                while True:
                    i = 10
            x_var = paddle.static.data(name='x', shape=image_shape, dtype='float32')
            paddle.flatten(x_var, start_axis=2, stop_axis=10)
        self.assertRaises(ValueError, test_ValueError3)

        def test_InputError():
            if False:
                i = 10
                return i + 15
            out = paddle.flatten(x)
        self.assertRaises(ValueError, test_InputError)

class TestStaticFlattenPythonAPI(unittest.TestCase):

    def execute_api(self, x, start_axis=0, stop_axis=-1):
        if False:
            for i in range(10):
                print('nop')
        return paddle.flatten(x, start_axis, stop_axis)

    def test_static_api(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        np_x = np.random.rand(2, 3, 4, 4).astype('float32')
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4, 4], dtype='float32')
            out = self.execute_api(x, start_axis=-2, stop_axis=-1)
        exe = paddle.static.Executor(place=paddle.XPUPlace(0))
        fetch_out = exe.run(main_prog, feed={'x': np_x}, fetch_list=[out])
        self.assertTrue((2, 3, 16) == fetch_out[0].shape)

class TestStaticInplaceFlattenPythonAPI(TestStaticFlattenPythonAPI):

    def execute_api(self, x, start_axis=0, stop_axis=-1):
        if False:
            for i in range(10):
                print('nop')
        return x.flatten_(start_axis, stop_axis)

class TestFlattenPython(unittest.TestCase):

    def test_python_api(self):
        if False:
            print('Hello World!')
        image_shape = (2, 3, 4, 4)
        x = np.arange(image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]).reshape(image_shape) / 100.0
        x = x.astype('float32')

        def test_InputError():
            if False:
                for i in range(10):
                    print('nop')
            out = paddle.flatten(x)
        self.assertRaises(ValueError, test_InputError)

        def test_Negative():
            if False:
                while True:
                    i = 10
            paddle.disable_static(paddle.XPUPlace(0))
            img = paddle.to_tensor(x)
            out = paddle.flatten(img, start_axis=-2, stop_axis=-1)
            return out.numpy().shape
        res_shape = test_Negative()
        self.assertTrue((2, 3, 16) == res_shape)
support_types = get_xpu_op_support_types('flatten_contiguous_range')
for stype in support_types:
    create_test_class(globals(), XPUTestFlattenOp, stype)
if __name__ == '__main__':
    unittest.main()