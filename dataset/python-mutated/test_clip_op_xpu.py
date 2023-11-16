import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
from paddle.base import Program, core, program_guard

class XPUTestClipOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'clip'
        self.use_dynamic_create_class = False

    class TestClipOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.init_dtype()
            self.set_xpu()
            self.op_type = 'clip'
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_data()
            self.set_attrs()
            self.set_inputs()
            self.outputs = {'Out': np.clip(self.inputs['X'], self.min_v, self.max_v)}

        def set_xpu(self):
            if False:
                i = 10
                return i + 15
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_data(self):
            if False:
                return 10
            self.shape = (4, 10, 10)
            self.max = 0.8
            self.min = 0.3

        def set_inputs(self):
            if False:
                return 10
            if 'Min' in self.inputs:
                min_v = self.inputs['Min']
            else:
                min_v = self.attrs['min']
            if 'Max' in self.inputs:
                max_v = self.inputs['Max']
            else:
                max_v = self.attrs['max']
            self.min_v = min_v
            self.max_v = max_v
            self.max_relative_error = 0.006
            input = np.random.random(self.shape).astype('float32')
            input[np.abs(input - min_v) < self.max_relative_error] = 0.5
            input[np.abs(input - max_v) < self.max_relative_error] = 0.5
            self.inputs['X'] = input

        def set_attrs(self):
            if False:
                while True:
                    i = 10
            self.attrs = {}
            self.attrs['min'] = self.min
            self.attrs['max'] = self.max

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = self.in_type

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            paddle.enable_static()
            self.check_output_with_place(self.place)
            paddle.disable_static()

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            if hasattr(self, 'no_need_check_grad') and self.no_need_check_grad:
                return
            if core.is_compiled_with_xpu():
                paddle.enable_static()
                self.check_grad_with_place(self.place, ['X'], 'Out', check_dygraph=True)
                paddle.disable_static()

    class TestClipOp1(TestClipOp):

        def init_data(self):
            if False:
                return 10
            self.shape = (8, 16, 8)
            self.max = 0.7
            self.min = 0.0

    class TestClipOp2(TestClipOp):

        def init_data(self):
            if False:
                return 10
            self.shape = (8, 16)
            self.max = 1.0
            self.min = 0.0

    class TestClipOp3(TestClipOp):

        def init_data(self):
            if False:
                return 10
            self.shape = (4, 8, 16)
            self.max = 0.7
            self.min = 0.2

    class TestClipOp4(TestClipOp):

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.shape = (4, 8, 8)
            self.max = 0.7
            self.min = 0.2
            self.inputs['Max'] = np.array([0.8]).astype('float32')
            self.inputs['Min'] = np.array([0.3]).astype('float32')

    class TestClipOp5(TestClipOp):

        def init_data(self):
            if False:
                print('Hello World!')
            self.shape = (4, 8, 16)
            self.max = 0.5
            self.min = 0.5

class TestClipOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        with program_guard(Program(), Program()):
            input_data = np.random.random((2, 4)).astype('float32')

            def test_Variable():
                if False:
                    for i in range(10):
                        print('nop')
                paddle.clip(x=input_data, min=-1.0, max=1.0)
            self.assertRaises(TypeError, test_Variable)
        paddle.disable_static()

class TestClipAPI(unittest.TestCase):

    def _executed_api(self, x, min=None, max=None):
        if False:
            print('Hello World!')
        return paddle.clip(x, min, max)

    def test_clip(self):
        if False:
            return 10
        paddle.enable_static()
        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        images = paddle.static.data(name='image', shape=data_shape, dtype='float32')
        min = paddle.static.data(name='min', shape=[1], dtype='float32')
        max = paddle.static.data(name='max', shape=[1], dtype='float32')
        place = base.XPUPlace(0) if base.core.is_compiled_with_xpu() else base.CPUPlace()
        exe = base.Executor(place)
        out_1 = self._executed_api(images, min=min, max=max)
        out_2 = self._executed_api(images, min=0.2, max=0.9)
        out_3 = self._executed_api(images, min=0.3)
        out_4 = self._executed_api(images, max=0.7)
        out_5 = self._executed_api(images, min=min)
        out_6 = self._executed_api(images, max=max)
        out_7 = self._executed_api(images, max=-1.0)
        out_8 = self._executed_api(images)
        (res1, res2, res3, res4, res5, res6, res7, res8) = exe.run(base.default_main_program(), feed={'image': data, 'min': np.array([0.2]).astype('float32'), 'max': np.array([0.8]).astype('float32')}, fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8])
        np.testing.assert_allclose(res1, data.clip(0.2, 0.8))
        np.testing.assert_allclose(res2, data.clip(0.2, 0.9))
        np.testing.assert_allclose(res3, data.clip(min=0.3))
        np.testing.assert_allclose(res4, data.clip(max=0.7))
        np.testing.assert_allclose(res5, data.clip(min=0.2))
        np.testing.assert_allclose(res6, data.clip(max=0.8))
        np.testing.assert_allclose(res7, data.clip(max=-1))
        np.testing.assert_allclose(res8, data)
        paddle.disable_static()

    def test_clip_dygraph(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        place = base.XPUPlace(0) if base.core.is_compiled_with_xpu() else base.CPUPlace()
        paddle.disable_static(place)
        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        images = paddle.to_tensor(data, dtype='float32')
        v_min = paddle.to_tensor(np.array([0.2], dtype=np.float32))
        v_max = paddle.to_tensor(np.array([0.8], dtype=np.float32))
        out_1 = self._executed_api(images, min=0.2, max=0.8)
        images = paddle.to_tensor(data, dtype='float32')
        out_2 = self._executed_api(images, min=0.2, max=0.9)
        images = paddle.to_tensor(data, dtype='float32')
        out_3 = self._executed_api(images, min=v_min, max=v_max)
        np.testing.assert_allclose(out_1.numpy(), data.clip(0.2, 0.8))
        np.testing.assert_allclose(out_2.numpy(), data.clip(0.2, 0.9))
        np.testing.assert_allclose(out_3.numpy(), data.clip(0.2, 0.8))

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        x1 = paddle.static.data(name='x1', shape=[1], dtype='int16')
        x2 = paddle.static.data(name='x2', shape=[1], dtype='int8')
        self.assertRaises(TypeError, paddle.clip, x=x1, min=0.2, max=0.8)
        self.assertRaises(TypeError, paddle.clip, x=x2, min=0.2, max=0.8)
        paddle.disable_static()

class TestInplaceClipAPI(TestClipAPI):

    def _executed_api(self, x, min=None, max=None):
        if False:
            return 10
        return x.clip_(min, max)
support_types = get_xpu_op_support_types('clip')
for stype in support_types:
    create_test_class(globals(), XPUTestClipOp, stype)
if __name__ == '__main__':
    unittest.main()