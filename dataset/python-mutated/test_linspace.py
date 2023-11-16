import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard
import paddle
from paddle import base
from paddle.base import Program, core, program_guard

class TestLinspaceOpCommonCase(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'linspace'
        self.python_api = paddle.linspace
        self._set_dtype()
        self._set_data()
        self.attrs = {'dtype': self.attr_dtype}

    def _set_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = 'float32'
        self.attr_dtype = int(core.VarDesc.VarType.FP32)

    def _set_data(self):
        if False:
            i = 10
            return i + 15
        self.outputs = {'Out': np.arange(0, 11).astype(self.dtype)}
        self.inputs = {'Start': np.array([0]).astype(self.dtype), 'Stop': np.array([10]).astype(self.dtype), 'Num': np.array([11]).astype('int32')}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_pir=True)

class TestLinspaceOpReverseCase(TestLinspaceOpCommonCase):

    def _set_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'Start': np.array([10]).astype(self.dtype), 'Stop': np.array([0]).astype(self.dtype), 'Num': np.array([11]).astype('int32')}
        self.outputs = {'Out': np.arange(10, -1, -1).astype(self.dtype)}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

class TestLinspaceOpNumOneCase(TestLinspaceOpCommonCase):

    def _set_data(self):
        if False:
            i = 10
            return i + 15
        self.inputs = {'Start': np.array([10]).astype(self.dtype), 'Stop': np.array([0]).astype(self.dtype), 'Num': np.array([1]).astype('int32')}
        self.outputs = {'Out': np.array([10], dtype=self.dtype)}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

class TestLinspaceOpCommonCaseFP16(TestLinspaceOpCommonCase):

    def _set_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)

class TestLinspaceOpReverseCaseFP16(TestLinspaceOpReverseCase):

    def _set_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)

class TestLinspaceOpNumOneCaseFP16(TestLinspaceOpNumOneCase):

    def _set_dtype(self):
        if False:
            return 10
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'not supported bf16')
class TestLinspaceOpCommonCaseBF16(TestLinspaceOpCommonCaseFP16):

    def _set_dtype(self):
        if False:
            return 10
        self.dtype = np.uint16
        self.attr_dtype = int(core.VarDesc.VarType.BF16)

    def _set_data(self):
        if False:
            i = 10
            return i + 15
        self.outputs = {'Out': convert_float_to_uint16(np.arange(0, 11).astype('float32'))}
        self.inputs = {'Start': convert_float_to_uint16(np.array([0]).astype('float32')), 'Stop': convert_float_to_uint16(np.array([10]).astype('float32')), 'Num': np.array([11]).astype('int32')}

    def test_check_output(self):
        if False:
            return 10
        return self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

class TestLinspaceOpReverseCaseBF16(TestLinspaceOpCommonCaseBF16):

    def _set_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'Start': convert_float_to_uint16(np.array([10]).astype('float32')), 'Stop': convert_float_to_uint16(np.array([0]).astype('float32')), 'Num': np.array([11]).astype('int32')}
        self.outputs = {'Out': convert_float_to_uint16(np.arange(10, -1, -1).astype('float32'))}

class TestLinspaceOpNumOneCaseBF16(TestLinspaceOpCommonCaseBF16):

    def _set_data(self):
        if False:
            while True:
                i = 10
        self.inputs = {'Start': convert_float_to_uint16(np.array([10]).astype('float32')), 'Stop': convert_float_to_uint16(np.array([0]).astype('float32')), 'Num': np.array([1]).astype('int32')}
        self.outputs = {'Out': convert_float_to_uint16(np.array([10], dtype='float32'))}

class TestLinspaceAPI(unittest.TestCase):

    def test_variable_input1(self):
        if False:
            i = 10
            return i + 15
        with paddle_static_guard():
            start = paddle.full(shape=[1], fill_value=0, dtype='float32')
            stop = paddle.full(shape=[1], fill_value=10, dtype='float32')
            num = paddle.full(shape=[1], fill_value=5, dtype='int32')
            out = paddle.linspace(start, stop, num, dtype='float32')
            exe = base.Executor(place=base.CPUPlace())
            res = exe.run(base.default_main_program(), fetch_list=[out])
            np_res = np.linspace(0, 10, 5, dtype='float32')
            self.assertEqual((res == np_res).all(), True)

    def test_variable_input2(self):
        if False:
            i = 10
            return i + 15
        start = paddle.full(shape=[1], fill_value=0, dtype='float32')
        stop = paddle.full(shape=[1], fill_value=10, dtype='float32')
        num = paddle.full(shape=[1], fill_value=5, dtype='int32')
        out = paddle.linspace(start, stop, num, dtype='float32')
        np_res = np.linspace(0, 10, 5, dtype='float32')
        self.assertEqual((out.numpy() == np_res).all(), True)

    def test_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle_static_guard():
            out_1 = paddle.linspace(0, 10, 5, dtype='float32')
            out_2 = paddle.linspace(0, 10, 5, dtype=np.float32)
            out_3 = paddle.linspace(0, 10, 5, dtype=core.VarDesc.VarType.FP32)
            exe = base.Executor(place=base.CPUPlace())
            (res_1, res_2, res_3) = exe.run(base.default_main_program(), fetch_list=[out_1, out_2, out_3])
            np.testing.assert_array_equal(res_1, res_2)

    def test_name(self):
        if False:
            return 10
        with paddle_static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                out = paddle.linspace(0, 10, 5, dtype='float32', name='linspace_res')
                assert 'linspace_res' in out.name

    def test_imperative(self):
        if False:
            return 10
        out1 = paddle.linspace(0, 10, 5, dtype='float32')
        np_out1 = np.linspace(0, 10, 5, dtype='float32')
        out2 = paddle.linspace(0, 10, 5, dtype='int32')
        np_out2 = np.linspace(0, 10, 5, dtype='int32')
        out3 = paddle.linspace(0, 10, 200, dtype='int32')
        np_out3 = np.linspace(0, 10, 200, dtype='int32')
        self.assertEqual((out1.numpy() == np_out1).all(), True)
        self.assertEqual((out2.numpy() == np_out2).all(), True)
        self.assertEqual((out3.numpy() == np_out3).all(), True)

class TestLinspaceOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10
        with paddle_static_guard():
            with program_guard(Program(), Program()):

                def test_dtype():
                    if False:
                        print('Hello World!')
                    paddle.linspace(0, 10, 1, dtype='int8')
                self.assertRaises(TypeError, test_dtype)

                def test_dtype1():
                    if False:
                        i = 10
                        return i + 15
                    paddle.linspace(0, 10, 1.33, dtype='int32')
                self.assertRaises(TypeError, test_dtype1)

                def test_start_type():
                    if False:
                        while True:
                            i = 10
                    paddle.linspace([0], 10, 1, dtype='float32')
                self.assertRaises(TypeError, test_start_type)

                def test_end_type():
                    if False:
                        return 10
                    paddle.linspace(0, [10], 1, dtype='float32')
                self.assertRaises(TypeError, test_end_type)

                def test_step_dtype():
                    if False:
                        return 10
                    paddle.linspace(0, 10, [0], dtype='float32')
                self.assertRaises(TypeError, test_step_dtype)

                def test_start_dtype():
                    if False:
                        while True:
                            i = 10
                    start = paddle.static.data(shape=[1], dtype='float64', name='start')
                    paddle.linspace(start, 10, 1, dtype='float32')
                self.assertRaises(ValueError, test_start_dtype)

                def test_end_dtype():
                    if False:
                        for i in range(10):
                            print('nop')
                    end = paddle.static.data(shape=[1], dtype='float64', name='end')
                    paddle.linspace(0, end, 1, dtype='float32')
                self.assertRaises(ValueError, test_end_dtype)

                def test_num_dtype():
                    if False:
                        i = 10
                        return i + 15
                    num = paddle.static.data(shape=[1], dtype='int32', name='step')
                    paddle.linspace(0, 10, num, dtype='float32')
                self.assertRaises(TypeError, test_step_dtype)
if __name__ == '__main__':
    unittest.main()