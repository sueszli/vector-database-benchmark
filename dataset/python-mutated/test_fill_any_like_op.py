import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.framework.dtype as dtypes
from paddle.base import core
from paddle.framework import in_pir_mode

def fill_any_like_wrapper(x, value, out_dtype=None, name=None):
    if False:
        i = 10
        return i + 15
    if isinstance(out_dtype, int):
        if not in_pir_mode():
            tmp_dtype = dtypes.dtype(out_dtype)
        else:
            from paddle.base.libpaddle import DataType
            tmp_dtype = DataType(paddle.pir.core.vartype_to_datatype[out_dtype])
    else:
        tmp_dtype = out_dtype
        if in_pir_mode() and isinstance(out_dtype, paddle.framework.core.VarDesc.VarType):
            tmp_dtype = paddle.pir.core.vartype_to_datatype[tmp_dtype]
    return paddle.full_like(x, value, tmp_dtype, name)

class TestFillAnyLikeOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'fill_any_like'
        self.prim_op_type = 'comp'
        self.python_api = fill_any_like_wrapper
        self.public_python_api = fill_any_like_wrapper
        self.dtype = np.int32
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.attrs = {'value': self.value}
        self.outputs = {'Out': self.value * np.ones_like(self.inputs['X'])}
        self.if_enable_cinn()

    def init(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_prim=True, check_pir=True)

    def if_enable_cinn(self):
        if False:
            return 10
        pass

class TestFillAnyLikeOpFloat32(TestFillAnyLikeOp):

    def init(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32
        self.value = 0.0

    def if_enable_cinn(self):
        if False:
            for i in range(10):
                print('nop')
        pass

@unittest.skipIf(not core.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(), 'core is not compiled with CUDA')
class TestFillAnyLikeOpBfloat16(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'fill_any_like'
        self.prim_op_type = 'comp'
        self.python_api = fill_any_like_wrapper
        self.public_python_api = fill_any_like_wrapper
        self.dtype = np.uint16
        self.value = 0.0
        self.inputs = {'X': np.random.random((219, 232)).astype(np.float32)}
        self.attrs = {'value': self.value, 'dtype': core.VarDesc.VarType.BF16}
        self.outputs = {'Out': convert_float_to_uint16(self.value * np.ones_like(self.inputs['X']))}
        self.if_enable_cinn()

    def test_check_output(self):
        if False:
            return 10
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_prim=True, check_pir=True)

    def if_enable_cinn(self):
        if False:
            i = 10
            return i + 15
        pass

class TestFillAnyLikeOpValue1(TestFillAnyLikeOp):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = 1.0

    def if_enable_cinn(self):
        if False:
            while True:
                i = 10
        pass

class TestFillAnyLikeOpValue2(TestFillAnyLikeOp):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = 1e-10

    def if_enable_cinn(self):
        if False:
            return 10
        pass

class TestFillAnyLikeOpValue3(TestFillAnyLikeOp):

    def init(self):
        if False:
            print('Hello World!')
        self.value = 1e-100

    def if_enable_cinn(self):
        if False:
            print('Hello World!')
        pass

class TestFillAnyLikeOpType(TestFillAnyLikeOp):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'fill_any_like'
        self.prim_op_type = 'comp'
        self.python_api = fill_any_like_wrapper
        self.public_python_api = fill_any_like_wrapper
        self.dtype = np.int32
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.attrs = {'value': self.value, 'dtype': int(core.VarDesc.VarType.FP32)}
        self.outputs = {'Out': self.value * np.ones_like(self.inputs['X']).astype(np.float32)}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        if False:
            i = 10
            return i + 15
        pass

class TestFillAnyLikeOpFloat16(TestFillAnyLikeOp):

    def init(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16

    def if_enable_cinn(self):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()