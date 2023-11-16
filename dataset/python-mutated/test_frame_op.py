import unittest
import numpy as np
from numpy.lib.stride_tricks import as_strided
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle.base import core

def frame_from_librosa(x, frame_length, hop_length, axis=-1):
    if False:
        while True:
            i = 10
    if axis == -1 and (not x.flags['C_CONTIGUOUS']):
        x = np.ascontiguousarray(x)
    elif axis == 0 and (not x.flags['F_CONTIGUOUS']):
        x = np.asfortranarray(x)
    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)
    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * x.itemsize]
    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * x.itemsize] + list(strides)
    else:
        raise ValueError(f'Frame axis={axis} must be either 0 or -1')
    return as_strided(x, shape=shape, strides=strides)

class TestFrameOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'frame'
        self.python_api = paddle.signal.frame
        self.init_dtype()
        self.init_shape()
        self.init_attrs()
        self.inputs = {'X': np.random.random(size=self.shape)}
        self.outputs = {'Out': frame_from_librosa(x=self.inputs['X'], **self.attrs)}

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'float64'

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.shape = (150,)

    def init_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'frame_length': 50, 'hop_length': 15, 'axis': -1}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        self.check_output()
        paddle.disable_static()

    def test_check_grad_normal(self):
        if False:
            return 10
        paddle.enable_static()
        self.check_grad(['X'], 'Out')
        paddle.disable_static()

class TestCase1(TestFrameOp):

    def initTestCase(self):
        if False:
            while True:
                i = 10
        input_shape = (150,)
        input_type = 'float64'
        attrs = {'frame_length': 50, 'hop_length': 15, 'axis': 0}
        return (input_shape, input_type, attrs)

class TestCase2(TestFrameOp):

    def initTestCase(self):
        if False:
            return 10
        input_shape = (8, 150)
        input_type = 'float64'
        attrs = {'frame_length': 50, 'hop_length': 15, 'axis': -1}
        return (input_shape, input_type, attrs)

class TestCase3(TestFrameOp):

    def initTestCase(self):
        if False:
            print('Hello World!')
        input_shape = (150, 8)
        input_type = 'float64'
        attrs = {'frame_length': 50, 'hop_length': 15, 'axis': 0}
        return (input_shape, input_type, attrs)

class TestCase4(TestFrameOp):

    def initTestCase(self):
        if False:
            while True:
                i = 10
        input_shape = (4, 2, 150)
        input_type = 'float64'
        attrs = {'frame_length': 50, 'hop_length': 15, 'axis': -1}
        return (input_shape, input_type, attrs)

class TestCase5(TestFrameOp):

    def initTestCase(self):
        if False:
            i = 10
            return i + 15
        input_shape = (150, 4, 2)
        input_type = 'float64'
        attrs = {'frame_length': 50, 'hop_length': 15, 'axis': 0}
        return (input_shape, input_type, attrs)

class TestFrameFP16OP(TestFrameOp):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestFrameBF16OP(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'frame'
        self.python_api = paddle.signal.frame
        (self.shape, self.dtype, self.attrs) = self.initTestCase()
        x = np.random.random(size=self.shape).astype(np.float32)
        out = frame_from_librosa(x, **self.attrs).copy()
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def initTestCase(self):
        if False:
            return 10
        input_shape = (150,)
        input_dtype = np.uint16
        attrs = {'frame_length': 50, 'hop_length': 15, 'axis': -1}
        return (input_shape, input_dtype, attrs)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        place = core.CUDAPlace(0)
        self.check_output_with_place(place)
        paddle.disable_static()

    def test_check_grad_normal(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out')
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()