import unittest
import numpy as np
from numpy.lib.stride_tricks import as_strided
from op_test import OpTest
import paddle

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

def stft_np(x, window, n_fft, hop_length, **kwargs):
    if False:
        while True:
            i = 10
    frames = frame_from_librosa(x, n_fft, hop_length)
    frames = np.multiply(frames.transpose([0, 2, 1]), window).transpose([0, 2, 1])
    res = np.fft.rfft(frames, axis=1)
    return res

class TestStftOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'stft'
        (self.shape, self.type, self.attrs) = self.initTestCase()
        self.inputs = {'X': np.random.random(size=self.shape).astype(self.type), 'Window': np.hamming(self.attrs['n_fft']).astype(self.type)}
        self.outputs = {'Out': stft_np(x=self.inputs['X'], window=self.inputs['Window'], **self.attrs)}

    def initTestCase(self):
        if False:
            print('Hello World!')
        input_shape = (2, 100)
        input_type = 'float64'
        attrs = {'n_fft': 50, 'hop_length': 15, 'normalized': False, 'onesided': True}
        return (input_shape, input_type, attrs)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        self.check_output(check_dygraph=False)
        paddle.disable_static()

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        self.check_grad(['X'], 'Out', check_dygraph=False)
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()