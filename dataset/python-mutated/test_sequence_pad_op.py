import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.base import core

class TestSequencePadOp(OpTest):

    def set_attr(self):
        if False:
            print('Hello World!')
        self.x_shape = [12, 10]
        self.x_len_lod = [[2, 3, 4, 3]]
        self.pad_value = [1.0]
        self.padded_length = -1
        self.dtype = 'float64'

    def set_data(self):
        if False:
            return 10
        x_data = np.random.uniform(0.1, 0.5, self.x_shape).astype(self.dtype)
        pad_value_data = np.array(self.pad_value).astype(self.dtype)
        self.inputs = {'X': (x_data, self.x_len_lod), 'PadValue': pad_value_data}
        self.attrs = {'padded_length': self.padded_length}

    def compute(self):
        if False:
            return 10
        padded_length = self.padded_length
        x_len_lod_0 = self.x_len_lod[0]
        if padded_length == -1:
            max_seq_len = 0
            for l in x_len_lod_0:
                max_seq_len = max(max_seq_len, l)
            padded_length = max_seq_len
        x_data = self.inputs['X'][0]
        pad_value_data = self.inputs['PadValue']
        if pad_value_data.shape == (1,):
            pad_value_data = np.broadcast_to(pad_value_data, shape=x_data.shape[1:])
        padded_sequences = []
        start_idx = 0
        for l in x_len_lod_0:
            end_idx = start_idx + l
            seq = x_data[start_idx:end_idx]
            to_pad_len = padded_length - l
            for _ in range(to_pad_len):
                seq = np.append(seq, pad_value_data[np.newaxis, :], axis=0)
            padded_sequences.append(seq)
            start_idx = end_idx
        out_data = np.array(padded_sequences)
        length = np.array(self.x_len_lod[0]).reshape(-1)
        self.outputs = {'Out': out_data, 'Length': length}

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'sequence_pad'
        self.set_attr()
        self.set_data()
        self.compute()

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestSequencePadOp2(TestSequencePadOp):

    def set_attr(self):
        if False:
            while True:
                i = 10
        self.x_shape = [12, 10]
        self.x_len_lod = [[2, 3, 4, 3]]
        self.pad_value = np.random.random(10)
        self.padded_length = -1
        self.dtype = 'float64'

class TestSequencePadOp3(TestSequencePadOp):

    def set_attr(self):
        if False:
            print('Hello World!')
        self.x_shape = [12, 10]
        self.x_len_lod = [[2, 3, 4, 3]]
        self.pad_value = [1.0]
        self.padded_length = 7
        self.dtype = 'float64'

class TestSequencePadOp4(TestSequencePadOp):

    def set_attr(self):
        if False:
            print('Hello World!')
        self.x_shape = [12, 10]
        self.x_len_lod = [[2, 3, 4, 3]]
        self.pad_value = np.random.random(10)
        self.padded_length = 7
        self.dtype = 'float64'

class TestSequencePadOp5(TestSequencePadOp):

    def set_attr(self):
        if False:
            return 10
        self.x_shape = [12, 2, 5]
        self.x_len_lod = [[2, 3, 4, 3]]
        self.pad_value = [1.0]
        self.padded_length = -1
        self.dtype = 'float64'

class TestSequencePadOp6(TestSequencePadOp):

    def set_attr(self):
        if False:
            while True:
                i = 10
        self.x_shape = [12, 2, 5]
        self.x_len_lod = [[2, 3, 4, 3]]
        self.pad_value = np.random.random((2, 5))
        self.padded_length = -1
        self.dtype = 'float64'

class TestSequencePadOp7(TestSequencePadOp):

    def set_attr(self):
        if False:
            return 10
        self.x_shape = [12, 2, 5]
        self.x_len_lod = [[2, 3, 4, 3]]
        self.pad_value = [1.0]
        self.padded_length = 7
        self.dtype = 'float64'

class TestSequencePadOp8(TestSequencePadOp):

    def set_attr(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [12, 2, 5]
        self.x_len_lod = [[0, 8, 0, 4, 0]]
        self.pad_value = [1.0]
        self.padded_length = 10
        self.dtype = 'float64'

class TestSequencePadOpError(unittest.TestCase):

    def test_error(self):
        if False:
            while True:
                i = 10

        def test_x_variable():
            if False:
                i = 10
                return i + 15
            x = np.random.random((2, 4)).astype('float32')
            pad_value = paddle.assign(np.array([0.0], dtype=np.float32))
            paddle.static.nn.sequence_lod.sequence_pad(x=x, pad_value=pad_value)
        self.assertRaises(TypeError, test_x_variable)

        def test_pad_value_variable():
            if False:
                for i in range(10):
                    print('nop')
            x1 = paddle.static.data(name='x1', shape=[-1, 10, 5], dtype='float32', lod_level=1)
            pad_value1 = np.array([0.0], dtype=np.float32)
            paddle.static.nn.sequence_lod.sequence_pad(x=x1, pad_value=pad_value1)
        self.assertRaises(TypeError, test_pad_value_variable)

        def test_dtype():
            if False:
                i = 10
                return i + 15
            x2 = paddle.static.data(name='x2', shape=[-1, 10, 5], dtype='int16', lod_level=1)
            pad_value2 = paddle.assign(np.array([0.0], dtype=np.int32))
            paddle.static.nn.sequence_lod.sequence_pad(x=x2, pad_value=pad_value2)
        self.assertRaises(TypeError, test_dtype)

    def test_length_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name='x', shape=[10, 5], dtype='float32', lod_level=1)
        pad_value = paddle.assign(np.array([0.0], dtype=np.float32))
        (out, length) = paddle.static.nn.sequence_lod.sequence_pad(x=x, pad_value=pad_value)
        self.assertEqual(length.dtype, core.VarDesc.VarType.INT64)
if __name__ == '__main__':
    unittest.main()