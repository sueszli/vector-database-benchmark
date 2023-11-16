import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.base.framework import Program, convert_np_dtype_to_dtype_, program_guard

def sequence_mask_wraper(x, maxlen_tensor=None, maxlen=-1, mask_dtype='int64'):
    if False:
        print('Hello World!')
    if maxlen_tensor is not None:
        maxlen = maxlen_tensor
    return paddle.nn.functional.sequence_mask(x, maxlen=maxlen, dtype=mask_dtype)

class SequenceMaskTestBase(OpTest):

    def initDefaultParameters(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'sequence_mask'
        self.python_api = sequence_mask_wraper
        self.maxlen = 10
        self.mask_dtype = 'int64'
        self.x = [[0, 3, 4], [5, 7, 9]]

    def initParameters(self):
        if False:
            return 10
        pass

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.initDefaultParameters()
        self.initParameters()
        if not isinstance(self.x, np.ndarray):
            self.x = np.array(self.x)
        self.inputs = {'X': self.x}
        self.outputs = {'Y': self.calc_ground_truth_mask()}
        self.attrs = {'maxlen': self.maxlen, 'out_dtype': convert_np_dtype_to_dtype_(self.mask_dtype)}

    def calc_ground_truth_mask(self):
        if False:
            for i in range(10):
                print('nop')
        maxlen = np.max(self.x) if self.maxlen < 0 else self.maxlen
        shape = self.x.shape + (maxlen,)
        index_broadcast = np.broadcast_to(np.reshape(range(maxlen), newshape=[1] * self.x.ndim + [-1]), shape=shape)
        x_broadcast = np.broadcast_to(np.reshape(self.x, newshape=self.x.shape + (-1,)), shape=shape)
        return (index_broadcast < x_broadcast).astype(self.mask_dtype)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

class SequenceMaskTest1(SequenceMaskTestBase):

    def initParameters(self):
        if False:
            i = 10
            return i + 15
        self.mask_dtype = 'bool'

class SequenceMaskTest2(SequenceMaskTestBase):

    def initParameters(self):
        if False:
            print('Hello World!')
        self.mask_dtype = 'uint8'

class SequenceMaskTest3(SequenceMaskTestBase):

    def initParameters(self):
        if False:
            return 10
        self.mask_dtype = 'int32'

class SequenceMaskTest4(SequenceMaskTestBase):

    def initParameters(self):
        if False:
            for i in range(10):
                print('nop')
        self.mask_dtype = 'float32'

class SequenceMaskTest5(SequenceMaskTestBase):

    def initParameters(self):
        if False:
            return 10
        self.mask_dtype = 'float64'

class SequenceMaskTest6(SequenceMaskTestBase):

    def initParameters(self):
        if False:
            print('Hello World!')
        self.maxlen = -1

class SequenceMaskTestBase_tensor_attr(OpTest):

    def initDefaultParameters(self):
        if False:
            print('Hello World!')
        self.op_type = 'sequence_mask'
        self.python_api = sequence_mask_wraper
        self.maxlen = 10
        self.maxlen_tensor = np.ones(1, 'int32') * 10
        self.mask_dtype = 'int64'
        self.x = [[0, 3, 4], [5, 7, 9]]

    def initParameters(self):
        if False:
            while True:
                i = 10
        pass

    def setUp(self):
        if False:
            while True:
                i = 10
        self.initDefaultParameters()
        self.initParameters()
        if not isinstance(self.x, np.ndarray):
            self.x = np.array(self.x)
        self.inputs = {'X': self.x, 'MaxLenTensor': self.maxlen_tensor}
        self.outputs = {'Y': self.calc_ground_truth_mask()}
        self.attrs = {'out_dtype': convert_np_dtype_to_dtype_(self.mask_dtype)}

    def calc_ground_truth_mask(self):
        if False:
            i = 10
            return i + 15
        maxlen = np.max(self.x) if self.maxlen < 0 else self.maxlen
        shape = self.x.shape + (maxlen,)
        index_broadcast = np.broadcast_to(np.reshape(range(maxlen), newshape=[1] * self.x.ndim + [-1]), shape=shape)
        x_broadcast = np.broadcast_to(np.reshape(self.x, newshape=self.x.shape + (-1,)), shape=shape)
        return (index_broadcast < x_broadcast).astype(self.mask_dtype)

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

class SequenceMaskTest1_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        if False:
            return 10
        self.mask_dtype = 'bool'

class SequenceMaskTest2_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        if False:
            i = 10
            return i + 15
        self.mask_dtype = 'uint8'

class SequenceMaskTest3_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        if False:
            print('Hello World!')
        self.mask_dtype = 'int32'

class SequenceMaskTest4_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        if False:
            i = 10
            return i + 15
        self.mask_dtype = 'float32'

class SequenceMaskTest5_tensor_attr(SequenceMaskTestBase_tensor_attr):

    def initParameters(self):
        if False:
            print('Hello World!')
        self.mask_dtype = 'float64'

class TestSequenceMaskOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        with program_guard(Program(), Program()):
            input_data = np.random.uniform(1, 5, [4]).astype('float32')

            def test_Variable():
                if False:
                    print('Hello World!')
                paddle.static.nn.sequence_lod.sequence_mask(input_data, maxlen=4)
            self.assertRaises(TypeError, test_Variable)

class TestSequenceMaskWithEmptyTensor(unittest.TestCase):

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        lengths = paddle.to_tensor(np.array([], dtype=np.int64))
        mask = paddle.nn.functional.sequence_mask(lengths)
        self.assertEqual(list(mask.shape), [0, 0])
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()