import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.framework import core
SEED = 2021
np.random.seed(SEED)

def get_c_embedding(start, end, table, ids):
    if False:
        i = 10
        return i + 15
    index = ids.flatten()
    input_mask = (index < start) | (index >= end)
    masked_input = index - start
    masked_input[input_mask] = 0
    output = table[masked_input]
    output[input_mask] = 0.0
    return output

def c_embedding_wrapper(table, index, start_index=0):
    if False:
        print('Hello World!')
    return paddle._legacy_C_ops.c_embedding(table, index, 'start_index', start_index)

class TestCEmbeddingCPU(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_dtype()
        self.initcase()
        if core.is_compiled_with_xpu():
            self.__class__.use_xpu = True
        elif core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def initcase(self):
        if False:
            return 10
        self.op_type = 'c_embedding'
        self.python_api = c_embedding_wrapper
        table = np.random.random((17, 64)).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(self.ids_dtype)
        self.start_index = 10
        self.end_index = self.start_index + 17
        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape((2, 4, 64))}
        self.attrs = {'start_index': self.start_index}
        if core.is_compiled_with_xpu():
            self.__class__.use_xpu = True

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad_with_place(core.CPUPlace(), ['W'], 'Out')

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'float32'
        self.ids_dtype = 'int64'

class TestCEmbeddingOpBase(TestCEmbeddingCPU):

    def setUp(self):
        if False:
            return 10
        self.init_dtype()
        self.initcase()

    def test_check_output(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))
        elif core.is_compiled_with_xpu():
            self.check_output_with_place(core.XPUPlace(0))

    def test_check_grad(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(core.CUDAPlace(0), ['W'], 'Out')
        elif core.is_compiled_with_xpu():
            self.check_grad_with_place(core.XPUPlace(0), ['W'], 'Out')

    def init_dtype(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            self.dtype = 'float64'
            self.ids_dtype = 'int64'
        elif core.is_compiled_with_xpu():
            self.dtype = 'float32'
            self.ids_dtype = 'int64'

class TestCEmbeddingOpFP32(TestCEmbeddingOpBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_dtype()
        self.initcase()

    def initcase(self):
        if False:
            print('Hello World!')
        self.op_type = 'c_embedding'
        self.python_api = c_embedding_wrapper
        table = np.random.random((17, 64)).astype(self.dtype)
        ids = np.random.randint(low=0, high=17 * 2, size=(2, 4)).astype(self.ids_dtype)
        self.start_index = 10
        ids[0][1] = 12
        ids[0][2] = 12
        ids[1][2] = 12
        ids[1][3] = 12
        self.end_index = self.start_index + 17
        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape((2, 4, 64))}
        self.attrs = {'start_index': self.start_index}
        if core.is_compiled_with_xpu():
            self.__class__.use_xpu = True
        elif core.is_compiled_with_cuda():
            self.__class__.exist_fp64_check_grad = True

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = 'float32'
        self.ids_dtype = 'int32'
if __name__ == '__main__':
    unittest.main()