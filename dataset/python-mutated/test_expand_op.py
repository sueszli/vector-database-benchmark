import unittest
import numpy as np
from op_test import OpTest
from paddle import base

class TestExpandOpRank1(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'expand'
        self.init_data()
        self.dtype = 'float32' if base.core.is_compiled_with_rocm() else 'float64'
        self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype)}
        self.attrs = {'expand_times': self.expand_times}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = [100]
        self.expand_times = [2]

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestExpandOpRank2_Corner(TestExpandOpRank1):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = [120]
        self.expand_times = [2]

class TestExpandOpRank2(TestExpandOpRank1):

    def init_data(self):
        if False:
            print('Hello World!')
        self.ori_shape = [12, 14]
        self.expand_times = [2, 3]

class TestExpandOpRank3_Corner(TestExpandOpRank1):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.ori_shape = (2, 10, 5)
        self.expand_times = (1, 1, 1)

class TestExpandOpRank3(TestExpandOpRank1):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (2, 4, 15)
        self.expand_times = (2, 1, 4)

class TestExpandOpRank4(TestExpandOpRank1):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = (2, 4, 5, 7)
        self.expand_times = (3, 2, 1, 2)

class TestExpandOpRank1_tensor_attr(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'expand'
        self.init_data()
        self.dtype = 'float32' if base.core.is_compiled_with_rocm() else 'float64'
        expand_times_tensor = []
        for (index, ele) in enumerate(self.expand_times):
            expand_times_tensor.append(('x' + str(index), np.ones(1).astype('int32') * ele))
        self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'expand_times_tensor': expand_times_tensor}
        self.attrs = {'expand_times': self.infer_expand_times}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        if False:
            print('Hello World!')
        self.ori_shape = [100]
        self.expand_times = [2]
        self.infer_expand_times = [-1]

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestExpandOpRank2_Corner_tensor_attr(TestExpandOpRank1_tensor_attr):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = [12, 14]
        self.expand_times = [1, 1]
        self.infer_expand_times = [1, -1]

class TestExpandOpRank2_attr_tensor(TestExpandOpRank1_tensor_attr):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = [12, 14]
        self.expand_times = [2, 3]
        self.infer_expand_times = [-1, 3]

class TestExpandOpRank1_tensor(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'expand'
        self.init_data()
        self.dtype = 'float32' if base.core.is_compiled_with_rocm() else 'float64'
        self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype), 'ExpandTimes': np.array(self.expand_times).astype('int32')}
        self.attrs = {}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        if False:
            while True:
                i = 10
        self.ori_shape = [100]
        self.expand_times = [2]

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestExpandOpRank2_tensor(TestExpandOpRank1_tensor):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.ori_shape = [12, 14]
        self.expand_times = [2, 3]

class TestExpandOpInteger(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'expand'
        self.inputs = {'X': np.random.randint(10, size=(2, 4, 5)).astype('int32')}
        self.attrs = {'expand_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_dygraph=False)

class TestExpandOpBoolean(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'expand'
        self.inputs = {'X': np.random.randint(2, size=(2, 4, 5)).astype('bool')}
        self.attrs = {'expand_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

class TestExpandOpInt64_t(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'expand'
        self.inputs = {'X': np.random.randint(10, size=(2, 4, 5)).astype('int64')}
        self.attrs = {'expand_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)
if __name__ == '__main__':
    unittest.main()