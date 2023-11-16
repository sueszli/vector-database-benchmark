import unittest
import numpy as np
from op_test import OpTest
import paddle
paddle.enable_static()

class TestBitwiseAnd(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'bitwise_and'
        self.python_api = paddle.tensor.logic.bitwise_and
        self.init_dtype()
        self.init_shape()
        self.init_bound()
        x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
        y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
        out = np.bitwise_and(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        pass

    def init_dtype(self):
        if False:
            return 10
        self.dtype = np.int32

    def init_shape(self):
        if False:
            print('Hello World!')
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        if False:
            return 10
        self.low = -100
        self.high = 100

class TestBitwiseAnd_ZeroDim1(TestBitwiseAnd):

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = []
        self.y_shape = []

class TestBitwiseAnd_ZeroDim2(TestBitwiseAnd):

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []

class TestBitwiseAnd_ZeroDim3(TestBitwiseAnd):

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseAndUInt8(TestBitwiseAnd):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.uint8

    def init_bound(self):
        if False:
            i = 10
            return i + 15
        self.low = 0
        self.high = 100

class TestBitwiseAndInt8(TestBitwiseAnd):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int8

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseAndInt16(TestBitwiseAnd):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.int16

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]

class TestBitwiseAndInt64(TestBitwiseAnd):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.int64

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseAndBool(TestBitwiseAnd):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'bitwise_and'
        self.python_api = paddle.tensor.logic.bitwise_and
        self.init_shape()
        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_and(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

class TestBitwiseOr(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'bitwise_or'
        self.python_api = paddle.tensor.logic.bitwise_or
        self.init_dtype()
        self.init_shape()
        self.init_bound()
        x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
        y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
        out = np.bitwise_or(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        pass

    def init_dtype(self):
        if False:
            return 10
        self.dtype = np.int32

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        if False:
            print('Hello World!')
        self.low = -100
        self.high = 100

class TestBitwiseOr_ZeroDim1(TestBitwiseOr):

    def init_shape(self):
        if False:
            return 10
        self.x_shape = []
        self.y_shape = []

class TestBitwiseOr_ZeroDim2(TestBitwiseOr):

    def init_shape(self):
        if False:
            print('Hello World!')
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []

class TestBitwiseOr_ZeroDim3(TestBitwiseOr):

    def init_shape(self):
        if False:
            return 10
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseOrUInt8(TestBitwiseOr):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint8

    def init_bound(self):
        if False:
            for i in range(10):
                print('nop')
        self.low = 0
        self.high = 100

class TestBitwiseOrInt8(TestBitwiseOr):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int8

    def init_shape(self):
        if False:
            return 10
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseOrInt16(TestBitwiseOr):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.int16

    def init_shape(self):
        if False:
            return 10
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]

class TestBitwiseOrInt64(TestBitwiseOr):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.int64

    def init_shape(self):
        if False:
            print('Hello World!')
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseOrBool(TestBitwiseOr):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'bitwise_or'
        self.python_api = paddle.tensor.logic.bitwise_or
        self.init_shape()
        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_or(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

class TestBitwiseXor(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'bitwise_xor'
        self.python_api = paddle.tensor.logic.bitwise_xor
        self.init_dtype()
        self.init_shape()
        self.init_bound()
        x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
        y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
        out = np.bitwise_xor(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        pass

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.int32

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        if False:
            i = 10
            return i + 15
        self.low = -100
        self.high = 100

class TestBitwiseXor_ZeroDim1(TestBitwiseXor):

    def init_shape(self):
        if False:
            print('Hello World!')
        self.x_shape = []
        self.y_shape = []

class TestBitwiseXor_ZeroDim2(TestBitwiseXor):

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []

class TestBitwiseXor_ZeroDim3(TestBitwiseXor):

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseXorUInt8(TestBitwiseXor):

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.uint8

    def init_bound(self):
        if False:
            return 10
        self.low = 0
        self.high = 100

class TestBitwiseXorInt8(TestBitwiseXor):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int8

    def init_shape(self):
        if False:
            while True:
                i = 10
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseXorInt16(TestBitwiseXor):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.int16

    def init_shape(self):
        if False:
            return 10
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]

class TestBitwiseXorInt64(TestBitwiseXor):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.int64

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]

class TestBitwiseXorBool(TestBitwiseXor):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'bitwise_xor'
        self.python_api = paddle.tensor.logic.bitwise_xor
        self.init_shape()
        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_xor(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

class TestBitwiseNot(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'bitwise_not'
        self.python_api = paddle.tensor.logic.bitwise_not
        self.init_dtype()
        self.init_shape()
        self.init_bound()
        x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
        out = np.bitwise_not(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        if False:
            return 10
        pass

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.int32

    def init_shape(self):
        if False:
            while True:
                i = 10
        self.x_shape = [2, 3, 4, 5]

    def init_bound(self):
        if False:
            i = 10
            return i + 15
        self.low = -100
        self.high = 100

class TestBitwiseNot_ZeroDim(TestBitwiseNot):

    def init_shape(self):
        if False:
            while True:
                i = 10
        self.x_shape = []

class TestBitwiseNotUInt8(TestBitwiseNot):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.uint8

    def init_bound(self):
        if False:
            print('Hello World!')
        self.low = 0
        self.high = 100

class TestBitwiseNotInt8(TestBitwiseNot):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.int8

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [4, 5]

class TestBitwiseNotInt16(TestBitwiseNot):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int16

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [2, 3, 4, 5]

class TestBitwiseNotInt64(TestBitwiseNot):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.int64

    def init_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [1, 4, 1]

class TestBitwiseNotBool(TestBitwiseNot):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'bitwise_not'
        self.python_api = paddle.tensor.logic.bitwise_not
        self.init_shape()
        x = np.random.choice([True, False], self.x_shape)
        out = np.bitwise_not(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}
if __name__ == '__main__':
    unittest.main()