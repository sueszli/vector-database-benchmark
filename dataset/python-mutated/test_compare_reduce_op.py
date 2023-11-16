import unittest
import numpy as np
import op_test
import paddle

def create_test_not_equal_class(op_type, typename, callback):
    if False:
        return 10

    class Cls(op_test.OpTest):

        def setUp(self):
            if False:
                return 10
            x = np.random.random(size=(10, 7)).astype(typename)
            y = np.random.random(size=(10, 7)).astype(typename)
            z = callback(x, y)
            self.python_api = paddle.tensor.equal_all
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output(check_pir=True)
    cls_name = '{}_{}_{}'.format(op_type, typename, 'not_equal_all')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls

def create_test_not_shape_equal_class(op_type, typename, callback):
    if False:
        for i in range(10):
            print('nop')

    class Cls(op_test.OpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            x = np.random.random(size=(10, 7)).astype(typename)
            y = np.random.random(size=10).astype(typename)
            z = callback(x, y)
            self.python_api = paddle.tensor.equal_all
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            if False:
                while True:
                    i = 10
            self.check_output(check_pir=True)
    cls_name = '{}_{}_{}'.format(op_type, typename, 'not_shape_equal_all')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls

def create_test_equal_class(op_type, typename, callback):
    if False:
        return 10

    class Cls(op_test.OpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            x = y = np.random.random(size=(10, 7)).astype(typename)
            z = callback(x, y)
            self.python_api = paddle.tensor.equal_all
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output(check_pir=True)
    cls_name = '{}_{}_{}'.format(op_type, typename, 'equal_all')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls

def create_test_dim1_class(op_type, typename, callback):
    if False:
        i = 10
        return i + 15

    class Cls(op_test.OpTest):

        def setUp(self):
            if False:
                i = 10
                return i + 15
            x = y = np.random.random(size=1).astype(typename)
            x = np.array([True, False, True]).astype(typename)
            x = np.array([False, False, True]).astype(typename)
            z = callback(x, y)
            self.python_api = paddle.tensor.equal_all
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            if False:
                print('Hello World!')
            self.check_output(check_pir=True)
    cls_name = '{}_{}_{}'.format(op_type, typename, 'equal_all')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls
np_equal = lambda _x, _y: np.array(np.array_equal(_x, _y))
for _type_name in {'float32', 'float64', 'int32', 'int64', 'bool'}:
    create_test_not_equal_class('equal_all', _type_name, np_equal)
    create_test_equal_class('equal_all', _type_name, np_equal)
    create_test_dim1_class('equal_all', _type_name, np_equal)

class TestEqualReduceAPI(unittest.TestCase):

    def test_dynamic_api(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        x = paddle.ones(shape=[10, 10], dtype='int32')
        y = paddle.ones(shape=[10, 10], dtype='int32')
        out = paddle.equal_all(x, y)
        assert out.item() is True
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()