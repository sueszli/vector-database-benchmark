"""Simple call to a builtin function."""
import unittest
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def dict_call(x):
    if False:
        i = 10
        return i + 15
    return dict(foo=x)

def dict_call_aliased(x):
    if False:
        return 10

    def fake_dict(x):
        if False:
            for i in range(10):
                print('nop')
        return x
    dict = fake_dict
    return dict(x)

def dict_call_dynamic(x):
    if False:
        while True:
            i = 10

    def gen_dict():
        if False:
            return 10
        return dict
    d = gen_dict()
    return d(foo=x)

def len_call(x):
    if False:
        while True:
            i = 10
    return len(x)

def nested_call(x):
    if False:
        return 10
    return list(range(len(x)))

def nested_cast(x):
    if False:
        while True:
            i = 10
    return float(int(x))

def len_call_aliased(x):
    if False:
        print('Hello World!')

    def fake_len(x):
        if False:
            return 10
        return x
    len = fake_len
    return len(x)

def len_call_dynamic(x):
    if False:
        i = 10
        return i + 15

    def gen_len():
        if False:
            i = 10
            return i + 15
        return len
    l = gen_len()
    return l(x)

def len_call_on_mock():
    if False:
        for i in range(10):
            print('nop')
    x = unittest.mock.MagicMock()
    return len(x)

class ReferenceTest(reference_test_base.TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        self.assertFunctionMatchesEager(dict_call, 1)
        self.assertFunctionMatchesEager(len_call, [1, 2])
        self.assertFunctionMatchesEager(dict_call_aliased, 1)
        self.assertFunctionMatchesEager(len_call_aliased, [1, 2])
        self.assertFunctionMatchesEager(dict_call_dynamic, 1)
        self.assertFunctionMatchesEager(len_call_dynamic, [1, 2])
        self.assertFunctionMatchesEager(nested_call, [])
        self.assertFunctionMatchesEager(nested_call, [1, 2, 3])

    def test_basic_tensor(self):
        if False:
            i = 10
            return i + 15
        self.all_inputs_tensors = True
        self.assertFunctionMatchesEager(dict_call, 1)
        self.assertFunctionMatchesEager(len_call, [1, 2])
        self.assertFunctionMatchesEager(dict_call_aliased, 1)
        self.assertFunctionMatchesEager(len_call_aliased, [1, 2])
        self.assertFunctionMatchesEager(dict_call_dynamic, 1)
        self.assertFunctionMatchesEager(len_call_dynamic, [1, 2])
        self.assertFunctionMatchesEager(nested_call, [])
        self.assertFunctionMatchesEager(nested_call, [1, 2, 3])
if __name__ == '__main__':
    tf.test.main()