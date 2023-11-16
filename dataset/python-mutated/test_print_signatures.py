"""
TestCases for print_signatures.py

sample lines from API_DEV.spec:
    paddle.autograd.backward (ArgSpec(args=['tensors', 'grad_tensors', 'retain_graph'], varargs=None, keywords=None, defaults=(None, False)), ('document', '33a4434c9d123331499334fbe0274870'))
    paddle.autograd.PyLayer (paddle.autograd.py_layer.PyLayer, ('document', 'c26adbbf5f1eb43d16d4a399242c979e'))
    paddle.autograd.PyLayer.apply (ArgSpec(args=['cls'], varargs=args, keywords=kwargs, defaults=None), ('document', 'cb78696dc032fb8af2cba8504153154d'))
"""
import functools
import hashlib
import unittest
from print_signatures import is_primitive, md5

def func_example(param_a, param_b):
    if False:
        print('Hello World!')
    '\n    example function\n    '
    pass

def func_example_2(func=functools.partial(func_example, 1)):
    if False:
        while True:
            i = 10
    '\n    example function 2\n    '
    pass

class ClassExample:
    """
    example Class
    """

    def example_method(self):
        if False:
            while True:
                i = 10
        '\n        class method\n        '
        pass

class Test_all_in_print_signatures(unittest.TestCase):

    def test_md5(self):
        if False:
            while True:
                i = 10
        algo = hashlib.md5()
        algo.update(func_example.__doc__.encode('utf-8'))
        digest = algo.hexdigest()
        self.assertEqual(digest, md5(func_example.__doc__))

class Test_is_primitive(unittest.TestCase):

    def test_single(self):
        if False:
            while True:
                i = 10
        self.assertTrue(is_primitive(2))
        self.assertTrue(is_primitive(2.1))
        self.assertTrue(is_primitive('2.1.1'))
        self.assertFalse(is_primitive(b'hello paddle'))
        self.assertFalse(is_primitive(1j))
        self.assertTrue(is_primitive(True))

    def test_collection(self):
        if False:
            print('Hello World!')
        self.assertTrue(is_primitive([]))
        self.assertTrue(is_primitive(()))
        self.assertTrue(is_primitive(set()))
        self.assertTrue(is_primitive([1, 2]))
        self.assertTrue(is_primitive((1.1, 2.2)))
        self.assertTrue(is_primitive({1, 2.3}))
        self.assertFalse(is_primitive(range(3)))
        self.assertFalse(is_primitive({}))
        self.assertFalse(is_primitive([1, 1j]))
if __name__ == '__main__':
    unittest.main()