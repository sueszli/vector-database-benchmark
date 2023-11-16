import unittest
import mock
import numpy
import chainer
from chainer import testing

class TestFunctionHook(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.h = chainer.FunctionHook()

    def test_name(self):
        if False:
            return 10
        self.assertEqual(self.h.name, 'FunctionHook')

    def test_forward_preprocess(self):
        if False:
            while True:
                i = 10
        self.assertTrue(hasattr(self.h, 'forward_preprocess'))

    def test_forward_postprocess(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(hasattr(self.h, 'forward_postprocess'))

    def test_backward_preprocess(self):
        if False:
            print('Hello World!')
        self.assertTrue(hasattr(self.h, 'backward_preprocess'))

    def test_backward_postprocess(self):
        if False:
            return 10
        self.assertTrue(hasattr(self.h, 'backward_postprocess'))

    def check_hook_methods_called(self, func):
        if False:
            for i in range(10):
                print('nop')

        def check_method_called(name):
            if False:
                return 10
            with mock.patch.object(self.h, name) as patched:
                with self.h:
                    func()
                patched.assert_called()
        check_method_called('forward_preprocess')
        check_method_called('forward_postprocess')
        check_method_called('backward_preprocess')
        check_method_called('backward_postprocess')

    def test_all_called_with_backward(self):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(numpy.random.rand(2, 3).astype(numpy.float32))
        y = chainer.functions.sum(x * x)
        self.check_hook_methods_called(y.backward)

    def test_all_called_with_grad(self):
        if False:
            return 10
        x = chainer.Variable(numpy.random.rand(2, 3).astype(numpy.float32))
        y = chainer.functions.sum(x * x)
        self.check_hook_methods_called(lambda : chainer.grad([y], [x]))
testing.run_module(__name__, __file__)