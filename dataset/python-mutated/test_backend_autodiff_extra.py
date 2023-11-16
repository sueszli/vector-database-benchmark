from builtins import zip
import numpy as np
import pprint
import pytest
from neon import NervanaObject
from neon.backends.autodiff import Autodiff
from utils import allclose_with_out

class CustomFunc(object):

    @staticmethod
    def sig(x):
        if False:
            while True:
                i = 10
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sig2(x):
        if False:
            return 10
        return 1.0 / (1.0 + np.exp2(-x))

    @staticmethod
    def tanh2(x):
        if False:
            i = 10
            return i + 15
        return (np.exp2(2.0 * x) - 1.0) / (np.exp2(2.0 * x) + 1.0)

    @staticmethod
    def argmax(x, axis=1, keepdims=True):
        if False:
            while True:
                i = 10
        '\n        calls numpy argmax with keepdims\n        '
        new_shape = list(x.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)
        return np.argmax(x, axis=axis).reshape(new_shape)

    @staticmethod
    def argmin(x, axis=1, keepdims=True):
        if False:
            print('Hello World!')
        '\n        calls numpy argmin with keepdims\n        '
        new_shape = list(x.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)
        return np.argmin(x, axis=axis).reshape(new_shape)

@pytest.mark.usefixtures('backend_default')
class TestAutodiff(object):

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.m = 2
        self.n = 2
        self.be = NervanaObject.be
        self.dtype = self.be.default_dtype
        self.test_epoch = 1
        self.delta = 1e-05

    def _rand_gen(self, *flags):
        if False:
            print('Hello World!')
        "\n        flags: 'int', 'pos', 'scalar', 'row', 'col'\n        "
        val = None
        m = self.m
        n = self.n
        if 'scalar' in flags:
            m = 1
            n = 1
        if 'row' in flags:
            m = 1
        if 'col' in flags:
            n = 1
        if 'int' in flags:
            if 'pos' in flags:
                val = np.random.randint(5.0, size=(m, n)).astype(float) + 1.0
            else:
                val = np.random.randint(5.0, size=(m, n)).astype(float) - 2.0
        elif 'pos' in flags:
            val = np.absolute(np.random.randn(m, n)) + 0.1
        else:
            val = np.random.randn(m, n)
        val[val > 5.0] = 5.0
        val[val < -5.0] = -5.0
        return val

    @staticmethod
    def _numpy_call(f_str, tensors):
        if False:
            print('Hello World!')
        '\n        evaluate function f from f_str and tensros\n\n        f_str: name of varialbe in the form of x0 - xn\n        tensors: numpy tensor vals\n        '
        f_str = f_str.replace('be', 'np')
        f_str = f_str.replace('np.sig', 'CustomFunc.sig')
        f_str = f_str.replace('np.sig2', 'CustomFunc.sig2')
        f_str = f_str.replace('np.tanh2', 'CustomFunc.tanh2')
        f_str = f_str.replace('np.argmax', 'CustomFunc.argmax')
        f_str = f_str.replace('np.argmin', 'CustomFunc.argmin')
        f_str = f_str.replace('axis=0', 'axis=0, keepdims=True')
        f_str = f_str.replace('axis=1', 'axis=1, keepdims=True')
        count = 0
        for tensor in tensors:
            exec('x%s = tensor' % count, globals(), locals())
            count += 1
        result = None
        result = eval(f_str)
        return result

    def _get_autodiff_grads_and_val(self, f_str, tensor_vals, get_op_tree=False, next_error=None):
        if False:
            return 10
        '\n        get autodiff grads from optree string expression\n        f_str: the string of expression to be executed\n        tensors: numpy tensor vals\n        '
        be = self.be
        count = 0
        tensors = []
        for tensor_val in tensor_vals:
            exec('x%s = self.be.array(tensor_val, name="x%s", dtype=self.dtype)' % (count, count))
            exec('tensors.append(x%s)' % count)
            count += 1
        f = None
        f = eval(f_str)
        f_val = be.empty(f.shape)
        f_val[:] = f
        if next_error is not None:
            next_error = self.be.array(next_error)
        ad = Autodiff(f, be, next_error=next_error)
        if get_op_tree:
            gradients = list(ad.get_grad_op_tree(tensors))
        else:
            gradients = list(ad.get_grad_asnumpyarray(tensors))
        return [gradients, f_val.get()]

    def _get_numerical_grads_and_val(self, f_str, tensors, next_error=None):
        if False:
            print('Hello World!')
        '`\n        get autodiff grads from numpy string expression\n        tensors: numpy tensor vals\n        '
        gradients = []
        for tensor in tensors:
            gradients.append(np.zeros(tensor.shape))
        f_val = TestAutodiff._numpy_call(f_str, tensors)
        if next_error is None:
            next_error = np.ones_like(f_val)
        for (tensor, gradient) in zip(tensors, gradients):
            gradient_flat = np.copy(gradient.reshape((-1,)))
            ind = 0
            for x in np.nditer(tensor, op_flags=['readwrite']):
                x_backup = np.copy(x)
                x[...] = x + self.delta
                f_inc = np.sum(TestAutodiff._numpy_call(f_str, tensors) * next_error)
                x[...] = x_backup
                x[...] = x - self.delta
                f_dec = np.sum(TestAutodiff._numpy_call(f_str, tensors) * next_error)
                x[...] = x_backup
                gradient_flat[ind] = (f_inc - f_dec) / (2.0 * self.delta)
                ind += 1
            gradient[:] = gradient_flat.reshape(gradient.shape)
        return [gradients, f_val]

    def _assert_grad_equal(self, f_str, tensors, rtol=0.01, atol=1e-05, next_error=None):
        if False:
            i = 10
            return i + 15

        def debug_msg(count):
            if False:
                for i in range(10):
                    print('nop')
            msg = ''
            msg += 'Error at tensor x%s' % (count,) + '\n'
            msg += pprint.pformat(tensors) + '\n'
            grad_op_trees = self._get_autodiff_grads_and_val(f_str, tensors, get_op_tree=True)
            grad_op_tree = grad_op_trees[0][count - 1]
            msg += grad_op_tree.pp() + '\n'
            return msg
        autodiff_grads_and_val = self._get_autodiff_grads_and_val(f_str, tensors, next_error=next_error)
        numerical_grads_and_val = self._get_numerical_grads_and_val(f_str, tensors, next_error=next_error)
        assert len(autodiff_grads_and_val) == len(numerical_grads_and_val)
        numerical_grads_and_val[1] = numerical_grads_and_val[1].reshape(autodiff_grads_and_val[1].shape)
        assert allclose_with_out(autodiff_grads_and_val[1].astype(self.dtype), numerical_grads_and_val[1].astype(self.dtype), rtol=rtol, atol=atol)
        count = 0
        for (autodiff_grad, numerical_grad) in zip(autodiff_grads_and_val[0], numerical_grads_and_val[0]):
            count += 1
            if not np.allclose(autodiff_grad.astype(self.dtype), numerical_grad.astype(self.dtype), rtol=rtol, atol=atol):
                raise ValueError(debug_msg(count))

    def test_reduction_shape(self):
        if False:
            return 10
        be = self.be
        x0 = be.array(np.array([[1, 2], [4, 5]]), name='x0')
        f = be.sum(x0, axis=0)
        assert f.shape == (1, 2)
        f = be.sum(x0, axis=1)
        assert f.shape == (2, 1)
        f = be.sum(x0)
        assert f.shape == (1, 1)

    def test_reduction(self):
        if False:
            while True:
                i = 10
        for _ in range(self.test_epoch):
            (x0, x1, x2, x3) = [self._rand_gen() for _ in range(4)]
            f_str = '  (x0 + x2) + be.sum(x0, axis=0)- (x0 - x1) - be.mean(x3, axis=1)+ (x2 + x3) + be.var(x0, axis=0)+ (x2 + x3) + be.std(x0)- (x2 - x3) - be.max(x3, axis=1)- (x2 - x3) - be.min(x3, axis=0)- (x2 - x3) - be.argmax(x3, axis=1)- (x2 - x3) - be.argmin(x3, axis=0)'
            self._assert_grad_equal(f_str, [x0, x1, x2, x3], rtol=0.01)

    def test_batchnorm(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(self.test_epoch):
            x0 = np.random.randn(10, 64)
            x1 = np.random.randn(10, 1)
            x2 = np.random.randn(10, 1)
            next_error = np.random.randn(10, 64) / 64.0
            f_str = '((x0 - be.mean(x0, axis=1)) / be.sqrt(be.var(x0, axis=1) + 1e-6)) * x1 + x2'
            self._assert_grad_equal(f_str, [x0, x1, x2], rtol=0.1, atol=0.01, next_error=next_error)

    def test_positive(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(self.test_epoch):
            (x0, x1, x2, x3) = [self._rand_gen('pos') for _ in range(4)]
            f_str = '0.9 ** 0.9 ** x0 + x1 ** 0.9 ** 0.9+ (be.sqrt(x0 + x1 + x2) + x3)- (be.exp(x0 + x1 + x2) + x3)+ (be.exp2(x0 + x1 + x2) + x3)- (be.log(x0 + x1 + x2) + x3)+ (be.log2(x0 + x1 + x2) + x3)'
            self._assert_grad_equal(f_str, [x0, x1, x2, x3], rtol=0.01)

    def test_real(self):
        if False:
            i = 10
            return i + 15
        for _ in range(self.test_epoch):
            (x0, x1, x2, x3) = [self._rand_gen() for _ in range(4)]
            f_str = 'x0 + be.absolute(x1 + x2) + x3- (x0 + be.square(x1 + x2) + x3)+ (x0 + be.sig(x1 + x2) + x3)- (x0 + be.sig2(x1 + x2) + x3)+ (x0 + be.tanh(x1 + x2) + x3)- (x0 + be.tanh2(x1 + x2) + x3)+ (x0 + be.maximum(x0 + x1, x2 + x3) + x3)'
            self._assert_grad_equal(f_str, [x0, x1, x2, x3], rtol=0.01)

    def test_unbroadcast(self):
        if False:
            print('Hello World!')
        for _ in range(self.test_epoch):
            x0 = self._rand_gen('scalar')
            x1 = self._rand_gen()
            f_str = 'x0 + x0 + x1 + x1'
            self._assert_grad_equal(f_str, [x0, x1], rtol=0.01)
            x0 = self._rand_gen('col')
            x1 = self._rand_gen()
            f_str = 'x0 + x0 + x1 + x1 + x0'
            self._assert_grad_equal(f_str, [x0, x1], rtol=0.01)
            x0 = self._rand_gen('row')
            x1 = self._rand_gen()
            f_str = 'x0 + x0 + x1 + x1 + x0'
            self._assert_grad_equal(f_str, [x0, x1], rtol=0.01)
            x0 = self._rand_gen('scalar')
            x1 = self._rand_gen('row')
            x2 = self._rand_gen('col')
            x3 = self._rand_gen()
            f_str = 'x0 + x1 + x3 * x2 + x0 + be.tanh(x1) + x3'
            self._assert_grad_equal(f_str, [x0, x1, x2, x3], rtol=0.01)

    def test_hard_coded(self):
        if False:
            return 10
        '\n        The most basic test case\n        '
        be = self.be
        x0 = be.array(np.ones((3, 3)) * 1, name='x0', dtype=self.dtype)
        x1 = be.array(np.ones((3, 3)) * 2, name='x1', dtype=self.dtype)
        x2 = be.array(np.ones((3, 3)) * 3, name='x2', dtype=self.dtype)
        x3 = be.array(np.ones((3, 3)) * 5, name='x3', dtype=self.dtype)
        f = x0 * x0 - x1 * x0 + x0 * x2 - x2 * x1 * x0 + x3 * x3 * x3
        ad = Autodiff(f, be)
        x0_grad = be.array(np.ones((3, 3)) * -3, dtype=self.dtype)
        x1_grad = be.array(np.ones((3, 3)) * -4, dtype=self.dtype)
        x2_grad = be.array(np.ones((3, 3)) * -1, dtype=self.dtype)
        x3_grad = be.array(np.ones((3, 3)) * 75, dtype=self.dtype)
        assert allclose_with_out(ad.get_grad_asnumpyarray([x0])[0], x0_grad.get(), atol=1e-05)
        assert allclose_with_out(ad.get_grad_asnumpyarray([x1])[0], x1_grad.get(), atol=1e-05)
        assert allclose_with_out(ad.get_grad_asnumpyarray([x2])[0], x2_grad.get(), atol=1e-05)
        assert allclose_with_out(ad.get_grad_asnumpyarray([x3])[0], x3_grad.get(), atol=1e-05)