import collections
import sys
import typing
import unittest
sys.path.insert(0, '.')
import config
import numpy as np
import utils
from utils import matmul, mul, nested, o2, reduce, reduce_dim
import paddle
import paddle.nn.functional as F
from paddle.incubate.autograd.utils import as_tensors

def make_v(f, inputs):
    if False:
        return 10
    outputs = as_tensors(f(*inputs))
    return [paddle.ones_like(x) for x in outputs]

class TestAutogradFunctional(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.RAW_INPUTS = {'a': [1.0], 'b': [1.0, 2.0], 'c': [3.0, 4.0], 'd': [[2.0], [3.0]], 'A': [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], 'B': [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]}

    def setUp(self):
        if False:
            return 10
        pass

    def gen_input(self, inp, stop_gradient=False):
        if False:
            print('Hello World!')
        if isinstance(inp, paddle.Tensor):
            return inp
        return paddle.to_tensor(self.RAW_INPUTS[inp], stop_gradient=stop_gradient)

    def gen_inputs(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(inputs, list):
            inputs = [self.gen_input(x) for x in inputs]
        else:
            inputs = [self.gen_input(inputs)]
        return inputs

    def gen_test_pairs(self, func, inputs, v=None, create_graph=False, allow_unused=False):
        if False:
            i = 10
            return i + 15

        def vjp_test():
            if False:
                i = 10
                return i + 15
            nonlocal v
            xs = self.gen_inputs(inputs)
            if v is not None:
                v = self.gen_inputs(v)
                (outputs, inputs_grad) = paddle.incubate.autograd.vjp(func, xs, v)
            else:
                (outputs, inputs_grad) = paddle.incubate.autograd.vjp(func, xs)
            return (outputs, inputs_grad)

        def grad_test():
            if False:
                print('Hello World!')
            nonlocal v
            xs = self.gen_inputs(inputs)
            if v is not None:
                v = self.gen_inputs(v)
            outputs = func(*xs)
            if v is not None:
                inputs_grad = paddle.grad(outputs, xs, v, create_graph=create_graph, allow_unused=allow_unused)
            else:
                inputs_grad = paddle.grad(outputs, xs, create_graph=create_graph, allow_unused=allow_unused)
            return (outputs, inputs_grad)
        return (vjp_test, grad_test)

    def gen_jvp_tests(self, func, inputs, v=None, create_graph=False, allow_unused=False):
        if False:
            print('Hello World!')

        def jvp_test():
            if False:
                print('Hello World!')
            nonlocal v
            xs = self.gen_inputs(inputs)
            if v is not None:
                v = self.gen_inputs(v)
                (outputs, outputs_grad) = paddle.incubate.autograd.jvp(func, xs, v, create_graph=create_graph, allow_unused=allow_unused)
            else:
                (outputs, outputs_grad) = paddle.incubate.autograd.jvp(func, xs, create_graph=create_graph, allow_unused=allow_unused)
            return (outputs, outputs_grad)
        return jvp_test

    def check_results(self, ref, res):
        if False:
            for i in range(10):
                print('nop')
        type_error = 'Result is different than expected in shape or type'
        value_error = 'Result is different than expected values'
        if ref is None:
            self.assertTrue(res is None, type_error)
        elif isinstance(ref, paddle.Tensor):
            self.assertTrue(isinstance(res, paddle.Tensor), type_error)
            np.testing.assert_allclose(res, ref)
        else:
            self.assertTrue(len(res) == len(ref), type_error)
            for i in range(len(ref)):
                self.check_results(ref[i], res[i])
        return True

class TestVJP(TestAutogradFunctional):

    def func_vjp_i1o1(self):
        if False:
            while True:
                i = 10
        test_cases = [[reduce, 'A'], [reduce_dim, 'A']]
        for (f, inputs) in test_cases:
            (vjp, grad) = self.gen_test_pairs(f, inputs)
            (vjp_result, grad_result) = (vjp(), grad())
            self.check_results(grad_result, vjp_result)

    def func_vjp_i2o1(self):
        if False:
            return 10
        test_cases = [[matmul, ['A', 'B']], [mul, ['b', 'c']]]
        for (f, inputs) in test_cases:
            (vjp, grad) = self.gen_test_pairs(f, inputs)
            (vjp_result, grad_result) = (vjp(), grad())
            self.check_results(grad_result, vjp_result)

    def func_vjp_i2o2(self):
        if False:
            for i in range(10):
                print('nop')
        test_cases = [[o2, ['A', 'A']]]
        for (f, inputs) in test_cases:
            inputs = self.gen_inputs(inputs)
            v = make_v(f, inputs)
            (vjp, grad) = self.gen_test_pairs(f, inputs, v=v)
            (vjp_result, grad_result) = (vjp(), grad())
            self.check_results(grad_result, vjp_result)

    def func_vjp_i2o2_omitting_v(self):
        if False:
            for i in range(10):
                print('nop')
        test_cases = [[o2, ['A', 'A']]]
        for (f, inputs) in test_cases:
            inputs = self.gen_inputs(inputs)
            (vjp, grad) = self.gen_test_pairs(f, inputs)
            (vjp_result, grad_result) = (vjp(), grad())
            self.check_results(grad_result, vjp_result)

    def func_vjp_nested(self):
        if False:
            return 10
        x = self.gen_input('a')
        test_cases = [[nested(x), 'a']]
        for (f, inputs) in test_cases:
            (vjp, grad) = self.gen_test_pairs(f, inputs)
            (vjp_result, grad_result) = (vjp(), grad())
            self.check_results(grad_result, vjp_result)

    def func_vjp_aliased_input(self):
        if False:
            print('Hello World!')
        x = self.gen_input('a')
        ref = self.gen_test_pairs(nested(x), 'a')[0]
        aliased = self.gen_test_pairs(nested(x), x)[0]
        (ref_result, aliased_result) = (ref(), aliased())
        self.check_results(ref_result, aliased_result)

    def test_all_cases(self):
        if False:
            while True:
                i = 10
        self.func_vjp_i1o1()
        self.func_vjp_i2o1()
        self.func_vjp_i2o2()
        self.func_vjp_i2o2_omitting_v()
        self.func_vjp_nested()
        self.func_vjp_aliased_input()

    def test_input_single_tensor(self):
        if False:
            return 10
        self.assertIsInstance(paddle.incubate.autograd.vjp(paddle.tanh, paddle.rand((3, 4)))[1], paddle.base.framework.Variable)

@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'xs', 'v', 'expected_exception'), (('v_shape_not_equal_ys', utils.square, np.random.rand(3), np.random.rand(1), RuntimeError),))
class TestVJPException(unittest.TestCase):

    def test_vjp(self):
        if False:
            return 10
        with self.assertRaises(self.expected_exception):
            paddle.incubate.autograd.vjp(self.fun, paddle.to_tensor(self.xs), paddle.to_tensor(self.v))

def jac(grad_fn, f, inputs):
    if False:
        for i in range(10):
            print('nop')
    assert grad_fn in [paddle.incubate.autograd.vjp, paddle.incubate.autograd.jvp]
    if grad_fn is paddle.incubate.autograd.jvp:
        vs = [paddle.zeros_like(x) for x in inputs]
    else:
        outputs = f(*inputs)
        if isinstance(outputs, paddle.Tensor):
            outputs = [outputs]
        vs = [paddle.zeros_like(y) for y in outputs]
    JJ_cols = []
    for (i, v) in enumerate(vs):
        v = v.flatten()
        for j in range(len(v)):
            _v = paddle.zeros_like(v).detach()
            _v[j] = 1.0
            _v = _v.reshape(vs[i].shape)
            _vs = vs.copy()
            _vs[i] = _v
            (_, grads) = grad_fn(f, inputs, _vs)
            if isinstance(grads, typing.Sequence):
                d_outs = paddle.concat([d_out.flatten() for d_out in grads])
            else:
                d_outs = grads.flatten()
            JJ_cols.append(d_outs)
    JJ = paddle.stack(JJ_cols)
    if grad_fn is paddle.incubate.autograd.vjp:
        JJ = JJ.t()
    return JJ

class TestJVP(TestAutogradFunctional):

    def func_jvp_i1o1(self):
        if False:
            print('Hello World!')
        test_cases = [[reduce, 'A'], [reduce_dim, 'A']]
        for (f, inputs) in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(paddle.incubate.autograd.jvp, f, inputs)
            reverse_jac = jac(paddle.incubate.autograd.vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def func_jvp_i2o1(self):
        if False:
            return 10
        test_cases = [[matmul, ['A', 'B']]]
        for (f, inputs) in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(paddle.incubate.autograd.jvp, f, inputs)
            reverse_jac = jac(paddle.incubate.autograd.vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def func_jvp_i2o2(self):
        if False:
            while True:
                i = 10
        test_cases = [[o2, ['A', 'A']]]
        for (f, inputs) in test_cases:
            inputs = self.gen_inputs(inputs)
            forward_jac = jac(paddle.incubate.autograd.jvp, f, inputs)
            reverse_jac = jac(paddle.incubate.autograd.vjp, f, inputs)
            self.check_results(forward_jac, reverse_jac)

    def func_jvp_i2o2_omitting_v(self):
        if False:
            while True:
                i = 10
        test_cases = [[o2, ['A', 'A']]]
        for (f, inputs) in test_cases:
            inputs = self.gen_inputs(inputs)
            results_omitting_v = paddle.incubate.autograd.jvp(f, inputs)
            v = [paddle.ones_like(x) for x in inputs]
            results_with_v = paddle.incubate.autograd.jvp(f, inputs, v)
            self.check_results(results_omitting_v, results_with_v)

    def test_all_cases(self):
        if False:
            while True:
                i = 10
        self.func_jvp_i1o1()
        self.func_jvp_i2o1()
        self.func_jvp_i2o2()
        self.func_jvp_i2o2_omitting_v()

@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'func', 'xs'), (('1d_in_1d_out', utils.square, np.array([2.0, 3.0])), ('3d_in_3d_out', utils.square, np.random.rand(2, 3, 4)), ('single_in_single_out', utils.square, np.random.rand(2, 3)), ('multi_in_single_out', paddle.matmul, (np.random.rand(2, 2), np.random.rand(2, 2)))))
class TestJacobianNoBatch(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._dtype = self.xs[0].dtype if isinstance(self.xs, typing.Sequence) else self.xs.dtype
        self._eps = config.TOLERANCE.get(str(self._dtype)).get('first_order_grad').get('eps')
        self._rtol = config.TOLERANCE.get(str(self._dtype)).get('first_order_grad').get('rtol')
        self._atol = config.TOLERANCE.get(str(self._dtype)).get('first_order_grad').get('atol')

    def test_jacobian(self):
        if False:
            for i in range(10):
                print('nop')
        xs = [paddle.to_tensor(x) for x in self.xs] if isinstance(self.xs, typing.Sequence) else paddle.to_tensor(self.xs)
        self._actual = paddle.incubate.autograd.Jacobian(self.func, xs, False)
        self._expected = self._get_expected()
        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (Index('all', (slice(0, None, None), slice(0, None, None))), Index('row', (0, slice(0, None, None))), Index('col', (slice(0, None, None), 0)), Index('multi-row', (slice(0, 2, 1), slice(0, None, None))))
        self.assertEqual(self._actual[:].numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(self._actual.__getitem__(index.value), self._expected.__getitem__(index.value), rtol=self._rtol, atol=self._atol, err_msg=f'Testcase {index.type} index not passed, value is {index.value}')

    def _get_expected(self):
        if False:
            print('Hello World!')
        xs = [paddle.to_tensor(x) for x in self.xs] if isinstance(self.xs, typing.Sequence) else paddle.to_tensor(self.xs)
        jac = utils._compute_numerical_jacobian(self.func, xs, self._eps, self._dtype)
        return utils._np_concat_matrix_sequence(jac, utils.MatrixFormat.NM)

@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'func', 'xs'), (('1d_in_1d_out', utils.square, np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0]])), ('3d_in_3d_out', utils.square, np.random.rand(2, 3, 4)), ('multi_in_single_out', utils.square, np.random.rand(2, 3))))
class TestJacobianBatchFirst(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._dtype = self.xs[0].dtype if isinstance(self.xs, typing.Sequence) else self.xs.dtype
        self._eps = config.TOLERANCE.get(str(self._dtype)).get('first_order_grad').get('eps')
        self._rtol = config.TOLERANCE.get(str(self._dtype)).get('first_order_grad').get('rtol')
        self._atol = config.TOLERANCE.get(str(self._dtype)).get('first_order_grad').get('atol')

    def test_jacobian(self):
        if False:
            return 10
        xs = [paddle.to_tensor(x) for x in self.xs] if isinstance(self.xs, typing.Sequence) else paddle.to_tensor(self.xs)
        self._actual = paddle.incubate.autograd.Jacobian(self.func, xs, True)
        self._expected = self._get_expected()
        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (Index('all', (slice(0, None, None), slice(0, None, None), slice(0, None, None))), Index('row', (slice(0, None, None), 0, slice(0, None, None))), Index('col', (slice(0, None, None), slice(0, None, None), 0)), Index('batch', (slice(0, 2, None), slice(0, None, None), slice(0, None, None))), Index('multi_row', (slice(0, 1, None), slice(0, 2, 1), slice(0, None, None))))
        self.assertEqual(self._actual[:].numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(self._actual.__getitem__(index.value), self._expected.__getitem__(index.value), rtol=self._rtol, atol=self._atol, err_msg=f'Testcase {index.type} index not passed, value is {index.value}')

    def _get_expected(self):
        if False:
            i = 10
            return i + 15
        xs = [paddle.to_tensor(x) for x in self.xs] if isinstance(self.xs, typing.Sequence) else paddle.to_tensor(self.xs)
        jac = utils._compute_numerical_batch_jacobian(self.func, xs, self._eps, self._dtype, False)
        jac = utils._np_concat_matrix_sequence(jac, utils.MatrixFormat.NBM)
        return utils._np_transpose_matrix_format(jac, utils.MatrixFormat.NBM, utils.MatrixFormat.BNM)

class TestHessianNoBatch(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        self.shape = (2, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = config.TOLERANCE.get(self.dtype).get('second_order_grad').get('eps')
        self.rtol = config.TOLERANCE.get(self.dtype).get('second_order_grad').get('rtol')
        self.atol = config.TOLERANCE.get(self.dtype).get('second_order_grad').get('atol')
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

    def func_single_input(self):
        if False:
            for i in range(10):
                print('nop')

        def func(x):
            if False:
                print('Hello World!')
            return paddle.sum(paddle.matmul(x, x))
        numerical_hessian = utils._compute_numerical_hessian(func, self.x, self.numerical_delta, self.np_dtype)
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)
        self.x.stop_gradient = False
        hessian = paddle.incubate.autograd.Hessian(func, self.x)
        np.testing.assert_allclose(hessian[:].numpy(), numerical_hessian, self.rtol, self.atol)

    def func_multi_input(self):
        if False:
            while True:
                i = 10

        def func(x, y):
            if False:
                i = 10
                return i + 15
            return paddle.sum(paddle.matmul(x, y))
        numerical_hessian = utils._compute_numerical_hessian(func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.incubate.autograd.Hessian(func, [self.x, self.y])
        np.testing.assert_allclose(hessian[:].numpy(), numerical_hessian, rtol=self.rtol, atol=self.atol)

    def func_allow_unused_true(self):
        if False:
            for i in range(10):
                print('nop')

        def func(x, y):
            if False:
                while True:
                    i = 10
            return paddle.sum(paddle.matmul(x, x))
        numerical_hessian = utils._compute_numerical_hessian(func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        hessian = paddle.incubate.autograd.Hessian(func, [self.x, self.y])
        np.testing.assert_allclose(hessian[:].numpy(), numerical_hessian, self.rtol, self.atol)

    def func_create_graph_true(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                while True:
                    i = 10
            return paddle.sum(F.sigmoid(x))
        numerical_hessian = utils._compute_numerical_hessian(func, self.x, self.numerical_delta, self.np_dtype)
        numerical_hessian = utils._np_concat_matrix_sequence(numerical_hessian)
        self.x.stop_gradient = False
        hessian = paddle.incubate.autograd.Hessian(func, self.x)
        assert not hessian[:].stop_gradient
        np.testing.assert_allclose(hessian[:].numpy(), numerical_hessian, self.rtol, self.atol)

    def func_out_not_single(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                i = 10
                return i + 15
            return x * x
        with self.assertRaises(RuntimeError):
            paddle.incubate.autograd.Hessian(func, paddle.ones([3]))

    def test_all_cases(self):
        if False:
            return 10
        self.setUpClass()
        self.func_single_input()
        self.func_multi_input()
        self.func_allow_unused_true()
        self.func_create_graph_true()
        self.func_out_not_single()

class TestHessianBatchFirst(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            print('Hello World!')
        self.x_shape = (5, 2)
        self.weight_shape = (2, 4)
        self.y_shape = (5, 2)
        (self.nbatch, self.nrow) = (5, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = config.TOLERANCE.get(self.dtype).get('second_order_grad').get('eps')
        self.rtol = config.TOLERANCE.get(self.dtype).get('second_order_grad').get('rtol')
        self.atol = config.TOLERANCE.get(self.dtype).get('second_order_grad').get('atol')
        self.x = paddle.rand(shape=self.x_shape, dtype=self.dtype)
        self.weight = paddle.rand(shape=self.weight_shape, dtype=self.dtype)
        self.y = paddle.rand(shape=self.y_shape, dtype=self.dtype)

    def func_single_input(self):
        if False:
            return 10

        def func(x):
            if False:
                return 10
            return paddle.matmul(x * x, self.weight)[:, 0:1]
        expected = utils._compute_numerical_batch_hessian(func, self.x, self.numerical_delta, self.np_dtype)
        H = paddle.incubate.autograd.Hessian(func, self.x, is_batched=True)
        actual = utils._np_transpose_matrix_format(H[:].numpy(), utils.MatrixFormat.BNM, utils.MatrixFormat.NBM)
        actual = actual.reshape((H.shape[1], -1))
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_multi_input(self):
        if False:
            i = 10
            return i + 15

        def func(x, y):
            if False:
                i = 10
                return i + 15
            return paddle.matmul(x * x * y * y, self.weight)[:, 0:1]
        xs_len = 2
        expected = utils._compute_numerical_batch_hessian(func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        expected = np.reshape(np.array(expected), (xs_len, xs_len, self.nrow, self.nbatch, self.nrow))
        expected = [list(row) for row in expected]
        expected = utils._np_concat_matrix_sequence(expected)
        self.x.stop_gradient = False
        self.y.stop_gradient = False
        H = paddle.incubate.autograd.Hessian(func, [self.x, self.y], is_batched=True)
        actual = utils._np_transpose_matrix_format(H[:].numpy(), utils.MatrixFormat.BNM, utils.MatrixFormat.NBM)
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_allow_unused(self):
        if False:
            i = 10
            return i + 15

        def func(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return paddle.matmul(x * x, self.weight)[:, 0:1]
        xs_len = 2
        expected = utils._compute_numerical_batch_hessian(func, [self.x, self.y], self.numerical_delta, self.np_dtype)
        expected = np.reshape(np.array(expected), (xs_len, xs_len, self.nrow, self.nbatch, self.nrow))
        expected = [list(row) for row in expected]
        expected = utils._np_concat_matrix_sequence(expected)
        expected = utils._np_transpose_matrix_format(expected, utils.MatrixFormat.NBM, utils.MatrixFormat.BNM)
        actual = paddle.incubate.autograd.Hessian(func, [self.x, self.y], is_batched=True)[:]
        np.testing.assert_allclose(actual, expected, rtol=self.rtol, atol=self.atol)

    def func_stop_gradient(self):
        if False:
            return 10

        def func(x):
            if False:
                return 10
            return paddle.matmul(x * x, self.weight)[:, 0:1]
        expected = utils._compute_numerical_batch_hessian(func, self.x, self.numerical_delta, self.np_dtype)
        x = self.x.clone()
        x.stop_gradient = True
        H = paddle.incubate.autograd.Hessian(func, self.x, is_batched=True)[:]
        actual = utils._np_transpose_matrix_format(H[:].numpy(), utils.MatrixFormat.BNM, utils.MatrixFormat.NBM)
        actual = actual.reshape((H.shape[1], -1))
        np.testing.assert_allclose(actual, expected, self.rtol, self.atol)

    def func_out_not_single(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                return 10
            return x * x
        with self.assertRaises(RuntimeError):
            paddle.incubate.autograd.Hessian(func, paddle.ones((3, 3)), is_batched=True)

    def test_all_cases(self):
        if False:
            while True:
                i = 10
        self.setUpClass()
        self.func_single_input()
        self.func_multi_input()
        self.func_allow_unused()
        self.func_stop_gradient()
        self.func_out_not_single()
if __name__ == '__main__':
    np.random.seed(2022)
    unittest.main()