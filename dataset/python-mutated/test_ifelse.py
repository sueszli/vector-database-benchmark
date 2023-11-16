import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir
from ifelse_simple_func import NetWithControlFlowIf, add_fn, base, dyfunc_empty_nonlocal, dyfunc_ifelse_ret_int1, dyfunc_ifelse_ret_int2, dyfunc_ifelse_ret_int3, dyfunc_ifelse_ret_int4, dyfunc_with_if_else, dyfunc_with_if_else2, dyfunc_with_if_else3, dyfunc_with_if_else_with_list_generator, if_tensor_case, if_with_and_or, if_with_and_or_1, if_with_and_or_2, if_with_and_or_3, if_with_and_or_4, if_with_class_var, loss_fn, nested_if_else, nested_if_else_2, nested_if_else_3
import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.jit.dy2static.utils import Dygraph2StaticException
np.random.seed(1)
if base.is_compiled_with_cuda():
    place = base.CUDAPlace(0)
else:
    place = base.CPUPlace()

class TestDy2staticException(Dy2StTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = None
        self.error = 'Your if/else have different number of return value.'

    @test_ast_only
    @test_legacy_and_pir
    def test_error(self):
        if False:
            return 10
        if self.dyfunc:
            with self.assertRaisesRegex(Dygraph2StaticException, self.error):
                paddle.jit.enable_to_static(True)
                self.assertTrue(paddle.jit.to_static(self.dyfunc)(self.x))
        paddle.base.dygraph.base.global_var._in_to_static_mode_ = False
        paddle.jit.enable_to_static(False)

class TestDygraphIfElse(Dy2StTestBase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `base.layers.cond`.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else

    def _run_static(self):
        if False:
            while True:
                i = 10
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard(place):
            x_v = base.dygraph.to_variable(self.x)
            if to_static:
                ret = paddle.jit.to_static(self.dyfunc)(x_v)
            else:
                ret = self.dyfunc(x_v)
            return ret.numpy()

    @test_legacy_and_pir
    def test_ast_to_func(self):
        if False:
            while True:
                i = 10
        self.assertTrue((self._run_dygraph() == self._run_static()).all())

class TestDygraphIfElse2(TestDygraphIfElse):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else2

class TestDygraphIfElse3(TestDygraphIfElse):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else3

class TestDygraphIfElse4(TestDygraphIfElse):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_empty_nonlocal

class TestDygraphIfElseWithListGenerator(TestDygraphIfElse):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else_with_list_generator

class TestDygraphNestedIfElse(Dy2StTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else

    def _run_static(self):
        if False:
            while True:
                i = 10
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        if False:
            while True:
                i = 10
        with base.dygraph.guard(place):
            x_v = paddle.to_tensor(self.x)
            if to_static:
                ret = paddle.jit.to_static(self.dyfunc)(x_v)
            else:
                ret = self.dyfunc(x_v)
            return ret.numpy()

    @test_legacy_and_pir
    def test_ast_to_func(self):
        if False:
            print('Hello World!')
        self.assertTrue((self._run_dygraph() == self._run_static()).all())

class TestDygraphNestedIfElse2(TestDygraphIfElse):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else_2

class TestDygraphNestedIfElse3(TestDygraphIfElse):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else_3

def dyfunc_ifExp_with_while(x):
    if False:
        while True:
            i = 10
    y = [x]

    def add_fn(x):
        if False:
            i = 10
            return i + 15
        x = x + 1
        return x

    def cond(i, ten, y):
        if False:
            return 10
        return i < ten

    def map_func(func, tensor_list):
        if False:
            while True:
                i = 10
        return [func(x) for x in tensor_list]

    def body(i, ten, y):
        if False:
            return 10
        y = map_func(lambda x: x if (i == 0) is not None else add_fn(x), y)
        i += 1
        return [i, ten, y]
    i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    ten = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=10)
    (i, ten, y) = paddle.static.nn.while_loop(cond, body, [i, ten, y])
    return y[0]

class TestDygraphIfElse6(TestDygraphIfElse):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_ifExp_with_while

def dyfunc_ifExp(x):
    if False:
        return 10
    y = [x]

    def add_fn(x):
        if False:
            print('Hello World!')
        x = x + 1
        return x

    def map_func(func, tensor_list):
        if False:
            return 10
        return [func(x) for x in tensor_list]
    i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    y = map_func(lambda x: x if i == 1 else add_fn(x), y)
    return y[0]

class TestDygraphIfElse7(TestDygraphIfElse):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_ifExp

class TestDygraphIfElseWithAndOr(TestDygraphIfElse):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or

class TestDygraphIfElseWithAndOr1(TestDygraphIfElse):

    def setUp(self):
        if False:
            return 10
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_1

class TestDygraphIfElseWithAndOr2(TestDygraphIfElse):

    def setUp(self):
        if False:
            return 10
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_2

class TestDygraphIfElseWithAndOr3(TestDygraphIfElse):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_3

class TestDygraphIfElseWithAndOr4(TestDygraphIfElse):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_4

class TestDygraphIfElseWithClassVar(TestDygraphIfElse):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_class_var

class TestDygraphIfTensor(Dy2StTestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_tensor_case

    def _run_static(self):
        if False:
            print('Hello World!')
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        if False:
            print('Hello World!')
        with base.dygraph.guard(place):
            x_v = paddle.to_tensor(self.x)
            if to_static:
                ret = paddle.jit.to_static(self.dyfunc)(x_v)
            else:
                ret = self.dyfunc(x_v)
            return ret.numpy()

    @test_legacy_and_pir
    def test_ast_to_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue((self._run_dygraph() == self._run_static()).all())

class TestDygraphIfElseNet(Dy2StTestBase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `base.layers.cond`.
    """

    def setUp(self):
        if False:
            return 10
        self.x = np.random.random([10, 16]).astype('float32')
        self.Net = NetWithControlFlowIf

    def _run_static(self):
        if False:
            print('Hello World!')
        return self._run(to_static=True)

    def _run_dygraph(self):
        if False:
            while True:
                i = 10
        return self._run(to_static=False)

    def _run(self, to_static=False):
        if False:
            return 10
        paddle.jit.enable_to_static(to_static)
        with base.dygraph.guard(place):
            net = self.Net()
            x_v = base.dygraph.to_variable(self.x)
            ret = net(x_v)
            return ret.numpy()

    @test_legacy_and_pir
    def test_ast_to_func(self):
        if False:
            print('Hello World!')
        self.assertTrue((self._run_dygraph() == self._run_static()).all())

def relu(x):
    if False:
        print('Hello World!')
    return F.relu(x)

def call_external_func(x, label=None):
    if False:
        i = 10
        return i + 15
    if paddle.mean(x) < 0:
        x_v = x - 1
    else:
        x_v = add_fn(x)
    x_v = relu(x_v)
    if label is not None:
        loss = loss_fn(x_v, label)
        return loss
    return x_v

class TestAst2FuncWithExternalFunc(TestDygraphIfElse):

    def setUp(self):
        if False:
            return 10
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = call_external_func

class NetWithExternalFunc(paddle.nn.Layer):

    @paddle.jit.to_static
    def forward(self, x, label=None):
        if False:
            i = 10
            return i + 15
        if paddle.mean(x) < 0:
            x_v = x - 1
        else:
            x_v = add_fn(x)
        x_v = softmax(x_v)
        if label is not None:
            loss = loss_fn(x_v, label)
            return loss
        return x_v

def softmax(x):
    if False:
        for i in range(10):
            print('nop')
    return paddle.nn.functional.softmax(x)

class TestNetWithExternalFunc(TestDygraphIfElseNet):

    def setUp(self):
        if False:
            return 10
        self.x = np.random.random([10, 16]).astype('float32')
        self.Net = NetWithExternalFunc

    @test_legacy_and_pir
    def test_ast_to_func(self):
        if False:
            while True:
                i = 10
        self.assertTrue((self._run_dygraph() == self._run_static()).all())

class DiffModeNet1(paddle.nn.Layer):

    def __init__(self, mode):
        if False:
            return 10
        super().__init__()
        self.mode = mode

    @paddle.jit.to_static
    def forward(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        if self.mode == 'train':
            out = x + y
        elif self.mode == 'infer':
            out = x - y
        else:
            raise ValueError('Illegal mode')
        return out

class DiffModeNet2(paddle.nn.Layer):

    def __init__(self, mode):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.mode = mode

    @paddle.jit.to_static
    def forward(self, x, y):
        if False:
            while True:
                i = 10
        if self.mode == 'train':
            out = x + y
            return out
        elif self.mode == 'infer':
            out = x - y
            return out
        else:
            raise ValueError('Illegal mode')

class TestDiffModeNet(Dy2StTestBase):
    """
    TestCase for the net with different modes
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = paddle.randn([10, 16], 'float32')
        self.y = paddle.randn([10, 16], 'float32')
        self.init_net()

    def init_net(self):
        if False:
            return 10
        self.Net = DiffModeNet1

    def _run(self, mode, to_static):
        if False:
            return 10
        paddle.jit.enable_to_static(to_static)
        net = self.Net(mode)
        ret = net(self.x, self.y)
        return ret.numpy()

    @test_legacy_and_pir
    def test_train_mode(self):
        if False:
            print('Hello World!')
        self.assertTrue((self._run(mode='train', to_static=True) == self._run(mode='train', to_static=False)).all())

    @test_legacy_and_pir
    def test_infer_mode(self):
        if False:
            print('Hello World!')
        self.assertTrue((self._run(mode='infer', to_static=True) == self._run(mode='infer', to_static=False)).all())

class TestDiffModeNet2(TestDiffModeNet):

    def init_net(self):
        if False:
            for i in range(10):
                print('nop')
        self.Net = DiffModeNet2

class TestNewVarCreateInOneBranch(Dy2StTestBase):

    @test_legacy_and_pir
    def test_var_used_in_another_for(self):
        if False:
            return 10

        def case_func(training):
            if False:
                while True:
                    i = 10
            if training:
                targets = [1, 2, 3]
                targets_list = [targets]
            num_step = 3
            for i in range(num_step):
                if i > 0:
                    (rois, rosi_num) = (1, 2)
                    if training:
                        (ros, rosi_num, targets) = (-1, -2, [-1, -2, -3])
                        targets_list.append(targets)
            return rosi_num
        self.assertEqual(paddle.jit.to_static(case_func)(False), 2)
        self.assertEqual(paddle.jit.to_static(case_func)(True), -2)

class TestDy2StIfElseRetInt1(Dy2StTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.random([5]).astype('float32')
        self.dyfunc = paddle.jit.to_static(dyfunc_ifelse_ret_int1)
        self.out = self.get_dy2stat_out()

    def get_dy2stat_out(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.jit.enable_to_static(True)
        static_func = paddle.jit.to_static(self.dyfunc)
        out = static_func(self.x)
        paddle.jit.enable_to_static(False)
        return out

    @test_ast_only
    @test_legacy_and_pir
    def test_ast_to_func(self):
        if False:
            print('Hello World!')
        self.setUp()
        self.assertIsInstance(self.out[0], (paddle.Tensor, core.eager.Tensor))
        self.assertIsInstance(self.out[1], int)

class TestDy2StIfElseRetInt2(TestDy2staticException):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.random([5]).astype('float32')
        self.error = 'Your if/else have different number of return value.'
        self.dyfunc = dyfunc_ifelse_ret_int2

class TestDy2StIfElseRetInt3(TestDy2StIfElseRetInt1):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.random([5]).astype('float32')
        self.dyfunc = paddle.jit.to_static(dyfunc_ifelse_ret_int3)
        self.out = self.get_dy2stat_out()

    @test_ast_only
    @test_legacy_and_pir
    def test_ast_to_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUp()
        self.assertIsInstance(self.out, (paddle.Tensor, core.eager.Tensor))

class TestDy2StIfElseRetInt4(TestDy2StIfElseRetInt1):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = np.random.random([5]).astype('float32')
        self.dyfunc = paddle.jit.to_static(dyfunc_ifelse_ret_int4)

    @test_ast_only
    @test_legacy_and_pir
    def test_ast_to_func(self):
        if False:
            i = 10
            return i + 15
        paddle.jit.enable_to_static(True)
        with self.assertRaises(Dygraph2StaticException):
            static_func = paddle.jit.to_static(self.dyfunc)
            out = static_func(self.x)
        paddle.base.dygraph.base.global_var._in_to_static_mode_ = False
        paddle.jit.enable_to_static(False)

class IfElseNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.param = self.create_parameter(shape=[3, 2], dtype='float32', is_bias=False)

    @paddle.jit.to_static
    def forward(self, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        a = paddle.matmul(a, self.param)
        a = paddle.reshape(a, (2, 4))
        cond = paddle.to_tensor([10])
        if cond == 10:
            a_argmax = a.argmax(axis=-1)
            b = b + self.param
        else:
            print(c)
        return b

class TestDy2StIfElseBackward(Dy2StTestBase):

    def test_run_backward(self):
        if False:
            return 10
        a = paddle.randn((4, 3), dtype='float32')
        a.stop_gradient = False
        b = paddle.to_tensor([10]).astype('float32')
        b.stop_gradient = False
        c = paddle.to_tensor([2])
        c.stop_gradient = False
        net = IfElseNet()
        net.train()
        out = net(a, b, c)
        out.backward()
        np.testing.assert_allclose((b + net.param).numpy(), out.numpy(), rtol=1e-05)
if __name__ == '__main__':
    unittest.main()