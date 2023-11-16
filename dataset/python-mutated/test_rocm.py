import unittest
import os
import random
import math
import time
import numpy as np
import tqdm
import jittor as jt
from jittor import init, Module, nn, Function
from jittor.models import vgg
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
from .test_core import expect_error
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re

def test_rocm(use_rocm=1):
    if False:
        print('Hello World!')

    @unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
    class TestCudaBase(unittest.TestCase):

        def setUp(self):
            if False:
                return 10
            jt.flags.use_rocm = use_rocm

        def tearDown(self):
            if False:
                while True:
                    i = 10
            jt.flags.use_rocm = 0
    return TestCudaBase

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestROCm(unittest.TestCase):

    @jt.flag_scope(use_rocm=1)
    def test_array(self):
        if False:
            while True:
                i = 10
        a = jt.array([1, 2, 3])
        np.testing.assert_allclose(a.numpy(), [1, 2, 3])

    @jt.flag_scope(use_rocm=1)
    def test_add(self):
        if False:
            print('Hello World!')
        a = jt.array([1, 2, 3])
        b = a + a
        np.testing.assert_allclose(b.numpy(), [2, 4, 6])

    @jt.flag_scope(use_rocm=1)
    def test_add_float(self):
        if False:
            while True:
                i = 10
        a = jt.array([1.0, 2.0, 3.0])
        b = a + a
        np.testing.assert_allclose(b.numpy(), [2, 4, 6])

    @jt.flag_scope(use_rocm=1)
    def test_array_cast(self):
        if False:
            while True:
                i = 10
        x = np.random.rand(10)
        y = jt.float32(x)
        np.testing.assert_allclose(x, y.numpy())

    def test_meminfo(self):
        if False:
            i = 10
            return i + 15
        jt.display_memory_info()

    @jt.flag_scope(use_rocm=1)
    def test_cuda_flags(self):
        if False:
            i = 10
            return i + 15
        a = jt.random((10, 10))
        a.sync()

    @jt.flag_scope(use_rocm=1)
    def test_rocm_custom_op_from_cuda(self):
        if False:
            while True:
                i = 10
        my_op = jt.compile_custom_op('\n        struct MyCudaOp : Op {\n            Var* output;\n            MyCudaOp(NanoVector shape, string dtype="float");\n            \n            const char* name() const override { return "my_cuda"; }\n            DECLARE_jit_run;\n        };\n        ', '\n        #ifndef JIT\n        MyCudaOp::MyCudaOp(NanoVector shape, string dtype) {\n            flags.set(NodeFlags::_cuda);\n            output = create_output(shape, dtype);\n        }\n\n        void MyCudaOp::jit_prepare(JK& jk) {\n            add_jit_define(jk, "T", output->dtype());\n        }\n\n        #else // JIT\n        #ifdef JIT_cuda\n\n        __global__ void kernel(index_t n, T *x) {\n            int index = blockIdx.x * blockDim.x + threadIdx.x;\n            int stride = blockDim.x * gridDim.x;\n            for (int i = index; i < n; i += stride)\n                x[i] = (T)-i;\n        }\n\n        void MyCudaOp::jit_run() {\n            index_t num = output->num;\n            auto* __restrict__ x = output->ptr<T>();\n            int blockSize = 256;\n            int numBlocks = (num + blockSize - 1) / blockSize;\n            kernel<<<numBlocks, blockSize>>>(num, x);\n        }\n        #endif // JIT_cuda\n        #endif // JIT\n        ', 'my_cuda')
        a = my_op([3, 4, 5], 'float')
        na = a.data
        assert a.shape == [3, 4, 5] and a.dtype == 'float'
        assert (-na.flatten() == range(3 * 4 * 5)).all(), na

    def test_rocm_fused_op(self):
        if False:
            for i in range(10):
                print('nop')
        a = jt.array([1, 2, 3])
        a.sync()
        with jt.flag_scope(use_rocm=1):
            ((a + a) * 2).data

class Model(Module):

    def __init__(self, input_size):
        if False:
            return 10
        self.linear1 = nn.Linear(input_size, 10)
        self.relu1 = nn.Relu()
        self.linear2 = nn.Linear(10, 1)

    def execute(self, x):
        if False:
            return 10
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestExample(unittest.TestCase):

    @jt.flag_scope(use_rocm=1)
    def test1(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        jt.set_seed(3)
        n = 1000
        batch_size = 50
        lr = 0.05

        def get_data(n):
            if False:
                i = 10
                return i + 15
            for i in range(n):
                x = np.random.rand(batch_size, 1).astype('float32')
                y = x * x
                yield (jt.float32(x), jt.float32(y))
        model = Model(input_size=1)
        ps = model.parameters()
        for (i, (x, y)) in enumerate(get_data(n)):
            jt.sync_all(True)
            pred_y = model(x).name('pred_y')
            loss = (pred_y - y).sqr().name('loss')
            loss_mean = loss.mean()
            gs = jt.grad(loss_mean, ps)
            for (p, g) in zip(ps, gs):
                p -= g * lr
            if i > 2:
                assert prev == jt.liveness_info(), f'memory leak {prev} {jt.liveness_info()}'
            prev = jt.liveness_info()
        possible_results = [0.0009948202641680837, 0.001381353591568768, 0.00110957445576787, 0.001124994712881744]
        loss_mean = loss_mean.data
        assert any((abs(loss_mean - r) < 1e-06 for r in possible_results))
        jt.clean()
from .test_unary_op import TestUnaryOp

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestROCmUnaryOp(TestUnaryOp, test_rocm(1)):
    pass
from .test_binary_op import TestBinaryOp

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestROCmBinaryOp(TestBinaryOp, test_rocm(1)):
    pass
from .test_reduce_op import TestReduceOp

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestROCmReduceOp(TestReduceOp, test_rocm(1)):
    pass
from .test_reindex_op import TestReindexOp

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestROCmReindexOp(TestReindexOp, test_rocm(1)):
    pass
from .test_where_op import TestWhereOp

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestROCmWhereOp(TestWhereOp, test_rocm(1)):
    pass

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestROCmCodeOp(unittest.TestCase):

    @jt.flag_scope(use_rocm=1)
    def test_cuda(self):
        if False:
            while True:
                i = 10
        a = jt.random([100000])
        b = jt.random([100000])
        c = jt.code(a.shape, a.dtype, [a, b], cuda_src='\n            __global__ static void kernel1(@ARGS_DEF) {\n                @PRECALC\n                int i = threadIdx.x + blockIdx.x * blockDim.x;\n                int stride = blockDim.x * gridDim.x;\n                for (; i<in0_shape0; i+=stride)\n                    @out(i) = @in0(i)*@in1(i);\n            }\n                kernel1<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);\n            ', cuda_grad_src=['\n            __global__ static void kernel2(@ARGS_DEF) {\n                @PRECALC\n                int i = threadIdx.x + blockIdx.x * blockDim.x;\n                int stride = blockDim.x * gridDim.x;\n                for (; i<in0_shape0; i+=stride)\n                    @out(i) = @dout(i)*@in1(i);\n            }\n                kernel2<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);\n            ', '\n            __global__ static void kernel3(@ARGS_DEF) {\n                @PRECALC\n                int i = threadIdx.x + blockIdx.x * blockDim.x;\n                int stride = blockDim.x * gridDim.x;\n                for (; i<in0_shape0; i+=stride)\n                    @out(i) = @dout(i)*@in0(i);\n            }\n                kernel3<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);\n            '])
        (da, db) = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data * b.data), (c.data, a.data * b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    @jt.flag_scope(use_rocm=1)
    def test_cuda2(self):
        if False:
            while True:
                i = 10
        a = jt.random((100, 100))
        b = jt.random((100, 100))
        c = jt.code(a.shape, a.dtype, [a, b], cuda_src='\n                __global__ static void kernel1(@ARGS_DEF) {\n                    @PRECALC\n                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)\n                        @out(i,j) = @in0(i,j)*@in1(i,j);\n                }\n                kernel1<<<32, 32>>>(@ARGS);\n            ', cuda_grad_src=['\n                __global__ static void kernel(@ARGS_DEF) {\n                    @PRECALC\n                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)\n                        @out(i,j) = @dout(i,j)*@in1(i,j);\n                }\n                kernel<<<32, 32>>>(@ARGS);\n            ', '\n                __global__ static void kernel(@ARGS_DEF) {\n                    @PRECALC\n                    @pout(0,0);\n                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)\n                        @out(i,j) = @dout(i,j)*@in0(i,j);\n                }\n                kernel<<<32, 32>>>(@ARGS);\n            '])
        (da, db) = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data * b.data), (c.data, a.data * b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    @jt.flag_scope(use_rocm=1)
    def test_cuda2_use_func(self):
        if False:
            while True:
                i = 10

        class Func(Function):

            def execute(self, a, b):
                if False:
                    for i in range(10):
                        print('nop')
                self.save_vars = (a, b)
                return jt.code(a.shape, a.dtype, [a, b], cuda_src='\n                        __global__ static void kernel1(@ARGS_DEF) {\n                            @PRECALC\n                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)\n                                @out(i,j) = @in0(i,j)*@in1(i,j);\n                        }\n                        kernel1<<<32, 32>>>(@ARGS);\n                    ')

            def grad(self, grad):
                if False:
                    for i in range(10):
                        print('nop')
                (a, b) = self.save_vars
                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad], cuda_src='\n                        __global__ static void kernel2(@ARGS_DEF) {\n                            @PRECALC\n                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)\n                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x) {\n                                @out0(i,j) = @in2(i,j)*@in1(i,j);\n                                @out1(i,j) = @in2(i,j)*@in0(i,j);\n                            }\n                        }\n                        kernel2<<<32, 32>>>(@ARGS);\n                    ')
        a = jt.random((100, 100))
        b = jt.random((100, 100))
        func = Func()
        c = func(a, b)
        (da, db) = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data * b.data), (c.data, a.data * b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

@unittest.skipIf(not jt.compiler.has_rocm, 'No ROCm found')
class TestBMM(unittest.TestCase):

    def test_bmm_rocm(self):
        if False:
            print('Hello World!')

        def check(batch, n, m, k):
            if False:
                i = 10
                return i + 15

            def calc(use_rocm, a, b, mask):
                if False:
                    print('Hello World!')
                jt.flags.use_rocm = use_rocm
                a = jt.array(a)
                b = jt.array(b)
                mask = jt.array(mask)
                c = nn.bmm(a, b)
                (da, db) = jt.grad(c * mask, [a, b])
                return (c.data, da.data, db.data)
            mask = np.random.rand(batch, n, k).astype('float32')
            a = np.random.rand(batch, n, m).astype('float32')
            b = np.random.rand(batch, m, k).astype('float32')
            (a1, a2, a3) = calc(0, a, b, mask)
            (b1, b2, b3) = calc(1, a, b, mask)
            assert np.allclose(a1, b1)
            assert np.allclose(a2, b2)
            assert np.allclose(a3, b3)
        check(10, 3, 4, 5)
        check(10, 8, 8, 8)
        check(10, 8, 1, 8)
        check(10, 8, 8, 1)
        check(10, 1, 8, 8)
        check(1, 7, 8, 8)

class Model(Module):

    def __init__(self, input_size):
        if False:
            while True:
                i = 10
        self.linear1 = nn.Linear(input_size, 10)
        self.relu1 = nn.Relu()
        self.linear2 = nn.Linear(10, 1)

    def execute(self, x):
        if False:
            return 10
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)
from jittor.models import resnet

class MnistNet(Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.model = resnet.Resnet18()
        self.layer = nn.Linear(1000, 10)

    def execute(self, x):
        if False:
            i = 10
            return i + 15
        x = self.model(x)
        x = self.layer(x)
        return x

@unittest.skipIf(not jt.compiler.has_rocm, 'skip_this_test')
class TestResnetFp32(unittest.TestCase):

    def setup_seed(self, seed):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)

    @jt.flag_scope(use_cuda=1)
    def test_resnet(self):
        if False:
            return 10
        self.setup_seed(1)
        self.batch_size = int(os.environ.get('TEST_BATCH_SIZE', '100'))
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.learning_rate = 0.1
        if jt.flags.amp_reg:
            self.learning_rate = 0.01
        self.train_loader = MNIST(train=True, transform=trans.Resize(224)).set_attrs(batch_size=self.batch_size, shuffle=True)
        self.train_loader.num_workers = 4
        loss_list = []
        acc_list = []
        mnist_net = MnistNet()
        global prev
        SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)
        self.train_loader.endless = True
        for (data, target) in self.train_loader:
            batch_id = self.train_loader.batch_id
            epoch_id = self.train_loader.epoch_id
            data = data.float_auto()
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            break
        jt.sync_all(True)
        for _ in range(10):
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            SGD.step(loss)

            def callback(epoch_id, batch_id, loss, output, target):
                if False:
                    i = 10
                    return i + 15
                pred = np.argmax(output, axis=1)
                acc = np.mean(target == pred)
            jt.fetch(epoch_id, _, loss, output, target, callback)
        jt.sync_all(True)
        all_time = time.time()
        prev = time.time()
        print('starting')
        for _ in range(100):
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            SGD.step(loss)

            def callback(epoch_id, batch_id, loss, output, target):
                if False:
                    for i in range(10):
                        print('nop')
                global prev
                pred = np.argmax(output, axis=1)
                acc = np.mean(target == pred)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f} \tTime:{:.3f}'.format(epoch_id, batch_id, 600, 1.0 * batch_id / 6.0, loss[0], acc, time.time() - prev))
                prev = time.time()
            jt.fetch(epoch_id, _, loss, output, target, callback)
        jt.sync_all(True)
        print(f'all = {time.time() - all_time}')
if __name__ == '__main__':
    unittest.main()