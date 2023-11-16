import unittest
import numpy as np
import paddle
from paddle import base
from paddle.nn import Embedding
from paddle.tensor import random

class AutoPruneLayer0(paddle.nn.Layer):

    def __init__(self, input_size):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear1 = paddle.nn.Linear(input_size, 5, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=2)), bias_attr=False)
        self.linear2 = paddle.nn.Linear(5, 5, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=2)), bias_attr=False)

    def forward(self, x, y):
        if False:
            print('Hello World!')
        a = self.linear1(x)
        b = self.linear2(y)
        c = paddle.matmul(a, b)
        d = paddle.mean(c)
        return d

class AutoPruneLayer1(paddle.nn.Layer):

    def __init__(self, input_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self.linear1 = paddle.nn.Linear(input_size, 5, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=2)), bias_attr=False)
        self.linear2 = paddle.nn.Linear(5, 5, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=2)), bias_attr=False)

    def forward(self, x, y):
        if False:
            return 10
        a = self.linear1(x)
        b = self.linear2(y)
        b.stop_gradient = True
        c = paddle.matmul(a, b)
        d = paddle.mean(c)
        return d

class AutoPruneLayer2(paddle.nn.Layer):

    def __init__(self, input_size):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear = paddle.nn.Linear(input_size, 10)
        self.linear2 = paddle.nn.Linear(1, 1)

    def forward(self, x, label):
        if False:
            i = 10
            return i + 15
        feature = self.linear(x)
        label = self.linear2(label)
        label = paddle.cast(label, dtype='float32')
        label = paddle.cast(label, dtype='int64')
        loss = paddle.nn.functional.cross_entropy(input=feature, label=label, reduction='none', use_softmax=False)
        loss = paddle.mean(loss)
        return loss

class AutoPruneLayer3(paddle.nn.Layer):

    def __init__(self, input_size):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear = paddle.nn.Linear(input_size, 20)

    def forward(self, x, label, test_num):
        if False:
            for i in range(10):
                print('nop')
        feature = self.linear(x)
        (part1, part2) = paddle.split(feature, num_or_sections=[10, 10], axis=1)
        loss = paddle.nn.functional.cross_entropy(input=part1, label=label, reduction='none', use_softmax=False)
        loss = paddle.mean(loss)
        if test_num == 1:
            return (loss, part2)
        else:
            return (loss, part1, part2)

class MyLayer(paddle.nn.Layer):

    def __init__(self, input_size, vocab_size, size, dtype='float32'):
        if False:
            i = 10
            return i + 15
        super().__init__(dtype=dtype)
        self.embed0 = Embedding(vocab_size, size)
        self.embed1 = Embedding(vocab_size, size)
        self.linear_0 = paddle.nn.Linear(input_size, size)
        self.linear_1 = paddle.nn.Linear(input_size, size)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        loss = paddle.mean(self.linear_0(x) + self.linear_1(x))
        return loss

    def linear0(self, x):
        if False:
            while True:
                i = 10
        loss = paddle.mean(self.linear_0(x))
        return loss

    def embed_linear0(self, x):
        if False:
            i = 10
            return i + 15
        loss = paddle.mean(self.linear_0(self.embed0(x)))
        return loss

class MyLayer2(paddle.nn.Layer):

    def __init__(self, input_size, vocab_size, size, dtype='float32'):
        if False:
            return 10
        super().__init__(dtype=dtype)
        self.embed0 = Embedding(vocab_size, size)
        self.embed1 = Embedding(vocab_size, size)
        self.linear_0 = paddle.nn.Linear(input_size, size)
        self.linear_1 = paddle.nn.Linear(input_size, size)

    def forward(self, indices):
        if False:
            i = 10
            return i + 15
        loss = paddle.mean(self.linear_0(self.embed0(indices)) + self.linear_1(self.embed1(indices)))
        return loss

    def linear0(self, x):
        if False:
            return 10
        loss = paddle.mean(self.linear_0(x))
        return loss

    def embed_linear0(self, x):
        if False:
            for i in range(10):
                print('nop')
        loss = paddle.mean(self.linear_0(self.embed0(x)))
        return loss

class TestImperativeAutoPrune(unittest.TestCase):

    def test_auto_prune(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            case1 = AutoPruneLayer0(input_size=5)
            value1 = np.arange(25).reshape(5, 5).astype('float32')
            value2 = np.arange(25).reshape(5, 5).astype('float32')
            v1 = base.dygraph.to_variable(value1)
            v2 = base.dygraph.to_variable(value2)
            loss = case1(v1, v2)
            loss.backward()
            self.assertIsNotNone(case1.linear2.weight._grad_ivar())
            self.assertIsNotNone(case1.linear1.weight._grad_ivar())

    def test_auto_prune2(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            case2 = AutoPruneLayer1(input_size=5)
            value1 = np.arange(25).reshape(5, 5).astype('float32')
            value2 = np.arange(25).reshape(5, 5).astype('float32')
            v1 = base.dygraph.to_variable(value1)
            v2 = base.dygraph.to_variable(value2)
            loss = case2(v1, v2)
            loss.backward()
            self.assertIsNone(case2.linear2.weight._grad_ivar())
            self.assertIsNotNone(case2.linear1.weight._grad_ivar())

    def test_auto_prune3(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            case3 = AutoPruneLayer3(input_size=784)
            value1 = np.arange(784).reshape(1, 784).astype('float32')
            value2 = np.arange(1).reshape(1, 1).astype('int64')
            v1 = base.dygraph.to_variable(value1)
            v2 = base.dygraph.to_variable(value2)
            (loss, part2) = case3(v1, v2, 1)
            part2.retain_grads()
            loss.backward()
            self.assertIsNotNone(case3.linear.weight._grad_ivar())
            self.assertTrue((part2.gradient() == 0).all())

    def test_auto_prune4(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            case4 = AutoPruneLayer3(input_size=784)
            value1 = np.arange(784).reshape(1, 784).astype('float32')
            value2 = np.arange(1).reshape(1, 1).astype('int64')
            v1 = base.dygraph.to_variable(value1)
            v2 = base.dygraph.to_variable(value2)
            (loss, part2) = case4(v1, v2, 1)
            part2.retain_grads()
            part2.backward()
            self.assertIsNotNone(case4.linear.weight._grad_ivar())
            self.assertTrue((part2.gradient() == 1).all())

    def test_auto_prune5(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            case4 = AutoPruneLayer3(input_size=784)
            value1 = np.arange(784).reshape(1, 784).astype('float32')
            value2 = np.arange(1).reshape(1, 1).astype('int64')
            v1 = base.dygraph.to_variable(value1)
            v2 = base.dygraph.to_variable(value2)
            (loss, part1, part2) = case4(v1, v2, 2)
            part2.retain_grads()
            part1.backward()
            self.assertIsNotNone(case4.linear.weight._grad_ivar())
            self.assertTrue((part2.gradient() == 0).all())

    def test_auto_prune6(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype('float32')
            value1 = np.arange(6).reshape(2, 3).astype('float32')
            value2 = np.arange(10).reshape(2, 5).astype('float32')
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(3, 3)
            a = base.dygraph.to_variable(value0)
            b = base.dygraph.to_variable(value1)
            c = base.dygraph.to_variable(value2)
            out1 = linear(a)
            out2 = linear2(b)
            out1.stop_gradient = True
            out = paddle.concat([out1, out2, c], axis=1)
            out.backward()
            self.assertIsNone(linear.weight.gradient())
            self.assertIsNone(out1.gradient())

    def test_auto_prune7(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype('float32')
            value1 = np.arange(6).reshape(2, 3).astype('float32')
            value2 = np.arange(10).reshape(2, 5).astype('float32')
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(3, 3)
            a = base.dygraph.to_variable(value0)
            b = base.dygraph.to_variable(value1)
            c = base.dygraph.to_variable(value2)
            out1 = linear(a)
            out2 = linear2(b)
            out1.stop_gradient = True
            out = paddle.concat([out1, out2, c], axis=1)
            out.backward()
            self.assertIsNone(linear.weight.gradient())
            self.assertIsNone(out1.gradient())

    def test_auto_prune8(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype('float32')
            value1 = np.arange(6).reshape(2, 3).astype('float32')
            value2 = np.arange(10).reshape(2, 5).astype('float32')
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(5, 3)
            a = base.dygraph.to_variable(value0)
            b = base.dygraph.to_variable(value1)
            c = base.dygraph.to_variable(value2)
            out1 = linear(a)
            linear_origin = linear.weight.numpy()
            out2 = linear2(out1)
            linear2_origin = linear2.weight.numpy()
            linear2.weight.stop_gradient = True
            out2.backward()
            optimizer = paddle.optimizer.SGD(learning_rate=0.003, parameters=linear.parameters() + linear2.parameters())
            optimizer.minimize(out2)
            np.testing.assert_array_equal(linear2_origin, linear2.weight.numpy())
            self.assertFalse(np.array_equal(linear_origin, linear.weight.numpy()))

    def test_auto_prune9(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype('float32')
            value1 = np.arange(6).reshape(2, 3).astype('float32')
            value2 = np.arange(10).reshape(2, 5).astype('float32')
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(5, 3)
            a = base.dygraph.to_variable(value0)
            b = base.dygraph.to_variable(value1)
            c = base.dygraph.to_variable(value2)
            out1 = linear(a)
            linear_origin = linear.weight.numpy()
            out2 = linear2(out1)
            linear2_origin = linear2.weight.numpy()
            out2.stop_gradient = True
            out2.backward()
            optimizer = paddle.optimizer.SGD(learning_rate=0.003, parameters=linear.parameters() + linear2.parameters())
            optimizer.minimize(out2)
            np.testing.assert_array_equal(linear2_origin, linear2.weight.numpy())
            np.testing.assert_array_equal(linear_origin, linear.weight.numpy())
            try:
                linear2.weight.gradient()
            except ValueError as e:
                assert type(e) == ValueError

    def test_auto_prune10(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype('float32')
            value1 = np.arange(6).reshape(2, 3).astype('float32')
            value2 = np.arange(10).reshape(2, 5).astype('float32')
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(3, 3)
            a = base.dygraph.to_variable(value0)
            b = base.dygraph.to_variable(value1)
            c = base.dygraph.to_variable(value2)
            out1 = linear(a)
            out2 = linear2(b)
            out1.stop_gradient = True
            out = paddle.concat([out1, out2, c], axis=1)
            base.set_flags({'FLAGS_sort_sum_gradient': True})
            out.backward()
            self.assertIsNone(linear.weight.gradient())
            self.assertIsNone(out1.gradient())

    def test_auto_prune_with_optimizer(self):
        if False:
            return 10
        vocab_size = 100
        size = 20
        batch_size = 16
        indices = np.random.randint(low=0, high=100, size=(batch_size, 1)).astype('int64')
        embed = np.random.randn(batch_size, size).astype('float32')
        place = base.CPUPlace()
        with base.dygraph.guard(place):
            model = MyLayer(size, vocab_size, size)
            grad_clip = paddle.nn.ClipGradByGlobalNorm(0.001)
            optimizer = paddle.optimizer.Adam(0.001, parameters=model.parameters(), grad_clip=grad_clip)
            indices = base.dygraph.to_variable(indices)
            embed = base.dygraph.to_variable(embed)
            dummy_loss = model(embed)
            loss = model.embed_linear0(indices)
            loss.backward()
            (_, params_grads) = optimizer.minimize(loss)
            for (items_0, *items_len) in params_grads:
                assert items_0.name is not model.embed1.weight.name
                assert items_0.name is not model.linear_1.weight.name
            assert model.embed1.weight._grad_ivar() is None
            assert model.linear_1.weight._grad_ivar() is None
        with base.dygraph.guard(place):
            model = MyLayer2(size, vocab_size, size)
            grad_clip = paddle.nn.ClipGradByGlobalNorm(0.001)
            optimizer = paddle.optimizer.Adam(0.001, parameters=model.parameters(), grad_clip=grad_clip)
            indices = base.dygraph.to_variable(indices)
            emebd = base.dygraph.to_variable(embed)
            dummy_loss = model(indices)
            loss = model.embed_linear0(indices)
            loss.backward()
            optimizer.minimize(loss)
            for items in params_grads:
                assert items[0].name is not model.embed1.weight.name
                assert items[0].name is not model.linear_1.weight.name
            assert model.embed1.weight._grad_ivar() is None
            assert model.linear_1.weight._grad_ivar() is None

    def test_case2_prune_no_grad_branch(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            value1 = np.arange(784).reshape(1, 784)
            value2 = np.arange(1).reshape(1, 1)
            v1 = base.dygraph.to_variable(value1).astype('float32')
            v2 = base.dygraph.to_variable(value2).astype('float32')
            case3 = AutoPruneLayer2(input_size=784)
            loss = case3(v1, v2)
            loss.backward()
            self.assertIsNone(case3.linear2.weight._grad_ivar())
            self.assertIsNotNone(case3.linear.weight._grad_ivar())

    def test_case3_prune_no_grad_branch2(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            value1 = np.arange(1).reshape(1, 1)
            linear = paddle.nn.Linear(1, 1)
            label = base.dygraph.to_variable(value1).astype('float32')
            label = linear(label)
            label = paddle.cast(label, dtype='float32')
            label = paddle.cast(label, dtype='int64')
            out = paddle.nn.functional.one_hot(label, 100)
            loss = paddle.mean(out)
            loss.backward()
            self.assertIsNone(linear.weight._grad_ivar())

    def test_case4_with_no_grad_op_maker(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            out = random.gaussian(shape=[20, 30])
            loss = paddle.mean(out)
            loss.backward()
            self.assertIsNone(out._grad_ivar())
if __name__ == '__main__':
    unittest.main()