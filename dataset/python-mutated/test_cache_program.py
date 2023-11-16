import unittest
from collections import Counter
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase
from test_fetch_feed import Linear, Pool2D
import paddle
from paddle import base
from paddle.jit.api import to_static
from paddle.jit.dy2static import convert_to_static

class TestCacheProgram(Dy2StTestBase):

    def setUp(self):
        if False:
            return 10
        self.batch_num = 5
        self.dygraph_class = Pool2D
        self.data = np.random.random((1, 2, 4, 4)).astype('float32')

    def test_cache(self):
        if False:
            for i in range(10):
                print('nop')
        (prev_ops, cur_ops) = (Counter(), Counter())
        (prev_out, cur_out) = (None, None)
        with base.dygraph.guard(base.CPUPlace()):
            static_net = self.dygraph_class()
            for batch_id in range(self.batch_num):
                out = static_net(paddle.to_tensor(self.data))
                prev_out = cur_out
                cur_out = out
                prev_ops = cur_ops
                cur_ops = Counter([op.type for op in base.default_main_program().block(0).ops])
                if batch_id > 0:
                    prev_out_numpy = prev_out[0].numpy() if isinstance(prev_out, (tuple, list)) else prev_out.numpy()
                    cur_out_numpy = cur_out[0].numpy() if isinstance(cur_out, (tuple, list)) else cur_out.numpy()
                    np.testing.assert_allclose(prev_out_numpy, cur_out_numpy, rtol=1e-05, err_msg='Output in previous batch is {}\n Output in current batch is \n{}'.format(prev_out_numpy, cur_out_numpy))
                    self.assertEqual(prev_ops, cur_ops)

class TestCacheProgram2(TestCacheProgram):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_num = 5
        self.dygraph_class = Linear
        self.data = np.random.random((4, 10)).astype('float32')

class TestCacheProgramWithOptimizer(Dy2StTestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dygraph_class = Linear
        self.data = np.random.random((4, 10)).astype('float32')
        self.batch_num = 5

    def train_static(self):
        if False:
            while True:
                i = 10
        return self.train(to_static=True)

    def train_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        return self.train(to_static=False)

    def train(self, to_static=False):
        if False:
            print('Hello World!')
        paddle.jit.enable_to_static(to_static)
        with base.dygraph.guard(base.CPUPlace()):
            dygraph_net = self.dygraph_class()
            adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=dygraph_net.parameters())
            loss_data = []
            for batch_id in range(self.batch_num):
                input = base.dygraph.to_variable(self.data)
                (pred, avg_loss) = dygraph_net(input)
                loss_data.append(avg_loss.numpy())
                avg_loss.backward()
                adam.minimize(avg_loss)
                dygraph_net.clear_gradients()
        return loss_data

    def test_with_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        dygraph_loss = self.train_dygraph()
        static_loss = self.train_static()
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05, err_msg=f'dygraph is {dygraph_loss}\n static_res is \n{static_loss}')

def simple_func(x):
    if False:
        i = 10
        return i + 15
    inputs = base.dygraph.to_variable(x)
    mean = paddle.mean(inputs)
    return mean

class TestConvertWithCache(Dy2StTestBase):

    def test_cache(self):
        if False:
            for i in range(10):
                print('nop')
        static_func = convert_to_static(simple_func)
        cached_func = convert_to_static(simple_func)
        self.assertTrue(id(static_func), id(cached_func))

@to_static
def sum_even_until_limit(max_len, limit):
    if False:
        i = 10
        return i + 15
    ret_sum = base.dygraph.to_variable(np.zeros(1).astype('int32'))
    for i in range(max_len):
        if i % 2 > 0:
            continue
        elif i > limit:
            break
        ret_sum += i
    return ret_sum

def sum_under_while(limit):
    if False:
        for i in range(10):
            print('nop')
    i = base.dygraph.to_variable(np.zeros(1).astype('int32'))
    ret_sum = base.dygraph.to_variable(np.zeros(1).astype('int32'))
    while i <= limit:
        ret_sum += i
        i += 1
    return ret_sum

class TestToOutputWithCache(Dy2StTestBase):

    def test_output(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            ret = sum_even_until_limit(80, 10)
            self.assertEqual(ret.numpy(), 30)
            ret = to_static(sum_under_while)(100)
            self.assertEqual(ret.numpy(), 5050)
if __name__ == '__main__':
    unittest.main()