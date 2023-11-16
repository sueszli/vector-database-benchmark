import os
import tempfile
import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, compare_legacy_with_pir
import paddle

def forward_post_hook1(layer, input, output):
    if False:
        for i in range(10):
            print('nop')
    return output + output

def forward_pre_hook1(layer, input):
    if False:
        print('Hello World!')
    input_return = (input[0] * 2,)
    return input_return

class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.fc1 = paddle.nn.Linear(10, 10)
        self.fc1.register_forward_post_hook(forward_post_hook1)
        self.fc2 = paddle.nn.Linear(10, 10)
        self.fc2.register_forward_pre_hook(forward_pre_hook1)
        self.register_forward_pre_hook(forward_pre_hook1)
        self.register_forward_post_hook(forward_post_hook1)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.fc1(x)
        x = self.fc2(x)
        out = paddle.mean(x)
        return out

class TestNestLayerHook(Dy2StTestBase):

    def setUp(self):
        if False:
            return 10
        paddle.seed(2022)
        self.x = paddle.randn([4, 10])
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'net_hook')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    @compare_legacy_with_pir
    def train_net(self, to_static=False):
        if False:
            i = 10
            return i + 15
        paddle.seed(2022)
        net = SimpleNet()
        if to_static:
            net = paddle.jit.to_static(net)
        out = net(self.x)
        if to_static:
            paddle.jit.save(net, self.path)
        return float(out)

    def load_train(self):
        if False:
            i = 10
            return i + 15
        net = paddle.jit.load(self.path)
        out = net(self.x)
        return float(out)

    def test_hook(self):
        if False:
            return 10
        dy_out = self.train_net(to_static=False)
        st_out = self.train_net(to_static=True)
        load_out = self.load_train()
        print(st_out, dy_out, load_out)
        np.testing.assert_allclose(st_out, dy_out, rtol=1e-05, err_msg=f'dygraph_res is {dy_out}\nstatic_res is {st_out}')
        np.testing.assert_allclose(st_out, load_out, rtol=1e-05, err_msg=f'load_out is {load_out}\nstatic_res is {st_out}')
if __name__ == '__main__':
    unittest.main()