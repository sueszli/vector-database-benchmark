import random
import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, compare_legacy_with_pir
import paddle
from paddle.jit.api import to_static
SEED = 102
random.seed(SEED)

class IRSelectedRowsTestNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.embedding = paddle.nn.Embedding(4, 3, sparse=False)
        w0 = paddle.to_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype='float32')
        self.embedding.weight.set_value(w0)
        self.linear = paddle.nn.Linear(in_features=3, out_features=3, weight_attr=paddle.ParamAttr(need_clip=True), bias_attr=paddle.ParamAttr(need_clip=False))

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.embedding(x)
        x = self.linear(x)
        return x

def train(net, adam, x):
    if False:
        return 10
    loss_data = []
    for i in range(10):
        out = net(x)
        loss = paddle.mean(out)
        loss.backward()
        adam.step()
        adam.clear_grad()
        loss_data.append(loss.numpy())
    return loss_data

def train_dygraph():
    if False:
        for i in range(10):
            print('nop')
    paddle.seed(100)
    net = IRSelectedRowsTestNet()
    x = paddle.to_tensor([[0], [1], [3]], dtype='int64', stop_gradient=False)
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    adam = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.01, grad_clip=clip)
    return train(net, adam, x)

@compare_legacy_with_pir
def train_static():
    if False:
        for i in range(10):
            print('nop')
    paddle.seed(100)
    net = IRSelectedRowsTestNet()
    x = paddle.to_tensor([[0], [1], [3]], dtype='int64', stop_gradient=False)
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    adam = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.01, grad_clip=clip)
    return to_static(train, full_graph=True)(net, adam, x)

class TestSimnet(Dy2StTestBase):

    def test_dygraph_static_same_loss(self):
        if False:
            print('Hello World!')
        dygraph_loss = train_dygraph()
        static_loss = train_static()
        self.assertEqual(len(dygraph_loss), len(static_loss))
        for i in range(len(dygraph_loss)):
            self.assertAlmostEqual(dygraph_loss[i], static_loss[i].numpy())
if __name__ == '__main__':
    unittest.main()