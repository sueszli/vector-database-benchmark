import unittest
from dygraph_to_static_utils_new import Dy2StTestBase
import paddle
import paddle.distributed as dist
from paddle import nn

class Net(nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.emb1 = nn.Embedding(100, 16)
        self.emb2 = nn.Embedding(100, 16)

    def forward(self, ids):
        if False:
            return 10
        feat1 = self.emb1(ids)
        feat1.stop_gradient = True
        feat2 = self.emb2(ids)
        out = feat1 + feat2
        out = paddle.mean(out)
        return out

def train():
    if False:
        return 10
    paddle.distributed.init_parallel_env()
    net = Net()
    net = paddle.jit.to_static(net)
    sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
    dp_net = paddle.DataParallel(net)
    for i in range(4):
        x = paddle.randint(low=0, high=100, shape=[4, 10])
        loss = dp_net(x)
        loss.backward()
        sgd.step()
        loss.clear_gradient()
        print(loss)

class TestParamsNoGrad(Dy2StTestBase):

    def test_two_card(self):
        if False:
            while True:
                i = 10
        if paddle.is_compiled_with_cuda() and len(paddle.static.cuda_places()) > 1:
            dist.spawn(train, nprocs=2, gpus='0,1')
if __name__ == '__main__':
    unittest.main()