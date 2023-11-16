import numpy as np
import megengine
import megengine.autodiff as ad
import megengine.optimizer as optimizer
from megengine import Parameter, tensor
from megengine.module import Module

class Simple(Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.a = Parameter([1.0], dtype=np.float32)

    def forward(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        x = x[y] * self.a
        return x

class Simple2(Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.a = Parameter([1.0], dtype=np.float32)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = x[1, ..., :, 0:4:2, 0:2] * self.a
        return x

def test_advance_indexing():
    if False:
        print('Hello World!')
    net = Simple()
    gm = ad.GradManager().attach(net.parameters())
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    optim.clear_grad()
    dshape = (10, 10)
    raw_data = np.arange(100).reshape(dshape).astype(np.float32)
    raw_mask = (np.random.random_sample(dshape) > 0.5).astype(np.bool_)
    data = tensor(raw_data)
    mask = tensor(raw_mask)
    answer = 1.0 - raw_data[raw_mask].sum()
    with gm:
        loss = net(data, mask).sum()
        gm.backward(loss)
    optim.step()
    np.testing.assert_almost_equal(net.a.numpy(), np.array([answer]).astype(np.float32))

def test_advance_indexing_with_subtensor():
    if False:
        while True:
            i = 10
    net = Simple2()
    gm = ad.GradManager().attach(net.parameters())
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    optim.clear_grad()
    dshape = (2, 3, 4, 3, 4, 2)
    raw_data = np.arange(576).reshape(dshape).astype(np.float32)
    data = tensor(raw_data)
    answer = 1.0 - raw_data[1, ..., :, 0:4:2, 0:2].sum()
    with gm:
        loss = net(data).sum()
        gm.backward(loss)
    optim.step()
    np.testing.assert_almost_equal(net.a.numpy(), np.array([answer]).astype(np.float32))