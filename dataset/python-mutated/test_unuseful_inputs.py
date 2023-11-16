import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
import paddle
from paddle import nn
from paddle.jit import to_static
np.random.seed(1)

def apply_to_static(support_to_static, model, image_shape=None):
    if False:
        i = 10
        return i + 15
    if support_to_static:
        specs = None
        model = to_static(model, input_spec=specs)
    return model

class Layer0(nn.Layer):

    def __init__(self, level):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._linear1 = nn.Linear(10, 5)
        self._linear2 = nn.Linear(10, 5)
        self.layer1 = Layer1(level)
        apply_to_static(True, self.layer1)

    def forward(self, x):
        if False:
            while True:
                i = 10
        out1 = self._linear1(x)
        out2 = self._linear2(x)
        a = [out1, out2]
        b = self.layer1(a)
        return b

class Layer1(nn.Layer):

    def __init__(self, level):
        if False:
            print('Hello World!')
        super().__init__()
        self.level = level
        self._linear = nn.Linear(5, 2)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        inp = x[self.level]
        val = self._linear(inp)
        return val

class TestDuplicateOutput(Dy2StTestBase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `base.layers.cond`.
    """

    @test_legacy_and_pir
    def test_case(self):
        if False:
            i = 10
            return i + 15
        layer = Layer0(0)
        a = paddle.rand(shape=[10, 10])
        out = layer(a)
        loss = out.mean()
        loss.backward()
if __name__ == '__main__':
    unittest.main()