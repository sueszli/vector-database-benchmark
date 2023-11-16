import unittest
import paddle
from paddle.nn import Conv2D
from paddle.nn.quant import Stub
from paddle.quantization import QAT, QuantConfig
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddle.quantization.quanters.abs_max import FakeQuanterWithAbsMaxObserverLayer
quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)

class Model(paddle.nn.Layer):

    def __init__(self, num_classes=10):
        if False:
            return 10
        super().__init__()
        self.quant_in = Stub()
        self.conv = Conv2D(3, 6, 3, stride=1, padding=1)
        self.quant = Stub(quanter)
        self.quant_out = Stub()

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        out = self.conv(inputs)
        out = self.quant(out)
        out = paddle.nn.functional.relu(out)
        return self.quant_out(out)

class TestStub(unittest.TestCase):

    def test_stub(self):
        if False:
            i = 10
            return i + 15
        model = Model()
        q_config = QuantConfig(activation=quanter, weight=quanter)
        qat = QAT(q_config)
        q_config.add_layer_config(model.quant_in, activation=None, weight=None)
        quant_model = qat.quantize(model)
        image = paddle.rand([1, 3, 32, 32], dtype='float32')
        out = model(image)
        out = quant_model(image)
        out.backward()
        quanter_count = 0
        for _layer in quant_model.sublayers(True):
            if isinstance(_layer, FakeQuanterWithAbsMaxObserverLayer):
                quanter_count += 1
        self.assertEqual(quanter_count, 5)
if __name__ == '__main__':
    unittest.main()