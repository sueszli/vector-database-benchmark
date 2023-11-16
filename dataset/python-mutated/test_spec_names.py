import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir
import paddle
from paddle.nn import Layer

class Net(Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fc = paddle.nn.Linear(16, 3)

    def forward(self, x, y, m, n):
        if False:
            return 10
        inputs = [x, y, m, n]
        outs = []
        for var in inputs:
            out = paddle.reshape(x, [-1, 16])
            out = self.fc(out)
            outs.append(out)
        out = paddle.stack(outs)
        return paddle.sum(out)

class TestArgsSpecName(Dy2StTestBase):

    def read_from_dataset(self):
        if False:
            return 10
        self.x = paddle.randn([4, 2, 8])
        self.y = paddle.randn([4, 2, 8])
        self.m = paddle.randn([4, 2, 8])
        self.n = paddle.randn([4, 2, 8])

    @test_legacy_and_pir
    @test_ast_only
    def test_spec_name_hash(self):
        if False:
            i = 10
            return i + 15
        net = Net()
        net = paddle.jit.to_static(net)
        self.read_from_dataset()
        self.run_test(net, [self.x, self.y, self.m, self.n], 1, [0, 1, 2, 3])
        self.read_from_dataset()
        self.run_test(net, [self.x, self.x, self.m, self.n], 1, [0, 0, 1, 2])
        self.read_from_dataset()
        self.run_test(net, [self.x, self.x, self.m, self.m], 1, [0, 0, 1, 1])
        self.read_from_dataset()
        self.run_test(net, [self.n, self.n, self.y, self.y], 1, [0, 0, 1, 1])
        self.read_from_dataset()
        self.run_test(net, [self.x, self.y, self.x, self.y], 1, [0, 1, 0, 1])
        self.read_from_dataset()
        self.run_test(net, [self.m, self.n, self.m, self.n], 1, [0, 1, 0, 1])
        self.read_from_dataset()
        self.run_test(net, [self.x, self.x, self.x, self.x], 1, [0, 0, 0, 0])
        self.read_from_dataset()
        self.run_test(net, [self.m, self.m, self.m, self.m], 1, [0, 0, 0, 0])

    def run_test(self, net, inputs, trace_count, mode):
        if False:
            print('Hello World!')
        out = net(*inputs)
        self.assertEqual(net.forward.get_traced_count(), trace_count)
if __name__ == '__main__':
    unittest.main()