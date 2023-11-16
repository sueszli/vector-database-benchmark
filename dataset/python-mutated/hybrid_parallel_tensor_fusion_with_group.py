import unittest
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.tensor_fusion_helper import HOOK_ACTION, fused_parameters

class SimpleDPNet(paddle.nn.Layer):

    def __init__(self, vocab_size, hidden_size, inner_size, output_size):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear1 = paddle.nn.Linear(hidden_size, inner_size)
        self.linear2 = paddle.nn.Linear(inner_size, hidden_size)
        self.linear3 = paddle.nn.Linear(hidden_size, output_size)
        self.embedding = paddle.nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        if False:
            return 10
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x

class TestDistSharding(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {'sharding_degree': 1, 'dp_degree': 2, 'mp_degree': 1, 'pp_degree': 1}
        fleet.init(is_collective=True, strategy=self.strategy)

    def test_fusion(self):
        if False:
            for i in range(10):
                print('nop')
        model = SimpleDPNet(20, 10, 8, 10)
        parameters = model.parameters()
        parameters[0].optimize_attr = {'lr': 1}
        param_group = [{'params': parameters}, {'params': parameters}]
        fused_parameters(param_group, act=HOOK_ACTION.ALL_REDUCE, comm_overlap=True, group_params=True)
if __name__ == '__main__':
    unittest.main()