import random
import unittest
import numpy as np
import paddle

class TestDygraphFleetAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        if False:
            print('Hello World!')
        self.dtype = 'float32'
        self.shape = (2, 10, 5)

    def test_dygraph_fleet_api(self):
        if False:
            return 10
        import paddle.distributed as dist
        from paddle.distributed import fleet
        strategy = fleet.DistributedStrategy()
        strategy.amp = True
        strategy.recompute = True
        fleet.init(is_collective=True, strategy=strategy)
        net = paddle.nn.Sequential(paddle.nn.Linear(10, 1), paddle.nn.Linear(1, 2))
        net = dist.fleet.distributed_model(net)
        data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
        data = paddle.to_tensor(data)
        net(data)
if __name__ == '__main__':
    unittest.main()