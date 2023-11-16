import unittest
import paddle
from paddle.incubate.distributed.fleet.collective import CollectiveOptimizer, DistributedStrategy

class CollectiveOptimizerTest(unittest.TestCase):

    def test_ds_as_None(self):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.Adam()
        dist_optimizer = CollectiveOptimizer(optimizer, strategy=None)

    def test_recompute_checkpoints(self):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.Adam()
        dist_strategy = DistributedStrategy()
        dist_strategy.forward_recompute = True
        dist_strategy.recompute_checkpoints = 'NoneListTest'
        self.assertRaises(ValueError, CollectiveOptimizer, optimizer, dist_strategy)
        dist_strategy.recompute_checkpoints = []
        dist_optimizer = CollectiveOptimizer(optimizer, dist_strategy)
        self.assertRaises(ValueError, dist_optimizer.minimize, None)

    def test_recompute_strategy(self):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.Adam()
        optimizer = paddle.incubate.optimizer.RecomputeOptimizer(optimizer)
        dist_strategy = DistributedStrategy()
        dist_strategy.forward_recompute = True
        dist_strategy.recompute_checkpoints = ['Test']
        dist_optimizer = CollectiveOptimizer(optimizer, strategy=dist_strategy)
        self.assertRaises(ValueError, dist_optimizer.minimize, None)

    def test_amp_strategy(self):
        if False:
            i = 10
            return i + 15
        optimizer = paddle.optimizer.Adam()
        optimizer = paddle.static.amp.decorate(optimizer, init_loss_scaling=1.0, use_dynamic_loss_scaling=True)
        dist_strategy = DistributedStrategy()
        dist_strategy.use_amp = True
        dist_optimizer = CollectiveOptimizer(optimizer, strategy=dist_strategy)
        self.assertRaises(ValueError, dist_optimizer.minimize, None)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()