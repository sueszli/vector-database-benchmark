import random
import sys
import unittest
sys.path.append('../legacy_test')
import numpy as np
from hybrid_parallel_pp_layer import AlexNet, AlexNetPipeDesc
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.mix_precision_utils import MixPrecisionLayer, MixPrecisionOptimizer

def set_random_seed(seed, dp_id, rank_id):
    if False:
        i = 10
        return i + 15
    'Set random seed for reproducability.'
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)
batch_size = 4
micro_batch_size = 2

class TestDistPPTraining(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {'dp_degree': self.data_parallel_size, 'mp_degree': self.model_parallel_size, 'pp_degree': self.pipeline_parallel_size}
        strategy.pipeline_configs = {'accumulate_steps': batch_size // micro_batch_size, 'micro_batch_size': micro_batch_size}
        fleet.init(is_collective=True, strategy=strategy)

    def build_optimizer(self, model):
        if False:
            i = 10
            return i + 15
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2], values=[0.001, 0.002], verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
        return (scheduler, optimizer)

    def wrapper_mix_precision(self, model, optimizer):
        if False:
            while True:
                i = 10
        return (model, optimizer)

    def test_pp_model(self):
        if False:
            print('Hello World!')
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)
        model_a = AlexNet(10)
        (scheduler_a, optimizer_a) = self.build_optimizer(model_a)
        param_len = len(model_a.parameters())
        parameters = []
        for param in model_a.parameters():
            parameters.append(param.numpy())
        model_b = AlexNetPipeDesc(num_stages=self.pipeline_parallel_size)
        (scheduler_b, optimizer_b) = self.build_optimizer(model_b)
        (model_b, optimizer_b) = self.wrapper_mix_precision(model_b, optimizer_b)
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)
        for (idx, param) in enumerate(model_b.parameters()):
            param.set_value(parameters[idx + pp_id * (param_len // 2)])
        train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=batch_size, drop_last=True)
        for (step_id, data) in enumerate(train_reader()):
            x_data = np.array([x[0] for x in data]).astype('float32').reshape(batch_size, 1, 28, 28)
            y_data = np.array([x[1] for x in data]).astype('int64').reshape(batch_size, 1)
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            img.stop_gradient = True
            label.stop_gradient = True
            if step_id >= 5:
                return True
            loss_a = model_a(img, label)
            loss_a.backward()
            optimizer_a.step()
            optimizer_a.clear_grad()
            scheduler_a.step()
            loss_b = model_b.train_batch([img, label], optimizer_b, scheduler_b)
            print('loss: ', loss_a.numpy(), loss_b.numpy())
            np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy(), rtol=5e-05)

class TestDistPPDelayScaleLoss(TestDistPPTraining):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {'dp_degree': self.data_parallel_size, 'mp_degree': self.model_parallel_size, 'pp_degree': self.pipeline_parallel_size, 'pp_configs': {'delay_scale_loss': True, 'enable_timer': True}}
        strategy.pipeline_configs = {'accumulate_steps': batch_size // micro_batch_size, 'micro_batch_size': micro_batch_size}
        fleet.init(is_collective=True, strategy=strategy)

class TestDistPPMainGrad(TestDistPPTraining):

    def wrapper_mix_precision(self, model, optimizer):
        if False:
            print('Hello World!')
        model = MixPrecisionLayer(model, dtype='float16')
        optimizer = MixPrecisionOptimizer(optimizer)
        return (model._layers, optimizer)

    def build_optimizer(self, model):
        if False:
            i = 10
            return i + 15
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2], values=[0.001, 0.002], verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters(), grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0))
        return (scheduler, optimizer)
if __name__ == '__main__':
    unittest.main()