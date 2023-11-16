import os
import unittest
import numpy as np
from hybrid_parallel_pp_transformer import ModelPipe, set_random_seed
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
batch_size = 8
length = 8
micro_batch_size = 2
vocab_size = 128
transformer_layer_num = 8

class TestDistPPSaveTraining(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {'dp_degree': self.data_parallel_size, 'mp_degree': self.model_parallel_size, 'pp_degree': self.pipeline_parallel_size}
        strategy.pipeline_configs = {'accumulate_steps': batch_size // micro_batch_size, 'micro_batch_size': micro_batch_size}
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        if False:
            return 10
        print(f'pwd {os.getcwd()}')
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)
        model = ModelPipe(topology, transformer_layer_num=transformer_layer_num)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2], values=[0.001, 0.002], verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
        output_dir = '{}/mp_00_sharding_00_pp_{:0>2d}'.format('./pp_transformer', pp_id)
        try:
            os.makedirs(output_dir)
        except:
            pass
        for step_id in range(2):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True
            loss = model.train_batch([x, x], optimizer, scheduler)
        paddle.save(model.state_dict(), os.path.join(output_dir, 'model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(output_dir, 'model_state.pdopt'))
        meta_dict = {'epoch': 0, 'step': 2, 'cuda_rng_state': paddle.get_cuda_rng_state()}
        paddle.save(meta_dict, os.path.join(output_dir, 'meta_state.pdopt'))
if __name__ == '__main__':
    unittest.main()