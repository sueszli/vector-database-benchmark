import random
import unittest
import numpy as np
from hybrid_parallel_mp_model import SimpleDPNet, SimpleMPNet, TestDistMPTraining, parallel_matmul, set_random_seed
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4

class SimpleMPMultimodalNet(SimpleMPNet):

    def forward(self, x, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.to_tensor(x)
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = parallel_matmul(x, self.embedding.weight, False)
        return x

class SimpleDPMultimodalNet(SimpleDPNet):

    def forward(self, x, **kwargs):
        if False:
            print('Hello World!')
        x = paddle.to_tensor(x)
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x

class TestMPBroadcastObj(TestDistMPTraining):

    def build_model_optimizer(self):
        if False:
            while True:
                i = 10
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        mp_id = hcg.get_model_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)
        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))
        model_a = SimpleMPMultimodalNet(vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2, mp_id)
        optimizer_a = self.build_optimizer(model_a)
        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)
        model_b = SimpleDPMultimodalNet(vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2)
        optimizer_b = self.build_optimizer(model_b)
        return (model_a, optimizer_a, model_b, optimizer_b)

    def train_batch(self, batch, model, optimizer, is_mp):
        if False:
            for i in range(10):
                print('nop')
        (img, text) = batch
        output = model(img, text=text)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return loss

    def test_mp_model(self):
        if False:
            print('Hello World!')
        (model_a, optimizer_a, model_b, optimizer_b) = self.build_model_optimizer()
        for _ in range(5):
            img = np.random.randint(0, vocab_size, (batch_size, seq_length))
            text = [random.sample('zyxwvutsrqponmlkjihgfedcba', 5) for i in range(batch_size)]
            batch = (img, text)
            loss_a = self.train_batch(batch, model_a, optimizer_a, True)
            loss_b = self.train_batch(batch, model_b, optimizer_b, False)
            np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy(), rtol=1e-06)
if __name__ == '__main__':
    unittest.main()