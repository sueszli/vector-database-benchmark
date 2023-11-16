import random
import unittest
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.nn import Layer

def set_random_seed(seed, dp_id, rank_id):
    if False:
        print('Hello World!')
    'Set random seed for reproducability.'
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)
batch_size = 8
length = 8
micro_batch_size = 2
vocab_size = 128
hidden_size = 16
d_model = hidden_size
dim_feedforward = 4 * d_model

class EmbeddingNet(Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        if False:
            return 10
        attention_mask = paddle.tensor.triu(paddle.ones((length, length), dtype='float32') * -1000000000.0, 1)
        no_used = paddle.ones((3, 3), dtype='int32')
        w_emb = self.word_embeddings(x)
        p_emb = self.position_embeddings(x)
        w_emb = w_emb + p_emb
        attention_mask.stop_gradient = True
        no_used.stop_gradient = True
        return (w_emb, attention_mask, no_used, p_emb)

class TransformerNet(Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-05)

    def forward(self, x, mask):
        if False:
            while True:
                i = 10
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.scale(product, scale=d_model ** (-0.5))
        weights = F.softmax(product + mask)
        tgt = paddle.matmul(weights, v)
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = residual + tgt
        out = self.linear2(F.gelu(self.linear1(tgt), approximate=True))
        return out

class EmbeddingPipe(EmbeddingNet):

    def forward(self, x):
        if False:
            while True:
                i = 10
        return super().forward(x)

class TransformerNetPipe(TransformerNet):

    def forward(self, args):
        if False:
            print('Hello World!')
        (x, mask, no_used, p_emb) = (args[0], args[1], args[2], args[3])
        output = super().forward(x, mask)
        output = output + p_emb
        mask.stop_gradient = True
        return (output, mask, no_used, p_emb)

class CriterionPipe(Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def forward(self, out, label):
        if False:
            while True:
                i = 10
        loss = out.mean()
        return loss

class ModelPipe(PipelineLayer):

    def __init__(self, topology, transformer_layer_num: int=6):
        if False:
            for i in range(10):
                print('nop')
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingPipe))
        for x in range(transformer_layer_num):
            self.descs.append(LayerDesc(TransformerNetPipe))
        self.descs.append(lambda x: x[0])
        super().__init__(layers=self.descs, loss_fn=CriterionPipe(), topology=topology, seg_method='layer:TransformerNetPipe')

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

    def test_pp_model(self):
        if False:
            while True:
                i = 10
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)
        model = ModelPipe(topology)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2], values=[0.001, 0.002], verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
        for step_id in range(5):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True
            e_loss = model.eval_batch([x, x], True)
            loss = model.train_batch([x, x], optimizer, scheduler)
            if pp_id != 0:
                np.testing.assert_allclose(loss.numpy(), e_loss.numpy())
if __name__ == '__main__':
    unittest.main()