import math
import paddle
import paddle.nn.functional as F
from ..utils import limit_by_capacity
from .naive_gate import NaiveGate

class GShardGate(NaiveGate):

    def __init__(self, d_model, num_expert, world_size, topk=2, capacity=(1.2, 2.4), random_routing=True, group=None):
        if False:
            i = 10
            return i + 15
        assert topk == 2, 'topk should be 2 in gshard'
        super().__init__(d_model, num_expert, world_size)
        self.capacity = capacity
        self.random_routing = random_routing
        self.group = group

    def forward(self, x):
        if False:
            while True:
                i = 10
        (topk_val, topk_idx, gate_score) = super().forward(x, return_all_scores=True)
        s = gate_score.shape[0]
        top1_idx = topk_idx.flatten()
        c_e = paddle.scatter(paddle.zeros(shape=[self.tot_expert]), top1_idx, paddle.ones_like(top1_idx, dtype='float32'), overwrite=False) / s
        m_e = paddle.mean(F.softmax(gate_score, axis=1), axis=0)
        loss = paddle.mean(c_e * m_e) * self.num_expert ** 2
        self.set_loss(loss)
        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        (_new_lec, _new_gec, topk_idx) = limit_by_capacity(topk_idx, self.num_expert, self.world_size, capacity, group=self.group)
        if self.random_routing:
            rand_routing_prob = paddle.rand(shape=[gate_score.shape[0]], dtype='float32')
            topk_idx = paddle.distributed.models.moe.utils._random_routing(topk_idx, topk_val, rand_routing_prob)
        return (topk_val, topk_idx)