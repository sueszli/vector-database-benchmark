import math
import paddle
import paddle.nn.functional as F
from ..utils import limit_by_capacity
from .naive_gate import NaiveGate

class SwitchGate(NaiveGate):

    def __init__(self, d_model, num_expert, world_size, topk=1, switch_eps=0.1, capacity=(1.2, 2.4), group=None):
        if False:
            return 10
        assert topk == 1, 'topk should be 1 in switch'
        super().__init__(d_model, num_expert, world_size, topk=1)
        self.switch_eps = switch_eps
        self.capacity = capacity
        self.group = group

    def forward(self, inp):
        if False:
            i = 10
            return i + 15
        score = self.gate(inp)
        if self.training:
            noise = paddle.rand(shape=score.shape)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            score += noise
        score = F.softmax(score, axis=-1)
        (top1_score, top1_idx) = paddle.topk(score, k=1, axis=-1, largest=True)
        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * inp.shape[0])
        (_new_lec, _new_gec, top1_idx) = limit_by_capacity(top1_idx, self.num_expert, self.world_size, capacity, group=self.group)
        valid_idx = top1_idx[top1_idx > -1]
        valid_idx_tmp = paddle.reshape(valid_idx, shape=[len(valid_idx), 1])
        fraction_expert = paddle.scatter_nd_add(x=paddle.zeros(shape=[self.tot_expert]), index=valid_idx_tmp, updates=paddle.ones_like(valid_idx, dtype=paddle.float32).reshape(shape=[len(valid_idx)])) / valid_idx.numel()
        prob_expert = score.sum(axis=0) / valid_idx.numel()
        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.set_loss(loss)
        return (top1_score, top1_idx)