import paddle
from paddle import nn
from .base_gate import BaseGate

class NaiveGate(BaseGate):

    def __init__(self, d_model, num_expert, world_size, topk=2):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.gate.weight.name = 'gate_' + self.gate.weight.name
        self.gate.bias.name = 'gate_' + self.gate.bias.name
        self.top_k = topk

    def forward(self, inp, return_all_scores=False):
        if False:
            i = 10
            return i + 15
        gate = self.gate(inp)
        (gate_top_k_val, gate_top_k_idx) = paddle.topk(gate, k=self.top_k, axis=-1, largest=True, sorted=False)
        if return_all_scores:
            return (gate_top_k_val, gate_top_k_idx, gate)
        return (gate_top_k_val, gate_top_k_idx)