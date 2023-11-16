import torch
import torch.nn as nn
from allennlp.common.params import Params
from allennlp.common.testing import multi_device
from allennlp.nn.checkpoint.checkpoint_wrapper import CheckpointWrapper
from allennlp.nn.checkpoint.fairscale_checkpoint_wrapper import FairScaleCheckpointWrapper
forward_calls = 0

class FeedForwardForTesting(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        global forward_calls
        forward_calls += 1
        return self.linear2(self.dropout(self.linear1(x)))

def setup_function(_):
    if False:
        while True:
            i = 10
    global forward_calls
    forward_calls = 0

class ModuleForTesting(nn.Module):

    def __init__(self, checkpoint_wrapper: FairScaleCheckpointWrapper) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.ffn = nn.Sequential(checkpoint_wrapper.wrap_module(nn.Linear(3, 3)), nn.Linear(3, 3), checkpoint_wrapper.wrap_module(FeedForwardForTesting()), checkpoint_wrapper.wrap_module(nn.Linear(3, 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        return self.ffn(x)

@multi_device
def test_fairscale_checkpoint_wrapper(device: str):
    if False:
        return 10
    device_ = torch.device(device)
    checkpoint_wrapper: FairScaleCheckpointWrapper = CheckpointWrapper.from_params(Params({'type': 'fairscale', 'offload_to_cpu': False if device == 'cpu' else True}))
    module = ModuleForTesting(checkpoint_wrapper).to(device_)
    optim = torch.optim.Adam(module.parameters(), lr=0.0001)
    module.train()
    x = torch.randn(2, 3).to(device_)
    loss = module(x).sum()
    assert forward_calls == 1, f'incorrect # of forward calls: {forward_calls}'
    loss.backward()
    assert forward_calls == 2, f'incorrect # of forward calls: {forward_calls}'
    for param in module.parameters():
        assert param.grad is not None
    optim.step()
    optim.zero_grad(set_to_none=True)
    module.eval()
    x = torch.randn(2, 3).to(device_)
    module(x)