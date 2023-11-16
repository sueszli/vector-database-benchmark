import torch
from torch import nn
from torch.distributed.pipeline.sync import Pipe
from torch.testing._internal.common_utils import run_tests

def test_simple_linears(setup_rpc):
    if False:
        print('Hello World!')

    def sum_grad(parameters):
        if False:
            return 10
        return sum([p.grad.sum() for p in parameters if p.grad is not None])

    def zero_grad(parameters):
        if False:
            print('Hello World!')
        for p in parameters:
            p.grad = None
    inputs = torch.rand(8, 1)
    model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 4), nn.Linear(4, 2), nn.Linear(2, 1))
    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()
    grad_without_pipe = sum_grad(model.parameters())
    zero_grad(model.parameters())
    model = Pipe(model, chunks=4)
    outputs = model(inputs).local_value()
    loss = outputs.mean()
    loss.backward()
    grad_with_pipe = sum_grad(model.parameters())
    assert torch.allclose(grad_with_pipe, grad_without_pipe)
if __name__ == '__main__':
    run_tests()