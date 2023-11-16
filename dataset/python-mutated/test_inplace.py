import pytest
import torch
from torch import nn
from torch.distributed.pipeline.sync import Pipe
from torch.testing._internal.common_utils import run_tests

def test_inplace_on_requires_grad(setup_rpc):
    if False:
        print('Hello World!')
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(inplace=True))
    model = Pipe(model, checkpoint='always')
    x = torch.rand(1)
    y = model(x).local_value()
    message = 'a leaf Variable that requires grad .* used in an in-place operation.'
    with pytest.raises(RuntimeError, match=message):
        y.backward()

@pytest.mark.xfail(strict=True)
def test_inplace_on_not_requires_grad(setup_rpc):
    if False:
        print('Hello World!')
    model = nn.Sequential(nn.ReLU(inplace=True))
    model = Pipe(model, [1], devices=['cpu'], checkpoint='always')
    x = torch.rand(1)
    y = model(x).local_value()
    del model
    message = 'a leaf Variable that requires grad .* used in an in-place operation.'
    with pytest.raises(RuntimeError, match=message):
        y.backward()

@pytest.mark.xfail(strict=True)
def test_inplace_incorrect_grad(setup_rpc):
    if False:
        for i in range(10):
            print('nop')

    class M(nn.Module):

        def forward(self, foo_bar):
            if False:
                return 10
            (foo, bar) = foo_bar
            bar.add_(1)
            return foo * bar
    model = nn.Sequential(M())
    model = Pipe(model, [1], devices=['cpu'], checkpoint='always')
    foo = torch.tensor([1.0], requires_grad=True)
    bar = torch.tensor([1.0])
    output = model((foo, bar)).local_value()
    del model
    output.backward()
    assert foo.grad.item() == 2.0
if __name__ == '__main__':
    run_tests()