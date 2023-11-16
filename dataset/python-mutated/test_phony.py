import torch
from torch.distributed.pipeline.sync.phony import get_phony
from torch.testing._internal.common_utils import run_tests

def test_phony_size():
    if False:
        while True:
            i = 10
    p = get_phony(torch.device('cpu'), requires_grad=False)
    assert p.size() == (0,)

def test_phony_requires_grad():
    if False:
        print('Hello World!')
    p1 = get_phony(torch.device('cpu'), requires_grad=True)
    p2 = get_phony(torch.device('cpu'), requires_grad=False)
    assert p1.requires_grad
    assert not p2.requires_grad

def test_cached_phony():
    if False:
        return 10
    p1 = get_phony(torch.device('cpu'), requires_grad=True)
    p2 = get_phony(torch.device('cpu'), requires_grad=True)
    assert p1 is p2
    p3 = get_phony(torch.device('cpu'), requires_grad=False)
    p4 = get_phony(torch.device('cpu'), requires_grad=False)
    assert p3 is p4
    assert p1 is not p3

def test_phony_in_autograd_function():
    if False:
        while True:
            i = 10

    class Phonify(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if False:
                print('Hello World!')
            phony = get_phony(input.device, requires_grad=False)
            return phony.detach()
    x = torch.rand(1, requires_grad=True)
    p1 = Phonify.apply(x)
    p2 = get_phony(torch.device('cpu'), requires_grad=True)
    assert p1 is not p2
    assert p1.grad_fn is not None
    assert p2.grad_fn is None
if __name__ == '__main__':
    run_tests()