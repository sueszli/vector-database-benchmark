import pytest
import torch
from torch import nn
from torch.distributed.pipeline.sync import Pipe, is_checkpointing, is_recomputing
from torch.distributed.pipeline.sync.skip import pop, skippable, stash
from torch.distributed.pipeline.sync.skip.tracker import current_skip_tracker
from torch.testing._internal.common_utils import run_tests

@skippable(stash=['skip'])
class Stash(nn.Module):

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        yield stash('skip', input)
        return input

@skippable(pop=['skip'])
class Pop(nn.Module):

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        skip = (yield pop('skip'))
        return input + skip

@pytest.mark.parametrize('train', [True, False], ids=['train', 'eval'])
@pytest.mark.parametrize('checkpoint', ['always', 'except_last', 'never'])
def test_delete_portal_tensor(train, checkpoint, setup_rpc):
    if False:
        i = 10
        return i + 15

    def portal_tensor_life_is(tensor_life, skip_tracker=None):
        if False:
            while True:
                i = 10
        if skip_tracker is None:
            skip_tracker = current_skip_tracker()
        portal = list(skip_tracker.portals.values())[0]
        if tensor_life == 0:
            return portal.tensor_life == 0 and portal.tensor is None
        else:
            return portal.tensor_life == tensor_life and portal.tensor is not None
    stash_ = Stash()

    @stash_.register_forward_hook
    def check_portal_tensor_after_stash(*_):
        if False:
            print('Hello World!')
        if is_checkpointing():
            assert portal_tensor_life_is(2)
        elif is_recomputing():
            assert portal_tensor_life_is(0)
        else:
            assert portal_tensor_life_is(1)
    pop_ = Pop()

    @pop_.register_forward_hook
    def check_portal_tensor_after_pop(*_):
        if False:
            i = 10
            return i + 15
        if is_checkpointing():
            assert portal_tensor_life_is(1)
        elif is_recomputing():
            assert portal_tensor_life_is(0)
        else:
            assert portal_tensor_life_is(0)

    class NoPortalTensorAtBackward(nn.Module):

        class F(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    return 10
                ctx.skip_tracker = current_skip_tracker()
                return input.detach()

            @staticmethod
            def backward(ctx, grad):
                if False:
                    return 10
                assert portal_tensor_life_is(0, skip_tracker=ctx.skip_tracker)
                return grad

        def forward(self, input):
            if False:
                return 10
            return self.F.apply(input)
    model = nn.Sequential(NoPortalTensorAtBackward(), stash_, pop_)
    model = Pipe(model, chunks=2, checkpoint=checkpoint)
    input = torch.rand(10, requires_grad=True)
    if train:
        model.train()
        output = model(input).local_value()
        output.norm().backward()
    else:
        model.eval()
        with torch.no_grad():
            model(input)

@pytest.mark.parametrize('train', [True, False], ids=['train', 'eval'])
def test_no_portal_without_pipe(train, monkeypatch, setup_rpc):
    if False:
        for i in range(10):
            print('nop')

    def deny(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise AssertionError('tried to create Portal without Pipe')
    monkeypatch.setattr('torch.distributed.pipeline.sync.skip.portal.Portal.__init__', deny)
    model = nn.Sequential(Stash(), Pop())
    input = torch.rand(10, requires_grad=True)
    if train:
        model.train()
        output = model(input)
        output.norm().backward()
    else:
        model.eval()
        with torch.no_grad():
            model(input)
if __name__ == '__main__':
    run_tests()