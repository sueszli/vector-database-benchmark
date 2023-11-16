from torch import nn
from torch.distributed.pipeline.sync.skip import Namespace, pop, skippable, stash
from torch.distributed.pipeline.sync.skip.layout import inspect_skip_layout
from torch.testing._internal.common_utils import run_tests

class Pass(nn.Module):

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        return input

@skippable(stash=['foo'])
class StashFoo(nn.Module):

    def forward(self, input):
        if False:
            print('Hello World!')
        yield stash('foo', input)
        return input

@skippable(pop=['foo'])
class PopFoo(nn.Module):

    def forward(self, input):
        if False:
            return 10
        foo = (yield stash('foo'))
        return input + foo

@skippable(stash=['bar'])
class StashBar(nn.Module):

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        yield stash('bar', input)
        return input

@skippable(pop=['bar'])
class PopBar(nn.Module):

    def forward(self, input):
        if False:
            while True:
                i = 10
        bar = (yield pop('bar'))
        return input + bar

def test_no_skippables():
    if False:
        return 10
    p1 = nn.Sequential(Pass())
    p2 = nn.Sequential(Pass())
    layout = inspect_skip_layout([p1, p2])
    policy = [list(layout.copy_policy(i)) for i in range(2)]
    assert policy == [[], []]

def test_inner_partition():
    if False:
        while True:
            i = 10
    p1 = nn.Sequential(StashFoo(), PopFoo())
    p2 = nn.Sequential(Pass())
    layout = inspect_skip_layout([p1, p2])
    policy = [list(layout.copy_policy(i)) for i in range(2)]
    assert policy == [[], []]

def test_adjoining_partitions():
    if False:
        for i in range(10):
            print('nop')
    p1 = nn.Sequential(StashFoo())
    p2 = nn.Sequential(PopFoo())
    layout = inspect_skip_layout([p1, p2])
    policy = [list(layout.copy_policy(i)) for i in range(2)]
    assert policy == [[], [(0, None, 'foo')]]

def test_far_partitions():
    if False:
        for i in range(10):
            print('nop')
    p1 = nn.Sequential(StashFoo())
    p2 = nn.Sequential(Pass())
    p3 = nn.Sequential(PopFoo())
    layout = inspect_skip_layout([p1, p2, p3])
    policy = [list(layout.copy_policy(i)) for i in range(3)]
    assert policy == [[], [], [(0, None, 'foo')]]

def test_pop_2_from_different_partitions():
    if False:
        return 10
    p1 = nn.Sequential(StashFoo())
    p2 = nn.Sequential(StashBar())
    p3 = nn.Sequential(PopBar(), PopFoo())
    layout = inspect_skip_layout([p1, p2, p3])
    policy = [list(layout.copy_policy(i)) for i in range(3)]
    assert policy == [[], [], [(0, None, 'foo'), (1, None, 'bar')]]

def test_namespace():
    if False:
        print('Hello World!')
    ns1 = Namespace()
    ns2 = Namespace()
    p1 = nn.Sequential(StashFoo().isolate(ns1))
    p2 = nn.Sequential(StashFoo().isolate(ns2))
    p3 = nn.Sequential(PopFoo().isolate(ns2), PopFoo().isolate(ns1))
    layout = inspect_skip_layout([p1, p2, p3])
    policy = [list(layout.copy_policy(i)) for i in range(3)]
    assert policy == [[], [], [(0, ns1, 'foo'), (1, ns2, 'foo')]]
if __name__ == '__main__':
    run_tests()