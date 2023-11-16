import pytest
import torch
import torch.cuda
from torch.distributed.pipeline.sync.microbatch import Batch, check, gather, scatter
from torch.testing._internal.common_utils import run_tests

def test_batch_atomic():
    if False:
        print('Hello World!')
    x = torch.tensor(42)
    b = Batch(x)
    assert b.atomic
    assert b.tensor is x
    with pytest.raises(AttributeError):
        b.tensors
    assert list(b) == [x]
    assert len(b) == 1
    assert b[0] is x

def test_batch_non_atomic():
    if False:
        for i in range(10):
            print('nop')
    (x, y) = (torch.tensor(42), torch.tensor(21))
    b = Batch((x, y))
    assert not b.atomic
    with pytest.raises(AttributeError):
        b.tensor
    assert list(b) == [x, y]
    assert len(b) == 2
    assert b[0] is x
    assert b[1] is y

def test_batch_call():
    if False:
        i = 10
        return i + 15
    a = Batch(torch.tensor(42))
    b = Batch((torch.tensor(42), torch.tensor(21)))

    def f(x):
        if False:
            while True:
                i = 10
        return x

    def g(x, y):
        if False:
            for i in range(10):
                print('nop')
        return (x, y)
    assert a.call(f).atomic
    assert not b.call(g).atomic

def test_batch_setitem_by_index():
    if False:
        for i in range(10):
            print('nop')
    a = Batch(torch.tensor(42))
    b = Batch((torch.tensor(42), torch.tensor(21)))
    a[0] = torch.tensor(0)
    b[0] = torch.tensor(0)
    assert a.atomic
    assert a[0].item() == 0
    assert not b.atomic
    assert len(b) == 2
    assert b[0].item() == 0
    assert b[1].item() == 21

def test_batch_setitem_by_slice():
    if False:
        i = 10
        return i + 15
    a = Batch(torch.tensor(42))
    b = Batch((torch.tensor(42), torch.tensor(21)))
    a[:] = (torch.tensor(0),)
    b[:] = (torch.tensor(0),)
    assert a.atomic
    assert a[0].item() == 0
    assert not b.atomic
    assert len(b) == 1
    assert b[0].item() == 0

def test_check():
    if False:
        for i in range(10):
            print('nop')
    check(torch.device('cpu'), torch.tensor(42))
    check(torch.device('cpu'), torch.tensor(4), torch.tensor(2))
    with pytest.raises(TypeError):
        check(torch.device('cpu'), 42)
    with pytest.raises(TypeError):
        check(torch.device('cpu'), 'str')
    with pytest.raises(TypeError):
        check(torch.device('cpu'), (torch.tensor(4), 2))

def test_gather_tensors():
    if False:
        while True:
            i = 10
    a = torch.zeros(1, 1)
    b = torch.zeros(1, 1)
    ab = gather([Batch(a), Batch(b)])
    assert ab.size() == (2, 1)

def test_gather_tuples():
    if False:
        print('Hello World!')
    a = (torch.zeros(1, 1), torch.zeros(2, 2))
    b = (torch.zeros(1, 1), torch.zeros(2, 2))
    ab = gather([Batch(a), Batch(b)])
    assert isinstance(ab, tuple)
    assert ab[0].size() == (2, 1)
    assert ab[1].size() == (4, 2)

def test_scatter_tensor():
    if False:
        for i in range(10):
            print('nop')
    ab = torch.zeros(2, 1)
    (a, b) = scatter(ab, chunks=2)
    assert a.tensor.size() == (1, 1)
    assert b.tensor.size() == (1, 1)

def test_scatter_multiple_tensors():
    if False:
        return 10
    ab = (torch.zeros(2, 1), torch.zeros(4, 2))
    (a, b) = scatter(*ab, chunks=2)
    assert list(a)[0].size() == (1, 1)
    assert list(b)[0].size() == (1, 1)
    assert list(a)[1].size() == (2, 2)
    assert list(b)[1].size() == (2, 2)
if __name__ == '__main__':
    run_tests()