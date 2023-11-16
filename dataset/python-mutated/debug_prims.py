import contextlib
from typing import Sequence
import torch
from torch._custom_op.impl import custom_op
from torch.utils._content_store import ContentStoreReader
LOAD_TENSOR_READER = None

@contextlib.contextmanager
def load_tensor_reader(loc):
    if False:
        while True:
            i = 10
    global LOAD_TENSOR_READER
    assert LOAD_TENSOR_READER is None
    LOAD_TENSOR_READER = ContentStoreReader(loc, cache=False)
    try:
        yield
    finally:
        LOAD_TENSOR_READER = None

def register_debug_prims():
    if False:
        i = 10
        return i + 15

    @custom_op('debugprims::load_tensor')
    def load_tensor(name: str, size: Sequence[int], stride: Sequence[int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if False:
            return 10
        ...

    @load_tensor.impl_factory()
    def load_tensor_factory(name, size, stride, dtype, device):
        if False:
            for i in range(10):
                print('nop')
        if LOAD_TENSOR_READER is None:
            from torch._dynamo.testing import rand_strided
            return rand_strided(size, stride, dtype, device)
        else:
            from torch._dynamo.utils import clone_input
            r = LOAD_TENSOR_READER.read_tensor(name, device=device)
            assert list(r.size()) == size, f'{r.size()} != {size}'
            assert list(r.stride()) == stride, f'{r.stride()} != {stride}'
            assert r.device == device, f'{r.device} != {device}'
            if r.dtype != dtype:
                r = clone_input(r, dtype=dtype)
            return r