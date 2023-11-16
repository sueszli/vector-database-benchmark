import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict
from typing import Dict, Optional, Set
import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch._C import default_generator
from torch.multiprocessing.reductions import StorageWeakRef

def lazy_compile(**compile_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Lazily wrap a function with torch.compile on the first call\n\n    This avoids eagerly importing dynamo.\n    '

    def decorate_fn(fn):
        if False:
            i = 10
            return i + 15

        @functools.wraps(fn)
        def compile_hook(*args, **kwargs):
            if False:
                while True:
                    i = 10
            compiled_fn = torch.compile(fn, **compile_kwargs)
            globals()[fn.__name__] = functools.wraps(fn)(compiled_fn)
            return compiled_fn(*args, **kwargs)
        return compile_hook
    return decorate_fn

@lazy_compile(dynamic=True)
def hash_storage_kernel(x):
    if False:
        print('Hello World!')
    a = torch.randint(-2 ** 31, 2 ** 31, x.shape, device=x.device, dtype=torch.int32).abs()
    a = (a % (2 ** 31 - 1) + 1).long()
    b = torch.randint(-2 ** 31, 2 ** 31, x.shape, device=x.device, dtype=torch.int32).abs().long()
    return prims.xor_sum((a * x + b).int(), [0])

def hash_storage(storage: torch.UntypedStorage, *, stable_hash: bool=False) -> str:
    if False:
        i = 10
        return i + 15
    import torch._dynamo
    from torch._dynamo.utils import is_compile_supported
    device_type = storage.device.type
    if stable_hash or not is_compile_supported(device_type):
        cpu_storage = storage.cpu()
        buf = (ctypes.c_byte * cpu_storage.nbytes()).from_address(cpu_storage.data_ptr())
        sha1 = hashlib.sha1()
        sha1.update(buf)
        return sha1.hexdigest()
    if device_type == 'cpu':
        generator = default_generator
    elif device_type == 'cuda':
        import torch.cuda
        generator = torch.cuda.default_generators[storage.device.index]
    else:
        raise AssertionError(f'unhandled device type {device_type}')
    state = generator.get_state()
    try:
        generator.manual_seed(0)
        x = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)
        pad = -x.numel() % 4
        if pad > 0:
            x = F.pad(x, (0, pad), 'constant', 0)
        x = x.view(torch.int32)
        ITER = 5
        cs = [hash_storage_kernel(x).item() for _ in range(ITER)]
        return struct.pack('>' + 'i' * ITER, *cs).hex()
    finally:
        generator.set_state(state)

class ContentStoreWriter:

    def __init__(self, loc: str, stable_hash: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.loc: str = loc
        self.seen_storage_hashes: Set[str] = set()
        self.stable_hash = stable_hash

    def write_storage(self, storage: torch.UntypedStorage) -> str:
        if False:
            return 10
        h = hash_storage(storage, stable_hash=self.stable_hash)
        if h in self.seen_storage_hashes:
            return h
        subfolder = os.path.join(self.loc, 'storages')
        os.makedirs(subfolder, exist_ok=True)
        target = os.path.join(subfolder, h)
        if os.path.exists(target):
            return h
        torch.save(storage, target)
        self.seen_storage_hashes.add(h)
        return h

    def compute_tensor_metadata(self, t: torch.Tensor, h=None):
        if False:
            return 10
        if h is None:
            h = hash_storage(t.untyped_storage(), stable_hash=self.stable_hash)
        return (t.dtype, h, t.storage_offset(), tuple(t.shape), t.stride(), torch._utils.get_tensor_metadata(t))

    def write_tensor(self, name: str, t: torch.Tensor) -> None:
        if False:
            for i in range(10):
                print('nop')
        storage = t.untyped_storage()
        h = self.write_storage(storage)
        (d, f) = os.path.split(name)
        payload = self.compute_tensor_metadata(t, h=h)
        subfolder = os.path.join(self.loc, 'tensors', d)
        os.makedirs(subfolder, exist_ok=True)
        torch.save(payload, os.path.join(subfolder, f))

class ContentStoreReader:

    def __init__(self, loc: str, *, cache=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.loc = loc
        self.storage_cache: Optional[Dict[Optional[torch.device], Dict[str, StorageWeakRef]]] = None
        if cache:
            self.storage_cache = defaultdict(dict)

    def read_storage(self, h: str, *, device=None) -> torch.UntypedStorage:
        if False:
            for i in range(10):
                print('nop')
        if device is not None:
            device = torch.device(device)
        ws = self.storage_cache[device].get(h) if self.storage_cache is not None else None
        s: Optional[torch.UntypedStorage]
        if ws is not None:
            s = torch.UntypedStorage._new_with_weak_ptr(ws.cdata)
            if s is not None:
                return s
        s = torch.load(os.path.join(self.loc, 'storages', h), weights_only=True, map_location=device)._untyped_storage
        assert s is not None
        if self.storage_cache is not None:
            self.storage_cache[device][h] = StorageWeakRef(s)
        return s

    def read_tensor_metadata(self, name: str):
        if False:
            i = 10
            return i + 15
        fn = os.path.join(self.loc, 'tensors', name)
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        return torch.load(fn, weights_only=True)

    def read_tensor(self, name: str, *, device=None) -> torch.Tensor:
        if False:
            return 10
        (dtype, h, storage_offset, size, stride, metadata) = self.read_tensor_metadata(name)
        storage = self.read_storage(h, device=device)
        t = torch.tensor([], dtype=dtype, device=storage.device)
        t.set_(storage, storage_offset, size, stride)
        torch._utils.set_tensor_metadata(t, metadata)
        return t