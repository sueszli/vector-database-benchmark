from contextlib import contextmanager
from torch._C._functorch import _vmap_add_layers, _vmap_remove_layers
_enabled = False

@contextmanager
def _enable_layers(dims):
    if False:
        for i in range(10):
            print('nop')
    global _enabled
    assert not _enabled
    input = sorted(((d._level, d.size) for d in dims if not isinstance(d, int)))
    n = len(input)
    try:
        _vmap_add_layers(input)
        _enabled = True
        yield
    finally:
        _enabled = False
        _vmap_remove_layers(n)