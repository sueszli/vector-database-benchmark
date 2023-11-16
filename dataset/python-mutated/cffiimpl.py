"""
Implementation of some CFFI functions
"""
from numba.core.imputils import Registry
from numba.core import types
from numba.np import arrayobj
registry = Registry('cffiimpl')

@registry.lower('ffi.from_buffer', types.Buffer)
def from_buffer(context, builder, sig, args):
    if False:
        return 10
    assert len(sig.args) == 1
    assert len(args) == 1
    [fromty] = sig.args
    [val] = args
    assert fromty.dtype == sig.return_type.dtype
    ary = arrayobj.make_array(fromty)(context, builder, val)
    return ary.data