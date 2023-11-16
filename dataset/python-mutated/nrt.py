from collections import namedtuple
from weakref import finalize as _finalize
from numba.core.runtime import nrtdynmod
from llvmlite import binding as ll
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typing.typeof import typeof_impl
from numba.core import types, config
from numba.core.runtime import _nrt_python as _nrt
_nrt_mstats = namedtuple('nrt_mstats', ['alloc', 'free', 'mi_alloc', 'mi_free'])

class _Runtime(object):

    def __init__(self):
        if False:
            return 10
        self._init = False

    @global_compiler_lock
    def initialize(self, ctx):
        if False:
            print('Hello World!')
        'Initializes the NRT\n\n        Must be called before any actual call to the NRT API.\n        Safe to be called multiple times.\n        '
        if self._init:
            return
        if config.NRT_STATS:
            _nrt.memsys_enable_stats()
        for py_name in _nrt.c_helpers:
            if py_name.startswith('_'):
                c_name = py_name
            else:
                c_name = 'NRT_' + py_name
            c_address = _nrt.c_helpers[py_name]
            ll.add_symbol(c_name, c_address)
        self._library = nrtdynmod.compile_nrt_functions(ctx)
        self._init = True

    def _init_guard(self):
        if False:
            while True:
                i = 10
        if not self._init:
            msg = 'Runtime must be initialized before use.'
            raise RuntimeError(msg)

    @staticmethod
    def shutdown():
        if False:
            return 10
        '\n        Shutdown the NRT\n        Safe to be called without calling Runtime.initialize first\n        '
        _nrt.memsys_shutdown()

    @property
    def library(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the Library object containing the various NRT functions.\n        '
        self._init_guard()
        return self._library

    def meminfo_new(self, data, pyobj):
        if False:
            print('Hello World!')
        '\n        Returns a MemInfo object that tracks memory at `data` owned by `pyobj`.\n        MemInfo will acquire a reference on `pyobj`.\n        The release of MemInfo will release a reference on `pyobj`.\n        '
        self._init_guard()
        mi = _nrt.meminfo_new(data, pyobj)
        return MemInfo(mi)

    def meminfo_alloc(self, size, safe=False):
        if False:
            i = 10
            return i + 15
        '\n        Allocate a new memory of `size` bytes and returns a MemInfo object\n        that tracks the allocation.  When there is no more reference to the\n        MemInfo object, the underlying memory will be deallocated.\n\n        If `safe` flag is True, the memory is allocated using the `safe` scheme.\n        This is used for debugging and testing purposes.\n        See `NRT_MemInfo_alloc_safe()` in "nrt.h" for details.\n        '
        self._init_guard()
        if size < 0:
            msg = f'Cannot allocate a negative number of bytes: {size}.'
            raise ValueError(msg)
        if safe:
            mi = _nrt.meminfo_alloc_safe(size)
        else:
            mi = _nrt.meminfo_alloc(size)
        if mi == 0:
            msg = f'Requested allocation of {size} bytes failed.'
            raise MemoryError(msg)
        return MemInfo(mi)

    def get_allocation_stats(self):
        if False:
            while True:
                i = 10
        '\n        Returns a namedtuple of (alloc, free, mi_alloc, mi_free) for count of\n        each memory operations.\n        '
        return _nrt_mstats(alloc=_nrt.memsys_get_stats_alloc(), free=_nrt.memsys_get_stats_free(), mi_alloc=_nrt.memsys_get_stats_mi_alloc(), mi_free=_nrt.memsys_get_stats_mi_free())
MemInfo = _nrt._MemInfo

@typeof_impl.register(MemInfo)
def typeof_meminfo(val, c):
    if False:
        i = 10
        return i + 15
    return types.MemInfoPointer(types.voidptr)
_nrt.memsys_use_cpython_allocator()
rtsys = _Runtime()
_finalize(rtsys, _Runtime.shutdown)
del _Runtime