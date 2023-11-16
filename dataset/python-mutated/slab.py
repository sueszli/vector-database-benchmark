from __future__ import annotations
from typing import Generator
import gdb
from pwndbg.gdblib import kernel
from pwndbg.gdblib import memory
from pwndbg.gdblib.kernel.macros import compound_head
from pwndbg.gdblib.kernel.macros import for_each_entry
from pwndbg.gdblib.kernel.macros import swab

def caches() -> Generator[SlabCache, None, None]:
    if False:
        i = 10
        return i + 15
    slab_caches = gdb.lookup_global_symbol('slab_caches').value()
    for slab_cache in for_each_entry(slab_caches, 'struct kmem_cache', 'list'):
        yield SlabCache(slab_cache)

def get_cache(target_name: str) -> SlabCache | None:
    if False:
        while True:
            i = 10
    slab_caches = gdb.lookup_global_symbol('slab_caches').value()
    for slab_cache in for_each_entry(slab_caches, 'struct kmem_cache', 'list'):
        if target_name == slab_cache['name'].string():
            return SlabCache(slab_cache)
    return None

def slab_struct_type() -> str:
    if False:
        for i in range(10):
            print('nop')
    try:
        gdb.lookup_type('struct slab')
        return 'slab'
    except gdb.error:
        return 'page'
OO_SHIFT = 16
OO_MASK = (1 << OO_SHIFT) - 1

def oo_order(x: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return int(x) >> OO_SHIFT

def oo_objects(x: int) -> int:
    if False:
        while True:
            i = 10
    return int(x) & OO_MASK
_flags = {'SLAB_DEBUG_FREE': 256, 'SLAB_RED_ZONE': 1024, 'SLAB_POISON': 2048, 'SLAB_HWCACHE_ALIGN': 8192, 'SLAB_CACHE_DMA': 16384, 'SLAB_STORE_USER': 65536, 'SLAB_RECLAIM_ACCOUNT': 131072, 'SLAB_PANIC': 262144, 'SLAB_DESTROY_BY_RCU': 524288, 'SLAB_MEM_SPREAD': 1048576, 'SLAB_TRACE': 2097152, 'SLAB_DEBUG_OBJECTS': 4194304, 'SLAB_NOLEAKTRACE': 8388608, 'SLAB_NOTRACK': 16777216, 'SLAB_FAILSLAB': 33554432}

def get_flags_list(flags: int) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    return [flag_name for (flag_name, mask) in _flags.items() if flags & mask]

class Freelist:

    def __init__(self, start_addr: int, offset: int, random: int=0) -> None:
        if False:
            while True:
                i = 10
        self.start_addr = start_addr
        self.offset = offset
        self.random = random

    def __iter__(self) -> Generator[int, None, None]:
        if False:
            for i in range(10):
                print('nop')
        current_object = self.start_addr
        while current_object:
            addr = int(current_object)
            yield current_object
            current_object = memory.pvoid(addr + self.offset)
            if self.random:
                current_object ^= self.random ^ swab(addr + self.offset)

    def __int__(self) -> int:
        if False:
            return 10
        return self.start_addr

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return sum((1 for _ in self))

    def find_next(self, addr: int) -> int:
        if False:
            while True:
                i = 10
        freelist_iter = iter(self)
        for obj in freelist_iter:
            if obj == addr:
                return next(freelist_iter, 0)
        return 0

class SlabCache:

    def __init__(self, slab_cache: gdb.Value) -> None:
        if False:
            return 10
        self._slab_cache = slab_cache

    @property
    def address(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(self._slab_cache)

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._slab_cache['name'].string()

    @property
    def offset(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(self._slab_cache['offset'])

    @property
    def random(self) -> int:
        if False:
            i = 10
            return i + 15
        if not kernel.kconfig():
            try:
                return int(self._slab_cache['random'])
            except gdb.error:
                return 0
        return int(self._slab_cache['random']) if 'SLAB_FREELIST_HARDENED' in kernel.kconfig() else 0

    @property
    def size(self) -> int:
        if False:
            while True:
                i = 10
        return int(self._slab_cache['size'])

    @property
    def object_size(self) -> int:
        if False:
            return 10
        return int(self._slab_cache['object_size'])

    @property
    def align(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(self._slab_cache['align'])

    @property
    def flags(self) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        return get_flags_list(int(self._slab_cache['flags']))

    @property
    def cpu_cache(self) -> CpuCache:
        if False:
            for i in range(10):
                print('nop')
        'returns cpu cache associated to current thread'
        cpu = gdb.selected_thread().num - 1
        cpu_cache = kernel.per_cpu(self._slab_cache['cpu_slab'], cpu=cpu)
        return CpuCache(cpu_cache, self, cpu)

    @property
    def cpu_caches(self) -> Generator[CpuCache, None, None]:
        if False:
            while True:
                i = 10
        'returns cpu caches for all cpus'
        for cpu in range(kernel.nproc()):
            cpu_cache = kernel.per_cpu(self._slab_cache['cpu_slab'], cpu=cpu)
            yield CpuCache(cpu_cache, self, cpu)

    @property
    def node_caches(self) -> Generator[NodeCache, None, None]:
        if False:
            for i in range(10):
                print('nop')
        'returns node caches for all NUMA nodes'
        for node in range(kernel.num_numa_nodes()):
            yield NodeCache(self._slab_cache['node'][node], self, node)

    @property
    def cpu_partial(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(self._slab_cache['cpu_partial'])

    @property
    def inuse(self) -> int:
        if False:
            return 10
        return int(self._slab_cache['inuse'])

    @property
    def __oo_x(self) -> int:
        if False:
            while True:
                i = 10
        return int(self._slab_cache['oo']['x'])

    @property
    def oo_order(self):
        if False:
            return 10
        return oo_order(self.__oo_x)

    @property
    def oo_objects(self):
        if False:
            print('Hello World!')
        return oo_objects(self.__oo_x)

class CpuCache:

    def __init__(self, cpu_cache: gdb.Value, slab_cache: SlabCache, cpu: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._cpu_cache = cpu_cache
        self.slab_cache = slab_cache
        self.cpu = cpu

    @property
    def address(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return int(self._cpu_cache)

    @property
    def freelist(self) -> Freelist:
        if False:
            print('Hello World!')
        return Freelist(int(self._cpu_cache['freelist']), self.slab_cache.offset, self.slab_cache.random)

    @property
    def active_slab(self) -> Slab | None:
        if False:
            for i in range(10):
                print('nop')
        slab_key = slab_struct_type()
        _slab = self._cpu_cache[slab_key]
        if not _slab:
            return None
        return Slab(_slab.dereference(), self, self.slab_cache)

    @property
    def partial_slabs(self) -> list[Slab]:
        if False:
            for i in range(10):
                print('nop')
        partial_slabs = []
        cur_slab = self._cpu_cache['partial']
        while cur_slab:
            _slab = cur_slab.dereference()
            partial_slabs.append(Slab(_slab, self, self.slab_cache, is_partial=True))
            cur_slab = _slab['next']
        return partial_slabs

class NodeCache:

    def __init__(self, node_cache: gdb.Value, slab_cache: SlabCache, node: int):
        if False:
            for i in range(10):
                print('nop')
        self._node_cache = node_cache
        self.slab_cache = slab_cache
        self.node = node

    @property
    def address(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(self._node_cache)

    @property
    def partial_slabs(self) -> list[Slab]:
        if False:
            return 10
        ret = []
        for slab in for_each_entry(self._node_cache['partial'], 'struct slab', 'slab_list'):
            ret.append(Slab(slab.dereference(), None, self.slab_cache, is_partial=True))
        return ret

class Slab:

    def __init__(self, slab: gdb.Value, cpu_cache: CpuCache | None, slab_cache: SlabCache, is_partial: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self._slab = slab
        self.cpu_cache = cpu_cache
        self.slab_cache = slab_cache
        self.is_partial = is_partial

    @property
    def slab_address(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return int(self._slab.address)

    @property
    def virt_address(self) -> int:
        if False:
            while True:
                i = 10
        return kernel.page_to_virt(self.slab_address)

    @property
    def object_count(self) -> int:
        if False:
            while True:
                i = 10
        return int(self._slab['objects'])

    @property
    def objects(self) -> Generator[int, None, None]:
        if False:
            i = 10
            return i + 15
        size = self.slab_cache.size
        start = self.virt_address
        end = start + self.object_count * size
        return (i for i in range(start, end, size))

    @property
    def frozen(self) -> int:
        if False:
            print('Hello World!')
        return int(self._slab['frozen'])

    @property
    def inuse(self) -> int:
        if False:
            while True:
                i = 10
        inuse = int(self._slab['inuse'])
        if not self.is_partial:
            for freelist in self.freelists:
                inuse -= len(freelist)
        return inuse

    @property
    def slabs(self) -> int:
        if False:
            i = 10
            return i + 15
        return int(self._slab[f'{slab_struct_type()}s'])

    @property
    def pobjects(self) -> int:
        if False:
            while True:
                i = 10
        if not self.is_partial:
            return 0
        try:
            return int(self._slab['pobjects'])
        except gdb.error:
            return self.slabs * self.slab_cache.oo_objects // 2

    @property
    def freelist(self) -> Freelist:
        if False:
            while True:
                i = 10
        return Freelist(int(self._slab['freelist']), self.slab_cache.offset, self.slab_cache.random)

    @property
    def freelists(self) -> list[Freelist]:
        if False:
            print('Hello World!')
        freelists = [self.freelist]
        if not self.is_partial:
            freelists.append(self.cpu_cache.freelist)
        return freelists

    @property
    def free_objects(self) -> set[int]:
        if False:
            return 10
        return {obj for freelist in self.freelists for obj in freelist}

def find_containing_slab_cache(addr: int) -> SlabCache | None:
    if False:
        while True:
            i = 10
    'Find the slab cache associated with the provided address.'
    min_pfn = 0
    max_pfn = int(gdb.lookup_global_symbol('max_pfn').value())
    page_size = kernel.page_size()
    start_addr = kernel.pfn_to_virt(min_pfn)
    end_addr = kernel.pfn_to_virt(max_pfn + page_size)
    if not start_addr <= addr < end_addr:
        return None
    page_type = gdb.lookup_type('struct page')
    page = memory.poi(page_type, kernel.virt_to_page(addr))
    head_page = compound_head(page)
    slab_type = gdb.lookup_type(f'struct {slab_struct_type()}')
    slab = head_page.cast(slab_type)
    return SlabCache(slab['slab_cache'])