from __future__ import annotations
import gdb
import pwndbg

def test_command_kbase():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_command_kchecksec():
    if False:
        return 10
    res = gdb.execute('kchecksec', to_string=True)

def test_command_kcmdline():
    if False:
        return 10
    res = gdb.execute('kcmdline', to_string=True)

def test_command_kconfig():
    if False:
        print('Hello World!')
    if not pwndbg.gdblib.kernel.has_debug_syms():
        res = gdb.execute('kconfig', to_string=True)
        assert 'may only be run when debugging a Linux kernel with debug' in res
        return
    res = gdb.execute('kconfig', to_string=True)
    assert 'CONFIG_IKCONFIG = y' in res
    res = gdb.execute('kconfig IKCONFIG', to_string=True)
    assert 'CONFIG_IKCONFIG = y' in res

def test_command_kversion():
    if False:
        return 10
    if not pwndbg.gdblib.kernel.has_debug_syms():
        res = gdb.execute('kversion', to_string=True)
        assert 'may only be run when debugging a Linux kernel with debug' in res
        return
    res = gdb.execute('kversion', to_string=True)
    assert 'Linux version' in res

def test_command_slab_list():
    if False:
        while True:
            i = 10
    if not pwndbg.gdblib.kernel.has_debug_syms():
        res = gdb.execute('slab list', to_string=True)
        assert 'may only be run when debugging a Linux kernel with debug' in res
        return
    res = gdb.execute('slab list', to_string=True)
    assert 'kmalloc' in res

def test_command_slab_info():
    if False:
        while True:
            i = 10
    if not pwndbg.gdblib.kernel.has_debug_syms():
        res = gdb.execute('slab info kmalloc-512', to_string=True)
        assert 'may only be run when debugging a Linux kernel with debug' in res
        return
    for cache in pwndbg.gdblib.kernel.slab.caches():
        cache_name = cache.name
        res = gdb.execute(f'slab info -v {cache_name}', to_string=True)
        assert cache_name in res
        assert 'Freelist' in res
        for cpu in range(pwndbg.gdblib.kernel.nproc()):
            assert f'[CPU {cpu}]' in res
    res = gdb.execute('slab info -v does_not_exit', to_string=True)
    assert 'not found' in res

def test_command_slab_contains():
    if False:
        while True:
            i = 10
    if not pwndbg.gdblib.kernel.has_debug_syms():
        res = gdb.execute('slab contains 0x123', to_string=True)
        assert 'may only be run when debugging a Linux kernel with debug' in res
        return
    (addr, slab_cache) = get_slab_object_address()
    res = gdb.execute(f'slab contains {addr}', to_string=True)
    assert f'{addr} @ {slab_cache}' in res

def get_slab_object_address():
    if False:
        while True:
            i = 10
    'helper function to get the address of some kmalloc slab object\n    and the associated slab cache name'
    import re
    caches = pwndbg.gdblib.kernel.slab.caches()
    for cache in caches:
        cache_name = cache.name
        info = gdb.execute(f'slab info -v {cache_name}', to_string=True)
        matches = re.findall('- (0x[0-9a-fA-F]+)', info)
        if len(matches) > 0:
            return (matches[0], cache_name)
    raise ValueError('Could not find any slab objects')