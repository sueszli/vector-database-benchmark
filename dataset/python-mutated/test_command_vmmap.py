from __future__ import annotations
import tempfile
import gdb
import pytest
import pwndbg
import tests
CRASH_SIMPLE_BINARY = tests.binaries.get('crash_simple.out.hardcoded')
BINARY_ISSUE_1565 = tests.binaries.get('issue_1565.out')

def get_proc_maps():
    if False:
        i = 10
        return i + 15
    '\n        Example info proc mappings:\n\n    pwndbg> info proc mappings\n    process 26781\n    Mapped address spaces:\n\n              Start Addr           End Addr       Size     Offset objfile\n                0x400000           0x401000     0x1000        0x0 /opt/pwndbg/tests/gdb-tests/tests/binaries/crash_simple.out\n          0x7ffff7ffa000     0x7ffff7ffd000     0x3000        0x0 [vvar]\n          0x7ffff7ffd000     0x7ffff7fff000     0x2000        0x0 [vdso]\n          0x7ffffffde000     0x7ffffffff000    0x21000        0x0 [stack]\n      0xffffffffff600000 0xffffffffff601000     0x1000        0x0 [vsyscall]\n    '
    maps = []
    with open('/proc/%d/maps' % pwndbg.gdblib.proc.pid) as f:
        for line in f.read().splitlines():
            (addrs, perms, offset, _inode, size, objfile) = line.split(maxsplit=6)
            (start, end) = map(lambda v: int(v, 16), addrs.split('-'))
            offset = offset.lstrip('0') or '0'
            size = end - start
            maps.append([hex(start), hex(end), perms, hex(size)[2:], offset, objfile])
    maps.sort()
    return maps

@pytest.mark.parametrize('unload_file', (False, True))
def test_command_vmmap_on_coredump_on_crash_simple_binary(start_binary, unload_file):
    if False:
        return 10
    "\n    Example vmmap when debugging binary:\n        LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA\n                  0x400000           0x401000 r-xp     1000 0      /opt/pwndbg/tests/gdb-tests/tests/binaries/crash_simple.out\n            0x7ffff7ffa000     0x7ffff7ffd000 r--p     3000 0      [vvar]\n            0x7ffff7ffd000     0x7ffff7fff000 r-xp     2000 0      [vdso]\n            0x7ffffffde000     0x7ffffffff000 rwxp    21000 0      [stack]\n        0xffffffffff600000 0xffffffffff601000 r-xp     1000 0      [vsyscall]\n\n    The same vmmap when debugging coredump:\n        LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA\n                  0x400000           0x401000 r-xp     1000 0      /opt/pwndbg/tests/gdb-tests/tests/binaries/crash_simple.out\n            0x7ffff7ffd000     0x7ffff7fff000 r-xp     2000 1158   load2\n            0x7ffffffde000     0x7ffffffff000 rwxp    21000 3158   [stack]\n        0xffffffffff600000 0xffffffffff601000 r-xp     1000 24158  [vsyscall]\n\n    Note that for a core-file, we display the [vdso] page as load2 and we are missing the [vvar] page.\n    This is... how it is. It just seems that core files (at least those I met) have no info about\n    the vvar page and also GDB can't access the [vvar] memory with its x/ command during core debugging.\n    "
    start_binary(CRASH_SIMPLE_BINARY)
    gdb.execute('continue')
    expected_maps = get_proc_maps()
    vmmaps = gdb.execute('vmmap', to_string=True).splitlines()
    assert len(vmmaps) == len(expected_maps) + 2
    assert vmmaps[0] == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA'
    vmmaps = [i.split() for i in vmmaps[2:]]
    assert vmmaps == expected_maps
    core = tempfile.mktemp()
    gdb.execute(f'generate-core-file {core}')
    if unload_file:
        gdb.execute('file')
    gdb.execute(f'core-file {core}')
    old_len_vmmaps = len(vmmaps)
    vmmaps = gdb.execute('vmmap', to_string=True).splitlines()
    assert vmmaps[0] == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA'
    vmmaps = [i.split() for i in vmmaps[2:]]
    has_proc_maps = 'warning: unable to find mappings in core file' not in gdb.execute('info proc mappings', to_string=True)
    if has_proc_maps:
        assert len(vmmaps) == old_len_vmmaps - 1
    else:
        assert len(vmmaps) == old_len_vmmaps - 2
        binary_map = next((i for i in expected_maps if CRASH_SIMPLE_BINARY in i[-1]))
        expected_maps.remove(binary_map)
    next((i for i in expected_maps if i[-1] == '[vdso]'))[-1] = 'load2'
    vdso_map = next((i for i in expected_maps if i[-1] == '[vvar]'))
    expected_maps.remove(vdso_map)

    def assert_maps():
        if False:
            i = 10
            return i + 15
        for (vmmap, expected_map) in zip(vmmaps, expected_maps):
            if vmmap[-1] == expected_map[-1] == '[vsyscall]':
                assert vmmap[:2] == expected_map[:2]
                assert vmmap[3] == expected_map[3] or vmmap[3] in ('r-xp', '--xp')
                assert vmmap[4:] == expected_map[4:]
                continue
            assert vmmap[:-1] == expected_map[:-1]
            if vmmap[-1].startswith('load'):
                continue
            assert vmmap[-1] == expected_map[-1]
    assert_maps()
    gdb.execute('file')
    vmmaps = gdb.execute('vmmap', to_string=True).splitlines()
    vmmaps = [i.split() for i in vmmaps[2:]]
    assert_maps()

def test_vmmap_issue_1565(start_binary):
    if False:
        print('Hello World!')
    '\n    https://github.com/pwndbg/pwndbg/issues/1565\n\n    In tests this bug is reported as:\n    >       gdb.execute("context")\n    E       gdb.error: Error occurred in Python: maximum recursion depth exceeded in comparison\n\n    In a normal GDB session this is reported as:\n        Exception occurred: context: maximum recursion depth exceeded while calling a Python object (<class \'RecursionError\'>)\n    '
    gdb.execute(f'file {BINARY_ISSUE_1565}')
    gdb.execute('break thread_function')
    gdb.execute('run')
    gdb.execute('next')
    gdb.execute('context')