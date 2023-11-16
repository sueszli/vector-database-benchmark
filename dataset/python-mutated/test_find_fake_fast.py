from __future__ import annotations
import re
import gdb
import pwndbg
import tests
HEAP_FIND_FAKE_FAST = tests.binaries.get('heap_find_fake_fast.out')
target_address = None

def check_result(result, expected_size):
    if False:
        return 10
    ptrsize = pwndbg.gdblib.arch.ptrsize
    matches = re.findall('\\bAddr: (0x[0-9a-f]+)', result)
    assert len(matches) == 1
    addr = int(matches[0], 16)
    matches = re.findall('\\bsize: (0x[0-9a-f]+)', result)
    assert len(matches) == 1
    size = int(matches[0], 16)
    assert size == expected_size
    assert addr <= target_address - 2 * ptrsize
    size &= ~15
    assert addr + ptrsize + size > target_address

def check_no_results(result):
    if False:
        for i in range(10):
            print('nop')
    matches = re.findall('\\bAddr: (0x[0-9a-f]+)', result)
    assert len(matches) == 0

def test_find_fake_fast_command(start_binary):
    if False:
        print('Hello World!')
    global target_address
    start_binary(HEAP_FIND_FAKE_FAST)
    gdb.execute('break break_here')
    gdb.execute('continue')
    unmapped_heap_info = pwndbg.heap.ptmalloc.heap_for_ptr(int(gdb.lookup_global_symbol('fake_chunk').value()))
    assert pwndbg.gdblib.memory.peek(unmapped_heap_info) is None
    gdb.execute('find_fake_fast fake_chunk+0x80')
    target_address = pwndbg.gdblib.symbol.address('target_address')
    assert target_address is not None
    print(hex(target_address))
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_result(result, 32)
    result = gdb.execute('find_fake_fast --align &target_address', to_string=True)
    check_result(result, 32)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_result(result, 40)
    result = gdb.execute('find_fake_fast --align &target_address', to_string=True)
    check_result(result, 40)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_result(result, 32)
    result = gdb.execute('find_fake_fast --align &target_address', to_string=True)
    check_no_results(result)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_no_results(result)
    result = gdb.execute('find_fake_fast --align &target_address', to_string=True)
    check_no_results(result)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_no_results(result)
    result = gdb.execute('find_fake_fast --align &target_address', to_string=True)
    check_no_results(result)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_no_results(result)
    result = gdb.execute('find_fake_fast --align &target_address', to_string=True)
    check_no_results(result)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_result(result, 128)
    result = gdb.execute('find_fake_fast --align &target_address', to_string=True)
    check_result(result, 128)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_no_results(result)
    result = gdb.execute('find_fake_fast --align &target_address', to_string=True)
    check_no_results(result)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_no_results(result)
    result = gdb.execute('find_fake_fast &target_address 0x100', to_string=True)
    check_result(result, 256)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_no_results(result)
    result = gdb.execute('find_fake_fast &target_address 0x100', to_string=True)
    check_result(result, 256)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_no_results(result)
    result = gdb.execute('find_fake_fast &target_address 0x100', to_string=True)
    check_no_results(result)
    gdb.execute('continue')
    result = gdb.execute('find_fake_fast &target_address', to_string=True)
    check_no_results(result)
    result = gdb.execute('find_fake_fast &target_address --glibc-fastbin-bug', to_string=True)
    check_result(result, 12302652056652480544)
    gdb.execute('continue')