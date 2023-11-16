from __future__ import annotations
import gdb
from pwnlib.util.cyclic import cyclic
import pwndbg.gdblib.arch
import pwndbg.gdblib.memory
import pwndbg.gdblib.regs
import tests
REFERENCE_BINARY = tests.binaries.get('reference-binary.out')

def test_command_cyclic_value(start_binary):
    if False:
        while True:
            i = 10
    '\n    Tests lookup on a constant value\n    '
    start_binary(REFERENCE_BINARY)
    ptr_size = pwndbg.gdblib.arch.ptrsize
    test_offset = 37
    pattern = cyclic(length=80, n=ptr_size)
    val = int.from_bytes(pattern[test_offset:test_offset + ptr_size], pwndbg.gdblib.arch.endian)
    out = gdb.execute(f'cyclic -l {hex(val)}', to_string=True)
    assert out == "Finding cyclic pattern of 8 bytes: b'aaafaaaa' (hex: 0x6161616661616161)\nFound at offset 37\n"

def test_command_cyclic_register(start_binary):
    if False:
        i = 10
        return i + 15
    '\n    Tests lookup on a register\n    '
    start_binary(REFERENCE_BINARY)
    ptr_size = pwndbg.gdblib.arch.ptrsize
    test_offset = 45
    pattern = cyclic(length=80, n=ptr_size)
    pwndbg.gdblib.regs.rdi = int.from_bytes(pattern[test_offset:test_offset + ptr_size], pwndbg.gdblib.arch.endian)
    out = gdb.execute('cyclic -l $rdi', to_string=True)
    assert out == "Finding cyclic pattern of 8 bytes: b'aaagaaaa' (hex: 0x6161616761616161)\nFound at offset 45\n"

def test_command_cyclic_address(start_binary):
    if False:
        print('Hello World!')
    '\n    Tests lookup on a memory address\n    '
    start_binary(REFERENCE_BINARY)
    addr = pwndbg.gdblib.regs.rsp
    ptr_size = pwndbg.gdblib.arch.ptrsize
    test_offset = 48
    pattern = cyclic(length=80, n=ptr_size)
    pwndbg.gdblib.memory.write(addr, pattern)
    out = gdb.execute(f"cyclic -l '{{unsigned long}}{hex(addr + test_offset)}'", to_string=True)
    assert out == "Finding cyclic pattern of 8 bytes: b'gaaaaaaa' (hex: 0x6761616161616161)\nFound at offset 48\n"

def test_command_cyclic_wrong_alphabet():
    if False:
        i = 10
        return i + 15
    out = gdb.execute('cyclic -l 1234', to_string=True)
    assert out == "Finding cyclic pattern of 4 bytes: b'\\xd2\\x04\\x00\\x00' (hex: 0xd2040000)\nPattern contains characters not present in the alphabet\n"

def test_command_cyclic_wrong_length():
    if False:
        for i in range(10):
            print('nop')
    out = gdb.execute('cyclic -l qwerty', to_string=True)
    assert out == 'Lookup pattern must be 4 bytes (use `-n <length>` to lookup pattern of different length)\n'