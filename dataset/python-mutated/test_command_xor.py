from __future__ import annotations
import gdb
import pwndbg.gdblib.memory
import pwndbg.gdblib.regs
import tests
from pwndbg.commands.xor import memfrob
REFERENCE_BINARY = tests.binaries.get('reference-binary.out')

def test_command_xor_with_gdb_execute(start_binary):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests simple xoring\n    '
    start_binary(REFERENCE_BINARY)
    before = pwndbg.gdblib.regs.rsp
    pwndbg.gdblib.memory.write(before, b'aaaaaaaa')
    gdb.execute("xor $rsp ' ' 4")
    after = pwndbg.gdblib.memory.read(before, 8)
    assert after == b'AAAAaaaa'

def test_command_xor_with_int(start_binary):
    if False:
        while True:
            i = 10
    '\n    Tests simple xoring\n    '
    start_binary(REFERENCE_BINARY)
    before = pwndbg.gdblib.regs.rsp
    assert isinstance(before, int)
    pwndbg.gdblib.memory.write(before, b'aaaaaaaa')
    gdb.execute(f"xor {before} ' ' 4")
    after = pwndbg.gdblib.memory.read(before, 8)
    assert after == b'AAAAaaaa'

def test_command_xor_with_hex(start_binary):
    if False:
        return 10
    '\n    Tests simple xoring\n    '
    start_binary(REFERENCE_BINARY)
    before = pwndbg.gdblib.regs.rsp
    before_hex = hex(before)
    assert isinstance(before_hex, str)
    pwndbg.gdblib.memory.write(before, b'aaaaaaaa')
    gdb.execute(f"xor {before_hex} ' ' 4")
    after = pwndbg.gdblib.memory.read(before, 8)
    assert after == b'AAAAaaaa'

def test_command_memfrob(start_binary):
    if False:
        return 10
    start_binary(REFERENCE_BINARY)
    before = pwndbg.gdblib.regs.rsp
    pwndbg.gdblib.memory.write(before, b'aaaaaaaa')
    memfrob(before, 4)
    after = pwndbg.gdblib.memory.read(before, 8)
    assert after == b'KKKKaaaa'