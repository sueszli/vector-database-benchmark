from __future__ import annotations
import gdb
import pwndbg.gdblib.regs
import tests
REFERENCE_BINARY = tests.binaries.get('reference-binary.out')

def test_command_distance(start_binary):
    if False:
        for i in range(10):
            print('nop')
    start_binary(REFERENCE_BINARY)
    rsp = pwndbg.gdblib.regs.rsp
    result = gdb.execute('distance $rsp $rsp+0x10', to_string=True)
    assert result == f'{rsp:#x}->{rsp + 16:#x} is 0x10 bytes (0x2 words)\n'