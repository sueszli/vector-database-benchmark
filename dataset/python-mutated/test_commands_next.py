from __future__ import annotations
import gdb
import pytest
import pwndbg.gdblib.regs
import tests
REFERENCE_BINARY = tests.binaries.get('reference-binary.out')
CRASH_SIMPLE_BINARY = tests.binaries.get('crash_simple.out.hardcoded')

def test_command_nextproginstr_binary_not_running():
    if False:
        return 10
    out = gdb.execute('nextproginstr', to_string=True)
    assert out == 'nextproginstr: The program is not being run.\n'

def test_command_nextproginstr(start_binary):
    if False:
        return 10
    start_binary(REFERENCE_BINARY)
    gdb.execute('break main')
    gdb.execute('continue')
    out = gdb.execute('nextproginstr', to_string=True)
    assert out == 'The pc is already at the binary objfile code. Not stepping.\n'
    exec_bin_pages = [p for p in pwndbg.gdblib.vmmap.get() if p.objfile == pwndbg.gdblib.proc.exe and p.execute]
    assert any((pwndbg.gdblib.regs.pc in p for p in exec_bin_pages))
    main_page = pwndbg.gdblib.vmmap.find(pwndbg.gdblib.regs.pc)
    gdb.execute('break puts')
    gdb.execute('continue')
    assert 'libc' in pwndbg.gdblib.vmmap.find(pwndbg.gdblib.regs.rip).objfile
    gdb.execute('nextproginstr')
    assert pwndbg.gdblib.regs.pc in main_page
    out = gdb.execute('nextproginstr', to_string=True)
    assert out == 'The pc is already at the binary objfile code. Not stepping.\n'

@pytest.mark.parametrize('command', ('nextcall', 'nextjump', 'nextproginstr', 'nextret', 'nextsyscall', 'stepret', 'stepsyscall'))
def test_next_command_doesnt_freeze_crashed_binary(start_binary, command):
    if False:
        for i in range(10):
            print('nop')
    start_binary(REFERENCE_BINARY)
    if command == 'nextproginstr':
        pwndbg.gdblib.regs.pc = 4660
    gdb.execute(command, to_string=True)