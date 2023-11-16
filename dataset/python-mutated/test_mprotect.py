from __future__ import annotations
import gdb
import pwndbg
import tests
SMALL_BINARY = tests.binaries.get('crash_simple.out.hardcoded')

def test_mprotect_executes_properly(start_binary):
    if False:
        return 10
    '\n    Tests the mprotect command\n    '
    start_binary(SMALL_BINARY)
    pc = pwndbg.gdblib.regs.pc
    gdb.execute('mprotect %d 4096 PROT_EXEC|PROT_READ|PROT_WRITE' % pc)
    vm = pwndbg.gdblib.vmmap.find(pc)
    assert vm.read and vm.write and vm.execute
    gdb.execute('mprotect $pc 0x1000 PROT_NONE')
    vm = pwndbg.gdblib.vmmap.find(pc)
    assert not (vm.read and vm.write and vm.execute)

def test_cannot_run_mprotect_when_not_running(start_binary):
    if False:
        print('Hello World!')
    assert 'mprotect: The program is not being run.\n' == gdb.execute('mprotect 0x0 0x1000 PROT_EXEC|PROT_READ|PROT_WRITE', to_string=True)