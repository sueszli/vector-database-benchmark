from __future__ import annotations
import gdb
import pwndbg
import tests
REFERENCE_BINARY = tests.binaries.get('reference-binary.out')

def test_command_ignore_no_breakpoint_set():
    if False:
        for i in range(10):
            print('nop')
    out = gdb.execute('ignore 1001', to_string=True)
    assert out == 'No breakpoints set.\n'

def test_command_ignore_no_breakpoint_set_remove():
    if False:
        print('Hello World!')
    gdb.execute('file ' + REFERENCE_BINARY)
    gdb.execute('break break_here')
    gdb.execute('delete 1')
    out = gdb.execute('ignore 1001', to_string=True)
    assert out == 'No breakpoints set.\n'

def test_command_ignore_no_breakpoint_found(start_binary):
    if False:
        print('Hello World!')
    start_binary(REFERENCE_BINARY)
    gdb.execute('break main')
    out = gdb.execute('ignore 2 1001', to_string=True)
    assert out == 'No breakpoint number 2.\n'

def test_command_ignore_breakpoint_last_found_one():
    if False:
        return 10
    gdb.execute('file ' + REFERENCE_BINARY)
    gdb.execute('break break_here')
    out = gdb.execute('ignore 1', to_string=True)
    assert out == 'Will ignore next 1 crossings of breakpoint 1.\n'
    gdb.execute('run')
    assert not pwndbg.gdblib.proc.alive
    gdb.execute('run')
    assert pwndbg.gdblib.proc.alive

def test_command_ignore_breakpoint_last_found_two():
    if False:
        i = 10
        return i + 15
    gdb.execute('file ' + REFERENCE_BINARY)
    gdb.execute('break break_here')
    gdb.execute('break main')
    out = gdb.execute('ignore 15', to_string=True)
    assert out == 'Will ignore next 15 crossings of breakpoint 2.\n'

def test_command_ignore_breakpoint_last_negative():
    if False:
        i = 10
        return i + 15
    gdb.execute('file ' + REFERENCE_BINARY)
    gdb.execute('break break_here')
    out = gdb.execute('ignore -100', to_string=True)
    assert out == 'Will ignore next 0 crossings of breakpoint 1.\n'