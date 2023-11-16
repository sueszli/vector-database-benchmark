from __future__ import annotations
import gdb
import tests
GOSAMPLE_X64 = tests.binaries.get('gosample.x64')
GOSAMPLE_X86 = tests.binaries.get('gosample.x86')

def test_typeinfo_go_x64(start_binary):
    if False:
        return 10
    "\n    Tests pwndbg's typeinfo knows about the Go x64 types.\n    Catches: Python Exception <class 'gdb.error'> No type named u8.:\n    Test catches the issue only if the binaries are not stripped.\n    "
    gdb.execute('file ' + GOSAMPLE_X64)
    start = gdb.execute('start', to_string=True)
    assert 'Python Exception' not in start

def test_typeinfo_go_x86(start_binary):
    if False:
        print('Hello World!')
    "\n    Tests pwndbg's typeinfo knows about the Go x32 types\n    Catches: Python Exception <class 'gdb.error'> No type named u8.:\n    Test catches the issue only if the binaries are not stripped.\n    "
    gdb.execute('file ' + GOSAMPLE_X86)
    start = gdb.execute('start', to_string=True)
    assert 'Python Exception' not in start