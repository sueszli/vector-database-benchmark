from __future__ import annotations
import gdb
import pwndbg.commands
import pwndbg.gdblib.regs
from pwndbg.commands import CommandCategory

class segment(gdb.Function):
    """Get the flat address of memory based off of the named segment register."""

    def __init__(self, name) -> None:
        if False:
            while True:
                i = 10
        super().__init__(name)
        self.name = name

    def invoke(self, arg=0):
        if False:
            for i in range(10):
                print('nop')
        result = getattr(pwndbg.gdblib.regs, self.name)
        return result + arg
segment('fsbase')
segment('gsbase')

@pwndbg.commands.ArgparsedCommand('Prints out the FS base address. See also $fsbase.', category=CommandCategory.REGISTER)
@pwndbg.commands.OnlyWhenRunning
@pwndbg.commands.OnlyWithArch(['i386', 'x86-64'])
def fsbase() -> None:
    if False:
        print('Hello World!')
    '\n    Prints out the FS base address. See also $fsbase.\n    '
    print(hex(int(pwndbg.gdblib.regs.fsbase)))

@pwndbg.commands.ArgparsedCommand('Prints out the GS base address. See also $gsbase.', category=CommandCategory.REGISTER)
@pwndbg.commands.OnlyWhenRunning
@pwndbg.commands.OnlyWithArch(['i386', 'x86-64'])
def gsbase() -> None:
    if False:
        print('Hello World!')
    '\n    Prints out the GS base address. See also $gsbase.\n    '
    print(hex(int(pwndbg.gdblib.regs.gsbase)))