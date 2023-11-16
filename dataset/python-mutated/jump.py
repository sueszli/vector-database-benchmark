from __future__ import annotations
from capstone import CS_GRP_JUMP
import pwndbg.disasm.x86
import pwndbg.gdblib.arch

def is_jump_taken(instruction):
    if False:
        for i in range(10):
            print('nop')
    '\n    Attempt to determine if a conditional instruction is executed.\n    Only valid for the current instruction.\n\n    Returns:\n        Returns True IFF the current instruction is a conditional\n        *or* jump instruction, and it is taken.\n\n        Returns False in all other cases.\n    '
    if CS_GRP_JUMP not in instruction.groups:
        return False
    if pwndbg.gdblib.regs.pc != instruction.address:
        return False
    return {'i386': pwndbg.disasm.x86.is_jump_taken, 'x86-64': pwndbg.disasm.x86.is_jump_taken}.get(pwndbg.gdblib.arch.current, lambda *a: False)(instruction)