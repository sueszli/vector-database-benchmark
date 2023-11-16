from __future__ import annotations
import argparse
import gdb
import pwnlib
from pwnlib import asm
import pwndbg.chain
import pwndbg.commands
import pwndbg.enhance
import pwndbg.gdblib.file
import pwndbg.wrappers.checksec
import pwndbg.wrappers.readelf
from pwndbg.commands import CommandCategory
from pwndbg.lib.regs import reg_sets
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='\nCalls the mprotect syscall and prints its result value.\n\nNote that the mprotect syscall may fail for various reasons\n(see `man mprotect`) and a non-zero error return value\ncan be decoded with the `errno <value>` command.\n\nExamples:\n    mprotect $rsp 4096 PROT_READ|PROT_WRITE|PROT_EXEC\n    mprotect some_symbol 0x1000 PROT_NONE\n')
parser.add_argument('addr', help='Page-aligned address to all mprotect on.', type=pwndbg.commands.sloppy_gdb_parse)
parser.add_argument('length', help='Count of bytes to call mprotect on. Needs to be multiple of page size.', type=int)
parser.add_argument('prot', help='Prot string as in mprotect(2). Eg. "PROT_READ|PROT_EXEC"', type=str)
SYS_MPROTECT = 125
prot_dict = {'PROT_NONE': 0, 'PROT_READ': 1, 'PROT_WRITE': 2, 'PROT_EXEC': 4}

def prot_str_to_val(protstr):
    if False:
        i = 10
        return i + 15
    'Heuristic to convert PROT_EXEC|PROT_WRITE to integer value.'
    prot_int = 0
    for (k, v) in prot_dict.items():
        if k in protstr:
            prot_int |= v
    return prot_int

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.MEMORY)
@pwndbg.commands.OnlyWhenRunning
def mprotect(addr, length, prot) -> None:
    if False:
        print('Hello World!')
    prot_int = prot_str_to_val(prot)
    shellcode_asm = pwnlib.shellcraft.syscall('SYS_mprotect', int(pwndbg.lib.memory.page_align(addr)), int(length), int(prot_int))
    shellcode = asm.asm(shellcode_asm)
    current_regs = reg_sets[pwndbg.gdblib.arch.current]
    regs_to_save = current_regs.args + (current_regs.retval, current_regs.pc)
    saved_registers = {reg: pwndbg.gdblib.regs[reg] for reg in regs_to_save}
    saved_instruction_bytes = pwndbg.gdblib.memory.read(saved_registers[current_regs.pc], len(shellcode))
    pwndbg.gdblib.memory.write(saved_registers[current_regs.pc], shellcode)
    gdb.execute('nextsyscall')
    gdb.execute('stepi')
    ret = pwndbg.gdblib.regs[current_regs.retval]
    print('mprotect returned %d (%s)' % (ret, current_regs.retval))
    pwndbg.gdblib.memory.write(saved_registers[current_regs.pc], saved_instruction_bytes)
    for (register, value) in saved_registers.items():
        setattr(pwndbg.gdblib.regs, register, value)