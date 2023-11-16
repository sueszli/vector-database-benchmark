"""
Commands for setting temporary breakpoints on the next
instruction of some type (call, branch, etc.)
"""
from __future__ import annotations
import re
from itertools import chain
import capstone
import gdb
import pwndbg.disasm
import pwndbg.gdblib.events
import pwndbg.gdblib.proc
import pwndbg.gdblib.regs
from pwndbg.color import message
jumps = {capstone.CS_GRP_CALL, capstone.CS_GRP_JUMP, capstone.CS_GRP_RET, capstone.CS_GRP_IRET}
interrupts = {capstone.CS_GRP_INT}

def clear_temp_breaks() -> None:
    if False:
        i = 10
        return i + 15
    if not pwndbg.gdblib.proc.alive:
        for bp in gdb.breakpoints():
            if bp.temporary and (not bp.visible):
                bp.delete()

def next_int(address=None):
    if False:
        while True:
            i = 10
    '\n    If there is a syscall in the current basic black,\n    return the instruction of the one closest to $PC.\n\n    Otherwise, return None.\n    '
    if address is None:
        ins = pwndbg.disasm.one(pwndbg.gdblib.regs.pc)
        if not ins:
            return None
        address = ins.next
    ins = pwndbg.disasm.one(address)
    while ins:
        ins_groups = set(ins.groups)
        if ins_groups & jumps:
            return None
        elif ins_groups & interrupts:
            return ins
        ins = pwndbg.disasm.one(ins.next)
    return None

def next_branch(address=None):
    if False:
        print('Hello World!')
    if address is None:
        ins = pwndbg.disasm.one(pwndbg.gdblib.regs.pc)
        if not ins:
            return None
        address = ins.next
    ins = pwndbg.disasm.one(address)
    while ins:
        if set(ins.groups) & jumps:
            return ins
        ins = pwndbg.disasm.one(ins.next)
    return None

def next_matching_until_branch(address=None, mnemonic=None, op_str=None):
    if False:
        i = 10
        return i + 15
    '\n    Finds the next instruction that matches the arguments between the given\n    address and the branch closest to it.\n    '
    if address is None:
        address = pwndbg.gdblib.regs.pc
    ins = pwndbg.disasm.one(address)
    while ins:
        mnemonic_match = ins.mnemonic.casefold() == mnemonic.casefold() if mnemonic else True
        op_str_match = True
        if op_str is not None:
            op_str_match = False
            ops = ''.join(ins.op_str.split()).casefold()
            if isinstance(op_str, str):
                op_str = ''.join(op_str.split()).casefold()
            elif isinstance(op_str, list):
                op_str = ''.join(chain.from_iterable((op.split() for op in op_str))).casefold()
            else:
                raise ValueError('op_str value is of an unsupported type')
            op_str_match = ops == op_str
        if mnemonic_match and op_str_match:
            return ins
        if set(ins.groups) & jumps:
            return None
        ins = pwndbg.disasm.one(ins.next)
    return None

def break_next_branch(address=None):
    if False:
        return 10
    ins = next_branch(address)
    if ins:
        gdb.Breakpoint('*%#x' % ins.address, internal=True, temporary=True)
        gdb.execute('continue', from_tty=False, to_string=True)
        return ins

def break_next_interrupt(address=None):
    if False:
        for i in range(10):
            print('nop')
    ins = next_int(address)
    if ins:
        gdb.Breakpoint('*%#x' % ins.address, internal=True, temporary=True)
        gdb.execute('continue', from_tty=False, to_string=True)
        return ins

def break_next_call(symbol_regex=None):
    if False:
        while True:
            i = 10
    symbol_regex = re.compile(symbol_regex) if symbol_regex else None
    while pwndbg.gdblib.proc.alive:
        if pwndbg.gdblib.proc.stopped_with_signal:
            return
        ins = break_next_branch()
        if not ins:
            break
        if capstone.CS_GRP_CALL not in ins.groups:
            continue
        if not symbol_regex or (ins.target_const and symbol_regex.match(hex(ins.target))) or (ins.symbol and symbol_regex.match(ins.symbol)):
            return ins

def break_next_ret(address=None):
    if False:
        print('Hello World!')
    while pwndbg.gdblib.proc.alive:
        if pwndbg.gdblib.proc.stopped_with_signal:
            return
        ins = break_next_branch(address)
        if not ins:
            break
        if capstone.CS_GRP_RET in ins.groups:
            return ins

def break_on_next_matching_instruction(mnemonic=None, op_str=None) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Breaks on next instuction that matches the arguments.\n    '
    if mnemonic is None and op_str is None:
        return False
    while pwndbg.gdblib.proc.alive:
        if pwndbg.gdblib.proc.stopped_with_signal:
            return False
        ins = next_matching_until_branch(mnemonic=mnemonic, op_str=op_str)
        if ins is not None:
            if ins.address != pwndbg.gdblib.regs.pc:
                print('Found instruction')
                gdb.Breakpoint('*%#x' % ins.address, internal=True, temporary=True)
                gdb.execute('continue', from_tty=False, to_string=True)
                return ins
            else:
                pass
        else:
            print('Moving to next branch')
            nb = next_branch(pwndbg.gdblib.regs.pc)
            if nb is not None:
                if nb.address != pwndbg.gdblib.regs.pc:
                    gdb.Breakpoint('*%#x' % nb.address, internal=True, temporary=True)
                    gdb.execute('continue', from_tty=False, to_string=True)
                else:
                    pass
        if pwndbg.gdblib.proc.alive:
            gdb.execute('si')
    return False

def break_on_program_code() -> bool:
    if False:
        for i in range(10):
            print('nop')
    "\n    Breaks on next instruction that belongs to process' objfile code\n\n    :return: True for success, False when process ended or when pc is not at the code or if a signal occurred\n    "
    exe = pwndbg.gdblib.proc.exe
    binary_exec_page_ranges = tuple(((p.start, p.end) for p in pwndbg.gdblib.vmmap.get() if p.objfile == exe and p.execute))
    pc = pwndbg.gdblib.regs.pc
    for (start, end) in binary_exec_page_ranges:
        if start <= pc < end:
            print(message.error('The pc is already at the binary objfile code. Not stepping.'))
            return False
    proc = pwndbg.gdblib.proc
    regs = pwndbg.gdblib.regs
    while proc.alive:
        if proc.stopped_with_signal:
            return False
        o = gdb.execute('si', from_tty=False, to_string=True)
        for (start, end) in binary_exec_page_ranges:
            if start <= regs.pc < end:
                return True
    return False

def break_on_next(address=None) -> None:
    if False:
        print('Hello World!')
    address = address or pwndbg.gdblib.regs.pc
    ins = pwndbg.disasm.one(address)
    gdb.Breakpoint('*%#x' % (ins.address + ins.size), temporary=True)
    gdb.execute('continue', from_tty=False, to_string=True)