"""
Given an address in memory which does not contain a pointer elsewhere
into memory, attempt to describe the data as best as possible.

Currently prints out code, integers, or strings, in a best-effort manner
dependent on page permissions, the contents of the data, and any
supplemental information sources (e.g. active IDA Pro connection).
"""
from __future__ import annotations
import string
import pwndbg.color.enhance as E
import pwndbg.disasm
import pwndbg.gdblib.arch
import pwndbg.gdblib.config
import pwndbg.gdblib.memory
import pwndbg.gdblib.strings
import pwndbg.gdblib.symbol
import pwndbg.gdblib.typeinfo
import pwndbg.lib.cache
from pwndbg import color
from pwndbg.color.syntax_highlight import syntax_highlight
bad_instrs = ['.byte', '.long', 'rex.R', 'rex.XB', '.inst', '(bad)']

def good_instr(i) -> bool:
    if False:
        while True:
            i = 10
    return not any((bad in i for bad in bad_instrs))

def int_str(value: int) -> str:
    if False:
        i = 10
        return i + 15
    retval = '%#x' % int(value & pwndbg.gdblib.arch.ptrmask)
    packed = pwndbg.gdblib.arch.pack(int(value))
    if all((c in string.printable.encode('utf-8') for c in packed)):
        if len(retval) > 4:
            retval = '{} ({!r})'.format(retval, str(packed.decode('ascii', 'ignore')))
    return retval

def enhance(value: int, code: bool=True, safe_linking: bool=False) -> str:
    if False:
        print('Hello World!')
    "\n    Given the last pointer in a chain, attempt to characterize\n\n    Note that 'the last pointer in a chain' may not at all actually be a pointer.\n\n    Additionally, optimizations are made based on various sources of data for\n    'value'. For example, if it is set to RWX, we try to get information on whether\n    it resides on the stack, or in a RW section that *happens* to be RWX, to\n    determine which order to print the fields.\n\n    Arguments:\n        value(obj): Value to enhance\n        code(bool): Hint that indicates the value may be an instruction\n        safe_linking(bool): Whether this chain use safe-linking\n    "
    value = int(value)
    name = pwndbg.gdblib.symbol.get(value) or None
    page = pwndbg.gdblib.vmmap.find(value)
    can_read = True
    if not page or None is pwndbg.gdblib.memory.peek(value):
        can_read = False
    if not can_read:
        return E.integer(int_str(value))
    instr = None
    exe = page and page.execute
    rwx = page and page.rwx
    if '[stack' in page.objfile or '[heap' in page.objfile:
        rwx = exe = False
    if pwndbg.ida.available() and (not pwndbg.ida.GetFunctionName(value)):
        rwx = exe = False
    if exe:
        instr = pwndbg.disasm.one(value)
        if instr:
            instr = f'{instr.mnemonic} {instr.op_str}'
            if pwndbg.gdblib.config.syntax_highlight:
                instr = syntax_highlight(instr)
    szval = pwndbg.gdblib.strings.get(value) or None
    szval0 = szval
    if szval:
        szval = E.string(repr(szval))
    if value + pwndbg.gdblib.arch.ptrsize > page.end:
        return E.integer(int_str(value))
    intval = int(pwndbg.gdblib.memory.poi(pwndbg.gdblib.typeinfo.pvoid, value))
    if safe_linking:
        intval ^= value >> 12
    intval0 = intval
    if 0 <= intval < 10:
        intval = E.integer(str(intval))
    else:
        intval = E.integer('%#x' % int(intval & pwndbg.gdblib.arch.ptrmask))
    retval = []
    if not code:
        instr = None
    if instr and 'stack' in page.objfile:
        retval = [intval, szval]
    elif instr and rwx and (intval0 < 4096):
        retval = [intval, szval]
    elif instr and exe:
        if not rwx:
            if szval:
                retval = [instr, szval]
            else:
                retval = [instr]
        else:
            retval = [instr, intval, szval]
    elif szval:
        if len(szval0) < pwndbg.gdblib.arch.ptrsize:
            retval = [intval, szval]
        else:
            retval = [szval]
    else:
        return E.integer(int_str(intval0))
    retval = tuple(filter(lambda x: x is not None, retval))
    if len(retval) == 0:
        return E.unknown('???')
    if len(retval) == 1:
        return retval[0]
    return retval[0] + E.comment(color.strip(f" /* {'; '.join(retval[1:])} */"))