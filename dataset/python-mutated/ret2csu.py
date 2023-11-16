from __future__ import absolute_import
from __future__ import division
from capstone import *
from capstone.x86 import *
from .rop import Padding
from ..log import getLogger
from ..util.packing import p64
log = getLogger(__name__)

def ret2csu(rop, elf, edi, rsi, rdx, rbx, rbp, r12, r13, r14, r15, call=None):
    if False:
        return 10
    'Build a ret2csu ROPchain\n\n    Arguments:\n        edi, rsi, rdx: Three primary registers to populate\n        rbx, rbp, r12, r13, r14, r15: Optional registers to populate\n        call: Pointer to the address of a function to call during\n            second gadget. If None then use the address of _fini in the\n            .dynamic section. .got.plt entries are a good target. Required\n            for PIE binaries.\n    '
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    md.skipdata = True
    if '__libc_csu_init' not in elf.symbols:
        textaddr = elf.offset_to_vaddr(elf.get_section_by_name('.text').header.sh_offset)
        entry = elf.entry
        data = elf.section('.text')[entry - textaddr:]
        mnemonic = elf.pie and 'lea' or 'mov'
        for insn in md.disasm(data, entry):
            if insn.mnemonic == mnemonic:
                if mnemonic == 'lea':
                    addr = insn.address + insn.size + insn.disp
                else:
                    addr = insn.operands[1].imm
                if insn.operands[0].reg == X86_REG_R8:
                    elf.sym['__libc_csu_fini'] = addr
                if insn.operands[0].reg == X86_REG_RCX:
                    elf.sym['__libc_csu_init'] = addr
                    break
            elif insn.mnemonic == 'xor' and insn.operands[0].reg == insn.operands[1].reg == X86_REG_ECX:
                log.error('This binary is compiled for glibc 2.34+ and does not have __libc_csu_init')
            elif insn.mnemonic in ('hlt', 'jmp', 'call', 'syscall'):
                log.error('No __libc_csu_init (no glibc _start)')
        else:
            log.error('Weird _start, definitely no __libc_csu_init')
    if not elf.pie and (not call):
        call = next(elf.search(p64(elf.dynamic_by_tag('DT_FINI')['d_ptr'])))
    elif elf.pie and (not call):
        log.error("No non-PIE binaries in [elfs], 'call' parameter is required")
    csu_function = elf.read(elf.sym['__libc_csu_init'], elf.sym['__libc_csu_fini'] - elf.sym['__libc_csu_init'])
    for insn in md.disasm(csu_function, elf.sym['__libc_csu_init']):
        if insn.mnemonic == 'pop' and insn.operands[0].reg == X86_REG_RBX:
            rop.raw(insn.address)
            break
    rop.raw(0)
    rop.raw(1)
    for insn in md.disasm(csu_function, elf.sym['__libc_csu_init']):
        if insn.mnemonic == 'mov' and insn.operands[0].reg == X86_REG_RDX and (insn.operands[1].reg == X86_REG_R13):
            rop.raw(call)
            rop.raw(rdx)
            rop.raw(rsi)
            rop.raw(edi)
            rop.raw(insn.address)
            break
        elif insn.mnemonic == 'mov' and insn.operands[0].reg == X86_REG_RDX and (insn.operands[1].reg == X86_REG_R14):
            rop.raw(edi)
            rop.raw(rsi)
            rop.raw(rdx)
            rop.raw(call)
            rop.raw(insn.address)
            break
        elif insn.mnemonic == 'mov' and insn.operands[0].reg == X86_REG_RDX and (insn.operands[1].reg == X86_REG_R15):
            rop.raw(call)
            rop.raw(edi)
            rop.raw(rsi)
            rop.raw(rdx)
            rop.raw(insn.address)
            break
    else:
        log.error('This CSU init variant is not supported by pwntools')
    rop.raw(Padding('<add rsp, 8>'))
    rop.raw(rbx)
    rop.raw(rbp)
    rop.raw(r12)
    rop.raw(r13)
    rop.raw(r14)
    rop.raw(r15)