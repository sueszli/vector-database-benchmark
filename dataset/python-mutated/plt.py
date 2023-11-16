"""Emulates instructions in the PLT to locate symbols more accurately.
"""
from __future__ import division
import logging
from pwnlib.args import args
from pwnlib.log import getLogger
from pwnlib.util import fiddling
from pwnlib.util import packing
log = getLogger(__name__)

def emulate_plt_instructions(elf, got, address, data, targets):
    if False:
        while True:
            i = 10
    'Emulates instructions in ``data``\n\n    Arguments:\n        elf(ELF): ELF that we are emulating\n        got(int): Address of the GOT, as expected in e.g. EBX\n        address(int): Address of ``data`` for emulation\n        data(str): Array of bytes to emulate\n        targets(list): List of target addresses\n\n    Returns:\n        :class:`dict`: Map of ``{address: target}`` for each address which\n            reaches one of the selected targets.\n    '
    rv = {}
    if not args.PLT_DEBUG:
        log.setLevel(logging.DEBUG + 1)
    if elf.endian == 'big' and elf.arch == 'mips':
        data = packing.unpack_many(data, bits=32, endian='little')
        data = packing.flat(data, bits=32, endian='big')
    (uc, ctx) = prepare_unicorn_and_context(elf, got, address, data)
    for (i, pc) in enumerate(range(address, address + len(data), 4)):
        if log.isEnabledFor(logging.DEBUG):
            log.debug('%s %#x', fiddling.enhex(data[i * 4:(i + 1) * 4]), pc)
            log.debug(elf.disasm(pc, 4))
        uc.context_restore(ctx)
        target = emulate_plt_instructions_inner(uc, elf, got, pc, data[i * 4:])
        if target in targets:
            log.debug('%#x -> %#x', pc, target)
            rv[pc] = target
    return rv

def prepare_unicorn_and_context(elf, got, address, data):
    if False:
        while True:
            i = 10
    import unicorn as U
    arch = {'aarch64': U.UC_ARCH_ARM64, 'amd64': U.UC_ARCH_X86, 'arm': U.UC_ARCH_ARM, 'i386': U.UC_ARCH_X86, 'mips': U.UC_ARCH_MIPS, 'mips64': U.UC_ARCH_MIPS, 'thumb': U.UC_ARCH_ARM, 'riscv32': U.UC_ARCH_RISCV, 'riscv64': U.UC_ARCH_RISCV}.get(elf.arch, None)
    if arch is None:
        log.warn('Could not emulate PLT instructions for %r' % elf)
        return {}
    emulation_bits = elf.bits
    if elf.arch == 'amd64' and elf.bits == 32:
        emulation_bits = 64
    mode = {32: U.UC_MODE_32, 64: U.UC_MODE_64}.get(emulation_bits)
    if elf.arch in ('arm', 'aarch64'):
        mode = U.UC_MODE_ARM
    uc = U.Uc(arch, mode)
    start = address & ~4095
    stop = address + len(data) + 4095 & ~4095
    if not 0 <= start <= stop <= 1 << elf.bits:
        return None
    uc.mem_map(start, stop - start)
    uc.mem_write(address, data)
    assert uc.mem_read(address, len(data)) == data
    magic_addr = 2088533116
    if elf.arch == 'mips':
        p_magic = packing.p32(magic_addr)
        start = got & ~4095
        try:
            uc.mem_map(start, 4096)
        except Exception:
            pass
        uc.mem_write(got, p_magic)
    return (uc, uc.context_save())

def emulate_plt_instructions_inner(uc, elf, got, pc, data):
    if False:
        while True:
            i = 10
    import unicorn as U
    stopped_addr = []
    magic_addr = 2088533116

    def hook_mem(uc, access, address, size, value, user_data):
        if False:
            return 10
        if elf.arch == 'mips' and address == got:
            return True
        user_data.append(address)
        uc.emu_stop()
        return False
    hooks = [uc.hook_add(U.UC_HOOK_MEM_READ | U.UC_HOOK_MEM_READ_UNMAPPED, hook_mem, stopped_addr)]
    if elf.arch == 'i386':
        uc.reg_write(U.x86_const.UC_X86_REG_EBX, got)
    if elf.arch == 'mips' and elf.bits == 32:
        OFFSET_GP_GOT = 32752
        uc.reg_write(U.mips_const.UC_MIPS_REG_GP, got + 32752)
    try:
        uc.emu_start(pc, until=-1, count=5)
    except U.UcError as error:
        UC_ERR = next((k for (k, v) in U.unicorn_const.__dict__.items() if error.errno == v and k.startswith('UC_ERR_')))
        log.debug('%#x: %s (%s)', pc, error, UC_ERR)
    if elf.arch == 'mips':
        pc = uc.reg_read(U.mips_const.UC_MIPS_REG_PC)
        if pc == magic_addr:
            t8 = uc.reg_read(U.mips_const.UC_MIPS_REG_T8)
            stopped_addr.append(elf._mips_got.get(t8, 0))
    retval = 0
    if stopped_addr:
        retval = stopped_addr.pop()
    for hook in hooks:
        uc.hook_del(hook)
    return retval