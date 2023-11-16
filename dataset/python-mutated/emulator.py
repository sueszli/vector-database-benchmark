"""
Emulation assistance from Unicorn.
"""
from __future__ import annotations
import binascii
import re
import capstone as C
import gdb
import unicorn as U
import unicorn.riscv_const
import pwndbg.disasm
import pwndbg.gdblib.arch
import pwndbg.gdblib.memory
import pwndbg.gdblib.regs

def parse_consts(u_consts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Unicorn "consts" is a python module consisting of a variable definition\n    for each known entity. We repack it here as a dict for performance.\n    '
    consts = {}
    for name in dir(u_consts):
        if name.startswith('UC_'):
            consts[name] = getattr(u_consts, name)
    return consts
arch_to_UC = {'i386': U.UC_ARCH_X86, 'x86-64': U.UC_ARCH_X86, 'mips': U.UC_ARCH_MIPS, 'sparc': U.UC_ARCH_SPARC, 'arm': U.UC_ARCH_ARM, 'aarch64': U.UC_ARCH_ARM64, 'rv32': U.UC_ARCH_RISCV, 'rv64': U.UC_ARCH_RISCV}
arch_to_UC_consts = {'i386': parse_consts(U.x86_const), 'x86-64': parse_consts(U.x86_const), 'mips': parse_consts(U.mips_const), 'sparc': parse_consts(U.sparc_const), 'arm': parse_consts(U.arm_const), 'aarch64': parse_consts(U.arm64_const), 'rv32': parse_consts(U.riscv_const), 'rv64': parse_consts(U.riscv_const)}
DEBUG = False
if DEBUG:

    def debug(fmt, args=()) -> None:
        if False:
            for i in range(10):
                print('nop')
        print(fmt % args)
else:

    def debug(fmt, args=()) -> None:
        if False:
            print('Hello World!')
        pass
arch_to_SYSCALL = {U.UC_ARCH_X86: [C.x86_const.X86_INS_SYSCALL, C.x86_const.X86_INS_SYSENTER, C.x86_const.X86_INS_SYSEXIT, C.x86_const.X86_INS_SYSRET, C.x86_const.X86_INS_IRET, C.x86_const.X86_INS_IRETD, C.x86_const.X86_INS_IRETQ, C.x86_const.X86_INS_INT, C.x86_const.X86_INS_INT1, C.x86_const.X86_INS_INT3], U.UC_ARCH_MIPS: [C.mips_const.MIPS_INS_SYSCALL], U.UC_ARCH_SPARC: [C.sparc_const.SPARC_INS_T], U.UC_ARCH_ARM: [C.arm_const.ARM_INS_SVC], U.UC_ARCH_ARM64: [C.arm64_const.ARM64_INS_SVC], U.UC_ARCH_PPC: [C.ppc_const.PPC_INS_SC], U.UC_ARCH_RISCV: [C.riscv_const.RISCV_INS_ECALL]}
blacklisted_regs = ['ip', 'cs', 'ds', 'es', 'fs', 'gs', 'ss', 'fsbase', 'gsbase']
'\ne = pwndbg.emu.emulator.Emulator()\ne.until_jump()\n'

class Emulator:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.arch = pwndbg.gdblib.arch.current
        if self.arch not in arch_to_UC:
            raise NotImplementedError(f'Cannot emulate code for {self.arch}')
        self.consts = arch_to_UC_consts[self.arch]
        self.const_regs = {}
        r = re.compile('^UC_.*_REG_(.*)$')
        for (k, v) in self.consts.items():
            m = r.match(k)
            if m:
                self.const_regs[m.group(1)] = v
        self.uc_mode = self.get_uc_mode()
        debug('# Instantiating Unicorn for %s', self.arch)
        debug('uc = U.Uc(%r, %r)', (arch_to_UC[self.arch], self.uc_mode))
        self.uc = U.Uc(arch_to_UC[self.arch], self.uc_mode)
        self.regs = pwndbg.gdblib.regs.current
        self._prev = None
        self._prev_size = None
        self._curr = None
        for reg in list(self.regs.retaddr) + list(self.regs.misc) + list(self.regs.common) + list(self.regs.flags):
            enum = self.get_reg_enum(reg)
            if not reg:
                debug('# Could not set register %r', reg)
                continue
            if reg in blacklisted_regs:
                debug('Skipping blacklisted register %r', reg)
                continue
            value = getattr(pwndbg.gdblib.regs, reg)
            if None in (enum, value):
                if reg not in blacklisted_regs:
                    debug('# Could not set register %r', reg)
                continue
            if value == 0:
                continue
            name = f'U.x86_const.UC_X86_REG_{reg.upper()}'
            debug('uc.reg_write(%(name)s, %(value)#x)', locals())
            self.uc.reg_write(enum, value)
        self.hook_add(U.UC_HOOK_MEM_UNMAPPED, self.hook_mem_invalid)
        self.hook_add(U.UC_HOOK_INTR, self.hook_intr)
        self.map_page(pwndbg.gdblib.regs.pc)
        if DEBUG:
            self.hook_add(U.UC_HOOK_CODE, self.trace_hook)

    def __getattr__(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        reg = self.get_reg_enum(name)
        if reg:
            return self.uc.reg_read(reg)
        raise AttributeError(f'AttributeError: {self!r} object has no attribute {name!r}')

    def update_pc(self, pc=None) -> None:
        if False:
            i = 10
            return i + 15
        if pc is None:
            pc = pwndbg.gdblib.regs.pc
        self.uc.reg_write(self.get_reg_enum(self.regs.pc), pc)

    def get_uc_mode(self):
        if False:
            while True:
                i = 10
        '\n        Retrieve the mode used by Unicorn for the current architecture.\n        '
        arch = pwndbg.gdblib.arch.current
        mode = 0
        if arch == 'armcm':
            mode |= U.UC_MODE_MCLASS | U.UC_MODE_THUMB if pwndbg.gdblib.regs.xpsr & 1 << 24 else U.UC_MODE_MCLASS
        elif arch in ('arm', 'aarch64'):
            mode |= U.UC_MODE_THUMB if pwndbg.gdblib.regs.cpsr & 1 << 5 else U.UC_MODE_ARM
        elif arch == 'mips' and 'isa32r6' in gdb.newest_frame().architecture().name():
            mode |= U.UC_MODE_MIPS32R6
        else:
            mode |= {4: U.UC_MODE_32, 8: U.UC_MODE_64}[pwndbg.gdblib.arch.ptrsize]
        if pwndbg.gdblib.arch.endian == 'little':
            mode |= U.UC_MODE_LITTLE_ENDIAN
        else:
            mode |= U.UC_MODE_BIG_ENDIAN
        return mode

    def map_page(self, page) -> bool:
        if False:
            print('Hello World!')
        page = pwndbg.lib.memory.page_align(page)
        size = pwndbg.lib.memory.PAGE_SIZE
        debug('# Mapping %#x-%#x', (page, page + size))
        try:
            data = pwndbg.gdblib.memory.read(page, size)
            data = bytes(data)
        except gdb.MemoryError:
            debug('Could not map page %#x during emulation! [exception]', page)
            return False
        if not data:
            debug('Could not map page %#x during emulation! [no data]', page)
            return False
        debug('uc.mem_map(%(page)#x, %(size)#x)', locals())
        self.uc.mem_map(page, size)
        debug('# Writing %#x bytes', len(data))
        debug('uc.mem_write(%(page)#x, ...)', locals())
        self.uc.mem_write(page, data)
        return True

    def hook_mem_invalid(self, uc, access, address, size: int, value, user_data) -> bool:
        if False:
            i = 10
            return i + 15
        debug('# Invalid access at %#x', address)
        start = pwndbg.lib.memory.page_align(address)
        size = pwndbg.lib.memory.page_size_align(address + size - start)
        stop = start + size
        for page in range(start, stop, pwndbg.lib.memory.PAGE_SIZE):
            if not self.map_page(page):
                return False
        return True

    def hook_intr(self, uc, intno, user_data) -> None:
        if False:
            i = 10
            return i + 15
        '\n        We never want to emulate through an interrupt.  Just stop.\n        '
        debug('Got an interrupt')
        self.uc.emu_stop()

    def get_reg_enum(self, reg):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the Unicorn Emulator enum code for the named register.\n\n        Also supports general registers like 'sp' and 'pc'.\n        "
        if not self.regs:
            return None
        if reg in self.regs.all:
            e = self.const_regs.get(reg.upper(), None)
            if e is not None:
                return e
        if hasattr(self.regs, reg):
            return self.get_reg_enum(getattr(self.regs, reg))
        elif reg == 'sp':
            return self.get_reg_enum(self.regs.stack)
        return None

    def hook_add(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        rv = self.uc.hook_add(*a, **kw)
        debug('%r = uc.hook_add(*%r, **%r)', (rv, a, kw))
        return rv

    def hook_del(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        debug('uc.hook_del(*%r, **%r)', (a, kw))
        return self.uc.hook_del(*a, **kw)

    def emu_start(self, *a, **kw):
        if False:
            return 10
        debug('uc.emu_start(*%r, **%r)', (a, kw))
        return self.uc.emu_start(*a, **kw)

    def emu_stop(self, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        debug('uc.emu_stop(*%r, **%r)', (a, kw))
        return self.uc.emu_stop(*a, **kw)

    def emulate_with_hook(self, hook, count=512) -> None:
        if False:
            for i in range(10):
                print('nop')
        ident = self.hook_add(U.UC_HOOK_CODE, hook)
        try:
            self.emu_start(self.pc, 0, count=count)
        finally:
            self.hook_del(ident)

    def mem_read(self, *a, **kw):
        if False:
            print('Hello World!')
        debug('uc.mem_read(*%r, **%r)', (a, kw))
        return self.uc.mem_read(*a, **kw)
    jump_types = {C.CS_GRP_CALL, C.CS_GRP_JUMP, C.CS_GRP_RET}

    def until_jump(self, pc=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Emulates instructions starting at the specified address until the\n        program counter is set to an address which does not linearly follow\n        the previously-emulated instruction.\n\n        Arguments:\n            pc(int): Address to start at.  If `None`, uses the current instruction.\n            types(list,set): List of instruction groups to stop at.\n                By default, it stops at all jumps, calls, and returns.\n\n        Return:\n            Returns a tuple containing the address of the jump instruction,\n            and its target in the format (address, target).\n\n            If emulation is forced to stop (e.g., because of a syscall or\n            invalid memory access) then address is the instruction which\n            could not be emulated through, and target will be None.\n\n        Notes:\n            This routine does not consider 'call $+5'\n        "
        if pc is not None:
            self.update_pc(pc)
        self._prev = None
        self._prev_size = None
        self._curr = None
        self.emulate_with_hook(self.until_jump_hook_code)
        return (self._prev, self._curr)

    def until_jump_hook_code(self, _uc, address, instruction_size: int, _user_data) -> None:
        if False:
            return 10
        if self._prev is None:
            pass
        elif self._prev + self._prev_size == address:
            pass
        else:
            self._curr = address
            debug('%#x %#X --> %#x', (self._prev, self._prev_size, self._curr))
            self.emu_stop()
            return
        self._prev = address
        self._prev_size = instruction_size

    def until_call(self, pc=None):
        if False:
            print('Hello World!')
        (addr, target) = self.until_jump(pc)
        while target and C.CS_GRP_CALL not in pwndbg.disasm.one(addr).groups:
            (addr, target) = self.until_jump(target)
        return (addr, target)

    def until_syscall(self, pc=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Emulates instructions starting at the specified address until the program\n        counter points at a syscall instruction (int 0x80, svc, etc.).\n        '
        self.until_syscall_address = None
        self.emulate_with_hook(self.until_syscall_hook_code)
        return (self.until_syscall_address, None)

    def until_syscall_hook_code(self, uc, address, size: int, user_data) -> None:
        if False:
            while True:
                i = 10
        data = binascii.hexlify(self.mem_read(address, size))
        debug('# Executing instruction at %(address)#x with bytes %(data)s', locals())
        self.until_syscall_address = address

    def single_step(self, pc=None):
        if False:
            for i in range(10):
                print('nop')
        'Steps one instruction.\n\n        Yields:\n            Each iteration, yields a tuple of (address, instruction_size).=\n\n            A StopIteration is raised if a fault or syscall or call instruction\n            is encountered.\n        '
        self._single_step = (None, None)
        pc = pc or self.pc
        insn = pwndbg.disasm.one(pc)
        if insn is None:
            debug("Can't disassemble instruction at %#x", pc)
            return self._single_step
        debug('# Single-stepping at %#x: %s %s', (pc, insn.mnemonic, insn.op_str))
        try:
            self.single_step_hook_hit_count = 0
            self.emulate_with_hook(self.single_step_hook_code, count=1)
        except U.unicorn.UcError as e:
            self._single_step = (None, None)
        return self._single_step

    def single_step_iter(self, pc=None):
        if False:
            while True:
                i = 10
        a = self.single_step(pc)
        while a:
            yield a
            a = self.single_step(pc)

    def single_step_hook_code(self, _uc, address, instruction_size: int, _user_data) -> None:
        if False:
            i = 10
            return i + 15
        if self.single_step_hook_hit_count == 0:
            debug('# single_step: %#-8x', address)
            self._single_step = (address, instruction_size)
            self.single_step_hook_hit_count += 1

    def dumpregs(self) -> None:
        if False:
            return 10
        for reg in list(self.regs.retaddr) + list(self.regs.misc) + list(self.regs.common) + list(self.regs.flags):
            enum = self.get_reg_enum(reg)
            if not reg or enum is None:
                debug('# Could not dump register %r', reg)
                continue
            name = f'U.x86_const.UC_X86_REG_{reg.upper()}'
            value = self.uc.reg_read(enum)
            debug('uc.reg_read(%(name)s) ==> %(value)x', locals())

    def trace_hook(self, _uc, address, instruction_size: int, _user_data) -> None:
        if False:
            i = 10
            return i + 15
        data = binascii.hexlify(self.mem_read(address, instruction_size))
        debug('# trace_hook: %#-8x %r', (address, data))