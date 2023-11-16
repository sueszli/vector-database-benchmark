import logging
from ..native.memory import MemoryException
from ..core.smtlib import issymbolic
from unicorn import *
from unicorn.x86_const import *
from unicorn.arm_const import *
from unicorn.arm64_const import *
from capstone import *
logger = logging.getLogger(__name__)

class EmulatorException(Exception):
    """
    Emulator exception
    """
    pass

class UnicornEmulator:
    """
    Helper class to emulate a single instruction via Unicorn.
    """

    def __init__(self, cpu):
        if False:
            i = 10
            return i + 15
        self._cpu = cpu
        text = cpu.memory.map_containing(cpu.PC)
        self._should_be_mapped = {text.start: (len(text), UC_PROT_READ | UC_PROT_EXEC)}
        self._should_be_written = {}
        if self._cpu.arch == CS_ARCH_ARM:
            self._uc_arch = UC_ARCH_ARM
            self._uc_mode = {CS_MODE_ARM: UC_MODE_ARM, CS_MODE_THUMB: UC_MODE_THUMB}[self._cpu.mode]
        elif self._cpu.arch == CS_ARCH_ARM64:
            self._uc_arch = UC_ARCH_ARM64
            self._uc_mode = UC_MODE_ARM
            if self._cpu.mode != UC_MODE_ARM:
                raise EmulatorException('Aarch64/Arm64 cannot have different uc mode than ARM.')
        elif self._cpu.arch == CS_ARCH_X86:
            self._uc_arch = UC_ARCH_X86
            self._uc_mode = {CS_MODE_32: UC_MODE_32, CS_MODE_64: UC_MODE_64}[self._cpu.mode]
        else:
            raise NotImplementedError(f'Unsupported architecture: {self._cpu.arch}')

    def reset(self):
        if False:
            i = 10
            return i + 15
        self._emu = Uc(self._uc_arch, self._uc_mode)
        self._to_raise = None

    def _create_emulated_mapping(self, uc, address):
        if False:
            i = 10
            return i + 15
        "\n        Create a mapping in Unicorn and note that we'll need it if we retry.\n        :param uc: The Unicorn instance.\n        :param address: The address which is contained by the mapping.\n        :rtype Map\n        "
        m = self._cpu.memory.map_containing(address)
        permissions = UC_PROT_NONE
        if 'r' in m.perms:
            permissions |= UC_PROT_READ
        if 'w' in m.perms:
            permissions |= UC_PROT_WRITE
        if 'x' in m.perms:
            permissions |= UC_PROT_EXEC
        uc.mem_map(m.start, len(m), permissions)
        self._should_be_mapped[m.start] = (len(m), permissions)
        return m

    def get_unicorn_pc(self):
        if False:
            for i in range(10):
                print('nop')
        if self._cpu.arch == CS_ARCH_ARM:
            return self._emu.reg_read(UC_ARM_REG_R15)
        elif self._cpu.arch == CS_ARCH_ARM64:
            return self._emu.reg_read(UC_ARM64_REG_PC)
        elif self._cpu.arch == CS_ARCH_X86:
            if self._cpu.mode == CS_MODE_32:
                return self._emu.reg_read(UC_X86_REG_EIP)
            elif self._cpu.mode == CS_MODE_64:
                return self._emu.reg_read(UC_X86_REG_RIP)
        else:
            raise EmulatorException(f'Getting PC after unicorn emulation for {self._cpu.arch} architecture is not implemented')

    def _hook_xfer_mem(self, uc, access, address, size, value, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle memory operations from unicorn.\n        '
        assert access in (UC_MEM_WRITE, UC_MEM_READ, UC_MEM_FETCH)
        if access == UC_MEM_WRITE:
            self._cpu.write_int(address, value, size * 8)
        elif access == UC_MEM_READ:
            value = self._cpu.read_bytes(address, size)
            if address in self._should_be_written:
                return True
            self._should_be_written[address] = value
            self._should_try_again = True
            return False
        return True

    def _hook_unmapped(self, uc, access, address, size, value, data):
        if False:
            while True:
                i = 10
        '\n        We hit an unmapped region; map it into unicorn.\n        '
        try:
            m = self._create_emulated_mapping(uc, address)
        except MemoryException as e:
            self._to_raise = e
            self._should_try_again = False
            return False
        self._should_try_again = True
        return False

    def _interrupt(self, uc, number, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle software interrupt (SVC/INT)\n        '
        from ..native.cpu.abstractcpu import Interruption
        self._to_raise = Interruption(number)
        return True

    def _to_unicorn_id(self, reg_name):
        if False:
            for i in range(10):
                print('nop')
        if self._cpu.arch == CS_ARCH_ARM:
            return globals()['UC_ARM_REG_' + reg_name]
        elif self._cpu.arch == CS_ARCH_ARM64:
            return globals()['UC_ARM64_REG_' + reg_name]
        elif self._cpu.arch == CS_ARCH_X86:
            return globals()['UC_X86_REG_' + reg_name]
        else:
            raise TypeError(f'Cannot convert {reg_name} to unicorn register id')

    def emulate(self, instruction, reset=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Emulate a single instruction.\n        '
        while True:
            if reset:
                self.reset()
                for base in self._should_be_mapped:
                    (size, perms) = self._should_be_mapped[base]
                    self._emu.mem_map(base, size, perms)
            for (address, values) in self._should_be_written.items():
                for (offset, byte) in enumerate(values, start=address):
                    if issymbolic(byte):
                        from ..native.cpu.abstractcpu import ConcretizeMemory
                        raise ConcretizeMemory(self._cpu.memory, offset, 8, 'Concretizing for emulation')
                self._emu.mem_write(address, b''.join(values))
            self._should_try_again = False
            self._step(instruction)
            if not self._should_try_again:
                break

    def _step(self, instruction):
        if False:
            i = 10
            return i + 15
        '\n        A single attempt at executing an instruction.\n        '
        logger.debug('0x%x:\t%s\t%s' % (instruction.address, instruction.mnemonic, instruction.op_str))
        ignore_registers = {'FIP', 'FOP', 'FDS', 'FCS', 'FDP', 'MXCSR_MASK'}
        registers = set(self._cpu.canonical_registers) - ignore_registers
        if self._cpu.arch == CS_ARCH_X86:
            registers -= set(['CF', 'PF', 'AF', 'ZF', 'SF', 'IF', 'DF', 'OF'])
            registers.add('EFLAGS')
            registers -= {'FS'}
        for reg in registers:
            val = self._cpu.read_register(reg)
            if issymbolic(val):
                from ..native.cpu.abstractcpu import ConcretizeRegister
                raise ConcretizeRegister(self._cpu, reg, 'Concretizing for emulation.', policy='ONE')
            self._emu.reg_write(self._to_unicorn_id(reg), val)
        instruction = self._cpu.decode_instruction(self._cpu.PC)
        text_bytes = self._cpu.read_bytes(self._cpu.PC, instruction.size)
        self._emu.mem_write(self._cpu.PC, b''.join(text_bytes))
        self._emu.hook_add(UC_HOOK_MEM_READ_UNMAPPED, self._hook_unmapped)
        self._emu.hook_add(UC_HOOK_MEM_WRITE_UNMAPPED, self._hook_unmapped)
        self._emu.hook_add(UC_HOOK_MEM_FETCH_UNMAPPED, self._hook_unmapped)
        self._emu.hook_add(UC_HOOK_MEM_READ, self._hook_xfer_mem)
        self._emu.hook_add(UC_HOOK_MEM_WRITE, self._hook_xfer_mem)
        self._emu.hook_add(UC_HOOK_INTR, self._interrupt)
        saved_PC = self._cpu.PC
        try:
            pc = self._cpu.PC
            if self._cpu.arch == CS_ARCH_ARM and self._uc_mode == UC_MODE_THUMB:
                pc |= 1
            self._emu.emu_start(pc, self._cpu.PC + instruction.size, count=1, timeout=1000000)
        except UcError as e:
            if not self._should_try_again:
                raise
        if self._should_try_again:
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('=' * 10)
            for register in registers:
                logger.debug(f'Register {register:3s}  Manticore: {self._cpu.read_register(register):08x}, Unicorn {self._emu.reg_read(self._to_unicorn_id(register)):08x}')
            logger.debug('>' * 10)
        for reg in registers:
            val = self._emu.reg_read(self._to_unicorn_id(reg))
            self._cpu.write_register(reg, val)
        mu_pc = self.get_unicorn_pc()
        if saved_PC == mu_pc:
            self._cpu.PC = saved_PC + instruction.size
        else:
            self._cpu.PC = mu_pc
        if self._to_raise:
            raise self._to_raise
        return