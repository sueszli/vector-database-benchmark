import inspect
import io
import logging
import struct
import types
from functools import wraps, partial
from itertools import islice
import unicorn
from .disasm import init_disassembler, Instruction
from ..memory import ConcretizeMemory, InvalidMemoryAccess, FileMap, AnonMap
from ..memory import LazySMemory, Memory
from ...core.smtlib import Operators, Constant, issymbolic, BitVec, Expression
from ...core.smtlib import visitors
from ...core.smtlib.solver import SelectedSolver
from ...utils.emulate import ConcreteUnicornEmulator
from ...utils.event import Eventful
from ...utils.fallback_emulator import UnicornEmulator
from capstone import CS_ARCH_ARM64, CS_ARCH_X86, CS_ARCH_ARM
from capstone.arm64 import ARM64_REG_ENDING
from capstone.x86 import X86_REG_ENDING
from capstone.arm import ARM_REG_ENDING
from typing import Any, Callable, Dict, Optional, Tuple
logger = logging.getLogger(__name__)
register_logger = logging.getLogger(f'{__name__}.registers')

def _sig_is_varargs(sig: inspect.Signature) -> bool:
    if False:
        for i in range(10):
            print('nop')
    VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
    return any((p.kind == VAR_POSITIONAL for p in sig.parameters.values()))

class CpuException(Exception):
    """Base cpu exception"""

class DecodeException(CpuException):
    """
    Raised when trying to decode an unknown or invalid instruction"""

    def __init__(self, pc, bytes):
        if False:
            print('Hello World!')
        super().__init__('Error decoding instruction @ 0x{:x}'.format(pc))
        self.pc = pc
        self.bytes = bytes

class InstructionNotImplementedError(CpuException):
    """
    Exception raised when you try to execute an instruction that is not yet
    implemented in the emulator. Add it to the Cpu-specific implementation.
    """

class InstructionEmulationError(CpuException):
    """
    Exception raised when failing to emulate an instruction outside of Manticore.
    """

class DivideByZeroError(CpuException):
    """A division by zero"""

class Interruption(CpuException):
    """A software interrupt."""

    def __init__(self, N):
        if False:
            print('Hello World!')
        super().__init__('CPU Software Interruption %08x' % N)
        self.N = N

class Syscall(CpuException):
    """ """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__('CPU Syscall')

class ConcretizeRegister(CpuException):
    """
    Raised when a symbolic register needs to be concretized.
    """

    def __init__(self, cpu: 'Cpu', reg_name: str, message: Optional[str]=None, policy: str='MINMAX'):
        if False:
            for i in range(10):
                print('nop')
        self.message = message if message else f'Concretizing {reg_name}'
        self.cpu = cpu
        self.reg_name = reg_name
        self.policy = policy

class ConcretizeArgument(CpuException):
    """
    Raised when a symbolic argument needs to be concretized.
    """

    def __init__(self, cpu, argnum, policy='MINMAX'):
        if False:
            while True:
                i = 10
        self.message = f'Concretizing argument #{argnum}.'
        self.cpu = cpu
        self.policy = policy
        self.argnum = argnum
SANE_SIZES = {8, 16, 32, 64, 80, 128, 256}

class Operand:
    """This class encapsulates how to access operands (regs/mem/immediates) for
    different CPUs
    """

    class MemSpec:
        """
        Auxiliary class wraps capstone operand 'mem' attribute. This will
        return register names instead of Ids
        """

        def __init__(self, parent):
            if False:
                i = 10
                return i + 15
            self.parent = parent
        segment = property(lambda self: self.parent._reg_name(self.parent.op.mem.segment))
        base = property(lambda self: self.parent._reg_name(self.parent.op.mem.base))
        index = property(lambda self: self.parent._reg_name(self.parent.op.mem.index))
        scale = property(lambda self: self.parent.op.mem.scale)
        disp = property(lambda self: self.parent.op.mem.disp)

    def __init__(self, cpu: 'Cpu', op):
        if False:
            print('Hello World!')
        '\n        This encapsulates the arch-independent way to access instruction\n        operands and immediates based on the disassembler operand descriptor in\n        use. This class knows how to browse an operand and get its details.\n\n        It also knows how to access the specific Cpu to get the actual values\n        from memory and registers.\n\n        :param Cpu cpu: A Cpu instance\n        :param Operand op: An wrapped Instruction Operand\n        :type op: X86Op or ArmOp\n        '
        assert isinstance(cpu, Cpu)
        self.cpu = cpu
        self.op = op
        self.mem = Operand.MemSpec(self)

    def _reg_name(self, reg_id: int):
        if False:
            while True:
                i = 10
        "\n        Translates a register ID from the disassembler object into the\n        register name based on manticore's alias in the register file\n\n        :param reg_id: Register ID\n        "
        if self.cpu.arch == CS_ARCH_ARM64 and reg_id >= ARM64_REG_ENDING or (self.cpu.arch == CS_ARCH_X86 and reg_id >= X86_REG_ENDING) or (self.cpu.arch == CS_ARCH_ARM and reg_id >= ARM_REG_ENDING):
            logger.warning('Trying to get register name for a non-register')
            return None
        cs_reg_name = self.cpu.instruction.reg_name(reg_id)
        if cs_reg_name is None or cs_reg_name.lower() == '(invalid)':
            return None
        return self.cpu._regfile._alias(cs_reg_name.upper())

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        return getattr(self.op, name)

    @property
    def type(self):
        if False:
            return 10
        'This property encapsulates the operand type.\n        It may be one of the following:\n            register\n            memory\n            immediate\n        '
        raise NotImplementedError

    @property
    def size(self):
        if False:
            return 10
        'Return bit size of operand'
        raise NotImplementedError

    @property
    def reg(self):
        if False:
            while True:
                i = 10
        return self._reg_name(self.op.reg)

    def address(self):
        if False:
            return 10
        'On a memory operand it returns the effective address'
        raise NotImplementedError

    def read(self):
        if False:
            while True:
                i = 10
        'It reads the operand value from the registers or memory'
        raise NotImplementedError

    def write(self, value):
        if False:
            print('Hello World!')
        'It writes the value of specific type to the registers or memory'
        raise NotImplementedError

class RegisterFile:

    def __init__(self, aliases=None):
        if False:
            print('Hello World!')
        self._aliases = aliases if aliases is not None else {}
        self._registers = {}

    def _alias(self, register):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get register canonical alias. ex. PC->RIP or PC->R15\n\n        :param str register: The register name\n        '
        return self._aliases.get(register, register)

    def write(self, register, value):
        if False:
            i = 10
            return i + 15
        '\n        Write value to the specified register\n\n        :param str register: a register id. Must be listed on all_registers\n        :param value: a value of the expected type\n        :type value: int or long or Expression\n        :return: the value actually written to the register\n        '
        raise NotImplementedError

    def read(self, register):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read value from specified register\n\n        :param str register: a register name. Must be listed on all_registers\n        :return: the register value\n        '
        raise NotImplementedError

    @property
    def all_registers(self):
        if False:
            i = 10
            return i + 15
        'Lists all possible register names (Including aliases)'
        return tuple(self._aliases)

    @property
    def canonical_registers(self):
        if False:
            print('Hello World!')
        'List the minimal most beautiful set of registers needed'
        raise NotImplementedError

    def __contains__(self, register):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check for register validity\n\n        :param register: a register name\n        '
        return self._alias(register) in self.all_registers

    def __copy__(self) -> 'RegisterFile':
        if False:
            for i in range(10):
                print('nop')
        'Custom shallow copy to create a snapshot of the register state.\n        Should be used as read-only'
        ...

class Abi:
    """
    Represents the ability to extract arguments from the environment and write
    back a result.

    Used for function call and system call models.
    """

    def __init__(self, cpu: 'Cpu'):
        if False:
            while True:
                i = 10
        '\n        :param CPU to initialize with\n        '
        self._cpu = cpu

    def get_arguments(self):
        if False:
            return 10
        '\n        Extract model arguments conforming to `convention`. Produces an iterable\n        of argument descriptors following the calling convention. A descriptor\n        is either a string describing a register, or an address (concrete or\n        symbolic).\n\n        :return: iterable returning syscall arguments.\n        :rtype: iterable\n        '
        raise NotImplementedError

    def get_result_reg(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract the location a return value will be written to. Produces\n        a string describing a register where the return value is written to.\n        :return: return register name\n        :rtype: string\n        '
        raise NotImplementedError

    def write_result(self, result):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write the result of a model back to the environment.\n\n        :param result: result of the model implementation\n        '
        raise NotImplementedError

    def ret(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle the "ret" semantics of the ABI, i.e. reclaiming stack space,\n        popping PC, etc.\n\n        A null operation by default.\n        '
        return

    def values_from(self, base):
        if False:
            return 10
        '\n        A reusable generator for increasing pointer-sized values from an address\n        (usually the stack).\n        '
        word_bytes = self._cpu.address_bit_size // 8
        while True:
            yield base
            base += word_bytes

    def get_argument_values(self, model: Callable, prefix_args: Tuple) -> Tuple:
        if False:
            while True:
                i = 10
        '\n        Extract arguments for model from the environment and return as a tuple that\n        is ready to be passed to the model.\n\n        :param model: Python model of the function\n        :param prefix_args: Parameters to pass to model before actual ones\n        :return: Arguments to be passed to the model\n        '
        if type(model) is partial:
            model = model.args[0]
        sig = inspect.signature(model)
        if _sig_is_varargs(sig):
            model_name = getattr(model, '__qualname__', '<no name>')
            logger.warning('ABI: %s: a vararg model must be a unary function.', model_name)
        nargs = len(sig.parameters) - len(prefix_args)

        def resolve_argument(arg):
            if False:
                print('Hello World!')
            if isinstance(arg, str):
                return self._cpu.read_register(arg)
            else:
                return self._cpu.read_int(arg)
        descriptors = self.get_arguments()
        argument_iter = map(resolve_argument, descriptors)
        from ..models import isvariadic
        if isvariadic(model):
            return prefix_args + (argument_iter,)
        else:
            return prefix_args + tuple(islice(argument_iter, nargs))

    def invoke(self, model, prefix_args=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Invoke a callable `model` as if it was a native function. If\n        :func:`~manticore.models.isvariadic` returns true for `model`, `model` receives a single\n        argument that is a generator for function arguments. Pass a tuple of\n        arguments for `prefix_args` you'd like to precede the actual\n        arguments.\n\n        :param callable model: Python model of the function\n        :param tuple prefix_args: Parameters to pass to model before actual ones\n        :return: The result of calling `model`\n        "
        prefix_args = prefix_args or ()
        arguments = self.get_argument_values(model, prefix_args)
        try:
            result = model(*arguments)
        except ConcretizeArgument as e:
            assert e.argnum >= len(prefix_args), "Can't concretize a constant arg"
            idx = e.argnum - len(prefix_args)
            descriptors = self.get_arguments()
            src = next(islice(descriptors, idx, idx + 1))
            msg = 'Concretizing due to model invocation'
            if isinstance(src, str):
                raise ConcretizeRegister(self._cpu, src, msg)
            else:
                raise ConcretizeMemory(self._cpu.memory, src, self._cpu.address_bit_size, msg)
        else:
            if result is not None:
                self.write_result(result)
            self.ret()
        return result
platform_logger = logging.getLogger('manticore.platforms.platform')

def unsigned_hexlify(i: Any) -> Any:
    if False:
        return 10
    if type(i) is int:
        if i < 0:
            return hex((1 << 64) + i)
        return hex(i)
    return i

class SyscallAbi(Abi):
    """
    A system-call specific ABI.

    Captures model arguments and return values for centralized logging.
    """

    def syscall_number(self):
        if False:
            i = 10
            return i + 15
        '\n        Extract the index of the invoked syscall.\n\n        :return: int\n        '
        raise NotImplementedError

    def get_argument_values(self, model, prefix_args):
        if False:
            print('Hello World!')
        self._last_arguments = super().get_argument_values(model, prefix_args)
        return self._last_arguments

    def invoke(self, model, prefix_args=None):
        if False:
            i = 10
            return i + 15
        self._last_arguments = ()
        if type(model) is partial:
            self._cpu._publish('will_execute_syscall', model.args[0])
        else:
            self._cpu._publish('will_execute_syscall', model)
        ret = super().invoke(model, prefix_args)
        if type(model) is partial:
            model = model.args[0]
        self._cpu._publish('did_execute_syscall', model.__func__.__name__ if isinstance(model, types.MethodType) else model.__name__, self._last_arguments, ret)
        if platform_logger.isEnabledFor(logging.DEBUG):
            max_arg_expansion = 32
            min_hex_expansion = 128
            args = []
            for arg in self._last_arguments:
                arg_s = unsigned_hexlify(arg) if not issymbolic(arg) and abs(arg) > min_hex_expansion else f'{arg}'
                if self._cpu.memory.access_ok(arg, 'r') and model.__func__.__name__ not in {'sys_mprotect', 'sys_mmap'}:
                    try:
                        s = self._cpu.read_string(arg, max_arg_expansion)
                        s = s.rstrip().replace('\n', '\\n') if s else s
                        arg_s = f'"{s}"' if s else arg_s
                    except Exception:
                        pass
                args.append(arg_s)
            args_s = ', '.join(args)
            ret_s = f'{unsigned_hexlify(ret)}' if abs(ret) > min_hex_expansion else f'{ret}'
            platform_logger.debug('%s(%s) = %s', model.__func__.__name__, args_s, ret_s)

class Cpu(Eventful):
    """
    Base class for all Cpu architectures. Functionality common to all
    architectures (and expected from users of a Cpu) should be here. Commonly
    used by platforms and py:class:manticore.core.Executor

    The following attributes need to be defined in any derived class

    - arch
    - mode
    - max_instr_width
    - address_bit_size
    - pc_alias
    - stack_alias
    """
    _published_events = {'write_register', 'read_register', 'write_memory', 'read_memory', 'decode_instruction', 'execute_instruction', 'invoke_syscall', 'set_descriptor', 'map_memory', 'protect_memory', 'unmap_memory', 'execute_syscall', 'solve'}

    def __init__(self, regfile: RegisterFile, memory: Memory, **kwargs):
        if False:
            i = 10
            return i + 15
        assert isinstance(regfile, RegisterFile)
        self._disasm = kwargs.pop('disasm', 'capstone')
        super().__init__(**kwargs)
        self._regfile = regfile
        self._memory = memory
        self._instruction_cache: Dict[int, Instruction] = {}
        self._icount = 0
        self._last_pc = None
        self._last_executed_pc = None
        self._concrete = kwargs.pop('concrete', False)
        self.emu = None
        self._break_unicorn_at: Optional[int] = None
        self._delayed_event = False
        if not hasattr(self, 'disasm'):
            self.disasm = init_disassembler(self._disasm, self.arch, self.mode)
        assert 'STACK' in self._regfile
        assert 'PC' in self._regfile

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = super().__getstate__()
        state['regfile'] = self._regfile
        state['memory'] = self._memory
        state['icount'] = self._icount
        state['last_pc'] = self._last_pc
        state['last_executed_pc'] = self._last_executed_pc
        state['disassembler'] = self._disasm
        state['concrete'] = self._concrete
        state['break_unicorn_at'] = self._break_unicorn_at
        state['delayed_event'] = self._delayed_event
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        Cpu.__init__(self, state['regfile'], state['memory'], disasm=state['disassembler'], concrete=state['concrete'])
        self._icount = state['icount']
        self._last_pc = state['last_pc']
        self._last_executed_pc = state['last_executed_pc']
        self._disasm = state['disassembler']
        self._concrete = state['concrete']
        self._break_unicorn_at = state['break_unicorn_at']
        self._delayed_event = state['delayed_event']
        super().__setstate__(state)

    @property
    def icount(self):
        if False:
            while True:
                i = 10
        return self._icount

    @property
    def last_executed_pc(self) -> Optional[int]:
        if False:
            return 10
        'The last PC that was executed.'
        return self._last_executed_pc

    @property
    def last_executed_insn(self) -> Optional[Instruction]:
        if False:
            return 10
        'The last instruction that was executed.'
        if not self.last_executed_pc:
            return None
        return self.decode_instruction(self.last_executed_pc)

    @property
    def regfile(self):
        if False:
            return 10
        'The RegisterFile of this cpu'
        return self._regfile

    @property
    def all_registers(self):
        if False:
            print('Hello World!')
        '\n        Returns all register names for this CPU. Any register returned can be\n        accessed via a `cpu.REG` convenience interface (e.g. `cpu.EAX`) for both\n        reading and writing.\n\n        :return: valid register names\n        :rtype: tuple[str]\n        '
        return self._regfile.all_registers

    @property
    def canonical_registers(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the list of all register names  for this CPU.\n\n        :rtype: tuple\n        :return: the list of register names for this CPU.\n        '
        return self._regfile.canonical_registers

    def write_register(self, register, value):
        if False:
            while True:
                i = 10
        '\n        Dynamic interface for writing cpu registers\n\n        :param str register: register name (as listed in `self.all_registers`)\n        :param value: register value\n        :type value: int or long or Expression\n        '
        self._publish('will_write_register', register, value)
        value = self._regfile.write(register, value)
        self._publish('did_write_register', register, value)
        return value

    def read_register(self, register):
        if False:
            i = 10
            return i + 15
        '\n        Dynamic interface for reading cpu registers\n\n        :param str register: register name (as listed in `self.all_registers`)\n        :return: register value\n        :rtype: int or long or Expression\n        '
        self._publish('will_read_register', register)
        value = self._regfile.read(register)
        self._publish('did_read_register', register, value)
        return value

    def __getattr__(self, name):
        if False:
            return 10
        '\n        A Pythonic version of read_register\n\n        :param str name: Name of the register\n        '
        if name != '_regfile':
            if name in self._regfile:
                return self.read_register(name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        A Pythonic version of write_register\n\n        :param str name: Name of the register to set\n        :param value: The value to set the register to\n        :type param: int or long or Expression\n        '
        try:
            if name in self._regfile:
                return self.write_register(name, value)
            object.__setattr__(self, name, value)
        except AttributeError:
            object.__setattr__(self, name, value)

    def emulate_until(self, target: int):
        if False:
            print('Hello World!')
        '\n        Tells the CPU to set up a concrete unicorn emulator and use it to execute instructions\n        until target is reached.\n\n        :param target: Where Unicorn should hand control back to Manticore. Set to 0 for all instructions.\n        '
        self._concrete = True
        self._break_unicorn_at = target
        if self.emu:
            self.emu.write_backs_disabled = False
            self.emu.load_state_from_manticore()
            self.emu._stop_at = target

    @property
    def memory(self) -> Memory:
        if False:
            print('Hello World!')
        return self._memory

    def write_int(self, where, expression, size=None, force=False):
        if False:
            print('Hello World!')
        '\n        Writes int to memory\n\n        :param int where: address to write to\n        :param expr: value to write\n        :type expr: int or BitVec\n        :param size: bit size of `expr`\n        :param force: whether to ignore memory permissions\n        '
        if size is None:
            size = self.address_bit_size
        assert size in SANE_SIZES
        self._publish('will_write_memory', where, expression, size)
        data = [Operators.CHR(Operators.EXTRACT(expression, offset, 8)) for offset in range(0, size, 8)]
        self._memory.write(where, data, force)
        self._publish('did_write_memory', where, expression, size)

    def _raw_read(self, where: int, size: int=1, force: bool=False) -> bytes:
        if False:
            print('Hello World!')
        '\n        Selects bytes from memory. Attempts to do so faster than via read_bytes.\n\n        :param where: address to read from\n        :param size: number of bytes to read\n        :param force: whether to ignore memory permissions\n        :return: the bytes in memory\n        '
        map = self.memory.map_containing(where)
        start = map._get_offset(where)
        if isinstance(map, FileMap):
            end = map._get_offset(where + size)
            if end > map._mapped_size:
                logger.warning(f'Missing {end - map._mapped_size} bytes at the end of {map._filename}')
            raw_data = map._data[map._get_offset(where):min(end, map._mapped_size)]
            if len(raw_data) < end:
                raw_data += b'\x00' * (end - len(raw_data))
            data = b''
            for offset in sorted(map._overlay.keys()):
                data += raw_data[len(data):offset]
                data += map._overlay[offset]
            data += raw_data[len(data):]
        elif isinstance(map, AnonMap):
            data = bytes(map._data[start:start + size])
        else:
            data = b''.join(self.memory.read(where, size, force=force))
        assert len(data) == size, 'Raw read resulted in wrong data read which should never happen'
        return data

    def read_int(self, where: int, size: int=None, force: bool=False, publish: bool=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reads int from memory\n\n        :param where: address to read from\n        :param size: number of bits to read\n        :param force: whether to ignore memory permissions\n        :param publish: whether to publish an event\n        :return: the value read\n        '
        if size is None:
            size = self.address_bit_size
        assert size in SANE_SIZES
        if publish:
            self._publish('will_read_memory', where, size)
        data = self._memory.read(where, size // 8, force)
        assert 8 * len(data) == size
        value = Operators.CONCAT(size, *map(Operators.ORD, reversed(data)))
        if publish:
            self._publish('did_read_memory', where, value, size)
        return value

    def write_bytes(self, where: int, data, force: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Write a concrete or symbolic (or mixed) buffer to memory\n\n        :param where: address to write to\n        :param data: data to write\n        :param force: whether to ignore memory permissions\n        '
        mp = self.memory.map_containing(where)
        if isinstance(mp, AnonMap) and isinstance(data, (str, bytes)) and (mp.end - mp.start + 1 >= len(data) >= 1024) and (not issymbolic(data)) and self._concrete:
            logger.debug('Using fast write')
            offset = mp._get_offset(where)
            if isinstance(data, str):
                data = bytes(data.encode('utf-8'))
            self._publish('will_write_memory', where, data, 8 * len(data))
            mp._data[offset:offset + len(data)] = data
            self._publish('did_write_memory', where, data, 8 * len(data))
        else:
            for i in range(len(data)):
                self.write_int(where + i, Operators.ORD(data[i]), 8, force)

    def read_bytes(self, where: int, size: int, force: bool=False, publish: bool=True):
        if False:
            print('Hello World!')
        '\n        Read from memory.\n\n        :param where: address to read data from\n        :param size: number of bytes\n        :param force: whether to ignore memory permissions\n        :param publish: whether to publish events\n        :return: data\n        '
        result = []
        for i in range(size):
            result.append(Operators.CHR(self.read_int(where + i, 8, force, publish=publish)))
        return result

    def write_string(self, where: int, string: str, max_length: Optional[int]=None, force: bool=False) -> None:
        if False:
            return 10
        '\n        Writes a string to memory, appending a NULL-terminator at the end.\n\n        :param where: Address to write the string to\n        :param string: The string to write to memory\n        :param max_length:\n\n        The size in bytes to cap the string at, or None [default] for no\n        limit. This includes the NULL terminator.\n\n        :param force: whether to ignore memory permissions\n        '
        if max_length is not None:
            string = string[:max_length - 1]
        self.write_bytes(where, string + '\x00', force)

    def read_string(self, where: int, max_length: Optional[int]=None, force: bool=False) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Read a NUL-terminated concrete buffer from memory. Stops reading at first symbolic byte.\n\n        :param where: Address to read string from\n        :param max_length:\n            The size in bytes to cap the string at, or None [default] for no\n            limit.\n        :param force: whether to ignore memory permissions\n        :return: string read\n        '
        s = io.BytesIO()
        while True:
            c = self.read_int(where, 8, force)
            if issymbolic(c) or c == 0:
                break
            if max_length is not None:
                if max_length == 0:
                    break
                max_length = max_length - 1
            s.write(Operators.CHR(c))
            where += 1
        return s.getvalue().decode()

    def push_bytes(self, data, force: bool=False):
        if False:
            return 10
        '\n        Write `data` to the stack and decrement the stack pointer accordingly.\n\n        :param data: Data to write\n        :param force: whether to ignore memory permissions\n        '
        self.STACK -= len(data)
        self.write_bytes(self.STACK, data, force)
        return self.STACK

    def pop_bytes(self, nbytes: int, force: bool=False):
        if False:
            print('Hello World!')
        '\n        Read `nbytes` from the stack, increment the stack pointer, and return\n        data.\n\n        :param nbytes: How many bytes to read\n        :param force: whether to ignore memory permissions\n        :return: Data read from the stack\n        '
        data = self.read_bytes(self.STACK, nbytes, force=force)
        self.STACK += nbytes
        return data

    def push_int(self, value: int, force: bool=False):
        if False:
            print('Hello World!')
        '\n        Decrement the stack pointer and write `value` to the stack.\n\n        :param value: The value to write\n        :param force: whether to ignore memory permissions\n        :return: New stack pointer\n        '
        self.STACK -= self.address_bit_size // 8
        self.write_int(self.STACK, value, force=force)
        return self.STACK

    def pop_int(self, force: bool=False):
        if False:
            print('Hello World!')
        '\n        Read a value from the stack and increment the stack pointer.\n\n        :param force: whether to ignore memory permissions\n        :return: Value read\n        '
        value = self.read_int(self.STACK, force=force)
        self.STACK += self.address_bit_size // 8
        return value

    def _wrap_operands(self, operands):
        if False:
            print('Hello World!')
        '\n        Private method to decorate an Operand to our needs based on the\n        underlying architecture.\n        See :class:`~manticore.core.cpu.abstractcpu.Operand` class\n        '
        raise NotImplementedError

    def decode_instruction(self, pc: int) -> Instruction:
        if False:
            i = 10
            return i + 15
        '\n        This will decode an instruction from memory pointed by `pc`\n\n        :param pc: address of the instruction\n        '
        if pc in self._instruction_cache:
            return self._instruction_cache[pc]
        text = b''
        exec_size = self.memory.max_exec_size(pc, self.max_instr_width)
        instr_memory = self.memory[pc:pc + exec_size]
        for i in range(exec_size):
            c = instr_memory[i]
            if issymbolic(c):
                if isinstance(self.memory, LazySMemory):
                    try:
                        vals = visitors.simplify_array_select(c)
                        c = bytes([vals[0]])
                    except visitors.ArraySelectSimplifier.ExpressionNotSimple:
                        self._publish('will_solve', self.memory.constraints, c, 'get_value')
                        solved = SelectedSolver.instance().get_value(self.memory.constraints, c)
                        self._publish('did_solve', self.memory.constraints, c, 'get_value', solved)
                        c = struct.pack('B', solved)
                elif isinstance(c, Constant):
                    c = bytes([c.value])
                else:
                    logger.error('Concretize executable memory %r %r', c, text)
                    raise ConcretizeMemory(self.memory, address=pc, size=8 * self.max_instr_width, policy='INSTRUCTION')
            text += c
        code = text.ljust(self.max_instr_width, b'\x00')
        try:
            insn = self.disasm.disassemble_instruction(code, pc)
        except StopIteration as e:
            raise DecodeException(pc, code)
        if insn.size > exec_size:
            logger.info('Trying to execute instructions from non-executable memory')
            raise InvalidMemoryAccess(pc, 'x')
        insn.operands = self._wrap_operands(insn.operands)
        self._instruction_cache[pc] = insn
        return insn

    @property
    def instruction(self):
        if False:
            return 10
        if self._last_pc is None:
            return self.decode_instruction(self.PC)
        else:
            return self.decode_instruction(self._last_pc)

    def canonicalize_instruction_name(self, instruction):
        if False:
            i = 10
            return i + 15
        '\n        Get the semantic name of an instruction.\n        '
        raise NotImplementedError

    def execute(self):
        if False:
            while True:
                i = 10
        '\n        Decode, and execute one instruction pointed by register PC\n        '
        curpc = self.PC
        if self._delayed_event:
            self._publish_instruction_as_executed(self.decode_instruction(self._last_pc))
            self._delayed_event = False
        if issymbolic(curpc):
            raise ConcretizeRegister(self, 'PC', policy='ALL')
        if not self.memory.access_ok(curpc, 'x'):
            raise InvalidMemoryAccess(curpc, 'x')
        self._publish('will_decode_instruction', curpc)
        insn = self.decode_instruction(curpc)
        self._last_pc = self.PC
        self._publish('will_execute_instruction', self._last_pc, insn)
        if insn.address != self.PC:
            self._last_executed_pc = insn.address
            return
        name = self.canonicalize_instruction_name(insn)
        if logger.level == logging.DEBUG:
            logger.debug(self.render_instruction(insn))
            for l in self.render_registers():
                register_logger.debug(l)
        try:
            if self._concrete and 'SYSCALL' in name:
                self.emu.sync_unicorn_to_manticore()
            if self._concrete and 'SYSCALL' not in name:
                self.emulate(insn)
                if self.PC == self._break_unicorn_at:
                    logger.debug('Switching from Unicorn to Manticore')
                    self._break_unicorn_at = None
                    self._concrete = False
            else:
                implementation = getattr(self, name, None)
                if implementation is not None:
                    implementation(*insn.operands)
                else:
                    text_bytes = ' '.join(('%02x' % x for x in insn.bytes))
                    logger.warning('Unimplemented instruction: 0x%016x:\t%s\t%s\t%s', insn.address, text_bytes, insn.mnemonic, insn.op_str)
                    self.backup_emulate(insn)
        except (Interruption, Syscall) as e:
            self._delayed_event = True
            raise e
        else:
            self._publish_instruction_as_executed(insn)

    def _publish_instruction_as_executed(self, insn):
        if False:
            i = 10
            return i + 15
        '\n        Notify listeners that an instruction has been executed.\n        '
        self._last_executed_pc = self._last_pc
        self._icount += 1
        self._publish('did_execute_instruction', self._last_pc, self.PC, insn)

    def emulate(self, insn):
        if False:
            print('Hello World!')
        '\n        Pick the right emulate function (maintains API compatiblity)\n\n        :param insn: single instruction to emulate/start emulation from\n        '
        if self._concrete:
            self.concrete_emulate(insn)
        else:
            self.backup_emulate(insn)

    def concrete_emulate(self, insn):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start executing in Unicorn from this point until we hit a syscall or reach break_unicorn_at\n\n        :param capstone.CsInsn insn: The instruction object to emulate\n        '
        if not self.emu:
            self.emu = ConcreteUnicornEmulator(self)
        if self.emu._stop_at is None:
            self.emu.write_backs_disabled = False
            self.emu._stop_at = self._break_unicorn_at
            self.emu.load_state_from_manticore()
        try:
            self.emu.emulate(insn)
        except unicorn.UcError as e:
            if e.errno == unicorn.UC_ERR_INSN_INVALID:
                text_bytes = ' '.join(('%02x' % x for x in insn.bytes))
                logger.error('Unimplemented instruction: 0x%016x:\t%s\t%s\t%s', insn.address, text_bytes, insn.mnemonic, insn.op_str)
            raise InstructionEmulationError(str(e))

    def backup_emulate(self, insn):
        if False:
            for i in range(10):
                print('nop')
        '\n        If we could not handle emulating an instruction, use Unicorn to emulate\n        it.\n\n        :param capstone.CsInsn instruction: The instruction object to emulate\n        '
        if not hasattr(self, 'backup_emu'):
            self.backup_emu = UnicornEmulator(self)
        try:
            self.backup_emu.emulate(insn)
        except unicorn.UcError as e:
            if e.errno == unicorn.UC_ERR_INSN_INVALID:
                text_bytes = ' '.join(('%02x' % x for x in insn.bytes))
                logger.error('Unimplemented instruction: 0x%016x:\t%s\t%s\t%s', insn.address, text_bytes, insn.mnemonic, insn.op_str)
            raise InstructionEmulationError(str(e))
        finally:
            del self.backup_emu

    def render_instruction(self, insn=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            insn = self.instruction
            return f'INSTRUCTION: 0x{insn.address:016x}:\t{insn.mnemonic}\t{insn.op_str}'
        except Exception as e:
            return "{can't decode instruction}"

    def render_register(self, reg_name):
        if False:
            return 10
        result = ''
        value = self.read_register(reg_name)
        if issymbolic(value):
            value = str(value)
            aux = f'{reg_name:3s}: {value:16s}'
            result += aux
        elif isinstance(value, int):
            result += f'{reg_name:3s}: 0x{value:016x}'
        else:
            result += f'{reg_name:3s}: {value!r}'
        return result

    def render_registers(self):
        if False:
            print('Hello World!')
        return map(self.render_register, sorted(self._regfile.canonical_registers))

    def __str__(self):
        if False:
            return 10
        '\n        Returns a string representation of cpu state\n\n        :rtype: str\n        :return: name and current value for all the registers.\n        '
        result = f'{self.render_instruction()}\n'
        result += '\n'.join(self.render_registers())
        return result

def instruction(old_method):
    if False:
        return 10

    @wraps(old_method)
    def new_method(cpu, *args, **kw_args):
        if False:
            print('Hello World!')
        cpu.PC += cpu.instruction.size
        return old_method(cpu, *args, **kw_args)
    new_method.old_method = old_method
    return new_method