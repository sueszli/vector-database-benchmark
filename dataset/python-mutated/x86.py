import collections
import logging
from functools import wraps
from typing import Tuple
import capstone as cs
from .abstractcpu import Abi, SyscallAbi, Cpu, CpuException, RegisterFile, Operand, instruction, ConcretizeRegister, Interruption, Syscall, DivideByZeroError
from ...core.smtlib import Operators, BitVec, Bool, BitVecConstant, operator, visitors, issymbolic
from ..memory import Memory, ConcretizeMemory
from functools import reduce
from typing import Any, Dict
logger = logging.getLogger(__name__)
OP_NAME_MAP = {'JNE': 'JNZ', 'JE': 'JZ', 'CMOVE': 'CMOVZ', 'CMOVNE': 'CMOVNZ', 'MOVUPS': 'MOV', 'MOVABS': 'MOV', 'MOVSB': 'MOVS', 'MOVSW': 'MOVS', 'MOVSQ': 'MOVS', 'SETNE': 'SETNZ', 'SETE': 'SETZ', 'LODSB': 'LODS', 'LODSW': 'LODS', 'LODSD': 'LODS', 'LODSQ': 'LODS', 'STOSB': 'STOS', 'STOSW': 'STOS', 'STOSD': 'STOS', 'STOSQ': 'STOS', 'SCASB': 'SCAS', 'SCASW': 'SCAS', 'SCASD': 'SCAS', 'SCASQ': 'SCAS', 'CMPSB': 'CMPS', 'CMPSW': 'CMPS', 'CMPSD': 'CMPS', 'VMOVSD': 'MOVSD', 'FUCOMPI': 'FUCOMIP'}

def rep(old_method):
    if False:
        i = 10
        return i + 15

    @wraps(old_method)
    def new_method(cpu, *args, **kw_args):
        if False:
            for i in range(10):
                print('nop')
        prefix = cpu.instruction.prefix
        if cs.x86.X86_PREFIX_REP in prefix:
            counter_name = {16: 'CX', 32: 'ECX', 64: 'RCX'}[cpu.instruction.addr_size * 8]
            count = cpu.read_register(counter_name)
            if issymbolic(count):
                raise ConcretizeRegister(cpu, counter_name, f'Concretizing {counter_name} on REP instruction', policy='SAMPLED')
            FLAG = count != 0
            if FLAG:
                old_method(cpu, *args, **kw_args)
                count = cpu.write_register(counter_name, count - 1)
                FLAG = count != 0
            if not FLAG:
                cpu.PC += cpu.instruction.size
        else:
            cpu.PC += cpu.instruction.size
            old_method(cpu, *args, **kw_args)
    return new_method

def repe(old_method):
    if False:
        print('Hello World!')

    @wraps(old_method)
    def new_method(cpu, *args, **kw_args):
        if False:
            i = 10
            return i + 15
        prefix = cpu.instruction.prefix
        if cs.x86.X86_PREFIX_REP in prefix or cs.x86.X86_PREFIX_REPNE in prefix:
            counter_name = {16: 'CX', 32: 'ECX', 64: 'RCX'}[cpu.instruction.addr_size * 8]
            count = cpu.read_register(counter_name)
            if issymbolic(count):
                raise ConcretizeRegister(cpu, counter_name, f'Concretizing {counter_name} on REP instruction', policy='SAMPLED')
            FLAG = count != 0
            if FLAG:
                old_method(cpu, *args, **kw_args)
                count = cpu.write_register(counter_name, count - 1)
                if cs.x86.X86_PREFIX_REP in prefix:
                    FLAG = Operators.AND(cpu.ZF == True, count != 0)
                elif cs.x86.X86_PREFIX_REPNE in prefix:
                    FLAG = Operators.AND(cpu.ZF == False, count != 0)
            cpu.PC += Operators.ITEBV(cpu.address_bit_size, FLAG, 0, cpu.instruction.size)
        else:
            cpu.PC += cpu.instruction.size
            old_method(cpu, *args, **kw_args)
    return new_method

class AMD64RegFile(RegisterFile):
    Regspec = collections.namedtuple('Regspec', 'register_id ty offset size reset')
    _flags = {'CF': 0, 'PF': 2, 'AF': 4, 'ZF': 6, 'SF': 7, 'IF': 9, 'DF': 10, 'OF': 11}
    _table = {'CS': Regspec('CS', int, 0, 16, False), 'DS': Regspec('DS', int, 0, 16, False), 'ES': Regspec('ES', int, 0, 16, False), 'SS': Regspec('SS', int, 0, 16, False), 'FS': Regspec('FS', int, 0, 16, False), 'GS': Regspec('GS', int, 0, 16, False), 'RAX': Regspec('RAX', int, 0, 64, True), 'RBX': Regspec('RBX', int, 0, 64, True), 'RCX': Regspec('RCX', int, 0, 64, True), 'RDX': Regspec('RDX', int, 0, 64, True), 'RSI': Regspec('RSI', int, 0, 64, True), 'RDI': Regspec('RDI', int, 0, 64, True), 'RSP': Regspec('RSP', int, 0, 64, True), 'RBP': Regspec('RBP', int, 0, 64, True), 'RIP': Regspec('RIP', int, 0, 64, True), 'R8': Regspec('R8', int, 0, 64, True), 'R9': Regspec('R9', int, 0, 64, True), 'R10': Regspec('R10', int, 0, 64, True), 'R11': Regspec('R11', int, 0, 64, True), 'R12': Regspec('R12', int, 0, 64, True), 'R13': Regspec('R13', int, 0, 64, True), 'R14': Regspec('R14', int, 0, 64, True), 'R15': Regspec('R15', int, 0, 64, True), 'EAX': Regspec('RAX', int, 0, 32, True), 'EBX': Regspec('RBX', int, 0, 32, True), 'ECX': Regspec('RCX', int, 0, 32, True), 'EDX': Regspec('RDX', int, 0, 32, True), 'ESI': Regspec('RSI', int, 0, 32, True), 'EDI': Regspec('RDI', int, 0, 32, True), 'ESP': Regspec('RSP', int, 0, 32, True), 'EBP': Regspec('RBP', int, 0, 32, True), 'EIP': Regspec('RIP', int, 0, 32, True), 'R8D': Regspec('R8', int, 0, 32, True), 'R9D': Regspec('R9', int, 0, 32, True), 'R10D': Regspec('R10', int, 0, 32, True), 'R11D': Regspec('R11', int, 0, 32, True), 'R12D': Regspec('R12', int, 0, 32, True), 'R13D': Regspec('R13', int, 0, 32, True), 'R14D': Regspec('R14', int, 0, 32, True), 'R15D': Regspec('R15', int, 0, 32, True), 'AX': Regspec('RAX', int, 0, 16, False), 'BX': Regspec('RBX', int, 0, 16, False), 'CX': Regspec('RCX', int, 0, 16, False), 'DX': Regspec('RDX', int, 0, 16, False), 'SI': Regspec('RSI', int, 0, 16, False), 'DI': Regspec('RDI', int, 0, 16, False), 'SP': Regspec('RSP', int, 0, 16, False), 'BP': Regspec('RBP', int, 0, 16, False), 'IP': Regspec('RIP', int, 0, 16, False), 'R8W': Regspec('R8', int, 0, 16, False), 'R9W': Regspec('R9', int, 0, 16, False), 'R10W': Regspec('R10', int, 0, 16, False), 'R11W': Regspec('R11', int, 0, 16, False), 'R12W': Regspec('R12', int, 0, 16, False), 'R13W': Regspec('R13', int, 0, 16, False), 'R14W': Regspec('R14', int, 0, 16, False), 'R15W': Regspec('R15', int, 0, 16, False), 'AL': Regspec('RAX', int, 0, 8, False), 'BL': Regspec('RBX', int, 0, 8, False), 'CL': Regspec('RCX', int, 0, 8, False), 'DL': Regspec('RDX', int, 0, 8, False), 'SIL': Regspec('RSI', int, 0, 8, False), 'DIL': Regspec('RDI', int, 0, 8, False), 'SPL': Regspec('RSP', int, 0, 8, False), 'BPL': Regspec('RBP', int, 0, 8, False), 'R8B': Regspec('R8', int, 0, 8, False), 'R9B': Regspec('R9', int, 0, 8, False), 'R10B': Regspec('R10', int, 0, 8, False), 'R11B': Regspec('R11', int, 0, 8, False), 'R12B': Regspec('R12', int, 0, 8, False), 'R13B': Regspec('R13', int, 0, 8, False), 'R14B': Regspec('R14', int, 0, 8, False), 'R15B': Regspec('R15', int, 0, 8, False), 'AH': Regspec('RAX', int, 8, 8, False), 'BH': Regspec('RBX', int, 8, 8, False), 'CH': Regspec('RCX', int, 8, 8, False), 'DH': Regspec('RDX', int, 8, 8, False), 'SIH': Regspec('RSI', int, 8, 8, False), 'DIH': Regspec('RDI', int, 8, 8, False), 'SPH': Regspec('RSP', int, 8, 8, False), 'BPH': Regspec('RBP', int, 8, 8, False), 'R8H': Regspec('R8', int, 8, 8, False), 'R9H': Regspec('R9', int, 8, 8, False), 'R10H': Regspec('R10', int, 8, 8, False), 'R11H': Regspec('R11', int, 8, 8, False), 'R12H': Regspec('R12', int, 8, 8, False), 'R13H': Regspec('R13', int, 8, 8, False), 'R14H': Regspec('R14', int, 8, 8, False), 'R15H': Regspec('R15', int, 8, 8, False), 'FP0': Regspec('FP0', float, 0, 80, False), 'FP1': Regspec('FP1', float, 0, 80, False), 'FP2': Regspec('FP2', float, 0, 80, False), 'FP3': Regspec('FP3', float, 0, 80, False), 'FP4': Regspec('FP4', float, 0, 80, False), 'FP5': Regspec('FP5', float, 0, 80, False), 'FP6': Regspec('FP6', float, 0, 80, False), 'FP7': Regspec('FP7', float, 0, 80, False), 'FPSW': Regspec('FPSW', int, 0, 16, False), 'TOP': Regspec('FPSW', int, 11, 3, False), 'FPTAG': Regspec('FPTAG', int, 0, 16, False), 'FPCW': Regspec('FPCW', int, 0, 16, False), 'FOP': Regspec('FOP', int, 0, 11, False), 'FIP': Regspec('FIP', int, 0, 64, False), 'FCS': Regspec('FCS', int, 0, 16, False), 'FDP': Regspec('FDP', int, 0, 64, False), 'FDS': Regspec('FDS', int, 0, 16, False), 'MXCSR': Regspec('MXCSR', int, 0, 32, False), 'MXCSR_MASK': Regspec('MXCSR_MASK', int, 0, 32, False), 'CF': Regspec('CF', bool, 0, 1, False), 'PF': Regspec('PF', bool, 0, 1, False), 'AF': Regspec('AF', bool, 0, 1, False), 'ZF': Regspec('ZF', bool, 0, 1, False), 'SF': Regspec('SF', bool, 0, 1, False), 'IF': Regspec('IF', bool, 0, 1, False), 'DF': Regspec('DF', bool, 0, 1, False), 'OF': Regspec('OF', bool, 0, 1, False), 'YMM0': Regspec('YMM0', int, 0, 256, False), 'YMM1': Regspec('YMM1', int, 0, 256, False), 'YMM2': Regspec('YMM2', int, 0, 256, False), 'YMM3': Regspec('YMM3', int, 0, 256, False), 'YMM4': Regspec('YMM4', int, 0, 256, False), 'YMM5': Regspec('YMM5', int, 0, 256, False), 'YMM6': Regspec('YMM6', int, 0, 256, False), 'YMM7': Regspec('YMM7', int, 0, 256, False), 'YMM8': Regspec('YMM8', int, 0, 256, False), 'YMM9': Regspec('YMM9', int, 0, 256, False), 'YMM10': Regspec('YMM10', int, 0, 256, False), 'YMM11': Regspec('YMM11', int, 0, 256, False), 'YMM12': Regspec('YMM12', int, 0, 256, False), 'YMM13': Regspec('YMM13', int, 0, 256, False), 'YMM14': Regspec('YMM14', int, 0, 256, False), 'YMM15': Regspec('YMM15', int, 0, 256, False), 'XMM0': Regspec('YMM0', int, 0, 128, False), 'XMM1': Regspec('YMM1', int, 0, 128, False), 'XMM2': Regspec('YMM2', int, 0, 128, False), 'XMM3': Regspec('YMM3', int, 0, 128, False), 'XMM4': Regspec('YMM4', int, 0, 128, False), 'XMM5': Regspec('YMM5', int, 0, 128, False), 'XMM6': Regspec('YMM6', int, 0, 128, False), 'XMM7': Regspec('YMM7', int, 0, 128, False), 'XMM8': Regspec('YMM8', int, 0, 128, False), 'XMM9': Regspec('YMM9', int, 0, 128, False), 'XMM10': Regspec('YMM10', int, 0, 128, False), 'XMM11': Regspec('YMM11', int, 0, 128, False), 'XMM12': Regspec('YMM12', int, 0, 128, False), 'XMM13': Regspec('YMM13', int, 0, 128, False), 'XMM14': Regspec('YMM14', int, 0, 128, False), 'XMM15': Regspec('YMM15', int, 0, 128, False)}
    _affects = {'RIP': ('EIP', 'IP'), 'EIP': ('IP', 'RIP'), 'IP': ('EIP', 'RIP'), 'RAX': ('AH', 'AL', 'AX', 'EAX'), 'EAX': ('AH', 'AL', 'AX', 'RAX'), 'AX': ('AH', 'AL', 'EAX', 'RAX'), 'AH': ('AX', 'EAX', 'RAX'), 'AL': ('AX', 'EAX', 'RAX'), 'RBX': ('BH', 'BL', 'BX', 'EBX'), 'EBX': ('BH', 'BL', 'BX', 'RBX'), 'BX': ('BH', 'BL', 'EBX', 'RBX'), 'BH': ('BX', 'EBX', 'RBX'), 'BL': ('BX', 'EBX', 'RBX'), 'RCX': ('CH', 'CL', 'CX', 'ECX'), 'ECX': ('CH', 'CL', 'CX', 'RCX'), 'CX': ('CH', 'CL', 'ECX', 'RCX'), 'CH': ('CX', 'ECX', 'RCX'), 'CL': ('CX', 'ECX', 'RCX'), 'RDX': ('DH', 'DL', 'DX', 'EDX'), 'EDX': ('DH', 'DL', 'DX', 'RDX'), 'DX': ('DH', 'DL', 'EDX', 'RDX'), 'DH': ('DX', 'EDX', 'RDX'), 'DL': ('DX', 'EDX', 'RDX'), 'RSI': ('ESI', 'SI', 'SIH', 'SIL'), 'ESI': ('RSI', 'SI', 'SIH', 'SIL'), 'SI': ('ESI', 'RSI', 'SIH', 'SIL'), 'SIH': ('ESI', 'RSI', 'SI'), 'SIL': ('ESI', 'RSI', 'SI'), 'RDI': ('DI', 'DIH', 'DIL', 'EDI'), 'EDI': ('DI', 'DIH', 'DIL', 'RDI'), 'DI': ('DIH', 'DIL', 'EDI', 'RDI'), 'DIH': ('DI', 'EDI', 'RDI'), 'DIL': ('DI', 'EDI', 'RDI'), 'RSP': ('ESP', 'SP', 'SPH', 'SPL'), 'ESP': ('RSP', 'SP', 'SPH', 'SPL'), 'SP': ('ESP', 'RSP', 'SPH', 'SPL'), 'SPH': ('ESP', 'RSP', 'SP'), 'SPL': ('ESP', 'RSP', 'SP'), 'RBP': ('BP', 'BPH', 'BPL', 'EBP'), 'EBP': ('BP', 'BPH', 'BPL', 'RBP'), 'BP': ('BPH', 'BPL', 'EBP', 'RBP'), 'BPH': ('BP', 'EBP', 'RBP'), 'BPL': ('BP', 'EBP', 'RBP'), 'CS': (), 'DS': (), 'ES': (), 'FS': (), 'GS': (), 'SS': (), 'RFLAGS': ('EFLAGS', 'AF', 'CF', 'DF', 'IF', 'OF', 'PF', 'SF', 'ZF'), 'EFLAGS': ('RFLAGS', 'AF', 'CF', 'DF', 'IF', 'OF', 'PF', 'SF', 'ZF'), 'AF': ('RFLAGS', 'EFLAGS'), 'CF': ('RFLAGS', 'EFLAGS'), 'DF': ('RFLAGS', 'EFLAGS'), 'IF': ('RFLAGS', 'EFLAGS'), 'OF': ('RFLAGS', 'EFLAGS'), 'PF': ('RFLAGS', 'EFLAGS'), 'SF': ('RFLAGS', 'EFLAGS'), 'ZF': ('RFLAGS', 'EFLAGS'), 'FPSW': ('TOP',), 'TOP': ('FPSW',), 'FPCW': (), 'FPTAG': (), 'FOP': (), 'FIP': (), 'FCS': (), 'FDP': (), 'FDS': (), 'MXCSR': (), 'MXCSR_MASK': (), 'FP0': (), 'FP1': (), 'FP2': (), 'FP3': (), 'FP4': (), 'FP5': (), 'FP6': (), 'FP7': (), 'R10': ('R10B', 'R10D', 'R10H', 'R10W'), 'R10B': ('R10', 'R10D', 'R10W'), 'R10D': ('R10', 'R10B', 'R10H', 'R10W'), 'R10H': ('R10', 'R10D', 'R10W'), 'R10W': ('R10', 'R10B', 'R10D', 'R10H'), 'R11': ('R11B', 'R11D', 'R11H', 'R11W'), 'R11B': ('R11', 'R11D', 'R11W'), 'R11D': ('R11', 'R11B', 'R11H', 'R11W'), 'R11H': ('R11', 'R11D', 'R11W'), 'R11W': ('R11', 'R11B', 'R11D', 'R11H'), 'R12': ('R12B', 'R12D', 'R12H', 'R12W'), 'R12B': ('R12', 'R12D', 'R12W'), 'R12D': ('R12', 'R12B', 'R12H', 'R12W'), 'R12H': ('R12', 'R12D', 'R12W'), 'R12W': ('R12', 'R12B', 'R12D', 'R12H'), 'R13': ('R13B', 'R13D', 'R13H', 'R13W'), 'R13B': ('R13', 'R13D', 'R13W'), 'R13D': ('R13', 'R13B', 'R13H', 'R13W'), 'R13H': ('R13', 'R13D', 'R13W'), 'R13W': ('R13', 'R13B', 'R13D', 'R13H'), 'R14': ('R14B', 'R14D', 'R14H', 'R14W'), 'R14B': ('R14', 'R14D', 'R14W'), 'R14D': ('R14', 'R14B', 'R14H', 'R14W'), 'R14H': ('R14', 'R14D', 'R14W'), 'R14W': ('R14', 'R14B', 'R14D', 'R14H'), 'R15': ('R15B', 'R15D', 'R15H', 'R15W'), 'R15B': ('R15', 'R15D', 'R15W'), 'R15D': ('R15', 'R15B', 'R15H', 'R15W'), 'R15H': ('R15', 'R15D', 'R15W'), 'R15W': ('R15', 'R15B', 'R15D', 'R15H'), 'R8': ('R8B', 'R8D', 'R8H', 'R8W'), 'R8B': ('R8', 'R8D', 'R8W'), 'R8D': ('R8', 'R8B', 'R8H', 'R8W'), 'R8H': ('R8', 'R8D', 'R8W'), 'R8W': ('R8', 'R8B', 'R8D', 'R8H'), 'R9': ('R9B', 'R9D', 'R9H', 'R9W'), 'R9B': ('R9', 'R9D', 'R9W'), 'R9D': ('R9', 'R9B', 'R9H', 'R9W'), 'R9H': ('R9', 'R9D', 'R9W'), 'R9W': ('R9', 'R9B', 'R9D', 'R9H'), 'XMM0': ('YMM0',), 'XMM1': ('YMM1',), 'XMM10': ('YMM10',), 'XMM11': ('YMM11',), 'XMM12': ('YMM12',), 'XMM13': ('YMM13',), 'XMM14': ('YMM14',), 'XMM15': ('YMM15',), 'XMM2': ('YMM2',), 'XMM3': ('YMM3',), 'XMM4': ('YMM4',), 'XMM5': ('YMM5',), 'XMM6': ('YMM6',), 'XMM7': ('YMM7',), 'XMM8': ('YMM8',), 'XMM9': ('YMM9',), 'YMM0': ('XMM0',), 'YMM1': ('XMM1',), 'YMM10': ('XMM10',), 'YMM11': ('XMM11',), 'YMM12': ('XMM12',), 'YMM13': ('XMM13',), 'YMM14': ('XMM14',), 'YMM15': ('XMM15',), 'YMM2': ('XMM2',), 'YMM3': ('XMM3',), 'YMM4': ('XMM4',), 'YMM5': ('XMM5',), 'YMM6': ('XMM6',), 'YMM7': ('XMM7',), 'YMM8': ('XMM8',), 'YMM9': ('XMM9',)}
    _canonical_registers = ('RAX', 'RCX', 'RDX', 'RBX', 'RSP', 'RBP', 'RSI', 'RDI', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'RIP', 'YMM0', 'YMM1', 'YMM2', 'YMM3', 'YMM4', 'YMM5', 'YMM6', 'YMM7', 'YMM8', 'YMM9', 'YMM10', 'YMM11', 'YMM12', 'YMM13', 'YMM14', 'YMM15', 'CS', 'DS', 'ES', 'SS', 'FS', 'GS', 'AF', 'CF', 'DF', 'IF', 'OF', 'PF', 'SF', 'ZF', 'FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7', 'FPSW', 'FPCW', 'FPTAG', 'FOP', 'FIP', 'FCS', 'FDP', 'FDS', 'MXCSR', 'MXCSR_MASK')

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        for reg in ('RAX', 'RCX', 'RDX', 'RBX', 'RSP', 'RBP', 'RSI', 'RDI', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'RIP', 'YMM0', 'YMM1', 'YMM2', 'YMM3', 'YMM4', 'YMM5', 'YMM6', 'YMM7', 'YMM8', 'YMM9', 'YMM10', 'YMM11', 'YMM12', 'YMM13', 'YMM14', 'YMM15', 'CS', 'DS', 'ES', 'SS', 'FS', 'GS', 'AF', 'CF', 'DF', 'IF', 'OF', 'PF', 'SF', 'ZF'):
            self._registers[reg] = 0
        for reg in ('FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7'):
            self._registers[reg] = (0, 0)
        for reg in ('FPSW', 'FPTAG', 'FPCW', 'FOP', 'FIP', 'FCS', 'FDP', 'FDS', 'MXCSR', 'MXCSR_MASK'):
            self._registers[reg] = 0
        self._cache = {}
        for name in ('AF', 'CF', 'DF', 'IF', 'OF', 'PF', 'SF', 'ZF'):
            self.write(name, False)
        self._all_registers = set(self._table.keys()) | set(['FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7', 'EFLAGS', 'RFLAGS']) | set(self._aliases.keys())

    @property
    def all_registers(self):
        if False:
            for i in range(10):
                print('nop')
        return self._all_registers

    @property
    def canonical_registers(self):
        if False:
            for i in range(10):
                print('nop')
        return self._canonical_registers

    def __contains__(self, register):
        if False:
            while True:
                i = 10
        return register in self._all_registers

    def _set_bv(self, register_id, register_size, offset, size, reset, value):
        if False:
            while True:
                i = 10
        if isinstance(value, int):
            value &= (1 << size) - 1
        elif not isinstance(value, BitVec) or value.size != size:
            raise TypeError
        if not reset:
            if register_size == size:
                new_value = 0
            elif offset == 0:
                new_value = self._registers[register_id] & ~((1 << size) - 1)
            else:
                new_value = self._registers[register_id] & ~((1 << size) - 1 << offset)
        else:
            new_value = 0
        new_value |= Operators.ZEXTEND(value, register_size) << offset
        self._registers[register_id] = new_value
        return value

    def _get_bv(self, register_id, register_size, offset, size):
        if False:
            print('Hello World!')
        if register_size == size:
            value = self._registers[register_id]
        else:
            value = Operators.EXTRACT(self._registers[register_id], offset, size)
        return value

    def _set_flag(self, register_id, register_size, offset, size, reset, value):
        if False:
            return 10
        assert size == 1
        if not isinstance(value, (bool, int, BitVec, Bool)):
            raise TypeError
        if isinstance(value, BitVec):
            if value.size != 1:
                raise TypeError
        if not isinstance(value, (bool, Bool)):
            value = value != 0
        self._registers[register_id] = value
        return value

    def _get_flag(self, register_id, register_size, offset, size):
        if False:
            while True:
                i = 10
        assert size == 1
        return self._registers[register_id]

    def _set_float(self, register_id, register_size, offset, size, reset, value):
        if False:
            for i in range(10):
                print('nop')
        assert size == 80
        assert offset == 0
        if isinstance(value, int):
            value &= 1208925819614629174706175
            exponent = value >> 64
            mantissa = value & 18446744073709551615
            value = (mantissa, exponent)
        elif not isinstance(value, tuple):
            raise TypeError
        self._registers[register_id] = value
        return value

    def _get_float(self, register_id, register_size, offset, size):
        if False:
            while True:
                i = 10
        assert size == 80
        assert offset == 0
        return self._registers[register_id]

    def _get_flags(self, reg):
        if False:
            return 10
        'Build EFLAGS/RFLAGS from flags'

        def make_symbolic(flag_expr):
            if False:
                for i in range(10):
                    print('nop')
            register_size = 32 if reg == 'EFLAGS' else 64
            (value, offset) = flag_expr
            return Operators.ITEBV(register_size, value, BitVecConstant(size=register_size, value=1 << offset), BitVecConstant(size=register_size, value=0))
        flags = []
        for (flag, offset) in self._flags.items():
            flags.append((self._registers[flag], offset))
        if any((issymbolic(flag) for (flag, offset) in flags)):
            res = reduce(operator.or_, map(make_symbolic, flags))
        else:
            res = 0
            for (flag, offset) in flags:
                res += flag << offset
        return res

    def _set_flags(self, reg, res):
        if False:
            for i in range(10):
                print('nop')
        'Set individual flags from a EFLAGS/RFLAGS value'
        for (flag, offset) in self._flags.items():
            self.write(flag, Operators.EXTRACT(res, offset, 1))

    def write(self, name, value):
        if False:
            return 10
        name = self._alias(name)
        if name in ('ST0', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7'):
            name = f"FP{self.read('TOP') + int(name[2]) & 7}"
        if 'FLAGS' in name:
            self._set_flags(name, value)
            self._update_cache(name, value)
            return value
        (register_id, ty, offset, size, reset) = self._table[name]
        if register_id != name:
            register_size = self._table[register_id].size
        else:
            register_size = size
        assert register_size >= offset + size
        typed_setter = {int: self._set_bv, bool: self._set_flag, float: self._set_float}[ty]
        value = typed_setter(register_id, register_size, offset, size, reset, value)
        self._update_cache(name, value)
        return value

    def _update_cache(self, name, value):
        if False:
            while True:
                i = 10
        self._cache[name] = value
        for affected in self._affects[name]:
            assert affected != name
            self._cache.pop(affected, None)

    def read(self, name):
        if False:
            return 10
        name = str(self._alias(name))
        if name in ('ST0', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7'):
            name = f"FP{self.read('TOP') + int(name[2]) & 7}"
        if name in self._cache:
            return self._cache[name]
        if 'FLAGS' in name:
            value = self._get_flags(name)
            self._cache[name] = value
            return value
        (register_id, ty, offset, size, reset) = self._table[name]
        if register_id != name:
            register_size = self._table[register_id].size
        else:
            register_size = size
        assert register_size >= offset + size
        typed_getter = {int: self._get_bv, bool: self._get_flag, float: self._get_float}[ty]
        value = typed_getter(register_id, register_size, offset, size)
        self._cache[name] = value
        return value

    def sizeof(self, reg):
        if False:
            print('Hello World!')
        return self._table[reg].size

    def __copy__(self):
        if False:
            print('Hello World!')
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._cache = self._cache.copy()
        result._registers = self._registers.copy()
        return result

class AMD64Operand(Operand):
    """This class deals with capstone X86 operands"""

    def __init__(self, cpu: Cpu, op):
        if False:
            while True:
                i = 10
        super().__init__(cpu, op)

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        type_map = {cs.x86.X86_OP_REG: 'register', cs.x86.X86_OP_MEM: 'memory', cs.x86.X86_OP_IMM: 'immediate'}
        return type_map[self.op.type]

    def address(self):
        if False:
            while True:
                i = 10
        (cpu, o) = (self.cpu, self.op)
        address = 0
        if self.mem.segment is not None:
            seg = self.mem.segment
            (base, size, ty) = cpu.get_descriptor(cpu.read_register(seg))
            address += base
        else:
            seg = 'DS'
            if self.mem.base is not None and self.mem.base in ['SP', 'ESP', 'EBP']:
                seg = 'SS'
            (base, size, ty) = cpu.get_descriptor(cpu.read_register(seg))
            address += base
        if self.mem.base is not None:
            base = self.mem.base
            address += cpu.read_register(base)
        if self.mem.index is not None:
            index = self.mem.index
            address += self.mem.scale * cpu.read_register(index)
        address += self.mem.disp
        return address & (1 << cpu.address_bit_size) - 1

    def read(self):
        if False:
            i = 10
            return i + 15
        (cpu, o) = (self.cpu, self.op)
        if self.type == 'register':
            value = cpu.read_register(self.reg)
            return value
        elif self.type == 'immediate':
            return o.imm
        elif self.type == 'memory':
            value = cpu.read_int(self.address(), self.size)
            return value
        else:
            raise NotImplementedError('read_operand unknown type', o.type)

    def write(self, value):
        if False:
            print('Hello World!')
        (cpu, o) = (self.cpu, self.op)
        if self.type == 'register':
            cpu.write_register(self.reg, value)
        elif self.type == 'memory':
            cpu.write_int(self.address(), value, self.size)
        else:
            raise NotImplementedError('write_operand unknown type', o.type)
        return value & (1 << self.size) - 1

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        return self.op.size * 8

    def __getattr__(self, name):
        if False:
            return 10
        return getattr(self.op, name)

class X86Cpu(Cpu):
    """
    A CPU model.
    """

    def __init__(self, regfile: RegisterFile, memory: Memory, *args, **kwargs):
        if False:
            return 10
        '\n        Builds a CPU model.\n        :param regfile: regfile object for this CPU.\n        :param memory: memory object for this CPU.\n        '
        super().__init__(regfile, memory, *args, **kwargs)
        self._segments: Dict[str, Any] = {}

    def __getstate__(self):
        if False:
            print('Hello World!')
        state = super().__getstate__()
        state['segments'] = self._segments
        return state

    def __setstate__(self, state):
        if False:
            return 10
        self._segments = state['segments']
        super().__setstate__(state)

    def set_descriptor(self, selector, base, limit, perms):
        if False:
            for i in range(10):
                print('nop')
        assert selector >= 0 and selector < 65535
        assert base >= 0 and base < 1 << self.address_bit_size
        assert limit >= 0 and limit < 65535 or limit & 4095 == 0
        self._publish('will_set_descriptor', selector, base, limit, perms)
        self._segments[selector] = (base, limit, perms)
        self._publish('did_set_descriptor', selector, base, limit, perms)

    def get_descriptor(self, selector):
        if False:
            while True:
                i = 10
        return self._segments.setdefault(selector, (0, 4294963200, 'rwx'))

    def _wrap_operands(self, operands):
        if False:
            for i in range(10):
                print('nop')
        return [AMD64Operand(self, op) for op in operands]

    def push(cpu, value, size):
        if False:
            print('Hello World!')
        '\n        Writes a value in the stack.\n\n        :param value: the value to put in the stack.\n        :param size: the size of the value.\n        '
        assert size in (8, 16, cpu.address_bit_size)
        cpu.STACK = cpu.STACK - size // 8
        (base, _, _) = cpu.get_descriptor(cpu.read_register('SS'))
        address = cpu.STACK + base
        cpu.write_int(address, value, size)

    def pop(cpu, size):
        if False:
            while True:
                i = 10
        '\n        Gets a value from the stack.\n\n        :rtype: int\n        :param size: the size of the value to consume from the stack.\n        :return: the value from the stack.\n        '
        assert size in (16, cpu.address_bit_size)
        (base, _, _) = cpu.get_descriptor(cpu.SS)
        address = cpu.STACK + base
        value = cpu.read_int(address, size)
        cpu.STACK = cpu.STACK + size // 8
        return value

    def invalidate_cache(cpu, address, size):
        if False:
            return 10
        'remove decoded instruction from instruction cache'
        cache = cpu.instruction_cache
        for offset in range(size):
            if address + offset in cache:
                del cache[address + offset]

    def canonicalize_instruction_name(self, instruction):
        if False:
            for i in range(10):
                print('nop')
        if instruction.opcode[0] in (164, 165):
            name = 'MOVS'
        else:
            name = instruction.insn_name().upper()
        name = OP_NAME_MAP.get(name, name)
        return name

    def read_register_as_bitfield(self, name):
        if False:
            print('Hello World!')
        'Read a register and return its value as a bitfield.\n        - if the register holds a bitvector, the bitvector object is returned.\n        - if the register holds a concrete value (int/float) it is returned as\n        a bitfield matching its representation in memory\n\n        This is mainly used to be able to write floating point registers to\n        memory.\n        '
        value = self.read_register(name)
        if isinstance(value, tuple):
            (mantissa, exponent) = value
            value = mantissa + (exponent << 64)
        return value

    def _calculate_CMP_flags(self, size, res, arg0, arg1):
        if False:
            for i in range(10):
                print('nop')
        SIGN_MASK = 1 << size - 1
        self.CF = Operators.ULT(arg0, arg1)
        self.AF = (arg0 ^ arg1 ^ res) & 16 != 0
        self.ZF = res == 0
        self.SF = res & SIGN_MASK != 0
        sign0 = arg0 & SIGN_MASK == SIGN_MASK
        sign1 = arg1 & SIGN_MASK == SIGN_MASK
        signr = res & SIGN_MASK == SIGN_MASK
        self.OF = Operators.AND(sign0 ^ sign1, sign0 ^ signr)
        self.PF = self._calculate_parity_flag(res)

    def _calculate_parity_flag(self, res):
        if False:
            print('Hello World!')
        return (res ^ res >> 1 ^ res >> 2 ^ res >> 3 ^ res >> 4 ^ res >> 5 ^ res >> 6 ^ res >> 7) & 1 == 0

    def _calculate_logic_flags(self, size, res):
        if False:
            while True:
                i = 10
        SIGN_MASK = 1 << size - 1
        self.CF = False
        self.AF = False
        self.ZF = res == 0
        self.SF = res & SIGN_MASK != 0
        self.OF = False
        self.PF = self._calculate_parity_flag(res)

    @staticmethod
    def CPUID_helper(PC: int, EAX: int, ECX: int) -> Tuple[int, int, int, int]:
        if False:
            return 10
        '\n        Takes values in eax and ecx to perform logic on what to return to (EAX,\n        EBX, ECX, EDX), in that order.\n        '
        conf = {0: (4, 1970169159, 1818588270, 1231384169), 1: (1635, 2048, 35136000, 126386433), 2: (1979931137, 15775231, 0, 12648448), 4: {0: (469778721, 29360191, 63, 0), 1: (469778722, 29360191, 63, 0), 2: (469778755, 29360191, 511, 0), 3: (470008163, 62914623, 4095, 6)}, 7: (0, 0, 0, 0), 8: (0, 0, 0, 0), 11: {0: (1, 2, 256, 5), 1: (4, 4, 513, 3)}, 13: {0: (0, 0, 0, 0), 1: (0, 0, 0, 0)}, 2147483648: (2147483648, 0, 0, 0)}
        if EAX not in conf:
            logger.warning('CPUID with EAX=%x not implemented @ %x', EAX, PC)
            return (0, 0, 0, 0)
        if isinstance(conf[EAX], tuple):
            return conf[EAX]
        if ECX not in conf[EAX]:
            logger.warning('CPUID with EAX=%x ECX=%x not implemented @ %x', EAX, ECX, PC)
            return (0, 0, 0, 0)
        return conf[EAX][ECX]

    @instruction
    def CPUID(cpu):
        if False:
            return 10
        "\n        CPUID instruction.\n\n        The ID flag (bit 21) in the EFLAGS register indicates support for the\n        CPUID instruction.  If a software procedure can set and clear this\n        flag, the processor executing the procedure supports the CPUID\n        instruction. This instruction operates the same in non-64-bit modes and\n        64-bit mode.  CPUID returns processor identification and feature\n        information in the EAX, EBX, ECX, and EDX registers.\n\n        The instruction's output is dependent on the contents of the EAX\n        register upon execution.\n\n        :param cpu: current CPU.\n        "
        (cpu.EAX, cpu.EBX, cpu.ECX, cpu.EDX) = X86Cpu.CPUID_helper(cpu.PC, cpu.EAX, cpu.ECX)

    @instruction
    def XGETBV(cpu):
        if False:
            print('Hello World!')
        '\n        XGETBV instruction.\n\n        Reads the contents of the extended cont register (XCR) specified in the ECX register into registers EDX:EAX.\n        Implemented only for ECX = 0.\n\n        :param cpu: current CPU.\n        '
        (cpu.EAX, cpu.EDX) = (7, 0)

    @instruction
    def AND(cpu, dest, src):
        if False:
            return 10
        '\n        Logical AND.\n\n        Performs a bitwise AND operation on the destination (first) and source\n        (second) operands and stores the result in the destination operand location.\n        Each bit of the result is set to 1 if both corresponding bits of the first and\n        second operands are 1; otherwise, it is set to 0.\n\n        The OF and CF flags are cleared; the SF, ZF, and PF flags are set according to the result::\n\n            DEST  =  DEST AND SRC;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        if src.size == 64 and src.type == 'immediate' and (dest.size == 64):
            arg1 = Operators.SEXTEND(src.read(), 32, 64)
        else:
            arg1 = src.read()
        res = dest.write(dest.read() & arg1)
        cpu._calculate_logic_flags(dest.size, res)

    @instruction
    def TEST(cpu, src1, src2):
        if False:
            i = 10
            return i + 15
        '\n        Logical compare.\n\n        Computes the bit-wise logical AND of first operand (source 1 operand)\n        and the second operand (source 2 operand) and sets the SF, ZF, and PF\n        status flags according to the result. The result is then discarded::\n\n            TEMP  =  SRC1 AND SRC2;\n            SF  =  MSB(TEMP);\n            IF TEMP  =  0\n            THEN ZF  =  1;\n            ELSE ZF  =  0;\n            FI:\n            PF  =  BitwiseXNOR(TEMP[0:7]);\n            CF  =  0;\n            OF  =  0;\n            (*AF is Undefined*)\n\n        :param cpu: current CPU.\n        :param src1: first operand.\n        :param src2: second operand.\n        '
        temp = src1.read() & src2.read()
        cpu.SF = temp & 1 << src1.size - 1 != 0
        cpu.ZF = temp == 0
        cpu.PF = cpu._calculate_parity_flag(temp)
        cpu.CF = False
        cpu.OF = False
        cpu.AF = False

    @instruction
    def NOT(cpu, dest):
        if False:
            for i in range(10):
                print('nop')
        "\n        One's complement negation.\n\n        Performs a bitwise NOT operation (each 1 is cleared to 0, and each 0\n        is set to 1) on the destination operand and stores the result in the destination\n        operand location::\n\n            DEST  =  NOT DEST;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        "
        res = dest.write(~dest.read())

    @instruction
    def XOR(cpu, dest, src):
        if False:
            return 10
        '\n        Logical exclusive OR.\n\n        Performs a bitwise exclusive-OR(XOR) operation on the destination (first)\n        and source (second) operands and stores the result in the destination\n        operand location.\n\n        Each bit of the result is 1 if the corresponding bits of the operands\n        are different; each bit is 0 if the corresponding bits are the same.\n\n        The OF and CF flags are cleared; the SF, ZF, and PF flags are set according to the result::\n\n            DEST  =  DEST XOR SRC;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        res = dest.write(dest.read() ^ src.read())
        cpu._calculate_logic_flags(dest.size, res)

    @instruction
    def OR(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Logical inclusive OR.\n\n        Performs a bitwise inclusive OR operation between the destination (first)\n        and source (second) operands and stores the result in the destination operand location.\n\n        Each bit of the result of the OR instruction is set to 0 if both corresponding\n        bits of the first and second operands are 0; otherwise, each bit is set\n        to 1.\n\n        The OF and CF flags are cleared; the SF, ZF, and PF flags are set according to the result::\n\n            DEST  =  DEST OR SRC;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        res = dest.write(dest.read() | src.read())
        cpu._calculate_logic_flags(dest.size, res)

    @instruction
    def AAA(cpu):
        if False:
            while True:
                i = 10
        '\n        ASCII adjust after addition.\n\n        Adjusts the sum of two unpacked BCD values to create an unpacked BCD\n        result. The AL register is the implied source and destination operand\n        for this instruction. The AAA instruction is only useful when it follows\n        an ADD instruction that adds (binary addition) two unpacked BCD values\n        and stores a byte result in the AL register. The AAA instruction then\n        adjusts the contents of the AL register to contain the correct 1-digit\n        unpacked BCD result.\n        If the addition produces a decimal carry, the AH register is incremented\n        by 1, and the CF and AF flags are set. If there was no decimal carry,\n        the CF and AF flags are cleared and the AH register is unchanged. In either\n        case, bits 4 through 7 of the AL register are cleared to 0.\n\n        This instruction executes as described in compatibility mode and legacy mode.\n        It is not valid in 64-bit mode.\n        ::\n                IF ((AL AND 0FH) > 9) OR (AF  =  1)\n                THEN\n                    AL  =  (AL + 6);\n                    AH  =  AH + 1;\n                    AF  =  1;\n                    CF  =  1;\n                ELSE\n                    AF  =  0;\n                    CF  =  0;\n                FI;\n                AL  =  AL AND 0FH;\n        :param cpu: current CPU.\n        '
        cpu.AF = Operators.OR(cpu.AL & 15 > 9, cpu.AF)
        cpu.CF = cpu.AF
        cpu.AH = Operators.ITEBV(8, cpu.AF, cpu.AH + 1, cpu.AH)
        cpu.AL = Operators.ITEBV(8, cpu.AF, cpu.AL + 6, cpu.AL)
        cpu.AL = cpu.AL & 15

    @instruction
    def AAD(cpu, imm):
        if False:
            print('Hello World!')
        '\n        ASCII adjust AX before division.\n\n        Adjusts two unpacked BCD digits (the least-significant digit in the\n        AL register and the most-significant digit in the AH register) so that\n        a division operation performed on the result will yield a correct unpacked\n        BCD value. The AAD instruction is only useful when it precedes a DIV instruction\n        that divides (binary division) the adjusted value in the AX register by\n        an unpacked BCD value.\n        The AAD instruction sets the value in the AL register to (AL + (10 * AH)), and then\n        clears the AH register to 00H. The value in the AX register is then equal to the binary\n        equivalent of the original unpacked two-digit (base 10) number in registers AH and AL.\n\n        The SF, ZF, and PF flags are set according to the resulting binary value in the AL register.\n\n        This instruction executes as described in compatibility mode and legacy mode.\n        It is not valid in 64-bit mode.::\n\n                tempAL  =  AL;\n                tempAH  =  AH;\n                AL  =  (tempAL + (tempAH * 10)) AND FFH;\n                AH  =  0\n\n        :param cpu: current CPU.\n        '
        cpu.AL += cpu.AH * imm.read()
        cpu.AH = 0
        cpu._calculate_logic_flags(8, cpu.AL)

    @instruction
    def AAM(cpu, imm=None):
        if False:
            while True:
                i = 10
        '\n        ASCII adjust AX after multiply.\n\n        Adjusts the result of the multiplication of two unpacked BCD values\n        to create a pair of unpacked (base 10) BCD values. The AX register is\n        the implied source and destination operand for this instruction. The AAM\n        instruction is only useful when it follows a MUL instruction that multiplies\n        (binary multiplication) two unpacked BCD values and stores a word result\n        in the AX register. The AAM instruction then adjusts the contents of the\n        AX register to contain the correct 2-digit unpacked (base 10) BCD result.\n\n        The SF, ZF, and PF flags are set according to the resulting binary value in the AL register.\n\n        This instruction executes as described in compatibility mode and legacy mode.\n        It is not valid in 64-bit mode.::\n\n                tempAL  =  AL;\n                AH  =  tempAL / 10;\n                AL  =  tempAL MOD 10;\n\n        :param cpu: current CPU.\n        '
        imm = imm.read()
        cpu.AH = Operators.UDIV(cpu.AL, imm)
        cpu.AL = Operators.UREM(cpu.AL, imm)
        cpu._calculate_logic_flags(8, cpu.AL)

    @instruction
    def AAS(cpu):
        if False:
            print('Hello World!')
        '\n        ASCII Adjust AL after subtraction.\n\n        Adjusts the result of the subtraction of two unpacked BCD values to  create a unpacked\n        BCD result. The AL register is the implied source and destination operand for this instruction.\n        The AAS instruction is only useful when it follows a SUB instruction that subtracts\n        (binary subtraction) one unpacked BCD value from another and stores a byte result in the AL\n        register. The AAA instruction then adjusts the contents of the AL register to contain the\n        correct 1-digit unpacked BCD result. If the subtraction produced a decimal carry, the AH register\n        is decremented by 1, and the CF and AF flags are set. If no decimal carry occurred, the CF and AF\n        flags are cleared, and the AH register is unchanged. In either case, the AL register is left with\n        its top nibble set to 0.\n\n        The AF and CF flags are set to 1 if there is a decimal borrow; otherwise, they are cleared to 0.\n\n        This instruction executes as described in compatibility mode and legacy mode.\n        It is not valid in 64-bit mode.::\n\n\n                IF ((AL AND 0FH) > 9) Operators.OR(AF  =  1)\n                THEN\n                    AX  =  AX - 6;\n                    AH  =  AH - 1;\n                    AF  =  1;\n                    CF  =  1;\n                ELSE\n                    CF  =  0;\n                    AF  =  0;\n                FI;\n                AL  =  AL AND 0FH;\n\n        :param cpu: current CPU.\n        '
        if cpu.AL & 15 > 9 or cpu.AF == 1:
            cpu.AX = cpu.AX - 6
            cpu.AH = cpu.AH - 1
            cpu.AF = True
            cpu.CF = True
        else:
            cpu.AF = False
            cpu.CF = False
        cpu.AL = cpu.AL & 15

    @instruction
    def ADC(cpu, dest, src):
        if False:
            return 10
        '\n        Adds with carry.\n\n        Adds the destination operand (first operand), the source operand (second operand),\n        and the carry (CF) flag and stores the result in the destination operand. The state\n        of the CF flag represents a carry from a previous addition. When an immediate value\n        is used as an operand, it is sign-extended to the length of the destination operand\n        format. The ADC instruction does not distinguish between signed or unsigned operands.\n        Instead, the processor evaluates the result for both data types and sets the OF and CF\n        flags to indicate a carry in the signed or unsigned result, respectively. The SF flag\n        indicates the sign of the signed result. The ADC instruction is usually executed as\n        part of a multibyte or multiword addition in which an ADD instruction is followed by an\n        ADC instruction::\n\n                DEST  =  DEST + SRC + CF;\n\n        The OF, SF, ZF, AF, CF, and PF flags are set according to the result.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        cpu._ADD(dest, src, carry=True)

    @instruction
    def ADD(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Add.\n\n        Adds the first operand (destination operand) and the second operand (source operand)\n        and stores the result in the destination operand. When an immediate value is used as\n        an operand, it is sign-extended to the length of the destination operand format.\n        The ADD instruction does not distinguish between signed or unsigned operands. Instead,\n        the processor evaluates the result for both data types and sets the OF and CF flags to\n        indicate a carry in the signed or unsigned result, respectively. The SF flag indicates\n        the sign of the signed result::\n\n                DEST  =  DEST + SRC;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        cpu._ADD(dest, src, carry=False)

    def _ADD(cpu, dest, src, carry=False):
        if False:
            i = 10
            return i + 15
        MASK = (1 << dest.size) - 1
        SIGN_MASK = 1 << dest.size - 1
        arg0 = dest.read()
        if src.size < dest.size:
            arg1 = Operators.SEXTEND(src.read(), src.size, dest.size)
        else:
            arg1 = src.read()
        to_add = arg1
        if carry:
            cv = Operators.ITEBV(dest.size, cpu.CF, 1, 0)
            to_add = arg1 + cv
        res = dest.write(arg0 + to_add & MASK)
        tempCF = Operators.OR(Operators.ULT(res, arg0 & MASK), Operators.ULT(res, arg1 & MASK))
        if carry:
            tempCF = Operators.OR(tempCF, Operators.AND(res == MASK, cpu.CF))
        cpu.CF = tempCF
        cpu.AF = (arg0 ^ arg1 ^ res) & 16 != 0
        cpu.ZF = res == 0
        cpu.SF = res & SIGN_MASK != 0
        cpu.OF = (arg0 ^ arg1 ^ SIGN_MASK) & (res ^ arg1) & SIGN_MASK != 0
        cpu.PF = cpu._calculate_parity_flag(res)

    @instruction
    def CMP(cpu, src1, src2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compares two operands.\n\n        Compares the first source operand with the second source operand and sets the status flags\n        in the EFLAGS register according to the results. The comparison is performed by subtracting\n        the second operand from the first operand and then setting the status flags in the same manner\n        as the SUB instruction. When an immediate value is used as an operand, it is sign-extended to\n        the length of the first operand::\n\n                temp  =  SRC1 - SignExtend(SRC2);\n                ModifyStatusFlags; (* Modify status flags in the same manner as the SUB instruction*)\n\n        The CF, OF, SF, ZF, AF, and PF flags are set according to the result.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        arg0 = src1.read()
        arg1 = Operators.SEXTEND(src2.read(), src2.size, src1.size)
        cpu._calculate_CMP_flags(src1.size, arg0 - arg1, arg0, arg1)

    @instruction
    def CMPXCHG(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compares and exchanges.\n\n        Compares the value in the AL, AX, EAX or RAX register (depending on the\n        size of the operand) with the first operand (destination operand). If\n        the two values are equal, the second operand (source operand) is loaded\n        into the destination operand. Otherwise, the destination operand is\n        loaded into the AL, AX, EAX or RAX register.\n\n        The ZF flag is set if the values in the destination operand and\n        register AL, AX, or EAX are equal; otherwise it is cleared. The CF, PF,\n        AF, SF, and OF flags are set according to the results of the comparison\n        operation::\n\n        (* accumulator  =  AL, AX, EAX or RAX,  depending on whether *)\n        (* a byte, word, a doubleword or a 64bit comparison is being performed*)\n        IF accumulator  ==  DEST\n        THEN\n            ZF  =  1\n            DEST  =  SRC\n        ELSE\n            ZF  =  0\n            accumulator  =  DEST\n        FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        size = dest.size
        reg_name = {8: 'AL', 16: 'AX', 32: 'EAX', 64: 'RAX'}[size]
        accumulator = cpu.read_register(reg_name)
        sval = src.read()
        dval = dest.read()
        cpu.write_register(reg_name, dval)
        dest.write(Operators.ITEBV(size, accumulator == dval, sval, dval))
        cpu._calculate_CMP_flags(size, accumulator - dval, accumulator, dval)

    @instruction
    def CMPXCHG8B(cpu, dest):
        if False:
            i = 10
            return i + 15
        '\n        Compares and exchanges bytes.\n\n        Compares the 64-bit value in EDX:EAX (or 128-bit value in RDX:RAX if\n        operand size is 128 bits) with the operand (destination operand). If\n        the values are equal, the 64-bit value in ECX:EBX (or 128-bit value in\n        RCX:RBX) is stored in the destination operand.  Otherwise, the value in\n        the destination operand is loaded into EDX:EAX (or RDX:RAX)::\n\n                IF (64-Bit Mode and OperandSize = 64)\n                THEN\n                    IF (RDX:RAX = DEST)\n                    THEN\n                        ZF = 1;\n                        DEST = RCX:RBX;\n                    ELSE\n                        ZF = 0;\n                        RDX:RAX = DEST;\n                    FI\n                ELSE\n                    IF (EDX:EAX = DEST)\n                    THEN\n                        ZF = 1;\n                        DEST = ECX:EBX;\n                    ELSE\n                        ZF = 0;\n                        EDX:EAX = DEST;\n                    FI;\n                FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        size = dest.size
        half_size = size // 2
        cmp_reg_name_l = {64: 'EAX', 128: 'RAX'}[size]
        cmp_reg_name_h = {64: 'EDX', 128: 'RDX'}[size]
        src_reg_name_l = {64: 'EBX', 128: 'RBX'}[size]
        src_reg_name_h = {64: 'ECX', 128: 'RCX'}[size]
        cmph = cpu.read_register(cmp_reg_name_h)
        cmpl = cpu.read_register(cmp_reg_name_l)
        srch = cpu.read_register(src_reg_name_h)
        srcl = cpu.read_register(src_reg_name_l)
        cmp0 = Operators.CONCAT(size, cmph, cmpl)
        src0 = Operators.CONCAT(size, srch, srcl)
        arg_dest = dest.read()
        cpu.ZF = arg_dest == cmp0
        dest.write(Operators.ITEBV(size, cpu.ZF, Operators.CONCAT(size, srch, srcl), arg_dest))
        cpu.write_register(cmp_reg_name_l, Operators.ITEBV(half_size, cpu.ZF, cmpl, Operators.EXTRACT(arg_dest, 0, half_size)))
        cpu.write_register(cmp_reg_name_h, Operators.ITEBV(half_size, cpu.ZF, cmph, Operators.EXTRACT(arg_dest, half_size, half_size)))

    @instruction
    def DAA(cpu):
        if False:
            return 10
        '\n        Decimal adjusts AL after addition.\n\n        Adjusts the sum of two packed BCD values to create a packed BCD result. The AL register\n        is the implied source and destination operand. If a decimal carry is detected, the CF\n        and AF flags are set accordingly.\n        The CF and AF flags are set if the adjustment of the value results in a decimal carry in\n        either digit of the result. The SF, ZF, and PF flags are set according to the result.\n\n        This instruction is not valid in 64-bit mode.::\n\n                IF (((AL AND 0FH) > 9) or AF  =  1)\n                THEN\n                    AL  =  AL + 6;\n                    CF  =  CF OR CarryFromLastAddition; (* CF OR carry from AL  =  AL + 6 *)\n                    AF  =  1;\n                ELSE\n                    AF  =  0;\n                FI;\n                IF ((AL AND F0H) > 90H) or CF  =  1)\n                THEN\n                    AL  =  AL + 60H;\n                    CF  =  1;\n                ELSE\n                    CF  =  0;\n                FI;\n\n        :param cpu: current CPU.\n        '
        cpu.AF = Operators.OR(cpu.AL & 15 > 9, cpu.AF)
        oldAL = cpu.AL
        cpu.AL = Operators.ITEBV(8, cpu.AF, cpu.AL + 6, cpu.AL)
        cpu.CF = Operators.ITE(cpu.AF, Operators.OR(cpu.CF, cpu.AL < oldAL), cpu.CF)
        cpu.CF = Operators.OR(cpu.AL & 240 > 144, cpu.CF)
        cpu.AL = Operators.ITEBV(8, cpu.CF, cpu.AL + 96, cpu.AL)
        '\n        #old not-symbolic aware version...\n        if ((cpu.AL & 0x0f) > 9) or cpu.AF:\n            oldAL = cpu.AL\n            cpu.AL =  cpu.AL + 6\n            cpu.CF = Operators.OR(cpu.CF, cpu.AL < oldAL)\n            cpu.AF  =  True\n        else:\n            cpu.AF  =  False\n\n        if ((cpu.AL & 0xf0) > 0x90) or cpu.CF:\n            cpu.AL  = cpu.AL + 0x60\n            cpu.CF  =  True\n        else:\n            cpu.CF  =  False\n        '
        cpu.ZF = cpu.AL == 0
        cpu.SF = cpu.AL & 128 != 0
        cpu.PF = cpu._calculate_parity_flag(cpu.AL)

    @instruction
    def DAS(cpu):
        if False:
            return 10
        '\n        Decimal adjusts AL after subtraction.\n\n        Adjusts the result of the subtraction of two packed BCD values to create a packed BCD result.\n        The AL register is the implied source and destination operand. If a decimal borrow is detected,\n        the CF and AF flags are set accordingly. This instruction is not valid in 64-bit mode.\n\n        The SF, ZF, and PF flags are set according to the result.::\n\n                IF (AL AND 0FH) > 9 OR AF  =  1\n                THEN\n                    AL  =  AL - 6;\n                    CF  =  CF OR BorrowFromLastSubtraction; (* CF OR borrow from AL  =  AL - 6 *)\n                    AF  =  1;\n                ELSE\n                    AF  =  0;\n                FI;\n                IF ((AL > 99H) or OLD_CF  =  1)\n                THEN\n                    AL  =  AL - 60H;\n                    CF  =  1;\n\n        :param cpu: current CPU.\n        '
        oldAL = cpu.AL
        oldCF = cpu.CF
        cpu.AF = Operators.OR(cpu.AL & 15 > 9, cpu.AF)
        cpu.AL = Operators.ITEBV(8, cpu.AF, cpu.AL - 6, cpu.AL)
        cpu.CF = Operators.ITE(cpu.AF, Operators.OR(oldCF, cpu.AL > oldAL), cpu.CF)
        cpu.CF = Operators.ITE(Operators.OR(oldAL > 153, oldCF), True, cpu.CF)
        cpu.AL = Operators.ITEBV(8, Operators.OR(oldAL > 153, oldCF), cpu.AL - 96, cpu.AL)
        '\n        if (cpu.AL & 0x0f) > 9 or cpu.AF:\n            cpu.AL = cpu.AL - 6;\n            cpu.CF = Operators.OR(oldCF, cpu.AL > oldAL)\n            cpu.AF = True\n        else:\n            cpu.AF  =  False\n\n        if ((oldAL > 0x99) or oldCF):\n            cpu.AL = cpu.AL - 0x60\n            cpu.CF = True\n        '
        cpu.ZF = cpu.AL == 0
        cpu.SF = cpu.AL & 128 != 0
        cpu.PF = cpu._calculate_parity_flag(cpu.AL)

    @instruction
    def DEC(cpu, dest):
        if False:
            i = 10
            return i + 15
        "\n        Decrements by 1.\n\n        Subtracts 1 from the destination operand, while preserving the state of\n        the CF flag. The destination operand can be a register or a memory\n        location. This instruction allows a loop counter to be updated without\n        disturbing the CF flag. (To perform a decrement operation that updates\n        the CF flag, use a SUB instruction with an immediate operand of 1.) The\n        instruction's 64-bit mode default operation size is 32 bits.\n\n        The OF, SF, ZF, AF, and PF flags are set according to the result::\n\n                DEST  =  DEST - 1;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        "
        arg0 = dest.read()
        res = dest.write(arg0 - 1)
        res &= (1 << dest.size) - 1
        SIGN_MASK = 1 << dest.size - 1
        cpu.AF = (arg0 ^ 1 ^ res) & 16 != 0
        cpu.ZF = res == 0
        cpu.SF = res & SIGN_MASK != 0
        cpu.OF = res == SIGN_MASK
        cpu.PF = cpu._calculate_parity_flag(res)

    @instruction
    def DIV(cpu, src):
        if False:
            while True:
                i = 10
        '\n        Unsigned divide.\n\n        Divides (unsigned) the value in the AX register, DX:AX register pair,\n        or EDX:EAX or RDX:RAX register pair (dividend) by the source operand\n        (divisor) and stores the result in the AX (AH:AL), DX:AX, EDX:EAX or\n        RDX:RAX registers. The source operand can be a general-purpose register\n        or a memory location. The action of this instruction depends of the\n        operand size (dividend/divisor). Division using 64-bit operand is\n        available only in 64-bit mode. Non-integral results are truncated\n        (chopped) towards 0. The reminder is always less than the divisor in\n        magnitude. Overflow is indicated with the #DE (divide error) exception\n        rather than with the CF flag::\n\n            IF SRC  =  0\n                THEN #DE; FI;(* divide error *)\n            IF OperandSize  =  8 (* word/byte operation *)\n                THEN\n                    temp  =  AX / SRC;\n                    IF temp > FFH\n                        THEN #DE; (* divide error *) ;\n                        ELSE\n                            AL  =  temp;\n                            AH  =  AX MOD SRC;\n                    FI;\n                ELSE IF OperandSize  =  16 (* doubleword/word operation *)\n                    THEN\n                        temp  =  DX:AX / SRC;\n                        IF temp > FFFFH\n                            THEN #DE; (* divide error *) ;\n                        ELSE\n                            AX  =  temp;\n                            DX  =  DX:AX MOD SRC;\n                        FI;\n                    FI;\n                ELSE If OperandSize = 32 (* quadword/doubleword operation *)\n                    THEN\n                        temp  =  EDX:EAX / SRC;\n                        IF temp > FFFFFFFFH\n                            THEN #DE; (* divide error *) ;\n                        ELSE\n                            EAX  =  temp;\n                            EDX  =  EDX:EAX MOD SRC;\n                        FI;\n                    FI;\n                ELSE IF OperandSize = 64 (*Doublequadword/quadword operation*)\n                    THEN\n                        temp = RDX:RAX / SRC;\n                        IF temp > FFFFFFFFFFFFFFFFH\n                            THEN #DE; (* Divide error *)\n                        ELSE\n                            RAX = temp;\n                            RDX = RDX:RAX MOD SRC;\n                        FI;\n                    FI;\n            FI;\n\n        :param cpu: current CPU.\n        :param src: source operand.\n        '
        size = src.size
        reg_name_h = {8: 'DL', 16: 'DX', 32: 'EDX', 64: 'RDX'}[size]
        reg_name_l = {8: 'AL', 16: 'AX', 32: 'EAX', 64: 'RAX'}[size]
        dividend = Operators.CONCAT(size * 2, cpu.read_register(reg_name_h), cpu.read_register(reg_name_l))
        divisor = Operators.ZEXTEND(src.read(), size * 2)
        if isinstance(divisor, int) and divisor == 0:
            raise DivideByZeroError()
        quotient = Operators.UDIV(dividend, divisor)
        MASK = (1 << size) - 1
        if isinstance(quotient, int) and quotient > MASK:
            raise DivideByZeroError()
        remainder = Operators.UREM(dividend, divisor)
        cpu.write_register(reg_name_l, Operators.EXTRACT(quotient, 0, size))
        cpu.write_register(reg_name_h, Operators.EXTRACT(remainder, 0, size))

    @instruction
    def IDIV(cpu, src):
        if False:
            i = 10
            return i + 15
        '\n        Signed divide.\n\n        Divides (signed) the value in the AL, AX, or EAX register by the source\n        operand and stores the result in the AX, DX:AX, or EDX:EAX registers.\n        The source operand can be a general-purpose register or a memory\n        location. The action of this instruction depends on the operand size.::\n\n        IF SRC  =  0\n        THEN #DE; (* divide error *)\n        FI;\n        IF OpernadSize  =  8 (* word/byte operation *)\n        THEN\n            temp  =  AX / SRC; (* signed division *)\n            IF (temp > 7FH) Operators.OR(temp < 80H)\n            (* if a positive result is greater than 7FH or a negative result is\n            less than 80H *)\n            THEN #DE; (* divide error *) ;\n            ELSE\n                AL  =  temp;\n                AH  =  AX SignedModulus SRC;\n            FI;\n        ELSE\n            IF OpernadSize  =  16 (* doubleword/word operation *)\n            THEN\n                temp  =  DX:AX / SRC; (* signed division *)\n                IF (temp > 7FFFH) Operators.OR(temp < 8000H)\n                (* if a positive result is greater than 7FFFH *)\n                (* or a negative result is less than 8000H *)\n                THEN #DE; (* divide error *) ;\n                ELSE\n                    AX  =  temp;\n                    DX  =  DX:AX SignedModulus SRC;\n                FI;\n            ELSE (* quadword/doubleword operation *)\n                temp  =  EDX:EAX / SRC; (* signed division *)\n                IF (temp > 7FFFFFFFH) Operators.OR(temp < 80000000H)\n                (* if a positive result is greater than 7FFFFFFFH *)\n                (* or a negative result is less than 80000000H *)\n                THEN #DE; (* divide error *) ;\n                ELSE\n                    EAX  =  temp;\n                    EDX  =  EDX:EAX SignedModulus SRC;\n                FI;\n            FI;\n        FI;\n\n        :param cpu: current CPU.\n        :param src: source operand.\n        '
        reg_name_h = {8: 'AH', 16: 'DX', 32: 'EDX', 64: 'RDX'}[src.size]
        reg_name_l = {8: 'AL', 16: 'AX', 32: 'EAX', 64: 'RAX'}[src.size]
        dividend = Operators.CONCAT(src.size * 2, cpu.read_register(reg_name_h), cpu.read_register(reg_name_l))
        divisor = src.read()
        if isinstance(divisor, int) and divisor == 0:
            raise DivideByZeroError()
        dst_size = src.size * 2
        divisor = Operators.SEXTEND(divisor, src.size, dst_size)
        mask = (1 << dst_size) - 1
        sign_mask = 1 << dst_size - 1
        dividend_sign = dividend & sign_mask != 0
        divisor_sign = divisor & sign_mask != 0
        if isinstance(divisor, int):
            if divisor_sign:
                divisor = ~divisor + 1 & mask
                divisor = -divisor
        if isinstance(dividend, int):
            if dividend_sign:
                dividend = ~dividend + 1 & mask
                dividend = -dividend
        quotient = Operators.SDIV(dividend, divisor)
        if isinstance(dividend, int) and isinstance(dividend, int):
            remainder = dividend - quotient * divisor
        else:
            remainder = Operators.SREM(dividend, divisor)
        cpu.write_register(reg_name_l, Operators.EXTRACT(quotient, 0, src.size))
        cpu.write_register(reg_name_h, Operators.EXTRACT(remainder, 0, src.size))

    @instruction
    def IMUL(cpu, *operands):
        if False:
            return 10
        '\n        Signed multiply.\n\n        Performs a signed multiplication of two operands. This instruction has\n        three forms, depending on the number of operands.\n            - One-operand form. This form is identical to that used by the MUL\n            instruction. Here, the source operand (in a general-purpose\n            register or memory location) is multiplied by the value in the AL,\n            AX, or EAX register (depending on the operand size) and the product\n            is stored in the AX, DX:AX, or EDX:EAX registers, respectively.\n            - Two-operand form. With this form the destination operand (the\n            first operand) is multiplied by the source operand (second\n            operand). The destination operand is a general-purpose register and\n            the source operand is an immediate value, a general-purpose\n            register, or a memory location. The product is then stored in the\n            destination operand location.\n            - Three-operand form. This form requires a destination operand (the\n            first operand) and two source operands (the second and the third\n            operands). Here, the first source operand (which can be a\n            general-purpose register or a memory location) is multiplied by the\n            second source operand (an immediate value). The product is then\n            stored in the destination operand (a general-purpose register).\n\n        When an immediate value is used as an operand, it is sign-extended to\n        the length of the destination operand format. The CF and OF flags are\n        set when significant bits are carried into the upper half of the\n        result. The CF and OF flags are cleared when the result fits exactly in\n        the lower half of the result. The three forms of the IMUL instruction\n        are similar in that the length of the product is calculated to twice\n        the length of the operands. With the one-operand form, the product is\n        stored exactly in the destination. With the two- and three- operand\n        forms, however, result is truncated to the length of the destination\n        before it is stored in the destination register. Because of this\n        truncation, the CF or OF flag should be tested to ensure that no\n        significant bits are lost. The two- and three-operand forms may also be\n        used with unsigned operands because the lower half of the product is\n        the same regardless if the operands are signed or unsigned. The CF and\n        OF flags, however, cannot be used to determine if the upper half of the\n        result is non-zero::\n\n        IF (NumberOfOperands == 1)\n        THEN\n            IF (OperandSize == 8)\n            THEN\n                AX = AL * SRC (* Signed multiplication *)\n                IF AL == AX\n                THEN\n                    CF = 0; OF = 0;\n                ELSE\n                    CF = 1; OF = 1;\n                FI;\n            ELSE\n                IF OperandSize == 16\n                THEN\n                    DX:AX = AX * SRC (* Signed multiplication *)\n                    IF sign_extend_to_32 (AX) == DX:AX\n                    THEN\n                        CF = 0; OF = 0;\n                    ELSE\n                        CF = 1; OF = 1;\n                    FI;\n                ELSE\n                    IF OperandSize == 32\n                    THEN\n                        EDX:EAX = EAX * SRC (* Signed multiplication *)\n                        IF EAX == EDX:EAX\n                        THEN\n                            CF = 0; OF = 0;\n                        ELSE\n                            CF = 1; OF = 1;\n                        FI;\n                    ELSE (* OperandSize = 64 *)\n                        RDX:RAX = RAX * SRC (* Signed multiplication *)\n                        IF RAX == RDX:RAX\n                        THEN\n                            CF = 0; OF = 0;\n                        ELSE\n                           CF = 1; OF = 1;\n                        FI;\n                    FI;\n                FI;\n        ELSE\n            IF (NumberOfOperands = 2)\n            THEN\n                temp = DEST * SRC (* Signed multiplication; temp is double DEST size *)\n                DEST = DEST * SRC (* Signed multiplication *)\n                IF temp != DEST\n                THEN\n                    CF = 1; OF = 1;\n                ELSE\n                    CF = 0; OF = 0;\n                FI;\n            ELSE (* NumberOfOperands = 3 *)\n                DEST = SRC1 * SRC2 (* Signed multiplication *)\n                temp = SRC1 * SRC2 (* Signed multiplication; temp is double SRC1 size *)\n                IF temp != DEST\n                THEN\n                    CF = 1; OF = 1;\n                ELSE\n                    CF = 0; OF = 0;\n                FI;\n            FI;\n        FI;\n\n        :param cpu: current CPU.\n        :param operands: variable list of operands.\n        '
        dest = operands[0]
        OperandSize = dest.size
        reg_name_h = {8: 'AH', 16: 'DX', 32: 'EDX', 64: 'RDX'}[OperandSize]
        reg_name_l = {8: 'AL', 16: 'AX', 32: 'EAX', 64: 'RAX'}[OperandSize]
        arg0 = dest.read()
        arg1 = None
        arg2 = None
        res = None
        if len(operands) == 1:
            arg1 = cpu.read_register(reg_name_l)
            temp = Operators.SEXTEND(arg0, OperandSize, OperandSize * 2) * Operators.SEXTEND(arg1, OperandSize, OperandSize * 2)
            temp = temp & (1 << OperandSize * 2) - 1
            cpu.write_register(reg_name_l, Operators.EXTRACT(temp, 0, OperandSize))
            cpu.write_register(reg_name_h, Operators.EXTRACT(temp, OperandSize, OperandSize))
            res = Operators.EXTRACT(temp, 0, OperandSize)
        elif len(operands) == 2:
            arg1 = operands[1].read()
            arg1 = Operators.SEXTEND(arg1, OperandSize, OperandSize * 2)
            temp = Operators.SEXTEND(arg0, OperandSize, OperandSize * 2) * arg1
            temp = temp & (1 << OperandSize * 2) - 1
            res = dest.write(Operators.EXTRACT(temp, 0, OperandSize))
        else:
            arg1 = operands[1].read()
            arg2 = operands[2].read()
            temp = Operators.SEXTEND(arg1, OperandSize, OperandSize * 2) * Operators.SEXTEND(arg2, operands[2].size, OperandSize * 2)
            temp = temp & (1 << OperandSize * 2) - 1
            res = dest.write(Operators.EXTRACT(temp, 0, OperandSize))
        cpu.CF = Operators.SEXTEND(res, OperandSize, OperandSize * 2) != temp
        cpu.OF = cpu.CF

    @instruction
    def INC(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Increments by 1.\n\n        Adds 1 to the destination operand, while preserving the state of the\n        CF flag. The destination operand can be a register or a memory location.\n        This instruction allows a loop counter to be updated without disturbing\n        the CF flag. (Use a ADD instruction with an immediate operand of 1 to\n        perform an increment operation that does updates the CF flag.)::\n\n                DEST  =  DEST +1;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        arg0 = dest.read()
        res = dest.write(arg0 + 1)
        res &= (1 << dest.size) - 1
        SIGN_MASK = 1 << dest.size - 1
        cpu.AF = (arg0 ^ 1 ^ res) & 16 != 0
        cpu.ZF = res == 0
        cpu.SF = res & SIGN_MASK != 0
        cpu.OF = res == SIGN_MASK
        cpu.PF = cpu._calculate_parity_flag(res)

    @instruction
    def MUL(cpu, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Unsigned multiply.\n\n        Performs an unsigned multiplication of the first operand (destination\n        operand) and the second operand (source operand) and stores the result\n        in the destination operand. The destination operand is an implied operand\n        located in register AL, AX or EAX (depending on the size of the operand);\n        the source operand is located in a general-purpose register or a memory location.\n\n        The result is stored in register AX, register pair DX:AX, or register\n        pair EDX:EAX (depending on the operand size), with the high-order bits\n        of the product contained in register AH, DX, or EDX, respectively. If\n        the high-order bits of the product are 0, the CF and OF flags are cleared;\n        otherwise, the flags are set::\n\n                IF byte operation\n                THEN\n                    AX  =  AL * SRC\n                ELSE (* word or doubleword operation *)\n                    IF OperandSize  =  16\n                    THEN\n                        DX:AX  =  AX * SRC\n                    ELSE (* OperandSize  =  32 *)\n                        EDX:EAX  =  EAX * SRC\n                    FI;\n                FI;\n\n        :param cpu: current CPU.\n        :param src: source operand.\n        '
        size = src.size
        (reg_name_low, reg_name_high) = {8: ('AL', 'AH'), 16: ('AX', 'DX'), 32: ('EAX', 'EDX'), 64: ('RAX', 'RDX')}[size]
        res = Operators.ZEXTEND(cpu.read_register(reg_name_low), 256) * Operators.ZEXTEND(src.read(), 256)
        cpu.write_register(reg_name_low, Operators.EXTRACT(res, 0, size))
        cpu.write_register(reg_name_high, Operators.EXTRACT(res, size, size))
        cpu.OF = Operators.EXTRACT(res, size, size) != 0
        cpu.CF = cpu.OF

    @instruction
    def NEG(cpu, dest):
        if False:
            print('Hello World!')
        "\n        Two's complement negation.\n\n        Replaces the value of operand (the destination operand) with its two's complement.\n        (This operation is equivalent to subtracting the operand from 0.) The destination operand is\n        located in a general-purpose register or a memory location::\n\n                IF DEST  =  0\n                THEN CF  =  0\n                ELSE CF  =  1;\n                FI;\n                DEST  =  - (DEST)\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        "
        source = dest.read()
        res = dest.write(-source)
        cpu._calculate_logic_flags(dest.size, res)
        cpu.CF = source != 0
        cpu.AF = res & 15 != 0

    @instruction
    def SBB(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Integer subtraction with borrow.\n\n        Adds the source operand (second operand) and the carry (CF) flag, and\n        subtracts the result from the destination operand (first operand). The\n        result of the subtraction is stored in the destination operand. The\n        destination operand can be a register or a memory location; the source\n        operand can be an immediate, a register, or a memory location.\n        (However, two memory operands cannot be used in one instruction.) The\n        state of the CF flag represents a borrow from a previous subtraction.\n        When an immediate value is used as an operand, it is sign-extended to\n        the length of the destination operand format.\n        The SBB instruction does not distinguish between signed or unsigned\n        operands. Instead, the processor evaluates the result for both data\n        types and sets the OF and CF flags to indicate a borrow in the signed\n        or unsigned result, respectively. The SF flag indicates the sign of the\n        signed result.  The SBB instruction is usually executed as part of a\n        multibyte or multiword subtraction in which a SUB instruction is\n        followed by a SBB instruction::\n\n                DEST  =  DEST - (SRC + CF);\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        cpu._SUB(dest, src, carry=True)

    @instruction
    def SUB(cpu, dest, src):
        if False:
            return 10
        '\n        Subtract.\n\n        Subtracts the second operand (source operand) from the first operand\n        (destination operand) and stores the result in the destination operand.\n        The destination operand can be a register or a memory location; the\n        source operand can be an immediate, register, or memory location.\n        (However, two memory operands cannot be used in one instruction.) When\n        an immediate value is used as an operand, it is sign-extended to the\n        length of the destination operand format.\n        The SUB instruction does not distinguish between signed or unsigned\n        operands. Instead, the processor evaluates the result for both\n        data types and sets the OF and CF flags to indicate a borrow in the\n        signed or unsigned result, respectively. The SF flag indicates the sign\n        of the signed result::\n\n            DEST  =  DEST - SRC;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        cpu._SUB(dest, src, carry=False)

    def _SUB(cpu, dest, src, carry=False):
        if False:
            return 10
        size = dest.size
        minuend = dest.read()
        if src.size < dest.size:
            subtrahend = Operators.SEXTEND(src.read(), src.size, size)
        else:
            subtrahend = src.read()
        if carry:
            cv = Operators.ITEBV(size, cpu.CF, 1, 0)
            subtrahend += cv
        res = dest.write(minuend - subtrahend) & (1 << size) - 1
        cpu._calculate_CMP_flags(dest.size, res, minuend, subtrahend)

    @instruction
    def XADD(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exchanges and adds.\n\n        Exchanges the first operand (destination operand) with the second operand\n        (source operand), then loads the sum of the two values into the destination\n        operand. The destination operand can be a register or a memory location;\n        the source operand is a register.\n        This instruction can be used with a LOCK prefix::\n\n                TEMP  =  SRC + DEST\n                SRC  =  DEST\n                DEST  =  TEMP\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        MASK = (1 << dest.size) - 1
        SIGN_MASK = 1 << dest.size - 1
        arg0 = dest.read()
        arg1 = src.read()
        temp = arg1 + arg0 & MASK
        src.write(arg0)
        dest.write(temp)
        tempCF = Operators.OR(Operators.ULT(temp, arg0), Operators.ULT(temp, arg1))
        cpu.CF = tempCF
        cpu.AF = (arg0 ^ arg1 ^ temp) & 16 != 0
        cpu.ZF = temp == 0
        cpu.SF = temp & SIGN_MASK != 0
        cpu.OF = (arg0 ^ arg1 ^ SIGN_MASK) & (temp ^ arg1) & SIGN_MASK != 0
        cpu.PF = cpu._calculate_parity_flag(temp)

    @instruction
    def BSWAP(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Byte swap.\n\n        Reverses the byte order of a 32-bit (destination) register: bits 0 through\n        7 are swapped with bits 24 through 31, and bits 8 through 15 are swapped\n        with bits 16 through 23. This instruction is provided for converting little-endian\n        values to big-endian format and vice versa.\n        To swap bytes in a word value (16-bit register), use the XCHG instruction.\n        When the BSWAP instruction references a 16-bit register, the result is\n        undefined::\n\n            TEMP  =  DEST\n            DEST[7..0]  =  TEMP[31..24]\n            DEST[15..8]  =  TEMP[23..16]\n            DEST[23..16]  =  TEMP[15..8]\n            DEST[31..24]  =  TEMP[7..0]\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        parts = []
        arg0 = dest.read()
        for i in range(0, dest.size, 8):
            parts.append(Operators.EXTRACT(arg0, i, 8))
        dest.write(Operators.CONCAT(8 * len(parts), *parts))

    @instruction
    def CMOVB(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Conditional move - Below/not above or equal.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.CF, src.read(), dest.read()))

    @instruction
    def CMOVA(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Conditional move - Above/not below or equal.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.AND(cpu.CF == False, cpu.ZF == False), src.read(), dest.read()))

    @instruction
    def CMOVAE(cpu, dest, src):
        if False:
            return 10
        '\n        Conditional move - Above or equal/not below.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.CF == False, src.read(), dest.read()))

    @instruction
    def CMOVBE(cpu, dest, src):
        if False:
            return 10
        '\n        Conditional move - Below or equal/not above.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.OR(cpu.CF, cpu.ZF), src.read(), dest.read()))

    @instruction
    def CMOVZ(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Conditional move - Equal/zero.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.ZF, src.read(), dest.read()))

    @instruction
    def CMOVNZ(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Conditional move - Not equal/not zero.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.ZF == False, src.read(), dest.read()))

    @instruction
    def CMOVP(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Conditional move - Parity/parity even.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.PF, src.read(), dest.read()))

    @instruction
    def CMOVNP(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Conditional move - Not parity/parity odd.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.PF == False, src.read(), dest.read()))

    @instruction
    def CMOVG(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Conditional move - Greater.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.AND(cpu.ZF == 0, cpu.SF == cpu.OF), src.read(), dest.read()))

    @instruction
    def CMOVGE(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Conditional move - Greater or equal/not less.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF ^ cpu.OF == 0, src.read(), dest.read()))

    @instruction
    def CMOVL(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Conditional move - Less/not greater or equal.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF ^ cpu.OF, src.read(), dest.read()))

    @instruction
    def CMOVLE(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Conditional move - Less or equal/not greater.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.OR(cpu.SF ^ cpu.OF, cpu.ZF), src.read(), dest.read()))

    @instruction
    def CMOVO(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Conditional move - Overflow.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.OF, src.read(), dest.read()))

    @instruction
    def CMOVNO(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Conditional move - Not overflow.\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.OF == False, src.read(), dest.read()))

    @instruction
    def CMOVS(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Conditional move - Sign (negative).\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF, src.read(), dest.read()))

    @instruction
    def CMOVNS(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Conditional move - Not sign (non-negative).\n\n        Tests the status flags in the EFLAGS register and moves the source operand\n        (second operand) to the destination operand (first operand) if the given\n        test condition is true.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF == False, src.read(), dest.read()))

    @instruction
    def LAHF(cpu):
        if False:
            print('Hello World!')
        '\n        Loads status flags into AH register.\n\n        Moves the low byte of the EFLAGS register (which includes status flags\n        SF, ZF, AF, PF, and CF) to the AH register. Reserved bits 1, 3, and 5\n        of the EFLAGS register are set in the AH register::\n\n                AH  =  EFLAGS(SF:ZF:0:AF:0:PF:1:CF);\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        used_regs = (cpu.SF, cpu.ZF, cpu.AF, cpu.PF, cpu.CF)
        is_expression = any((issymbolic(x) for x in used_regs))

        def make_flag(val, offset):
            if False:
                print('Hello World!')
            if is_expression:
                return Operators.ITEBV(size=8, cond=val, true_value=BitVecConstant(size=8, value=1 << offset), false_value=BitVecConstant(size=8, value=0))
            else:
                return val << offset
        cpu.AH = make_flag(cpu.SF, 7) | make_flag(cpu.ZF, 6) | make_flag(0, 5) | make_flag(cpu.AF, 4) | make_flag(0, 3) | make_flag(cpu.PF, 2) | make_flag(1, 1) | make_flag(cpu.CF, 0)

    @instruction
    def LDS(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Not implemented.\n\n        '
        raise NotImplementedError('LDS')

    @instruction
    def LES(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Not implemented.\n\n        '
        raise NotImplementedError('LES')

    @instruction
    def LFS(cpu, dest, src):
        if False:
            return 10
        '\n        Not implemented.\n\n        '
        raise NotImplementedError('LFS')

    @instruction
    def LGS(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Not implemented.\n\n        '
        raise NotImplementedError('LGS')

    @instruction
    def LSS(cpu, dest, src):
        if False:
            while True:
                i = 10
        "\n        Loads far pointer.\n\n        Loads a far pointer (segment selector and offset) from the second operand\n        (source operand) into a segment register and the first operand (destination\n        operand). The source operand specifies a 48-bit or a 32-bit pointer in\n        memory depending on the current setting of the operand-size attribute\n        (32 bits or 16 bits, respectively). The instruction opcode and the destination\n        operand specify a segment register/general-purpose register pair. The\n        16-bit segment selector from the source operand is loaded into the segment\n        register specified with the opcode (DS, SS, ES, FS, or GS). The 32-bit\n        or 16-bit offset is loaded into the register specified with the destination\n        operand.\n        In 64-bit mode, the instruction's default operation size is 32 bits. Using a\n        REX prefix in the form of REX.W promotes operation to specify a source operand\n        referencing an 80-bit pointer (16-bit selector, 64-bit offset) in memory.\n        If one of these instructions is executed in protected mode, additional\n        information from the segment descriptor pointed to by the segment selector\n        in the source operand is loaded in the hidden part of the selected segment\n        register.\n        Also in protected mode, a null selector (values 0000 through 0003) can\n        be loaded into DS, ES, FS, or GS registers without causing a protection\n        exception. (Any subsequent reference to a segment whose corresponding\n        segment register is loaded with a null selector, causes a general-protection\n        exception (#GP) and no memory reference to the segment occurs.)::\n\n                IF ProtectedMode\n                THEN IF SS is loaded\n                    THEN IF SegementSelector  =  null\n                        THEN #GP(0);\n                        FI;\n                    ELSE IF Segment selector index is not within descriptor table limits\n                        OR Segment selector RPL  CPL\n                        OR Access rights indicate nonwritable data segment\n                        OR DPL  CPL\n                        THEN #GP(selector);\n                        FI;\n                    ELSE IF Segment marked not present\n                        THEN #SS(selector);\n                        FI;\n                        SS  =  SegmentSelector(SRC);\n                        SS  =  SegmentDescriptor([SRC]);\n                    ELSE IF DS, ES, FS, or GS is loaded with non-null segment selector\n                        THEN IF Segment selector index is not within descriptor table limits\n                            OR Access rights indicate segment neither data nor readable code segment\n                            OR Segment is data or nonconforming-code segment\n                            AND both RPL and CPL > DPL)\n                            THEN #GP(selector);\n                            FI;\n                        ELSE IF Segment marked not present\n                            THEN #NP(selector);\n                            FI;\n                            SegmentRegister  =  SegmentSelector(SRC) AND RPL;\n                            SegmentRegister  =  SegmentDescriptor([SRC]);\n                        ELSE IF DS, ES, FS, or GS is loaded with a null selector:\n                            SegmentRegister  =  NullSelector;\n                            SegmentRegister(DescriptorValidBit)  =  0; (*hidden flag; not accessible by software*)\n                        FI;\n                    FI;\n                    IF (Real-Address or Virtual-8086 Mode)\n                    THEN\n                        SegmentRegister  =  SegmentSelector(SRC);\n                    FI;\n                    DEST  =  Offset(SRC);\n        "
        raise NotImplementedError('LSS')

    @instruction
    def LEA(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Loads effective address.\n\n        Computes the effective address of the second operand (the source operand) and stores it in the first operand\n        (destination operand). The source operand is a memory address (offset part) specified with one of the processors\n        addressing modes; the destination operand is a general-purpose register. The address-size and operand-size\n        attributes affect the action performed by this instruction. The operand-size\n        attribute of the instruction is determined by the chosen register; the address-size attribute is determined by the\n        attribute of the code segment.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(Operators.EXTRACT(src.address(), 0, dest.size))

    @instruction
    def MOV(cpu, dest, src, *rest):
        if False:
            i = 10
            return i + 15
        '\n        Move.\n\n        Copies the second operand (source operand) to the first operand (destination\n        operand). The source operand can be an immediate value, general-purpose\n        register, segment register, or memory location; the destination register\n        can be a general-purpose register, segment register, or memory location.\n        Both operands must be the same size, which can be a byte, a word, or a\n        doubleword.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        :param rest: workaround for a capstone bug, should never be provided\n        '
        dest.write(src.read())

    @instruction
    def MOVBE(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        "\n        Moves data after swapping bytes.\n\n        Performs a byte swap operation on the data copied from the second operand (source operand) and store the result\n        in the first operand (destination operand). The source operand can be a general-purpose register, or memory location; the destination register can be a general-purpose register, or a memory location; however, both operands can\n        not be registers, and only one operand can be a memory location. Both operands must be the same size, which can\n        be a word, a doubleword or quadword.\n        The MOVBE instruction is provided for swapping the bytes on a read from memory or on a write to memory; thus\n        providing support for converting little-endian values to big-endian format and vice versa.\n        In 64-bit mode, the instruction's default operation size is 32 bits. Use of the REX.R prefix permits access to additional registers (R8-R15). Use of the REX.W prefix promotes operation to 64 bits::\n\n                TEMP = SRC\n                IF ( OperandSize = 16)\n                THEN\n                    DEST[7:0] = TEMP[15:8];\n                    DEST[15:8] = TEMP[7:0];\n                ELSE IF ( OperandSize = 32)\n                    DEST[7:0] = TEMP[31:24];\n                    DEST[15:8] = TEMP[23:16];\n                    DEST[23:16] = TEMP[15:8];\n                    DEST[31:23] = TEMP[7:0];\n                ELSE IF ( OperandSize = 64)\n                    DEST[7:0] = TEMP[63:56];\n                    DEST[15:8] = TEMP[55:48];\n                    DEST[23:16] = TEMP[47:40];\n                    DEST[31:24] = TEMP[39:32];\n                    DEST[39:32] = TEMP[31:24];\n                    DEST[47:40] = TEMP[23:16];\n                    DEST[55:48] = TEMP[15:8];\n                    DEST[63:56] = TEMP[7:0];\n                FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        "
        size = dest.size
        arg0 = dest.read()
        temp = 0
        for pos in range(0, size, 8):
            temp = temp << 8 | arg0 & 255
            arg0 = arg0 >> 8
        dest.write(arg0)

    @instruction
    def SAHF(cpu):
        if False:
            print('Hello World!')
        '\n        Stores AH into flags.\n\n        Loads the SF, ZF, AF, PF, and CF flags of the EFLAGS register with values\n        from the corresponding bits in the AH register (bits 7, 6, 4, 2, and 0,\n        respectively). Bits 1, 3, and 5 of register AH are ignored; the corresponding\n        reserved bits (1, 3, and 5) in the EFLAGS register remain as shown below::\n\n                EFLAGS(SF:ZF:0:AF:0:PF:1:CF)  =  AH;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        eflags_size = 32
        val = cpu.AH & 213 | 2
        cpu.EFLAGS = Operators.ZEXTEND(val, eflags_size)

    @instruction
    def SETA(cpu, dest):
        if False:
            while True:
                i = 10
        '\n        Sets byte if above.\n\n        Sets the destination operand to 0 or 1 depending on the settings of the status flags (CF, SF, OF, ZF, and PF, 1, 0) in the\n        EFLAGS register. The destination operand points to a byte register or a byte in memory. The condition code suffix\n        (cc, 1, 0) indicates the condition being tested for::\n                IF condition\n                THEN\n                    DEST = 1;\n                ELSE\n                    DEST = 0;\n                FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.OR(cpu.CF, cpu.ZF) == False, 1, 0))

    @instruction
    def SETAE(cpu, dest):
        if False:
            i = 10
            return i + 15
        '\n        Sets byte if above or equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.CF == False, 1, 0))

    @instruction
    def SETB(cpu, dest):
        if False:
            while True:
                i = 10
        '\n        Sets byte if below.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.CF, 1, 0))

    @instruction
    def SETBE(cpu, dest):
        if False:
            return 10
        '\n        Sets byte if below or equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.OR(cpu.CF, cpu.ZF), 1, 0))

    @instruction
    def SETC(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Sets if carry.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.CF, 1, 0))

    @instruction
    def SETE(cpu, dest):
        if False:
            return 10
        '\n        Sets byte if equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.ZF, 1, 0))

    @instruction
    def SETG(cpu, dest):
        if False:
            return 10
        '\n        Sets byte if greater.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.AND(cpu.ZF == False, cpu.SF == cpu.OF), 1, 0))

    @instruction
    def SETGE(cpu, dest):
        if False:
            return 10
        '\n        Sets byte if greater or equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF == cpu.OF, 1, 0))

    @instruction
    def SETL(cpu, dest):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets byte if less.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF != cpu.OF, 1, 0))

    @instruction
    def SETLE(cpu, dest):
        if False:
            return 10
        '\n        Sets byte if less or equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.OR(cpu.ZF, cpu.SF != cpu.OF), 1, 0))

    @instruction
    def SETNA(cpu, dest):
        if False:
            return 10
        '\n        Sets byte if not above.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.OR(cpu.CF, cpu.ZF), 1, 0))

    @instruction
    def SETNAE(cpu, dest):
        if False:
            while True:
                i = 10
        '\n        Sets byte if not above or equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.CF, 1, 0))

    @instruction
    def SETNB(cpu, dest):
        if False:
            while True:
                i = 10
        '\n        Sets byte if not below.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.CF == False, 1, 0))

    @instruction
    def SETNBE(cpu, dest):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets byte if not below or equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.AND(cpu.CF == False, cpu.ZF == False), 1, 0))

    @instruction
    def SETNC(cpu, dest):
        if False:
            return 10
        '\n        Sets byte if not carry.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.CF == False, 1, 0))

    @instruction
    def SETNE(cpu, dest):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets byte if not equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.ZF == False, 1, 0))

    @instruction
    def SETNG(cpu, dest):
        if False:
            i = 10
            return i + 15
        '\n        Sets byte if not greater.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.OR(cpu.ZF, cpu.SF != cpu.OF), 1, 0))

    @instruction
    def SETNGE(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Sets if not greater or equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF != cpu.OF, 1, 0))

    @instruction
    def SETNL(cpu, dest):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets byte if not less.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF == cpu.OF, 1, 0))

    @instruction
    def SETNLE(cpu, dest):
        if False:
            while True:
                i = 10
        '\n        Sets byte if not less or equal.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, Operators.AND(cpu.ZF == False, cpu.SF == cpu.OF), 1, 0))

    @instruction
    def SETNO(cpu, dest):
        if False:
            i = 10
            return i + 15
        '\n        Sets byte if not overflow.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.OF == False, 1, 0))

    @instruction
    def SETNP(cpu, dest):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets byte if not parity.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.PF == False, 1, 0))

    @instruction
    def SETNS(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Sets byte if not sign.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF == False, 1, 0))

    @instruction
    def SETNZ(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Sets byte if not zero.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.ZF == False, 1, 0))

    @instruction
    def SETO(cpu, dest):
        if False:
            return 10
        '\n        Sets byte if overflow.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.OF, 1, 0))

    @instruction
    def SETP(cpu, dest):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets byte if parity.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.PF, 1, 0))

    @instruction
    def SETPE(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Sets byte if parity even.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.PF, 1, 0))

    @instruction
    def SETPO(cpu, dest):
        if False:
            i = 10
            return i + 15
        '\n        Sets byte if parity odd.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.PF == False, 1, 0))

    @instruction
    def SETS(cpu, dest):
        if False:
            while True:
                i = 10
        '\n        Sets byte if sign.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.SF, 1, 0))

    @instruction
    def SETZ(cpu, dest):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets byte if zero.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(Operators.ITEBV(dest.size, cpu.ZF, 1, 0))

    @instruction
    def XCHG(cpu, dest, src):
        if False:
            print('Hello World!')
        "\n        Exchanges register/memory with register.\n\n        Exchanges the contents of the destination (first) and source (second)\n        operands. The operands can be two general-purpose registers or a register\n        and a memory location. If a memory operand is referenced, the processor's\n        locking protocol is automatically implemented for the duration of the\n        exchange operation, regardless of the presence or absence of the LOCK\n        prefix or of the value of the IOPL.\n        This instruction is useful for implementing semaphores or similar data\n        structures for process synchronization.\n        The XCHG instruction can also be used instead of the BSWAP instruction\n        for 16-bit operands::\n\n                TEMP  =  DEST\n                DEST  =  SRC\n                SRC  =  TEMP\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        "
        temp = dest.read()
        dest.write(src.read())
        src.write(temp)

    @instruction
    def LEAVE(cpu):
        if False:
            for i in range(10):
                print('nop')
        "\n        High level procedure exit.\n\n        Releases the stack frame set up by an earlier ENTER instruction. The\n        LEAVE instruction copies the frame pointer (in the EBP register) into\n        the stack pointer register (ESP), which releases the stack space allocated\n        to the stack frame. The old frame pointer (the frame pointer for the calling\n        procedure that was saved by the ENTER instruction) is then popped from\n        the stack into the EBP register, restoring the calling procedure's stack\n        frame.\n        A RET instruction is commonly executed following a LEAVE instruction\n        to return program control to the calling procedure::\n\n                IF Stackaddress_bit_size  =  32\n                THEN\n                    ESP  =  EBP;\n                ELSE (* Stackaddress_bit_size  =  16*)\n                    SP  =  BP;\n                FI;\n                IF OperandSize  =  32\n                THEN\n                    EBP  =  Pop();\n                ELSE (* OperandSize  =  16*)\n                    BP  =  Pop();\n                FI;\n\n        :param cpu: current CPU.\n        "
        cpu.STACK = cpu.FRAME
        cpu.FRAME = cpu.pop(cpu.address_bit_size)

    @instruction
    def POP(cpu, dest):
        if False:
            while True:
                i = 10
        '\n        Pops a value from the stack.\n\n        Loads the value from the top of the stack to the location specified\n        with the destination operand and then increments the stack pointer.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        dest.write(cpu.pop(dest.size))

    @instruction
    def PUSH(cpu, src):
        if False:
            i = 10
            return i + 15
        '\n        Pushes a value onto the stack.\n\n        Decrements the stack pointer and then stores the source operand on the top of the stack.\n\n        :param cpu: current CPU.\n        :param src: source operand.\n        '
        size = src.size
        v = src.read()
        if size != 64 and size != cpu.address_bit_size // 2:
            v = Operators.SEXTEND(v, size, cpu.address_bit_size)
            size = cpu.address_bit_size
        cpu.push(v, size)

    @instruction
    def POPF(cpu):
        if False:
            return 10
        '\n        Pops stack into EFLAGS register.\n\n        :param cpu: current CPU.\n        '
        mask = 1 | 4 | 16 | 64 | 128 | 1024 | 2048
        val = cpu.pop(16)
        eflags_size = 32
        cpu.EFLAGS = Operators.ZEXTEND(val & mask, eflags_size)

    @instruction
    def POPFD(cpu):
        if False:
            while True:
                i = 10
        '\n        Pops stack into EFLAGS register.\n\n        :param cpu: current CPU.\n        '
        mask = 1 | 4 | 16 | 64 | 128 | 1024 | 2048
        cpu.EFLAGS = cpu.pop(32) & mask

    @instruction
    def POPFQ(cpu):
        if False:
            print('Hello World!')
        '\n        Pops stack into EFLAGS register.\n\n        :param cpu: current CPU.\n        '
        mask = 1 | 4 | 16 | 64 | 128 | 1024 | 2048
        cpu.EFLAGS = cpu.EFLAGS & ~mask | cpu.pop(64) & mask

    @instruction
    def PUSHF(cpu):
        if False:
            while True:
                i = 10
        '\n        Pushes FLAGS register onto the stack.\n\n        :param cpu: current CPU.\n        '
        cpu.push(cpu.EFLAGS, 16)

    @instruction
    def PUSHFD(cpu):
        if False:
            while True:
                i = 10
        '\n        Pushes EFLAGS register onto the stack.\n\n        :param cpu: current CPU.\n        '
        cpu.push(cpu.EFLAGS, 32)

    @instruction
    def PUSHFQ(cpu):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pushes RFLAGS register onto the stack.\n\n        :param cpu: current CPU.\n        '
        cpu.push(cpu.RFLAGS, 64)

    @instruction
    def INT(cpu, op0):
        if False:
            print('Hello World!')
        '\n        Calls to interrupt procedure.\n\n        The INT n instruction generates a call to the interrupt or exception handler specified\n        with the destination operand. The INT n instruction is the  general mnemonic for executing\n        a software-generated call to an interrupt handler. The INTO instruction is a special\n        mnemonic for calling overflow exception (#OF), interrupt vector number 4. The overflow\n        interrupt checks the OF flag in the EFLAGS register and calls the overflow interrupt handler\n        if the OF flag is set to 1.\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        '
        if op0.read() != 128:
            logger.warning('Unsupported interrupt')
        raise Interruption(op0.read())

    @instruction
    def INT3(cpu):
        if False:
            print('Hello World!')
        '\n        Breakpoint\n\n        :param cpu: current CPU.\n        '
        raise Interruption(3)

    @instruction
    def CALL(cpu, op0):
        if False:
            while True:
                i = 10
        '\n        Procedure call.\n\n        Saves procedure linking information on the stack and branches to the called procedure specified using the target\n        operand. The target operand specifies the address of the first instruction in the called procedure. The operand can\n        be an immediate value, a general-purpose register, or a memory location.\n\n        :param cpu: current CPU.\n        :param op0: target operand.\n        '
        proc = op0.read()
        cpu.push(cpu.PC, cpu.address_bit_size)
        cpu.PC = proc

    @instruction
    def RET(cpu, *operands):
        if False:
            return 10
        '\n        Returns from procedure.\n\n        Transfers program control to a return address located on the top of\n        the stack. The address is usually placed on the stack by a CALL instruction,\n        and the return is made to the instruction that follows the CALL instruction.\n        The optional source operand specifies the number of stack bytes to be\n        released after the return address is popped; the default is none.\n\n        :param cpu: current CPU.\n        :param operands: variable operands list.\n        '
        N = 0
        if len(operands) > 0:
            N = operands[0].read()
        cpu.PC = cpu.pop(cpu.address_bit_size)
        cpu.STACK += N

    @instruction
    def JA(cpu, target):
        if False:
            i = 10
            return i + 15
        '\n        Jumps short if above.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, Operators.AND(cpu.CF == False, cpu.ZF == False), target.read(), cpu.PC)

    @instruction
    def JAE(cpu, target):
        if False:
            while True:
                i = 10
        '\n        Jumps short if above or equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.CF == False, target.read(), cpu.PC)

    @instruction
    def JB(cpu, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jumps short if below.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.CF == True, target.read(), cpu.PC)

    @instruction
    def JBE(cpu, target):
        if False:
            i = 10
            return i + 15
        '\n        Jumps short if below or equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, Operators.OR(cpu.CF, cpu.ZF), target.read(), cpu.PC)

    @instruction
    def JC(cpu, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jumps short if carry.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.CF, target.read(), cpu.PC)

    @instruction
    def JCXZ(cpu, target):
        if False:
            while True:
                i = 10
        '\n        Jumps short if CX register is 0.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.CX == 0, target.read(), cpu.PC)

    @instruction
    def JECXZ(cpu, target):
        if False:
            return 10
        '\n        Jumps short if ECX register is 0.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.ECX == 0, target.read(), cpu.PC)

    @instruction
    def JRCXZ(cpu, target):
        if False:
            while True:
                i = 10
        '\n        Jumps short if RCX register is 0.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.RCX == 0, target.read(), cpu.PC)

    @instruction
    def JE(cpu, target):
        if False:
            return 10
        '\n        Jumps short if equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.ZF, target.read(), cpu.PC)

    @instruction
    def JG(cpu, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jumps short if greater.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, Operators.AND(cpu.ZF == False, cpu.SF == cpu.OF), target.read(), cpu.PC)

    @instruction
    def JGE(cpu, target):
        if False:
            while True:
                i = 10
        '\n        Jumps short if greater or equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.SF == cpu.OF, target.read(), cpu.PC)

    @instruction
    def JL(cpu, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jumps short if less.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.SF != cpu.OF, target.read(), cpu.PC)

    @instruction
    def JLE(cpu, target):
        if False:
            print('Hello World!')
        '\n        Jumps short if less or equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, Operators.OR(cpu.ZF, cpu.SF != cpu.OF), target.read(), cpu.PC)

    @instruction
    def JNA(cpu, target):
        if False:
            return 10
        '\n        Jumps short if not above.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, Operators.OR(cpu.CF, cpu.ZF), target.read(), cpu.PC)

    @instruction
    def JNAE(cpu, target):
        if False:
            print('Hello World!')
        '\n        Jumps short if not above or equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.CF, target.read(), cpu.PC)

    @instruction
    def JNB(cpu, target):
        if False:
            print('Hello World!')
        '\n        Jumps short if not below.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.CF == False, target.read(), cpu.PC)

    @instruction
    def JNBE(cpu, target):
        if False:
            return 10
        '\n        Jumps short if not below or equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, Operators.AND(cpu.CF == False, cpu.ZF == False), target.read(), cpu.PC)

    @instruction
    def JNC(cpu, target):
        if False:
            print('Hello World!')
        '\n        Jumps short if not carry.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, False == cpu.CF, target.read(), cpu.PC)

    @instruction
    def JNE(cpu, target):
        if False:
            i = 10
            return i + 15
        '\n        Jumps short if not equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, False == cpu.ZF, target.read(), cpu.PC)

    @instruction
    def JNG(cpu, target):
        if False:
            print('Hello World!')
        '\n        Jumps short if not greater.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, Operators.OR(cpu.ZF, cpu.SF != cpu.OF), target.read(), cpu.PC)

    @instruction
    def JNGE(cpu, target):
        if False:
            while True:
                i = 10
        '\n        Jumps short if not greater or equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.SF != cpu.OF, target.read(), cpu.PC)

    @instruction
    def JNL(cpu, target):
        if False:
            i = 10
            return i + 15
        '\n        Jumps short if not less.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.SF == cpu.OF, target.read(), cpu.PC)

    @instruction
    def JNLE(cpu, target):
        if False:
            while True:
                i = 10
        '\n        Jumps short if not less or equal.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, Operators.AND(False == cpu.ZF, cpu.SF == cpu.OF), target.read(), cpu.PC)

    @instruction
    def JNO(cpu, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jumps short if not overflow.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, False == cpu.OF, target.read(), cpu.PC)

    @instruction
    def JNP(cpu, target):
        if False:
            while True:
                i = 10
        '\n        Jumps short if not parity.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, False == cpu.PF, target.read(), cpu.PC)

    @instruction
    def JNS(cpu, target):
        if False:
            i = 10
            return i + 15
        '\n        Jumps short if not sign.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, False == cpu.SF, target.read(), cpu.PC)

    def JNZ(cpu, target):
        if False:
            print('Hello World!')
        '\n        Jumps short if not zero.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.JNE(target)

    @instruction
    def JO(cpu, target):
        if False:
            print('Hello World!')
        '\n        Jumps short if overflow.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.OF, target.read(), cpu.PC)

    @instruction
    def JP(cpu, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jumps short if parity.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.PF, target.read(), cpu.PC)

    @instruction
    def JPE(cpu, target):
        if False:
            print('Hello World!')
        '\n        Jumps short if parity even.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.PF, target.read(), cpu.PC)

    @instruction
    def JPO(cpu, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jumps short if parity odd.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, False == cpu.PF, target.read(), cpu.PC)

    @instruction
    def JS(cpu, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jumps short if sign.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.SF, target.read(), cpu.PC)

    @instruction
    def JZ(cpu, target):
        if False:
            i = 10
            return i + 15
        '\n        Jumps short if zero.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, cpu.ZF, target.read(), cpu.PC)

    @instruction
    def JMP(cpu, target):
        if False:
            return 10
        '\n        Jump.\n\n        Transfers program control to a different point in the instruction stream without\n        recording return information. The destination (target) operand specifies the address\n        of the instruction being jumped to. This operand can be an immediate value, a general-purpose register, or a memory location.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        cpu.PC = target.read()

    def LJMP(cpu, cs_selector, target):
        if False:
            return 10
        '\n        We are just going to ignore the CS selector for now.\n        '
        logger.info('LJMP: Jumping to: %r:%r', cs_selector.read(), target.read())
        cpu.CS = cs_selector.read()
        cpu.PC = target.read()

    def LOOP(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Loops according to ECX counter.\n\n        Performs a loop operation using the ECX or CX register as a counter.\n        Each time the LOOP instruction is executed, the count register is decremented,\n        then checked for 0. If the count is 0, the loop is terminated and program\n        execution continues with the instruction following the LOOP instruction.\n        If the count is not zero, a near jump is performed to the destination\n        (target) operand, which is presumably the instruction at the beginning\n        of the loop. If the address-size attribute is 32 bits, the ECX register\n        is used as the count register; otherwise the CX register is used::\n\n                IF address_bit_size  =  32\n                THEN\n                    Count is ECX;\n                ELSE (* address_bit_size  =  16 *)\n                    Count is CX;\n                FI;\n                Count  =  Count - 1;\n\n                IF (Count  0)  =  1\n                THEN\n                    EIP  =  EIP + SignExtend(DEST);\n                    IF OperandSize  =  16\n                    THEN\n                        EIP  =  EIP AND 0000FFFFH;\n                    FI;\n                ELSE\n                    Terminate loop and continue program execution at EIP;\n                FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        counter_name = {16: 'CX', 32: 'ECX', 64: 'RCX'}[cpu.address_bit_size]
        counter = cpu.write_register(counter_name, cpu.read_register(counter_name) - 1)
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, counter == 0, cpu.PC + dest.read() & (1 << dest.size) - 1, cpu.PC + cpu.instruction.size)

    def LOOPNZ(cpu, target):
        if False:
            i = 10
            return i + 15
        '\n        Loops if ECX counter is nonzero.\n\n        :param cpu: current CPU.\n        :param target: destination operand.\n        '
        counter_name = {16: 'CX', 32: 'ECX', 64: 'RCX'}[cpu.address_bit_size]
        counter = cpu.write_register(counter_name, cpu.read_register(counter_name) - 1)
        cpu.PC = Operators.ITEBV(cpu.address_bit_size, counter != 0, cpu.PC + target.read() & (1 << target.size) - 1, cpu.PC + cpu.instruction.size)

    @instruction
    def RCL(cpu, dest, src):
        if False:
            return 10
        '\n        Rotates through carry left.\n\n        Shifts (rotates) the bits of the first operand (destination operand) the number of bit positions specified in the\n        second operand (count operand) and stores the result in the destination operand. The destination operand can be\n        a register or a memory location; the count operand is an unsigned integer that can be an immediate or a value in\n        the CL register. In legacy and compatibility mode, the processor restricts the count to a number between 0 and 31\n        by masking all the bits in the count operand except the 5 least-significant bits.\n\n        The RCL instruction shifts the CF flag into the least-significant bit and shifts the most-significant bit into the CF flag.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = src.read()
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        tempCount = Operators.ZEXTEND((count & countMask) % (src.size + 1), OperandSize)
        value = dest.read()
        if isinstance(tempCount, int) and tempCount == 0:
            new_val = value
            dest.write(new_val)
        else:
            carry = Operators.ITEBV(OperandSize, cpu.CF, 1, 0)
            right = value >> OperandSize - tempCount
            new_val = value << tempCount | carry << tempCount - 1 | right >> 1
            dest.write(new_val)

            def sf(v, size):
                if False:
                    return 10
                return v & 1 << size - 1 != 0
            cpu.CF = sf(value << tempCount - 1, OperandSize)
            cpu.OF = Operators.ITE(tempCount == 1, sf(new_val, OperandSize) != cpu.CF, cpu.OF)

    @instruction
    def RCR(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rotates through carry right (RCR).\n\n        Shifts (rotates) the bits of the first operand (destination operand) the number of bit positions specified in the\n        second operand (count operand) and stores the result in the destination operand. The destination operand can be\n        a register or a memory location; the count operand is an unsigned integer that can be an immediate or a value in\n        the CL register. In legacy and compatibility mode, the processor restricts the count to a number between 0 and 31\n        by masking all the bits in the count operand except the 5 least-significant bits.\n\n        Rotate through carry right (RCR) instructions shift all the bits toward less significant bit positions, except\n        for the least-significant bit, which is rotated to the most-significant bit location. The RCR instruction shifts the\n        CF flag into the most-significant bit and shifts the least-significant bit into the CF flag.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = src.read()
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        tempCount = Operators.ZEXTEND((count & countMask) % (src.size + 1), OperandSize)
        value = dest.read()
        if isinstance(tempCount, int) and tempCount == 0:
            new_val = value
            dest.write(new_val)
        else:
            carry = Operators.ITEBV(OperandSize, cpu.CF, 1, 0)
            left = value >> tempCount - 1
            right = value << OperandSize - tempCount
            new_val = left >> 1 | carry << OperandSize - tempCount | right << 1
            dest.write(new_val)
            cpu.CF = Operators.ITE(tempCount != 0, left & 1 == 1, cpu.CF)
            s_MSB = new_val >> OperandSize - 1 & 1 == 1
            s_MSB2 = new_val >> OperandSize - 2 & 1 == 1
            cpu.OF = Operators.ITE(tempCount == 1, s_MSB ^ s_MSB2, cpu.OF)

    @instruction
    def ROL(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rotates left (ROL).\n\n        Shifts (rotates) the bits of the first operand (destination operand) the number of bit positions specified in the\n        second operand (count operand) and stores the result in the destination operand. The destination operand can be\n        a register or a memory location; the count operand is an unsigned integer that can be an immediate or a value in\n        the CL register. In legacy and compatibility mode, the processor restricts the count to a number between 0 and 31\n        by masking all the bits in the count operand except the 5 least-significant bits.\n\n        The rotate left shift all the bits toward more-significant bit positions, except for the most-significant bit, which\n        is rotated to the least-significant bit location.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = src.read()
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        tempCount = Operators.ZEXTEND((count & countMask) % OperandSize, OperandSize)
        value = dest.read()
        newValue = value << tempCount | value >> OperandSize - tempCount
        dest.write(newValue)
        cpu.CF = Operators.ITE(tempCount != 0, newValue & 1 == 1, cpu.CF)
        s_MSB = newValue >> OperandSize - 1 & 1 == 1
        cpu.OF = Operators.ITE(tempCount == 1, s_MSB ^ cpu.CF, cpu.OF)

    @instruction
    def ROR(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rotates right (ROR).\n\n        Shifts (rotates) the bits of the first operand (destination operand) the number of bit positions specified in the\n        second operand (count operand) and stores the result in the destination operand. The destination operand can be\n        a register or a memory location; the count operand is an unsigned integer that can be an immediate or a value in\n        the CL register. In legacy and compatibility mode, the processor restricts the count to a number between 0 and 31\n        by masking all the bits in the count operand except the 5 least-significant bits.\n\n        The rotate right (ROR) instruction shift all the bits toward less significant bit positions, except\n        for the least-significant bit, which is rotated to the most-significant bit location.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = src.read()
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        tempCount = Operators.ZEXTEND((count & countMask) % OperandSize, OperandSize)
        value = dest.read()
        newValue = value >> tempCount | value << OperandSize - tempCount
        dest.write(newValue)
        cpu.CF = Operators.ITE(tempCount != 0, newValue >> OperandSize - 1 & 1 == 1, cpu.CF)
        s_MSB = newValue >> OperandSize - 1 & 1 == 1
        s_MSB2 = newValue >> OperandSize - 2 & 1 == 1
        cpu.OF = Operators.ITE(tempCount == 1, s_MSB ^ s_MSB2, cpu.OF)

    @instruction
    def SAL(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        The shift arithmetic left.\n\n        Shifts the bits in the first operand (destination operand) to the left or right by the number of bits specified in the\n        second operand (count operand). Bits shifted beyond the destination operand boundary are first shifted into the CF\n        flag, then discarded. At the end of the shift operation, the CF flag contains the last bit shifted out of the destination\n        operand.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = src.read()
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        tempCount = Operators.ZEXTEND(count & countMask, dest.size)
        tempDest = value = dest.read()
        res = dest.write(Operators.ITEBV(dest.size, tempCount == 0, tempDest, value << tempCount))
        MASK = (1 << OperandSize) - 1
        SIGN_MASK = 1 << OperandSize - 1
        cpu.CF = Operators.OR(Operators.AND(tempCount == 0, cpu.CF), Operators.AND(tempCount != 0, tempDest & 1 << OperandSize - tempCount != 0))
        cpu.OF = Operators.ITE(tempCount != 0, cpu.CF ^ (res >> OperandSize - 1 & 1 == 1), cpu.OF)
        cpu.SF = Operators.OR(Operators.AND(tempCount == 0, cpu.SF), Operators.AND(tempCount != 0, res & SIGN_MASK != 0))
        cpu.ZF = Operators.OR(Operators.AND(tempCount == 0, cpu.ZF), Operators.AND(tempCount != 0, res == 0))
        cpu.PF = Operators.OR(Operators.AND(tempCount == 0, cpu.PF), Operators.AND(tempCount != 0, cpu._calculate_parity_flag(res)))

    def SHL(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        The shift logical left.\n\n        The shift arithmetic left (SAL) and shift logical left (SHL) instructions perform the same operation.\n\n        :param cpu: current cpu.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        return cpu.SAL(dest, src)

    @instruction
    def SAR(cpu, dest, src):
        if False:
            print('Hello World!')
        "\n        Shift arithmetic right.\n\n        The shift arithmetic right (SAR) and shift logical right (SHR) instructions shift the bits of the destination operand to\n        the right (toward less significant bit locations). For each shift count, the least significant bit of the destination\n        operand is shifted into the CF flag, and the most significant bit is either set or cleared depending on the instruction\n        type. The SHR instruction clears the most significant bit. the SAR instruction sets or clears the most significant bit\n        to correspond to the sign (most significant bit) of the original value in the destination operand. In effect, the SAR\n        instruction fills the empty bit position's shifted value with the sign of the unshifted value\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        "
        OperandSize = dest.size
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        count = src.read() & countMask
        value = dest.read()
        res = Operators.SAR(OperandSize, value, Operators.ZEXTEND(count, OperandSize))
        dest.write(res)
        SIGN_MASK = 1 << OperandSize - 1
        if issymbolic(count):
            cpu.CF = Operators.ITE(Operators.AND(count != 0, count <= OperandSize), value >> Operators.ZEXTEND(count - 1, OperandSize) & 1 != 0, cpu.CF)
        elif count != 0:
            if count > OperandSize:
                count = OperandSize
            cpu.CF = Operators.EXTRACT(value, count - 1, 1) != 0
        cpu.ZF = Operators.ITE(count != 0, res == 0, cpu.ZF)
        cpu.SF = Operators.ITE(count != 0, res & SIGN_MASK != 0, cpu.SF)
        cpu.OF = Operators.ITE(count == 1, False, cpu.OF)
        cpu.PF = Operators.ITE(count != 0, cpu._calculate_parity_flag(res), cpu.PF)

    @instruction
    def SHR(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Shift logical right.\n\n        The shift arithmetic right (SAR) and shift logical right (SHR)\n        instructions shift the bits of the destination operand to the right\n        (toward less significant bit locations). For each shift count, the\n        least significant bit of the destination operand is shifted into the CF\n        flag, and the most significant bit is either set or cleared depending\n        on the instruction type. The SHR instruction clears the most\n        significant bit.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = Operators.ZEXTEND(src.read() & OperandSize - 1, OperandSize)
        value = dest.read()
        res = dest.write(value >> count)
        MASK = (1 << OperandSize) - 1
        SIGN_MASK = 1 << OperandSize - 1
        if issymbolic(count):
            cpu.CF = Operators.ITE(count != 0, value >> Operators.ZEXTEND(count - 1, OperandSize) & 1 != 0, cpu.CF)
        elif count != 0:
            cpu.CF = Operators.EXTRACT(value, count - 1, 1) != 0
        cpu.ZF = Operators.ITE(count != 0, res == 0, cpu.ZF)
        cpu.SF = Operators.ITE(count != 0, res & SIGN_MASK != 0, cpu.SF)
        cpu.OF = Operators.ITE(count != 0, value >> OperandSize - 1 & 1 == 1, cpu.OF)
        cpu.PF = Operators.ITE(count != 0, cpu._calculate_parity_flag(res), cpu.PF)

    def _set_shiftd_flags(cpu, opsize, original, result, lastbit, count):
        if False:
            print('Hello World!')
        MASK = (1 << opsize) - 1
        SIGN_MASK = 1 << opsize - 1
        cpu.CF = Operators.OR(Operators.AND(cpu.CF, count == 0), Operators.AND(count != 0, lastbit))
        signchange = result & SIGN_MASK != original & SIGN_MASK
        cpu.OF = Operators.ITE(count == 1, signchange, cpu.OF)
        cpu.PF = Operators.ITE(count == 0, cpu.PF, cpu._calculate_parity_flag(result))
        cpu.SF = Operators.ITE(count == 0, cpu.SF, result & SIGN_MASK != 0)
        cpu.ZF = Operators.ITE(count == 0, cpu.ZF, result == 0)

    @instruction
    def SHRD(cpu, dest, src, count):
        if False:
            while True:
                i = 10
        '\n        Double precision shift right.\n\n        Shifts the first operand (destination operand) to the right the number of bits specified by the third operand\n        (count operand). The second operand (source operand) provides bits to shift in from the left (starting with\n        the most significant bit of the destination operand).\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        :param count: count operand\n        '
        OperandSize = dest.size
        MASK = (1 << OperandSize) - 1
        tempCount = Operators.ZEXTEND(count.read(), OperandSize) & OperandSize - 1
        if isinstance(tempCount, int) and tempCount == 0:
            pass
        else:
            arg0 = dest.read()
            arg1 = src.read()
            res = Operators.ITEBV(OperandSize, tempCount == 0, arg0, arg0 >> tempCount | arg1 << dest.size - tempCount)
            res = res & MASK
            dest.write(res)
            lastbit = 0 != arg0 >> tempCount - 1 & 1
            cpu._set_shiftd_flags(OperandSize, arg0, res, lastbit, tempCount)

    @instruction
    def SHLD(cpu, dest, src, count):
        if False:
            return 10
        '\n        Double precision shift right.\n\n        Shifts the first operand (destination operand) to the left the number of bits specified by the third operand\n        (count operand). The second operand (source operand) provides bits to shift in from the right (starting with\n        the least significant bit of the destination operand).\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        :param count: count operand\n        '
        OperandSize = dest.size
        tempCount = Operators.ZEXTEND(count.read(), OperandSize) & OperandSize - 1
        arg0 = dest.read()
        arg1 = src.read()
        MASK = (1 << OperandSize) - 1
        t0 = arg0 << tempCount
        t1 = arg1 >> OperandSize - tempCount
        res = Operators.ITEBV(OperandSize, tempCount == 0, arg0, t0 | t1)
        res = res & MASK
        dest.write(res)
        if isinstance(tempCount, int) and tempCount == 0:
            pass
        else:
            SIGN_MASK = 1 << OperandSize - 1
            lastbit = 0 != arg0 << tempCount - 1 & SIGN_MASK
            cpu._set_shiftd_flags(OperandSize, arg0, res, lastbit, tempCount)

    def _getMemoryBit(cpu, bitbase, bitoffset):
        if False:
            while True:
                i = 10
        'Calculate address and bit offset given a base address and a bit offset\n        relative to that address (in the form of asm operands)'
        assert bitbase.type == 'memory'
        assert bitbase.size >= bitoffset.size
        addr = bitbase.address()
        offt = Operators.SEXTEND(bitoffset.read(), bitoffset.size, bitbase.size)
        offt_is_neg = offt >= 1 << bitbase.size - 1
        offt_in_bytes = offt // 8
        bitpos = offt % 8
        new_addr = addr + Operators.ITEBV(bitbase.size, offt_is_neg, -offt_in_bytes, offt_in_bytes)
        return (new_addr, bitpos)

    @instruction
    def BSF(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Bit scan forward.\n\n        Searches the source operand (second operand) for the least significant\n        set bit (1 bit). If a least significant 1 bit is found, its bit index\n        is stored in the destination operand (first operand). The source operand\n        can be a register or a memory location; the destination operand is a register.\n        The bit index is an unsigned offset from bit 0 of the source operand.\n        If the contents source operand are 0, the contents of the destination\n        operand is undefined::\n\n                    IF SRC  =  0\n                    THEN\n                        ZF  =  1;\n                        DEST is undefined;\n                    ELSE\n                        ZF  =  0;\n                        temp  =  0;\n                        WHILE Bit(SRC, temp)  =  0\n                        DO\n                            temp  =  temp + 1;\n                            DEST  =  temp;\n                        OD;\n                    FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        value = src.read()
        flag = Operators.EXTRACT(value, 0, 1) == 1
        res = 0
        for pos in range(1, src.size):
            res = Operators.ITEBV(dest.size, flag, res, pos)
            flag = Operators.OR(flag, Operators.EXTRACT(value, pos, 1) == 1)
        cpu.ZF = value == 0
        dest.write(Operators.ITEBV(dest.size, cpu.ZF, dest.read(), res))

    @instruction
    def BSR(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Bit scan reverse.\n\n        Searches the source operand (second operand) for the most significant\n        set bit (1 bit). If a most significant 1 bit is found, its bit index is\n        stored in the destination operand (first operand). The source operand\n        can be a register or a memory location; the destination operand is a register.\n        The bit index is an unsigned offset from bit 0 of the source operand.\n        If the contents source operand are 0, the contents of the destination\n        operand is undefined::\n\n                IF SRC  =  0\n                THEN\n                    ZF  =  1;\n                    DEST is undefined;\n                ELSE\n                    ZF  =  0;\n                    temp  =  OperandSize - 1;\n                    WHILE Bit(SRC, temp)  =  0\n                    DO\n                        temp  =  temp - 1;\n                        DEST  =  temp;\n                    OD;\n                FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        value = src.read()
        flag = Operators.EXTRACT(value, src.size - 1, 1) == 1
        res = 0
        for pos in reversed(range(0, src.size)):
            res = Operators.ITEBV(dest.size, flag, res, pos)
            flag = Operators.OR(flag, Operators.EXTRACT(value, pos, 1) == 1)
        cpu.PF = cpu._calculate_parity_flag(res)
        cpu.ZF = value == 0
        dest.write(Operators.ITEBV(dest.size, cpu.ZF, dest.read(), res))

    @instruction
    def BT(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Bit Test.\n\n        Selects the bit in a bit string (specified with the first operand, called the bit base) at the\n        bit-position designated by the bit offset (specified by the second operand) and stores the value\n        of the bit in the CF flag. The bit base operand can be a register or a memory location; the bit\n        offset operand can be a register or an immediate value:\n            - If the bit base operand specifies a register, the instruction takes the modulo 16, 32, or 64\n              of the bit offset operand (modulo size depends on the mode and register size; 64-bit operands\n              are available only in 64-bit mode).\n            - If the bit base operand specifies a memory location, the operand represents the address of the\n              byte in memory that contains the bit base (bit 0 of the specified byte) of the bit string. The\n              range of the bit position that can be referenced by the offset operand depends on the operand size.\n\n        :param cpu: current CPU.\n        :param dest: bit base.\n        :param src: bit offset.\n        '
        if dest.type == 'register':
            cpu.CF = dest.read() >> src.read() % dest.size & 1 != 0
        elif dest.type == 'memory':
            (addr, pos) = cpu._getMemoryBit(dest, src)
            (base, size, ty) = cpu.get_descriptor(cpu.DS)
            value = cpu.read_int(addr + base, 8)
            cpu.CF = Operators.EXTRACT(value, pos, 1) == 1
        else:
            raise NotImplementedError(f'Unknown operand for BT: {dest.type}')

    @instruction
    def BTC(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Bit test and complement.\n\n        Selects the bit in a bit string (specified with the first operand, called\n        the bit base) at the bit-position designated by the bit offset operand\n        (second operand), stores the value of the bit in the CF flag, and complements\n        the selected bit in the bit string.\n\n        :param cpu: current CPU.\n        :param dest: bit base operand.\n        :param src: bit offset operand.\n        '
        if dest.type == 'register':
            value = dest.read()
            pos = src.read() % dest.size
            cpu.CF = value & 1 << pos == 1 << pos
            dest.write(value ^ 1 << pos)
        elif dest.type == 'memory':
            (addr, pos) = cpu._getMemoryBit(dest, src)
            (base, size, ty) = cpu.get_descriptor(cpu.DS)
            addr += base
            value = cpu.read_int(addr, 8)
            cpu.CF = value & 1 << pos == 1 << pos
            value = value ^ 1 << pos
            cpu.write_int(addr, value, 8)
        else:
            raise NotImplementedError(f'Unknown operand for BTC: {dest.type}')

    @instruction
    def BTR(cpu, dest, src):
        if False:
            return 10
        '\n        Bit test and reset.\n\n        Selects the bit in a bit string (specified with the first operand, called\n        the bit base) at the bit-position designated by the bit offset operand\n        (second operand), stores the value of the bit in the CF flag, and clears\n        the selected bit in the bit string to 0.\n\n        :param cpu: current CPU.\n        :param dest: bit base operand.\n        :param src: bit offset operand.\n        '
        if dest.type == 'register':
            value = dest.read()
            pos = src.read() % dest.size
            cpu.CF = value & 1 << pos == 1 << pos
            dest.write(value & ~(1 << pos))
        elif dest.type == 'memory':
            (addr, pos) = cpu._getMemoryBit(dest, src)
            (base, size, ty) = cpu.get_descriptor(cpu.DS)
            addr += base
            value = cpu.read_int(addr, 8)
            cpu.CF = value & 1 << pos == 1 << pos
            value = value & ~(1 << pos)
            cpu.write_int(addr, value, 8)
        else:
            raise NotImplementedError(f'Unknown operand for BTR: {dest.type}')

    @instruction
    def BTS(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Bit test and set.\n\n        Selects the bit in a bit string (specified with the first operand, called\n        the bit base) at the bit-position designated by the bit offset operand\n        (second operand), stores the value of the bit in the CF flag, and sets\n        the selected bit in the bit string to 1.\n\n        :param cpu: current CPU.\n        :param dest: bit base operand.\n        :param src: bit offset operand.\n        '
        if dest.type == 'register':
            value = dest.read()
            pos = src.read() % dest.size
            cpu.CF = value & 1 << pos == 1 << pos
            dest.write(value | 1 << pos)
        elif dest.type == 'memory':
            (addr, pos) = cpu._getMemoryBit(dest, src)
            (base, size, ty) = cpu.get_descriptor(cpu.DS)
            addr += base
            value = cpu.read_int(addr, 8)
            cpu.CF = value & 1 << pos == 1 << pos
            value = value | 1 << pos
            cpu.write_int(addr, value, 8)
        else:
            raise NotImplementedError(f'Unknown operand for BTS: {dest.type}')

    @instruction
    def POPCNT(cpu, dest, src):
        if False:
            return 10
        "\n        This instruction calculates of number of bits set to 1 in the second\n        operand (source) and returns the count in the first operand (a destination\n        register).\n        Count = 0;\n        For (i=0; i < OperandSize; i++) {\n            IF (SRC[ i] = 1) // i'th bit\n                THEN Count++;\n            FI;\n        }\n        DEST = Count;\n        Flags Affected\n        OF, SF, ZF, AF, CF, PF are all cleared.\n        ZF is set if SRC = 0, otherwise ZF is cleared\n        "
        count = 0
        source = src.read()
        for i in range(src.size):
            count += Operators.ITEBV(dest.size, source >> i & 1 == 1, 1, 0)
        dest.write(count)
        cpu.OF = False
        cpu.SF = False
        cpu.AF = False
        cpu.CF = False
        cpu.PF = False
        cpu.ZF = source == 0

    @instruction
    def CLD(cpu):
        if False:
            while True:
                i = 10
        '\n        Clears direction flag.\n        Clears the DF flag in the EFLAGS register. When the DF flag is set to 0, string operations\n        increment the index registers (ESI and/or EDI)::\n\n            DF  =  0;\n\n        :param cpu: current CPU.\n        '
        cpu.DF = False

    @instruction
    def STD(cpu):
        if False:
            print('Hello World!')
        '\n        Sets direction flag.\n\n        Sets the DF flag in the EFLAGS register. When the DF flag is set to 1, string operations decrement\n        the index registers (ESI and/or EDI)::\n\n            DF  =  1;\n\n        :param cpu: current CPU.\n        '
        cpu.DF = True

    @instruction
    def CLC(cpu):
        if False:
            return 10
        '\n        Clears CF\n        :param cpu: current CPU.\n        '
        cpu.CF = False

    @instruction
    def STC(cpu):
        if False:
            print('Hello World!')
        '\n        Sets CF\n        :param cpu: current CPU.\n        '
        cpu.CF = True

    @repe
    def CMPS(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Compares string operands.\n\n        Compares the byte, word, double word or quad specified with the first source\n        operand with the byte, word, double or quad word specified with the second\n        source operand and sets the status flags in the EFLAGS register according\n        to the results. Both the source operands are located in memory::\n\n                temp  = SRC1 - SRC2;\n                SetStatusFlags(temp);\n                IF (byte comparison)\n                THEN IF DF  =  0\n                    THEN\n                        (E)SI  =  (E)SI + 1;\n                        (E)DI  =  (E)DI + 1;\n                    ELSE\n                        (E)SI  =  (E)SI - 1;\n                        (E)DI  =  (E)DI - 1;\n                    FI;\n                ELSE IF (word comparison)\n                    THEN IF DF  =  0\n                        (E)SI  =  (E)SI + 2;\n                        (E)DI  =  (E)DI + 2;\n                    ELSE\n                        (E)SI  =  (E)SI - 2;\n                        (E)DI  =  (E)DI - 2;\n                    FI;\n                ELSE (* doubleword comparison*)\n                    THEN IF DF  =  0\n                        (E)SI  =  (E)SI + 4;\n                        (E)DI  =  (E)DI + 4;\n                    ELSE\n                        (E)SI  =  (E)SI - 4;\n                        (E)DI  =  (E)DI - 4;\n                    FI;\n                FI;\n\n        :param cpu: current CPU.\n        :param dest: first source operand.\n        :param src: second source operand.\n        '
        src_reg = {8: 'SI', 32: 'ESI', 64: 'RSI'}[cpu.address_bit_size]
        dest_reg = {8: 'DI', 32: 'EDI', 64: 'RDI'}[cpu.address_bit_size]
        (base, _, ty) = cpu.get_descriptor(cpu.DS)
        src_addr = cpu.read_register(src_reg) + base
        dest_addr = cpu.read_register(dest_reg) + base
        size = dest.size
        arg1 = cpu.read_int(dest_addr, size)
        arg0 = cpu.read_int(src_addr, size)
        res = arg0 - arg1 & (1 << size) - 1
        cpu._calculate_CMP_flags(size, res, arg0, arg1)
        increment = Operators.ITEBV(cpu.address_bit_size, cpu.DF, -size // 8, size // 8)
        cpu.write_register(src_reg, cpu.read_register(src_reg) + increment)
        cpu.write_register(dest_reg, cpu.read_register(dest_reg) + increment)

    @rep
    def LODS(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Loads string.\n\n        Loads a byte, word, or doubleword from the source operand into the AL, AX, or EAX register, respectively. The\n        source operand is a memory location, the address of which is read from the DS:ESI or the DS:SI registers\n        (depending on the address-size attribute of the instruction, 32 or 16, respectively). The DS segment may be over-\n        ridden with a segment override prefix.\n        After the byte, word, or doubleword is transferred from the memory location into the AL, AX, or EAX register, the\n        (E)SI register is incremented or decremented automatically according to the setting of the DF flag in the EFLAGS\n        register. (If the DF flag is 0, the (E)SI register is incremented; if the DF flag is 1, the ESI register is decremented.)\n        The (E)SI register is incremented or decremented by 1 for byte operations, by 2 for word operations, or by 4 for\n        doubleword operations.\n\n        :param cpu: current CPU.\n        :param dest: source operand.\n        '
        src_reg = {8: 'SI', 32: 'ESI', 64: 'RSI'}[cpu.address_bit_size]
        (base, _, ty) = cpu.get_descriptor(cpu.DS)
        src_addr = cpu.read_register(src_reg) + base
        size = dest.size
        arg0 = cpu.read_int(src_addr, size)
        dest.write(arg0)
        increment = Operators.ITEBV(cpu.address_bit_size, cpu.DF, -size // 8, size // 8)
        cpu.write_register(src_reg, cpu.read_register(src_reg) + increment)

    @rep
    def MOVS(cpu, dest, src):
        if False:
            return 10
        '\n        Moves data from string to string.\n\n        Moves the byte, word, or doubleword specified with the second operand (source operand) to the location specified\n        with the first operand (destination operand). Both the source and destination operands are located in memory. The\n        address of the source operand is read from the DS:ESI or the DS:SI registers (depending on the address-size\n        attribute of the instruction, 32 or 16, respectively). The address of the destination operand is read from the ES:EDI\n        or the ES:DI registers (again depending on the address-size attribute of the instruction). The DS segment may be\n        overridden with a segment override prefix, but the ES segment cannot be overridden.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        (base, size, ty) = cpu.get_descriptor(cpu.DS)
        src_addr = src.address() + base
        dest_addr = dest.address() + base
        src_reg = src.mem.base
        dest_reg = dest.mem.base
        size = dest.size
        dest.write(src.read())
        increment = Operators.ITEBV(cpu.address_bit_size, cpu.DF, -size // 8, size // 8)
        cpu.write_register(src_reg, cpu.read_register(src_reg) + increment)
        cpu.write_register(dest_reg, cpu.read_register(dest_reg) + increment)

    @repe
    def SCAS(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Scans String.\n\n        Compares the byte, word, or double word specified with the memory operand\n        with the value in the AL, AX, EAX, or RAX register, and sets the status flags\n        according to the results. The memory operand address is read from either\n        the ES:RDI, ES:EDI or the ES:DI registers (depending on the address-size\n        attribute of the instruction, 32 or 16, respectively)::\n\n                IF (byte comparison)\n                THEN\n                    temp  =  AL - SRC;\n                    SetStatusFlags(temp);\n                    THEN IF DF  =  0\n                        THEN (E)DI  =  (E)DI + 1;\n                        ELSE (E)DI  =  (E)DI - 1;\n                        FI;\n                    ELSE IF (word comparison)\n                        THEN\n                            temp  =  AX - SRC;\n                            SetStatusFlags(temp)\n                            THEN IF DF  =  0\n                                THEN (E)DI  =  (E)DI + 2;\n                                ELSE (E)DI  =  (E)DI - 2;\n                                FI;\n                     ELSE (* doubleword comparison *)\n                           temp  =  EAX - SRC;\n                           SetStatusFlags(temp)\n                           THEN IF DF  =  0\n                                THEN\n                                    (E)DI  =  (E)DI + 4;\n                                ELSE\n                                    (E)DI  =  (E)DI - 4;\n                                FI;\n                           FI;\n                     FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest_reg = dest.reg
        mem_reg = src.mem.base
        size = dest.size
        arg0 = dest.read()
        arg1 = src.read()
        res = arg0 - arg1
        cpu._calculate_CMP_flags(size, res, arg0, arg1)
        increment = Operators.ITEBV(cpu.address_bit_size, cpu.DF, -size // 8, size // 8)
        cpu.write_register(mem_reg, cpu.read_register(mem_reg) + increment)

    @rep
    def STOS(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Stores String.\n\n        Stores a byte, word, or doubleword from the AL, AX, or EAX register,\n        respectively, into the destination operand. The destination operand is\n        a memory location, the address of which is read from either the ES:EDI\n        or the ES:DI registers (depending on the address-size attribute of the\n        instruction, 32 or 16, respectively). The ES segment cannot be overridden\n        with a segment override prefix.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        size = src.size
        dest.write(src.read())
        dest_reg = dest.mem.base
        increment = Operators.ITEBV({'RDI': 64, 'EDI': 32, 'DI': 16}[dest_reg], cpu.DF, -size // 8, size // 8)
        cpu.write_register(dest_reg, cpu.read_register(dest_reg) + increment)

    @instruction
    def EMMS(cpu):
        if False:
            print('Hello World!')
        '\n        Empty MMX Technology State\n\n        Sets the values of all the tags in the x87 FPU tag word to empty (all\n        1s). This operation marks the x87 FPU data registers (which are aliased\n        to the MMX technology registers) as available for use by x87 FPU\n        floating-point instructions.\n\n            x87FPUTagWord <- FFFFH;\n        '
        cpu.FPTAG = 65535

    @instruction
    def STMXCSR(cpu, dest):
        if False:
            i = 10
            return i + 15
        'Store MXCSR Register State\n        Stores the contents of the MXCSR control and status register to the destination operand.\n        The destination operand is a 32-bit memory location. The reserved bits in the MXCSR register\n        are stored as 0s.'
        dest.write(8064)

    @instruction
    def PAUSE(cpu):
        if False:
            i = 10
            return i + 15
        pass

    @instruction
    def ANDN(cpu, dest, src1, src2):
        if False:
            for i in range(10):
                print('nop')
        'Performs a bitwise logical AND of inverted second operand (the first source operand)\n        with the third operand (the second source operand). The result is stored in the first\n        operand (destination operand).\n\n             DEST <- (NOT SRC1) bitwiseAND SRC2;\n             SF <- DEST[OperandSize -1];\n             ZF <- (DEST = 0);\n        Flags Affected\n             SF and ZF are updated based on result. OF and CF flags are cleared. AF and PF flags are undefined.\n        '
        value = ~src1.read() & src2.read()
        dest.write(value)
        cpu.ZF = value == 0
        cpu.SF = value & 1 << dest.size != 0
        cpu.OF = False
        cpu.CF = False

    @instruction
    def SHLX(cpu, dest, src, count):
        if False:
            i = 10
            return i + 15
        '\n        The shift arithmetic left.\n\n        Shifts the bits in the first operand (destination operand) to the left or right by the number of bits specified in the\n        second operand (count operand). Bits shifted beyond the destination operand boundary are first shifted into the CF\n        flag, then discarded. At the end of the shift operation, the CF flag contains the last bit shifted out of the destination\n        operand.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = count.read()
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        tempCount = Operators.ZEXTEND(count & countMask, dest.size)
        tempDest = value = src.read()
        res = dest.write(Operators.ITEBV(dest.size, tempCount == 0, tempDest, value << tempCount))

    @instruction
    def SHRX(cpu, dest, src, count):
        if False:
            while True:
                i = 10
        '\n        The shift arithmetic right.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = count.read()
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        tempCount = Operators.ZEXTEND(count & countMask, dest.size)
        tempDest = value = src.read()
        res = dest.write(Operators.ITEBV(dest.size, tempCount == 0, tempDest, value >> tempCount))

    @instruction
    def SARX(cpu, dest, src, count):
        if False:
            print('Hello World!')
        '\n        The shift arithmetic right.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        OperandSize = dest.size
        count = count.read()
        countMask = {8: 31, 16: 31, 32: 31, 64: 63}[OperandSize]
        tempCount = count & countMask
        tempDest = value = src.read()
        sign = value & 1 << OperandSize - 1
        while tempCount != 0:
            cpu.CF = value & 1 != 0
            value = value >> 1 | sign
            tempCount = tempCount - 1
        res = dest.write(value)

    @instruction
    def PMINUB(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        PMINUB: returns minimum of packed unsigned byte integers in the dest operand\n        see PMAXUB\n        '
        dest_value = dest.read()
        src_value = src.read()
        result = 0
        for pos in range(0, dest.size, 8):
            itema = dest_value >> pos & 255
            itemb = src_value >> pos & 255
            result |= Operators.ITEBV(dest.size, itema < itemb, itema, itemb) << pos
        dest.write(result)

    @instruction
    def PMAXUB(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        PMAXUB: returns maximum of packed unsigned byte integers in the dest operand\n\n        Performs a SIMD compare of the packed unsigned byte in the second source operand\n        and the first source operand and returns the maximum value for each pair of\n        integers to the destination operand.\n\n        Example :\n        $xmm1.v16_int8 = {..., 0xf2, 0xd1}\n        $xmm2.v16_int8 = {..., 0xd2, 0xf1}\n        # after pmaxub xmm1, xmm2, we get\n        $xmm1.v16_int8 = {..., 0xf2, 0xf1}\n        '
        dest_value = dest.read()
        src_value = src.read()
        result = 0
        for pos in range(0, dest.size, 8):
            itema = dest_value >> pos & 255
            itemb = src_value >> pos & 255
            result |= Operators.ITEBV(dest.size, itema > itemb, itema, itemb) << pos
        dest.write(result)

    @instruction
    def VPXOR(cpu, dest, arg0, arg1):
        if False:
            return 10
        res = dest.write(arg0.read() ^ arg1.read())

    @instruction
    def PXOR(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Logical exclusive OR.\n\n        Performs a bitwise logical exclusive-OR (XOR) operation on the quadword\n        source (second) and destination (first) operands and stores the result\n        in the destination operand location. The source operand can be an MMX(TM)\n        technology register or a quadword memory location; the destination operand\n        must be an MMX register. Each bit of the result is 1 if the corresponding\n        bits of the two operands are different; each bit is 0 if the corresponding\n        bits of the operands are the same::\n\n            DEST  =  DEST XOR SRC;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: quadword source operand.\n        '
        res = dest.write(dest.read() ^ src.read())

    def _PUNPCKL(cpu, dest, src, item_size):
        if False:
            while True:
                i = 10
        '\n        Generic PUNPCKL\n        '
        assert dest.size == src.size
        size = dest.size
        dest_value = dest.read()
        src_value = src.read()
        mask = (1 << item_size) - 1
        res = 0
        count = 0
        for pos in range(0, size // item_size):
            if count >= size:
                break
            item0 = Operators.ZEXTEND(dest_value >> pos * item_size & mask, size)
            item1 = Operators.ZEXTEND(src_value >> pos * item_size & mask, size)
            res |= item0 << count
            count += item_size
            res |= item1 << count
            count += item_size
        dest.write(res)

    def _PUNPCKH(cpu, dest, src, item_size):
        if False:
            i = 10
            return i + 15
        '\n        Generic PUNPCKH\n        '
        assert dest.size == src.size
        size = dest.size
        dest_value = dest.read()
        src_value = src.read()
        mask = (1 << item_size) - 1
        res = 0
        count = 0
        for pos in reversed(range(0, size // item_size)):
            if count >= size:
                break
            item0 = Operators.ZEXTEND(dest_value >> pos * item_size & mask, size)
            item1 = Operators.ZEXTEND(src_value >> pos * item_size & mask, size)
            res = res << item_size
            res |= item1
            res = res << item_size
            res |= item0
            count += item_size * 2
        dest.write(res)

    @instruction
    def PUNPCKHBW(cpu, dest, src):
        if False:
            while True:
                i = 10
        cpu._PUNPCKH(dest, src, 8)

    @instruction
    def PUNPCKHWD(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        cpu._PUNPCKH(dest, src, 16)

    @instruction
    def PUNPCKHDQ(cpu, dest, src):
        if False:
            return 10
        cpu._PUNPCKH(dest, src, 32)

    @instruction
    def PUNPCKHQDQ(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        cpu._PUNPCKH(dest, src, 64)

    @instruction
    def PUNPCKLBW(cpu, dest, src):
        if False:
            return 10
        '\n        Interleaves the low-order bytes of the source and destination operands.\n\n        Unpacks and interleaves the low-order data elements (bytes, words, doublewords, and quadwords)\n        of the destination operand (first operand) and source operand (second operand) into the\n        destination operand.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        cpu._PUNPCKL(dest, src, 8)

    @instruction
    def PUNPCKLWD(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Interleaves the low-order bytes of the source and destination operands.\n\n        Unpacks and interleaves the low-order data elements (bytes, words, doublewords, and quadwords)\n        of the destination operand (first operand) and source operand (second operand) into the\n        destination operand.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        cpu._PUNPCKL(dest, src, 16)

    @instruction
    def PUNPCKLQDQ(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Interleaves the low-order quad-words of the source and destination operands.\n\n        Unpacks and interleaves the low-order data elements (bytes, words, doublewords, and quadwords)\n        of the destination operand (first operand) and source operand (second operand) into the\n        destination operand.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        cpu._PUNPCKL(dest, src, 64)

    @instruction
    def PUNPCKLDQ(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Interleaves the low-order double-words of the source and destination operands.\n\n        Unpacks and interleaves the low-order data elements (bytes, words, doublewords, and quadwords)\n        of the destination operand (first operand) and source operand (second operand) into the\n        destination operand.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        cpu._PUNPCKL(dest, src, 32)

    @instruction
    def PSHUFW(cpu, op0, op1, op3):
        if False:
            while True:
                i = 10
        '\n        Packed shuffle words.\n\n        Copies doublewords from source operand (second operand) and inserts them in the destination operand\n        (first operand) at locations selected with the order operand (third operand).\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        :param op3: order operand.\n        '
        size = op0.size
        arg0 = op0.read()
        arg1 = op1.read()
        arg3 = Operators.ZEXTEND(op3.read(), size)
        assert size == 64
        arg0 |= arg1 >> (arg3 >> 0 & 3 * 16) & 65535
        arg0 |= (arg1 >> (arg3 >> 2 & 3 * 16) & 65535) << 16
        arg0 |= (arg1 >> (arg3 >> 4 & 3 * 16) & 65535) << 32
        arg0 |= (arg1 >> (arg3 >> 6 & 3 * 16) & 65535) << 48
        op0.write(arg0)

    @instruction
    def PSHUFLW(cpu, op0, op1, op3):
        if False:
            while True:
                i = 10
        '\n        Shuffle Packed Low Words\n\n        Copies words from the low quadword of the source operand (second operand)\n        and inserts them in the low quadword of the destination operand (first operand)\n        at word locations selected with the order operand (third operand).\n\n        This operation is similar to the operation used by the PSHUFD instruction.\n\n            Operation\n            Destination[0..15] = (Source >> (Order[0..1] * 16))[0..15];\n            Destination[16..31] = (Source >> (Order[2..3] * 16))[0..15];\n            Destination[32..47] = (Source >> (Order[4..5] * 16))[0..15];\n            Destination[48..63] = (Source >> (Order[6..7] * 16))[0..15];\n            Destination[64..127] = Source[64..127];\n        '
        size = op0.size
        arg0 = op0.read()
        arg1 = op1.read()
        arg3 = Operators.ZEXTEND(op3.read(), size)
        arg0 = arg1 & 340282366920938463444927863358058659840
        arg0 |= arg1 >> (arg3 >> 0 & 3) * 16 & 65535
        arg0 |= (arg1 >> (arg3 >> 2 & 3) * 16 & 65535) << 16
        arg0 |= (arg1 >> (arg3 >> 4 & 3) * 16 & 65535) << 32
        arg0 |= (arg1 >> (arg3 >> 6 & 3) * 16 & 65535) << 48
        op0.write(arg0)

    @instruction
    def PSHUFD(cpu, op0, op1, op3):
        if False:
            return 10
        '\n        Packed shuffle doublewords.\n\n        Copies doublewords from source operand (second operand) and inserts them in the destination operand\n        (first operand) at locations selected with the order operand (third operand).\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        :param op3: order operand.\n        '
        size = op0.size
        arg0 = op0.read()
        arg1 = op1.read()
        order = Operators.ZEXTEND(op3.read(), size)
        arg0 = arg0 & 115792089237316195423570985008687907852929702298719625575994209400481361428480
        arg0 |= arg1 >> (order >> 0 & 3) * 32 & 4294967295
        arg0 |= (arg1 >> (order >> 2 & 3) * 32 & 4294967295) << 32
        arg0 |= (arg1 >> (order >> 4 & 3) * 32 & 4294967295) << 64
        arg0 |= (arg1 >> (order >> 6 & 3) * 32 & 4294967295) << 96
        op0.write(arg0)

    @instruction
    def MOVDQU(cpu, op0, op1):
        if False:
            i = 10
            return i + 15
        '\n        Moves unaligned double quadword.\n\n        Moves a double quadword from the source operand (second operand) to the destination operand\n        (first operand)::\n\n            OP0  =  OP1;\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        '
        op0.write(op1.read())

    @instruction
    def MOVDQA(cpu, op0, op1):
        if False:
            return 10
        '\n        Moves aligned double quadword.\n\n        Moves a double quadword from the source operand (second operand) to the destination operand\n        (first operand)::\n            OP0  =  OP1;\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        @todo: check alignment.\n        '
        op0.write(op1.read())

    @instruction
    def PCMPEQB(cpu, op0, op1):
        if False:
            i = 10
            return i + 15
        '\n        Packed compare for equal.\n\n        Performs a SIMD compare for equality of the packed bytes, words, or doublewords in the\n        destination operand (first operand) and the source operand (second operand). If a pair of\n        data elements are equal, the corresponding data element in the destination operand is set\n        to all 1s; otherwise, it is set to all 0s. The source operand can be an MMX(TM) technology\n        register or a 64-bit memory location, or it can be an XMM register or a 128-bit memory location.\n        The destination operand can be an MMX or an XMM register.\n        The PCMPEQB instruction compares the bytes in the destination operand to the corresponding bytes\n        in the source operand.\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        '
        arg0 = op0.read()
        arg1 = op1.read()
        res = 0
        for i in range(0, op0.size, 8):
            res = Operators.ITEBV(op0.size, Operators.EXTRACT(arg0, i, 8) == Operators.EXTRACT(arg1, i, 8), res | 255 << i, res)
        op0.write(res)

    @instruction
    def PCMPEQD(cpu, op0, op1):
        if False:
            i = 10
            return i + 15
        '\n        PCMPEQD: Packed compare for equal with double words\n        see PCMPEQB\n        '
        arg0 = op0.read()
        arg1 = op1.read()
        res = 0
        for i in range(0, op0.size, 32):
            res = Operators.ITEBV(op0.size, Operators.EXTRACT(arg0, i, 32) == Operators.EXTRACT(arg1, i, 32), res | 4294967295 << i, res)
        op0.write(res)

    @instruction
    def PCMPGTD(cpu, op0, op1):
        if False:
            while True:
                i = 10
        '\n        PCMPGTD: Packed compare for greater than with double words\n        see PCMPEQB\n        '
        arg0 = op0.read()
        arg1 = op1.read()
        res = 0
        for i in range(0, op0.size, 32):
            res = Operators.ITEBV(op0.size, Operators.EXTRACT(arg0, i, 32) > Operators.EXTRACT(arg1, i, 32), res | 4294967295 << i, res)
        op0.write(res)

    @instruction
    def PADDD(cpu, op0, op1):
        if False:
            i = 10
            return i + 15
        '\n        PADDD: Packed add with double words\n\n        Performs a SIMD add of the packed integers from the source operand (second operand)\n        and the destination operand (first operand), and stores the packed integer results\n        in the destination operand\n\n        Example :\n        $xmm1.v16_int8 = {..., 0x00000003, 0x00000001}\n        $xmm2.v16_int8 = {..., 0x00000004, 0x00000002}\n        # after paddd xmm1, xmm2, we get\n        $xmm1.v16_int8 = {..., 0x00000007, 0x00000003}\n        '
        arg0 = op0.read()
        arg1 = op1.read()
        res = 0
        for i in range(0, op0.size, 32):
            res |= (Operators.EXTRACT(arg0, i, 32) + Operators.EXTRACT(arg1, i, 32) & 4294967295) << i
        op0.write(res)

    @instruction
    def PADDQ(cpu, op0, op1):
        if False:
            for i in range(10):
                print('nop')
        '\n        PADDQ: Packed add with quadruple words\n        see PADDD\n        '
        arg0 = op0.read()
        arg1 = op1.read()
        res = 0
        for i in range(0, op0.size, 64):
            res |= (Operators.EXTRACT(arg0, i, 64) + Operators.EXTRACT(arg1, i, 64) & 18446744073709551615) << i
        op0.write(res)

    @instruction
    def PSLLD(cpu, op0, op1):
        if False:
            i = 10
            return i + 15
        '\n        PSLLD: Packed shift left logical with double words\n\n        Shifts the destination operand (first operand) to the left by the number of bytes specified\n        in the count operand (second operand). The empty low-order bytes are cleared (set to all 0s).\n        If the value specified by the count operand is greater than 15, the destination operand is\n        set to all 0s. The count operand is an 8-bit immediate.\n\n        Example :\n        $xmm1.v16_int8 = {..., 0x00000003, 0x00000001}\n        # after pslld xmm1, 2, we get\n        $xmm1.v16_int8 = {..., 0x0000000c, 0x00000004}\n        '
        arg0 = op0.read()
        arg1 = op1.read()
        res = 0
        for i in range(0, op0.size, 32):
            res |= (Operators.EXTRACT(arg0, i, 32) << arg1 & 4294967295) << i
        op0.write(res)

    @instruction
    def PSLLQ(cpu, op0, op1):
        if False:
            print('Hello World!')
        '\n        PSLLQ: Packed shift left logical with quadruple words\n        see PSLLD\n        '
        arg0 = op0.read()
        arg1 = op1.read()
        res = 0
        for i in range(0, op0.size, 64):
            res |= (Operators.EXTRACT(arg0, i, 64) << arg1 & 18446744073709551615) << i
        op0.write(res)

    def _pcmpxstrx_srcdat_format(self, ctlbyte):
        if False:
            i = 10
            return i + 15
        if Operators.EXTRACT(ctlbyte, 0, 2) & 1 == 0:
            stepsize = 8
        else:
            stepsize = 16
        return stepsize

    def _pcmpxstri_output_selection(self, ctlbyte, res):
        if False:
            i = 10
            return i + 15
        stepsize = self._pcmpxstrx_srcdat_format(ctlbyte)
        if Operators.EXTRACT(ctlbyte, 6, 1) == 0:
            oecx = 0
            tres = res
            while tres & 1 == 0:
                oecx += 1
                tres >>= 1
            return oecx
        else:
            oecx = 128 // stepsize - 1
            tres = res
            msbmask = 1 << 128 // stepsize - 1
            while tres & msbmask == 0:
                oecx -= 1
                tres = tres << 1 & (msbmask << 1) - 1
            return oecx

    def _pcmpxstrm_output_selection(self, ctlbyte, res):
        if False:
            while True:
                i = 10
        if Operators.EXTRACT(ctlbyte, 6, 1) == 0:
            return res
        else:
            stepsize = self._pcmpxstrx_srcdat_format(ctlbyte)
            xmmres = 0
            for i in range(0, 128, stepsize):
                if res & 1 == 1:
                    xmmres |= (1 << stepsize) - 1 << i
                res >>= 1
            return xmmres

    def _pcmpistrx_varg(self, arg, ctlbyte):
        if False:
            i = 10
            return i + 15
        step = self._pcmpxstrx_srcdat_format(ctlbyte)
        result = []
        for i in range(0, 128, step):
            uc = Operators.EXTRACT(arg, i, step)
            if uc == 0:
                break
            result.append(uc)
        return result

    def _pcmpestrx_varg(self, arg, regname, ctlbyte):
        if False:
            for i in range(10):
                print('nop')
        reg = self.read_register(regname)
        if issymbolic(reg):
            raise ConcretizeRegister(self, regname, 'Concretize PCMPESTRx ECX/EDX')
        smask = 1 << self.regfile.sizeof(regname) - 1
        step = self._pcmpxstrx_srcdat_format(ctlbyte)
        if reg & smask == 1:
            val = Operators.NOT(reg - 1)
        else:
            val = reg
        if val > 128 // step:
            val = 128 // step
        result = []
        for i in range(val):
            uc = Operators.EXTRACT(arg, i * step, step)
            result.append(uc)
        return result

    def _pcmpxstrx_aggregation_operation(self, varg0, varg1, ctlbyte):
        if False:
            return 10
        needle = [e for e in varg0]
        haystack = [e for e in varg1]
        res = 0
        stepsize = self._pcmpxstrx_srcdat_format(ctlbyte)
        xmmsize = 128
        if Operators.EXTRACT(ctlbyte, 2, 2) == 0:
            for i in range(len(haystack)):
                if haystack[i] in needle:
                    res |= 1 << i
        elif Operators.EXTRACT(ctlbyte, 2, 2) == 1:
            assert len(needle) % 2 == 0
            for i in range(len(haystack)):
                for j in range(0, len(needle), 2):
                    if haystack[i] >= needle[j] and haystack[i] <= needle[j + 1]:
                        res |= 1 << i
                        break
        elif Operators.EXTRACT(ctlbyte, 2, 2) == 2:
            while len(needle) < xmmsize // stepsize:
                needle.append('\x00')
            while len(haystack) < xmmsize // stepsize:
                haystack.append('\x00')
            for i in range(xmmsize // stepsize):
                res = Operators.ITEBV(xmmsize, needle[i] == haystack[i], res | 1 << i, res)
        elif Operators.EXTRACT(ctlbyte, 2, 2) == 3:
            if len(haystack) < len(needle):
                return 0
            for i in range(len(haystack)):
                subneedle = needle[:xmmsize // stepsize - i if len(needle) + i > xmmsize // stepsize else len(needle)]
                res = Operators.ITEBV(xmmsize, haystack[i:i + len(subneedle)] == subneedle, res | 1 << i, res)
        return res

    def _pcmpxstrx_polarity(self, res1, ctlbyte, arg2len):
        if False:
            for i in range(10):
                print('nop')
        stepsize = self._pcmpxstrx_srcdat_format(ctlbyte)
        if Operators.EXTRACT(ctlbyte, 4, 2) == 0:
            res2 = res1
        if Operators.EXTRACT(ctlbyte, 4, 2) == 1:
            res2 = (1 << 128 // stepsize) - 1 ^ res1
        if Operators.EXTRACT(ctlbyte, 4, 2) == 2:
            res2 = res1
        if Operators.EXTRACT(ctlbyte, 4, 2) == 3:
            res2 = (1 << arg2len) - 1 ^ res1
        return res2

    def _pcmpxstrx_setflags(self, res, varg0, varg1, ctlbyte):
        if False:
            while True:
                i = 10
        stepsize = self._pcmpxstrx_srcdat_format(ctlbyte)
        self.ZF = len(varg1) < 128 // stepsize
        self.SF = len(varg0) < 128 // stepsize
        self.CF = res != 0
        self.OF = res & 1
        self.AF = False
        self.PF = False

    def _pcmpxstrx_operands(self, op0, op1, op2):
        if False:
            return 10
        arg0 = op0.read()
        arg1 = op1.read()
        ctlbyte = op2.read()
        if issymbolic(arg0):
            assert op0.type == 'register'
            raise ConcretizeRegister(self, op0.reg, 'Concretize for PCMPXSTRX')
        if issymbolic(arg1):
            if op1.type == 'register':
                raise ConcretizeRegister(self, op1.reg, 'Concretize for PCMPXSTRX')
            else:
                raise ConcretizeMemory(self.memory, op1.address(), op0.size)
        assert not issymbolic(ctlbyte)
        return (arg0, arg1, ctlbyte)

    @instruction
    def PCMPISTRI(cpu, op0, op1, op2):
        if False:
            while True:
                i = 10
        (arg0, arg1, ctlbyte) = cpu._pcmpxstrx_operands(op0, op1, op2)
        varg0 = cpu._pcmpistrx_varg(arg0, ctlbyte)
        varg1 = cpu._pcmpistrx_varg(arg1, ctlbyte)
        res = cpu._pcmpxstrx_aggregation_operation(varg0, varg1, ctlbyte)
        res = cpu._pcmpxstrx_polarity(res, ctlbyte, len(varg1))
        if res == 0:
            cpu.ECX = 128 // cpu._pcmpxstrx_srcdat_format(ctlbyte)
        else:
            cpu.ECX = cpu._pcmpxstri_output_selection(ctlbyte, res)
        cpu._pcmpxstrx_setflags(res, varg0, varg1, ctlbyte)

    @instruction
    def PCMPISTRM(cpu, op0, op1, op2):
        if False:
            for i in range(10):
                print('nop')
        (arg0, arg1, ctlbyte) = cpu._pcmpxstrx_operands(op0, op1, op2)
        varg0 = cpu._pcmpistrx_varg(arg0, ctlbyte)
        varg1 = cpu._pcmpistrx_varg(arg1, ctlbyte)
        res = cpu._pcmpxstrx_aggregation_operation(varg0, varg1, ctlbyte)
        res = cpu._pcmpxstrx_polarity(res, ctlbyte, len(varg1))
        cpu.XMM0 = cpu._pcmpxstrm_output_selection(ctlbyte, res)
        cpu._pcmpxstrx_setflags(res, varg0, varg1, ctlbyte)

    @instruction
    def PCMPESTRI(cpu, op0, op1, op2):
        if False:
            print('Hello World!')
        (arg0, arg1, ctlbyte) = cpu._pcmpxstrx_operands(op0, op1, op2)
        varg0 = cpu._pcmpestrx_varg(arg0, 'EAX', ctlbyte)
        varg1 = cpu._pcmpestrx_varg(arg1, 'EDX', ctlbyte)
        res = cpu._pcmpxstrx_aggregation_operation(varg0, varg1, ctlbyte)
        res = cpu._pcmpxstrx_polarity(res, ctlbyte, len(varg1))
        if res == 0:
            cpu.ECX = 128 // cpu._pcmpxstrx_srcdat_format(ctlbyte)
        else:
            cpu.ECX = cpu._pcmpxstri_output_selection(ctlbyte, res)
        cpu._pcmpxstrx_setflags(res, varg0, varg1, ctlbyte)

    @instruction
    def PCMPESTRM(cpu, op0, op1, op2):
        if False:
            while True:
                i = 10
        (arg0, arg1, ctlbyte) = cpu._pcmpxstrx_operands(op0, op1, op2)
        varg0 = cpu._pcmpestrx_varg(arg0, 'EAX', ctlbyte)
        varg1 = cpu._pcmpestrx_varg(arg1, 'EDX', ctlbyte)
        res = cpu._pcmpxstrx_aggregation_operation(varg0, varg1, ctlbyte)
        res = cpu._pcmpxstrx_polarity(res, ctlbyte, len(varg1))
        cpu.XMM0 = cpu._pcmpxstrm_output_selection(ctlbyte, res)
        cpu._pcmpxstrx_setflags(res, varg0, varg1, ctlbyte)

    @instruction
    def PMOVMSKB(cpu, op0, op1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Moves byte mask to general-purpose register.\n\n        Creates an 8-bit mask made up of the most significant bit of each byte of the source operand\n        (second operand) and stores the result in the low byte or word of the destination operand\n        (first operand). The source operand is an MMX(TM) technology or an XXM register; the destination\n        operand is a general-purpose register.\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        '
        arg0 = op0.read()
        arg1 = op1.read()
        res = 0
        for i in reversed(range(7, op1.size, 8)):
            res = res << 1 | arg1 >> i & 1
        op0.write(Operators.EXTRACT(res, 0, op0.size))

    @instruction
    def PSRLDQ(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Packed shift right logical double quadword.\n\n        Shifts the destination operand (first operand) to the right by the number\n        of bytes specified in the count operand (second operand). The empty high-order\n        bytes are cleared (set to all 0s). If the value specified by the count\n        operand is greater than 15, the destination operand is set to all 0s.\n        The destination operand is an XMM register. The count operand is an 8-bit\n        immediate::\n\n            TEMP  =  SRC;\n            if (TEMP > 15) TEMP  =  16;\n            DEST  =  DEST >> (temp * 8);\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: count operand.\n        '
        temp = Operators.EXTRACT(src.read(), 0, 8)
        temp = Operators.ITEBV(src.size, temp > 15, 16, temp)
        dest.write(dest.read() >> temp * 8)

    @instruction
    def NOP(cpu, arg0=None):
        if False:
            return 10
        '\n        No Operation.\n\n        Performs no operation. This instruction is a one-byte instruction that  takes up space in the\n        instruction stream but does not affect the machine.\n        The NOP instruction is an alias mnemonic for the XCHG (E)AX, (E)AX instruction.\n\n        :param cpu: current CPU.\n        :param arg0: this argument is ignored.\n        '
        pass

    @instruction
    def ENDBR32(cpu):
        if False:
            while True:
                i = 10
        '\n        The ENDBRANCH is a new instruction that is used to mark valid jump target\n        addresses of indirect calls and jumps in the program. This instruction\n        opcode is selected to be one that is a NOP on legacy machines such that\n        programs compiled with ENDBRANCH new instruction continue to function on\n        old machines without the CET enforcement. On processors that support CET\n        the ENDBRANCH is still a NOP and is primarily used as a marker instruction\n        by the processor pipeline to detect control flow violations.\n        This is the 32-bit variant.\n        :param cpu: current CPU.\n        '
        pass

    @instruction
    def ENDBR64(cpu):
        if False:
            i = 10
            return i + 15
        '\n        The ENDBRANCH is a new instruction that is used to mark valid jump target\n        addresses of indirect calls and jumps in the program. This instruction\n        opcode is selected to be one that is a NOP on legacy machines such that\n        programs compiled with ENDBRANCH new instruction continue to function on\n        old machines without the CET enforcement. On processors that support CET\n        the ENDBRANCH is still a NOP and is primarily used as a marker instruction\n        by the processor pipeline to detect control flow violations.\n        :param cpu: current CPU.\n        '
        pass

    @instruction
    def MOVD(cpu, op0, op1):
        if False:
            return 10
        cpu._writeCorrectSize(op0, op1)

    @instruction
    def MOVZX(cpu, op0, op1):
        if False:
            while True:
                i = 10
        '\n        Moves with zero-extend.\n\n        Copies the contents of the source operand (register or memory location) to the destination\n        operand (register) and zero extends the value to 16 or 32 bits. The size of the converted value\n        depends on the operand-size attribute::\n\n                OP0  =  ZeroExtend(OP1);\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        '
        op0.write(Operators.ZEXTEND(op1.read(), op0.size))

    @instruction
    def MOVSX(cpu, op0, op1):
        if False:
            while True:
                i = 10
        '\n        Moves with sign-extension.\n\n        Copies the contents of the source operand (register or memory location) to the destination\n        operand (register) and sign extends the value to 16::\n\n                OP0  =  SignExtend(OP1);\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        '
        op0.write(Operators.SEXTEND(op1.read(), op1.size, op0.size))

    @instruction
    def MOVSXD(cpu, op0, op1):
        if False:
            return 10
        'Move DWORD with sign extension to QWORD.'
        op0.write(Operators.SEXTEND(op1.read(), op1.size, op0.size))

    @instruction
    def CQO(cpu):
        if False:
            return 10
        '\n        RDX:RAX = sign-extend of RAX.\n        '
        res = Operators.SEXTEND(cpu.RAX, 64, 128)
        cpu.RAX = Operators.EXTRACT(res, 0, 64)
        cpu.RDX = Operators.EXTRACT(res, 64, 64)

    @instruction
    def CDQE(cpu):
        if False:
            while True:
                i = 10
        '\n        RAX = sign-extend of EAX.\n        '
        cpu.RAX = Operators.SEXTEND(cpu.EAX, 32, 64)

    @instruction
    def CDQ(cpu):
        if False:
            for i in range(10):
                print('nop')
        '\n        EDX:EAX = sign-extend of EAX\n        '
        cpu.EDX = Operators.EXTRACT(Operators.SEXTEND(cpu.EAX, 32, 64), 32, 32)

    @instruction
    def CWDE(cpu):
        if False:
            i = 10
            return i + 15
        '\n        Converts word to doubleword.\n\n        ::\n            DX = sign-extend of AX.\n\n        :param cpu: current CPU.\n        '
        bit = Operators.EXTRACT(cpu.AX, 15, 1)
        cpu.EAX = Operators.SEXTEND(cpu.AX, 16, 32)
        cpu.EDX = Operators.SEXTEND(bit, 1, 32)

    @instruction
    def CBW(cpu):
        if False:
            print('Hello World!')
        '\n        Converts byte to word.\n\n        Double the size of the source operand by means of sign extension::\n\n                AX = sign-extend of AL.\n\n        :param cpu: current CPU.\n        '
        cpu.AX = Operators.SEXTEND(cpu.AL, 8, 16)

    @instruction
    def RDTSC(cpu):
        if False:
            for i in range(10):
                print('nop')
        "\n        Reads time-stamp counter.\n\n        Loads the current value of the processor's time-stamp counter into the\n        EDX:EAX registers.  The time-stamp counter is contained in a 64-bit\n        MSR. The high-order 32 bits of the MSR are loaded into the EDX\n        register, and the low-order 32 bits are loaded into the EAX register.\n        The processor increments the time-stamp counter MSR every clock cycle\n        and resets it to 0 whenever the processor is reset.\n\n        :param cpu: current CPU.\n        "
        val = cpu.icount
        cpu.RAX = val & 4294967295
        cpu.RDX = val >> 32 & 4294967295

    def _writeCorrectSize(cpu, op0, op1):
        if False:
            print('Hello World!')
        if op0.size > op1.size:
            op0.write(Operators.ZEXTEND(op1.read(), op0.size))
        else:
            op0.write(Operators.EXTRACT(op1.read(), 0, op0.size))

    @instruction
    def VMOVD(cpu, op0, op1):
        if False:
            while True:
                i = 10
        cpu._writeCorrectSize(op0, op1)

    @instruction
    def VMOVUPS(cpu, op0, op1):
        if False:
            i = 10
            return i + 15
        arg1 = op1.read()
        op0.write(arg1)

    @instruction
    def VMOVAPS(cpu, op0, op1):
        if False:
            i = 10
            return i + 15
        arg1 = op1.read()
        op0.write(arg1)

    @instruction
    def VMOVQ(cpu, op0, op1):
        if False:
            return 10
        cpu._writeCorrectSize(op0, op1)

    @instruction
    def FNSTCW(cpu, dest):
        if False:
            print('Hello World!')
        '\n        Stores x87 FPU Control Word.\n\n        Stores the current value of the FPU control word at the specified destination in memory.\n        The FSTCW instruction checks for and handles pending unmasked floating-point exceptions\n        before storing the control word; the FNSTCW instruction does not::\n\n            DEST  =  FPUControlWord;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        '
        cpu.write_int(dest.address(), cpu.FPCW, 16)

    def sem_SYSCALL(cpu):
        if False:
            return 10
        '\n        Syscall semantics without @instruction for use in emulator\n        '
        cpu.RCX = cpu.RIP
        cpu.R11 = cpu.RFLAGS
        raise Syscall()

    def generic_FXSAVE(cpu, dest, reg_layout):
        if False:
            for i in range(10):
                print('nop')
        '\n        Saves the current state of the x87 FPU, MMX technology, XMM, and\n        MXCSR registers to a 512-byte memory location specified in the\n        destination operand.\n\n        The content layout of the 512 byte region depends\n        on whether the processor is operating in non-64-bit operating modes\n        or 64-bit sub-mode of IA-32e mode\n        '
        addr = dest.address()
        for (offset, reg, size) in reg_layout:
            cpu.write_int(addr + offset, cpu.read_register_as_bitfield(reg), size)

    def generic_FXRSTOR(cpu, dest, reg_layout):
        if False:
            while True:
                i = 10
        '\n        Reloads the x87 FPU, MMX technology, XMM, and MXCSR registers from\n        the 512-byte memory image specified in the source operand. This data should\n        have been written to memory previously using the FXSAVE instruction, and in\n        the same format as required by the operating modes. The first byte of the data\n        should be located on a 16-byte boundary.\n\n        There are three distinct layouts of the FXSAVE state map:\n        one for legacy and compatibility mode, a second\n        format for 64-bit mode FXSAVE/FXRSTOR with REX.W=0, and the third format is for\n        64-bit mode with FXSAVE64/FXRSTOR64\n        '
        addr = dest.address()
        for (offset, reg, size) in reg_layout:
            cpu.write_register(reg, cpu.read_int(addr + offset, size))

    @instruction
    def SYSCALL(cpu):
        if False:
            while True:
                i = 10
        '\n        Calls to interrupt procedure.\n\n        The INT n instruction generates a call to the interrupt or exception handler specified\n        with the destination operand. The INT n instruction is the general mnemonic for executing\n        a software-generated call to an interrupt handler. The INTO instruction is a special\n        mnemonic for calling overflow exception (#OF), interrupt vector number 4. The overflow\n        interrupt checks the OF flag in the EFLAGS register and calls the overflow interrupt handler\n        if the OF flag is set to 1.\n\n        :param cpu: current CPU.\n        '
        cpu.sem_SYSCALL()

    @instruction
    def MOVLPD(cpu, dest, src):
        if False:
            print('Hello World!')
        '\n        Moves low packed double-precision floating-point value.\n\n        Moves a double-precision floating-point value from the source operand (second operand) and the\n        destination operand (first operand). The source and destination operands can be an XMM register\n        or a 64-bit memory location. This instruction allows double-precision floating-point values to be moved\n        to and from the low quadword of an XMM register and memory. It cannot be used for register to register\n        or memory to memory moves. When the destination operand is an XMM register, the high quadword of the\n        register remains unchanged.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        value = src.read()
        if src.size == 64 and dest.size == 128:
            value = dest.read() & 340282366920938463444927863358058659840 | Operators.ZEXTEND(value, 128)
        dest.write(value)

    @instruction
    def MOVHPD(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Moves high packed double-precision floating-point value.\n\n        Moves a double-precision floating-point value from the source operand (second operand) and the\n        destination operand (first operand). The source and destination operands can be an XMM register\n        or a 64-bit memory location. This instruction allows double-precision floating-point values to be moved\n        to and from the high quadword of an XMM register and memory. It cannot be used for register to\n        register or memory to memory moves. When the destination operand is an XMM register, the low quadword\n        of the register remains unchanged.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        if src.size == 128:
            assert dest.size == 64
            dest.write(Operators.EXTRACT(src.read(), 64, 64))
        else:
            assert src.size == 64 and dest.size == 128
            value = Operators.EXTRACT(dest.read(), 0, 64)
            dest.write(Operators.CONCAT(128, src.read(), value))

    @instruction
    def MOVHPS(cpu, dest, src):
        if False:
            return 10
        '\n        Moves high packed single-precision floating-point value.\n\n        Moves two packed single-precision floating-point values from the source operand\n        (second operand) to the destination operand (first operand). The source and destination\n        operands can be an XMM register or a 64-bit memory location. The instruction allows\n        single-precision floating-point values to be moved to and from the high quadword of\n        an XMM register and memory. It cannot be used for register to register or memory to\n        memory moves. When the destination operand is an XMM register, the low quadword\n        of the register remains unchanged.\n        '
        if src.size == 128:
            assert dest.size == 64
            dest.write(Operators.EXTRACT(src.read(), 64, 64))
        else:
            assert src.size == 64 and dest.size == 128
            value = Operators.EXTRACT(dest.read(), 0, 64)
            dest.write(Operators.CONCAT(128, src.read(), value))

    @instruction
    def PSUBB(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Packed subtract.\n\n        Performs a SIMD subtract of the packed integers of the source operand (second operand) from the packed\n        integers of the destination operand (first operand), and stores the packed integer results in the\n        destination operand. The source operand can be an MMX(TM) technology register or a 64-bit memory location,\n        or it can be an XMM register or a 128-bit memory location. The destination operand can be an MMX or an XMM\n        register.\n        The PSUBB instruction subtracts packed byte integers. When an individual result is too large or too small\n        to be represented in a byte, the result is wrapped around and the low 8 bits are written to the\n        destination element.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        result = []
        value_a = dest.read()
        value_b = src.read()
        for i in reversed(range(0, dest.size, 8)):
            a = Operators.EXTRACT(value_a, i, 8)
            b = Operators.EXTRACT(value_b, i, 8)
            result.append(a - b & 255)
        dest.write(Operators.CONCAT(8 * len(result), *result))

    @instruction
    def PSUBQ(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        PSUBQ: Packed add with quadruple words\n        Packed subtract with quad\n\n        Subtracts the second operand (source operand) from the first operand (destination operand) and stores\n        the result in the destination operand. When packed quadword operands are used, a SIMD subtract is performed.\n        When a quadword result is too large to be represented in 64 bits (overflow), the result is wrapped around\n        and the low 64 bits are written to the destination element (that is, the carry is ignored).\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        result = []
        value_a = dest.read()
        value_b = src.read()
        for i in reversed(range(0, dest.size, 64)):
            a = Operators.EXTRACT(value_a, i, 64)
            b = Operators.EXTRACT(value_b, i, 64)
            result.append(a - b)
        dest.write(Operators.CONCAT(dest.size, *result))

    @instruction
    def POR(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Performs a bitwise logical OR operation on the source operand (second operand) and the destination operand\n        (first operand) and stores the result in the destination operand. The source operand can be an MMX technology\n        register or a 64-bit memory location or it can be an XMM register or a 128-bit memory location. The destination\n        operand can be an MMX technology register or an XMM register. Each bit of the result is set to 1 if either\n        or both of the corresponding bits of the first and second operands are 1; otherwise, it is set to 0.\n        '
        res = dest.write(dest.read() | src.read())

    @instruction
    def XORPS(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs a bitwise logical OR operation on the source operand (second operand) and the destination operand\n        (first operand) and stores the result in the destination operand. The source operand can be an MMX technology\n        register or a 64-bit memory location or it can be an XMM register or a 128-bit memory location. The destination\n        operand can be an MMX technology register or an XMM register. Each bit of the result is set to 1 if either\n        or both of the corresponding bits of the first and second operands are 1; otherwise, it is set to 0.\n        '
        res = dest.write(dest.read() ^ src.read())

    @instruction
    def VORPD(cpu, dest, src, src2):
        if False:
            i = 10
            return i + 15
        '\n        Performs a bitwise logical OR operation on the source operand (second operand) and second source operand (third operand)\n         and stores the result in the destination operand (first operand).\n        '
        res = dest.write(src.read() | src2.read())

    @instruction
    def VORPS(cpu, dest, src, src2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs a bitwise logical OR operation on the source operand (second operand) and second source operand (third operand)\n         and stores the result in the destination operand (first operand).\n        '
        res = dest.write(src.read() | src2.read())

    @instruction
    def PTEST(cpu, dest, src):
        if False:
            while True:
                i = 10
        'PTEST\n        PTEST set the ZF flag if all bits in the result are 0 of the bitwise AND\n        of the first source operand (first operand) and the second source operand\n        (second operand). Also this sets the CF flag if all bits in the result\n        are 0 of the bitwise AND of the second source operand (second operand)\n        and the logical NOT of the destination operand.\n        '
        cpu.OF = False
        cpu.AF = False
        cpu.PF = False
        cpu.SF = False
        cpu.ZF = Operators.EXTRACT(dest.read(), 0, 128) & Operators.EXTRACT(src.read(), 0, 128) == 0
        cpu.CF = Operators.EXTRACT(src.read(), 0, 128) & ~Operators.EXTRACT(dest.read(), 0, 128) == 0

    @instruction
    def VPTEST(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        cpu.OF = False
        cpu.AF = False
        cpu.PF = False
        cpu.SF = False
        cpu.ZF = dest.read() & src.read() == 0
        cpu.CF = dest.read() & ~src.read() == 0

    @instruction
    def MOVAPS(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Moves aligned packed single-precision floating-point values.\n\n        Moves a double quadword containing four packed single-precision floating-point numbers from the\n        source operand (second operand) to the destination operand (first operand). This instruction can be\n        used to load an XMM register from a 128-bit memory location, to store the contents of an XMM register\n        into a 128-bit memory location, or move data between two XMM registers.\n        When the source or destination operand is a memory operand, the operand must be aligned on a 16-byte\n        boundary or a general-protection exception (#GP) will be generated::\n\n                DEST  =  SRC;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        dest.write(src.read())

    @instruction
    def MOVQ(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Move quadword.\n\n        Copies a quadword from the source operand (second operand) to the destination operand (first operand).\n        The source and destination operands can be MMX(TM) technology registers, XMM registers, or 64-bit memory\n        locations. This instruction can be used to move a between two MMX registers or between an MMX register\n        and a 64-bit memory location, or to move data between two XMM registers or between an XMM register and\n        a 64-bit memory location. The instruction cannot be used to transfer data between memory locations.\n        When the source operand is an XMM register, the low quadword is moved; when the destination operand is\n        an XMM register, the quadword is stored to the low quadword of the register, and the high quadword is\n        cleared to all 0s::\n\n            MOVQ instruction when operating on MMX registers and memory locations:\n\n            DEST  =  SRC;\n\n            MOVQ instruction when source and destination operands are XMM registers:\n\n            DEST[63-0]  =  SRC[63-0];\n\n            MOVQ instruction when source operand is XMM register and destination operand is memory location:\n\n            DEST  =  SRC[63-0];\n\n            MOVQ instruction when source operand is memory location and destination operand is XMM register:\n\n            DEST[63-0]  =  SRC;\n            DEST[127-64]  =  0000000000000000H;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        if dest.size == src.size and dest.size == 64:
            dest.write(src.read())
        elif dest.size == src.size and dest.size == 128:
            src_lo = Operators.EXTRACT(src.read(), 0, 64)
            dest.write(Operators.ZEXTEND(src_lo, 128))
        elif dest.size == 128 and src.size == 64:
            dest.write(Operators.ZEXTEND(src.read(), dest.size))
        elif dest.size == 64 and src.size == 128:
            dest.write(Operators.EXTRACT(src.read(), 0, dest.size))
        else:
            msg = 'Invalid size in MOVQ'
            logger.error(msg)
            raise CpuException(msg)

    @instruction
    def MOVSD(cpu, dest, src):
        if False:
            return 10
        '\n        Move Scalar Double-Precision Floating-Point Value\n\n        Moves a scalar double-precision floating-point value from the source\n        operand (second operand) to the destination operand (first operand).\n        The source and destination operands can be XMM registers or 64-bit memory\n        locations. This instruction can be used to move a double-precision\n        floating-point value to and from the low quadword of an XMM register and\n        a 64-bit memory location, or to move a double-precision floating-point\n        value between the low quadwords of two XMM registers. The instruction\n        cannot be used to transfer data between memory locations.\n        When the source and destination operands are XMM registers, the high\n        quadword of the destination operand remains unchanged. When the source\n        operand is a memory location and destination operand is an XMM registers,\n        the high quadword of the destination operand is cleared to all 0s.\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        :param src: source operand.\n        '
        assert dest.type != 'memory' or src.type != 'memory'
        value = Operators.EXTRACT(src.read(), 0, 64)
        if dest.size > src.size:
            value = Operators.ZEXTEND(value, dest.size)
        dest.write(value)

    @instruction
    def MOVSS(cpu, dest, src):
        if False:
            while True:
                i = 10
        '\n        Moves a scalar single-precision floating-point value\n\n        Moves a scalar single-precision floating-point value from the source operand (second operand)\n        to the destination operand (first operand). The source and destination operands can be XMM\n        registers or 32-bit memory locations. This instruction can be used to move a single-precision\n        floating-point value to and from the low doubleword of an XMM register and a 32-bit memory\n        location, or to move a single-precision floating-point value between the low doublewords of\n        two XMM registers. The instruction cannot be used to transfer data between memory locations.\n        When the source and destination operands are XMM registers, the three high-order doublewords of the\n        destination operand remain unchanged. When the source operand is a memory location and destination\n        operand is an XMM registers, the three high-order doublewords of the destination operand are cleared to all 0s.\n\n        //MOVSS instruction when source and destination operands are XMM registers:\n        if(IsXMM(Source) && IsXMM(Destination))\n            Destination[0..31] = Source[0..31];\n            //Destination[32..127] remains unchanged\n            //MOVSS instruction when source operand is XMM register and destination operand is memory location:\n        else if(IsXMM(Source) && IsMemory(Destination))\n            Destination = Source[0..31];\n        //MOVSS instruction when source operand is memory location and destination operand is XMM register:\n        else {\n                Destination[0..31] = Source;\n                Destination[32..127] = 0;\n        }\n        '
        if dest.type == 'register' and src.type == 'register':
            assert dest.size == 128 and src.size == 128
            dest.write(dest.read() & ~4294967295 | src.read() & 4294967295)
        elif dest.type == 'memory':
            assert src.type == 'register'
            dest.write(Operators.EXTRACT(src.read(), 0, dest.size))
        else:
            assert src.type == 'memory' and dest.type == 'register'
            assert src.size == 32 and dest.size == 128
            dest.write(Operators.ZEXTEND(src.read(), 128))

    @instruction
    def VMOVDQA(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Move Aligned Double Quadword\n\n        Moves 128 bits of packed integer values from the source operand (second\n        operand) to the destination operand (first operand). This instruction\n        can be used to load an XMM register from a 128-bit memory location, to\n        store the contents of an XMM register into a 128-bit memory location, or\n        to move data between two XMM registers.\n\n        When the source or destination operand is a memory operand, the operand\n        must be aligned on a 16-byte boundary or a general-protection exception\n        (#GP) will be generated. To move integer data to and from unaligned\n        memory locations, use the VMOVDQU instruction.'
        dest.write(src.read())

    @instruction
    def VMOVDQU(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        '\n        Move Unaligned Double Quadword\n\n        Moves 128 bits of packed integer values from the source operand (second operand)\n        to the destination operand (first operand). This instruction can be used to load\n        an XMM register from a 128-bit memory location, to store the contents of an XMM\n        register into a 128-bit memory location, or to move data between two XMM registers.\n        When the source or destination operand is a memory operand, the operand may be\n        unaligned on a 16-byte boundary without causing a general-protection exception\n        (#GP) to be generated.\n\n            VMOVDQU (VEX.128 encoded version)\n            DEST[127:0] <- SRC[127:0]\n            DEST[VLMAX-1:128] <- 0\n            VMOVDQU (VEX.256 encoded version)\n            DEST[255:0] <- SRC[255:0]\n        '
        dest.write(src.read())

    @instruction
    def VEXTRACTF128(cpu, dest, src, offset):
        if False:
            while True:
                i = 10
        'Extract Packed Floating-Point Values\n\n        Extracts 128-bits of packed floating-point values from the source\n        operand (second operand) at an 128-bit offset from imm8[0] into the\n        destination operand (first operand). The destination may be either an\n        XMM register or an 128-bit memory location.\n        '
        offset = offset.read()
        dest.write(Operators.EXTRACT(src.read(), offset * 128, (offset + 1) * 128))

    @instruction
    def PREFETCHT0(cpu, arg):
        if False:
            print('Hello World!')
        '\n        Not implemented.\n\n        Performs no operation.\n        '

    @instruction
    def PREFETCHT1(cpu, arg):
        if False:
            i = 10
            return i + 15
        '\n        Not implemented.\n\n        Performs no operation.\n        '

    @instruction
    def PREFETCHT2(cpu, arg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Not implemented.\n\n        Performs no operation.\n        '

    @instruction
    def PREFETCHTNTA(cpu, arg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Not implemented.\n\n        Performs no operation.\n        '

    @instruction
    def PINSRW(cpu, dest, src, count):
        if False:
            i = 10
            return i + 15
        if dest.size == 64:
            sel = count.read() & 3
            mask = [65535, 4294901760, 281470681743360, 18446462598732840960][sel]
        else:
            assert dest.size == 128
            sel = count.read() & 7
            mask = [65535, 4294901760, 281470681743360, 18446462598732840960, 1208907372870555465154560, 79226953588444722964369244160, 5192217630372313364192902785269760, 340277174624079928635746076935438991360][sel]
        dest.write(dest.read() & ~mask | Operators.ZEXTEND(src.read(), dest.size) << sel * 16 & mask)

    @instruction
    def PEXTRW(cpu, dest, src, count):
        if False:
            i = 10
            return i + 15
        if src.size == 64:
            sel = Operators.ZEXTEND(Operators.EXTRACT(count.read(), 0, 2), src.size)
        else:
            sel = Operators.ZEXTEND(Operators.EXTRACT(count.read(), 0, 3), src.size)
        tmp = src.read() >> sel * 16 & 65535
        dest.write(Operators.EXTRACT(tmp, 0, dest.size))

    @instruction
    def PALIGNR(cpu, dest, src, offset):
        if False:
            while True:
                i = 10
        'ALIGNR concatenates the destination operand (the first operand) and the source\n        operand (the second operand) into an intermediate composite, shifts the composite\n        at byte granularity to the right by a constant immediate, and extracts the right-\n        aligned result into the destination.'
        dest.write(Operators.EXTRACT(Operators.CONCAT(dest.size * 2, dest.read(), src.read()), offset.read() * 8, dest.size))

    @instruction
    def PSLLDQ(cpu, dest, src):
        if False:
            while True:
                i = 10
        'Packed Shift Left Logical Double Quadword\n        Shifts the destination operand (first operand) to the left by the number\n         of bytes specified in the count operand (second operand). The empty low-order\n         bytes are cleared (set to all 0s). If the value specified by the count\n         operand is greater than 15, the destination operand is set to all 0s.\n         The destination operand is an XMM register. The count operand is an 8-bit\n         immediate.\n\n            TEMP  =  COUNT;\n            if (TEMP > 15) TEMP  =  16;\n            DEST  =  DEST << (TEMP * 8);\n        '
        count = Operators.ZEXTEND(src.read(), dest.size * 2)
        byte_count = Operators.ITEBV(src.size * 2, count > 15, 16, count)
        bit_count = byte_count * 8
        val = Operators.ZEXTEND(dest.read(), dest.size * 2)
        val = val << Operators.ZEXTEND(bit_count, dest.size * 2)
        dest.write(Operators.EXTRACT(val, 0, dest.size))

    @instruction
    def PSRLQ(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        'Shift Packed Data Right Logical\n\n        Shifts the bits in the individual quadword in the destination operand to the right by\n        the number of bits specified in the count operand . As the bits in the data elements\n        are shifted right, the empty high-order bits are cleared (set to 0). If the value\n        specified by the count operand is greater than  63, then the destination operand is set\n        to all 0s.\n\n        if(OperandSize == 64) {\n                        //PSRLQ instruction with 64-bit operand:\n                        if(Count > 63) Destination[64..0] = 0;\n                        else Destination = ZeroExtend(Destination >> Count);\n                }\n                else {\n                        //PSRLQ instruction with 128-bit operand:\n                        if(Count > 15) Destination[128..0] = 0;\n                        else {\n                                Destination[0..63] = ZeroExtend(Destination[0..63] >> Count);\n                                Destination[64..127] = ZeroExtend(Destination[64..127] >> Count);\n                        }\n                }\n        '
        count = src.read()
        count = Operators.ITEBV(src.size, Operators.UGT(count, 63), 64, count)
        count = Operators.EXTRACT(count, 0, 64)
        if dest.size == 64:
            dest.write(dest.read() >> count)
        else:
            hi = Operators.EXTRACT(dest.read(), 64, 64) >> count
            low = Operators.EXTRACT(dest.read(), 0, 64) >> count
            dest.write(Operators.CONCAT(128, hi, low))

    @instruction
    def PAND(cpu, dest, src):
        if False:
            for i in range(10):
                print('nop')
        dest.write(dest.read() & src.read())

    @instruction
    def LSL(cpu, limit_ptr, selector):
        if False:
            for i in range(10):
                print('nop')
        selector = selector.read()
        if issymbolic(selector):
            raise NotImplementedError('Do not yet implement symbolic LSL')
        if selector == 0 or selector not in cpu._segments:
            cpu.ZF = False
            logger.info('Invalid selector %s. Clearing ZF', selector)
            return
        (base, limit, ty) = cpu.get_descriptor(selector)
        logger.debug('LSL instruction not fully implemented')
        cpu.ZF = True
        limit_ptr.write(limit)

    @instruction
    def SYSENTER(cpu):
        if False:
            return 10
        '\n        Calls to system\n\n        Executes a fast call to a level 0 system procedure or routine\n\n        :param cpu: current CPU.\n        '
        raise Syscall()

    @instruction
    def TZCNT(cpu, dest, src):
        if False:
            i = 10
            return i + 15
        '\n        Count the number of trailing least significant zero bits in source\n        operand (second operand) and returns the result in destination\n        operand (first operand). TZCNT is an extension of the BSF instruction.\n\n        The key difference between TZCNT and BSF instruction is that TZCNT\n        provides operand size as output when source operand is zero while in\n        the case of BSF instruction, if source operand is zero, the content of\n        destination operand are undefined. On processors that do not support\n        TZCNT, the instruction byte encoding is executed as BSF\n        '
        value = src.read()
        flag = Operators.EXTRACT(value, 0, 1) == 1
        res = 0
        for pos in range(1, src.size):
            res = Operators.ITEBV(dest.size, flag, res, pos)
            flag = Operators.OR(flag, Operators.EXTRACT(value, pos, 1) == 1)
        cpu.CF = res == src.size
        cpu.ZF = res == 0
        dest.write(res)

    @instruction
    def VPSHUFB(cpu, op0, op1, op3):
        if False:
            print('Hello World!')
        '\n        Packed shuffle bytes.\n\n        Copies bytes from source operand (second operand) and inserts them in the destination operand\n        (first operand) at locations selected with the order operand (third operand).\n\n        :param cpu: current CPU.\n        :param op0: destination operand.\n        :param op1: source operand.\n        :param op3: order operand.\n        '
        size = op0.size
        arg0 = op0.read()
        arg1 = op1.read()
        arg3 = Operators.ZEXTEND(op3.read(), size)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 7, 1) == 1, 0, arg1 >> (arg3 >> 0 & 7 * 8) & 255)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 15, 1) == 1, 0, (arg1 >> (arg3 >> 8 & 7 * 8) & 255) << 8)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 23, 1) == 1, 0, (arg1 >> (arg3 >> 16 & 7 * 8) & 255) << 16)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 31, 1) == 1, 0, (arg1 >> (arg3 >> 24 & 7 * 8) & 255) << 24)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 39, 1) == 1, 0, (arg1 >> (arg3 >> 32 & 7 * 8) & 255) << 32)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 47, 1) == 1, 0, (arg1 >> (arg3 >> 40 & 7 * 8) & 255) << 40)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 55, 1) == 1, 0, (arg1 >> (arg3 >> 48 & 7 * 8) & 255) << 48)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 63, 1) == 1, 0, (arg1 >> (arg3 >> 56 & 7 * 8) & 255) << 56)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 71, 1) == 1, 0, (arg1 >> (arg3 >> 64 & 7 * 8) & 255) << 64)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 79, 1) == 1, 0, (arg1 >> (arg3 >> 72 & 7 * 8) & 255) << 72)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 87, 1) == 1, 0, (arg1 >> (arg3 >> 80 & 7 * 8) & 255) << 80)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 95, 1) == 1, 0, (arg1 >> (arg3 >> 88 & 7 * 8) & 255) << 88)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 103, 1) == 1, 0, (arg1 >> (arg3 >> 96 & 7 * 8) & 255) << 96)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 111, 1) == 1, 0, (arg1 >> (arg3 >> 104 & 7 * 8) & 255) << 104)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 119, 1) == 1, 0, (arg1 >> (arg3 >> 112 & 7 * 8) & 255) << 112)
        arg0 |= Operators.ITEBV(size, Operators.EXTRACT(arg3, 127, 1) == 1, 0, (arg1 >> (arg3 >> 120 & 7 * 8) & 255) << 120)
        op0.write(arg0)

    @instruction
    def VZEROUPPER(cpu):
        if False:
            print('Hello World!')
        cpu.YMM0 = cpu.YMM0 & 340282366920938463463374607431768211455
        cpu.YMM1 = cpu.YMM1 & 340282366920938463463374607431768211455
        cpu.YMM2 = cpu.YMM2 & 340282366920938463463374607431768211455
        cpu.YMM3 = cpu.YMM3 & 340282366920938463463374607431768211455
        cpu.YMM4 = cpu.YMM4 & 340282366920938463463374607431768211455
        cpu.YMM5 = cpu.YMM5 & 340282366920938463463374607431768211455
        cpu.YMM6 = cpu.YMM6 & 340282366920938463463374607431768211455
        cpu.YMM7 = cpu.YMM7 & 340282366920938463463374607431768211455
        if cpu.mode == cs.CS_MODE_64:
            cpu.YMM8 = cpu.YMM8 & 340282366920938463463374607431768211455
            cpu.YMM9 = cpu.YMM9 & 340282366920938463463374607431768211455
            cpu.YMM10 = cpu.YMM10 & 340282366920938463463374607431768211455
            cpu.YMM11 = cpu.YMM11 & 340282366920938463463374607431768211455
            cpu.YMM12 = cpu.YMM12 & 340282366920938463463374607431768211455
            cpu.YMM13 = cpu.YMM13 & 340282366920938463463374607431768211455
            cpu.YMM14 = cpu.YMM14 & 340282366920938463463374607431768211455
            cpu.YMM15 = cpu.YMM15 & 340282366920938463463374607431768211455

class I386LinuxSyscallAbi(SyscallAbi):
    """
    i386 Linux system call ABI
    """

    def syscall_number(self):
        if False:
            for i in range(10):
                print('nop')
        return self._cpu.EAX

    def get_arguments(self):
        if False:
            while True:
                i = 10
        for reg in ('EBX', 'ECX', 'EDX', 'ESI', 'EDI', 'EBP'):
            yield reg

    def get_result_reg(self):
        if False:
            for i in range(10):
                print('nop')
        return 'EAX'

    def write_result(self, result):
        if False:
            i = 10
            return i + 15
        self._cpu.EAX = result

class AMD64LinuxSyscallAbi(SyscallAbi):
    """
    AMD64 Linux system call ABI
    """

    def syscall_number(self):
        if False:
            i = 10
            return i + 15
        return self._cpu.RAX

    def get_arguments(self):
        if False:
            return 10
        for reg in ('RDI', 'RSI', 'RDX', 'R10', 'R8', 'R9'):
            yield reg

    def get_result_reg(self):
        if False:
            while True:
                i = 10
        return 'RAX'

    def write_result(self, result):
        if False:
            i = 10
            return i + 15
        self._cpu.RAX = result

class I386CdeclAbi(Abi):
    """
    i386 cdecl function call semantics
    """

    def get_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        base = self._cpu.STACK + self._cpu.address_bit_size // 8
        for address in self.values_from(base):
            yield address

    def get_result_reg(self):
        if False:
            return 10
        return 'EAX'

    def write_result(self, result):
        if False:
            i = 10
            return i + 15
        self._cpu.EAX = result

    def ret(self):
        if False:
            print('Hello World!')
        self._cpu.EIP = self._cpu.pop(self._cpu.address_bit_size)

class I386StdcallAbi(Abi):
    """
    x86 Stdcall function call convention. Callee cleans up the stack.
    """

    def __init__(self, cpu):
        if False:
            i = 10
            return i + 15
        super().__init__(cpu)
        self._arguments = 0

    def get_arguments(self):
        if False:
            return 10
        base = self._cpu.STACK + self._cpu.address_bit_size // 8
        for address in self.values_from(base):
            self._arguments += 1
            yield address

    def get_result_reg(self):
        if False:
            print('Hello World!')
        return 'EAX'

    def write_result(self, result):
        if False:
            while True:
                i = 10
        self._cpu.EAX = result

    def ret(self):
        if False:
            print('Hello World!')
        self._cpu.EIP = self._cpu.pop(self._cpu.address_bit_size)
        word_bytes = self._cpu.address_bit_size // 8
        self._cpu.ESP += self._arguments * word_bytes
        self._arguments = 0

class SystemVAbi(Abi):
    """
    x64 SystemV function call convention
    """

    def get_arguments(self):
        if False:
            while True:
                i = 10
        reg_args = ('RDI', 'RSI', 'RDX', 'RCX', 'R8', 'R9')
        for reg in reg_args:
            yield reg
        word_bytes = self._cpu.address_bit_size // 8
        for address in self.values_from(self._cpu.RSP + word_bytes):
            yield address

    def get_result_reg(self):
        if False:
            for i in range(10):
                print('nop')
        return 'RAX'

    def write_result(self, result):
        if False:
            while True:
                i = 10
        self._cpu.RAX = result

    def ret(self):
        if False:
            i = 10
            return i + 15
        self._cpu.RIP = self._cpu.pop(self._cpu.address_bit_size)

class AMD64Cpu(X86Cpu):
    max_instr_width = 15
    address_bit_size = 64
    machine = 'amd64'
    arch = cs.CS_ARCH_X86
    mode = cs.CS_MODE_64
    FXSAVE_layout = [(0, 'FPCW', 16), (2, 'FPSW', 16), (4, 'FPTAG', 8), (6, 'FOP', 16), (8, 'FIP', 32), (12, 'FCS', 16), (16, 'FDP', 32), (20, 'FDS', 16), (24, 'MXCSR', 32), (28, 'MXCSR_MASK', 32), (32, 'FP0', 80), (48, 'FP1', 80), (64, 'FP2', 80), (80, 'FP3', 80), (96, 'FP4', 80), (112, 'FP5', 80), (128, 'FP6', 80), (144, 'FP7', 80), (160, 'XMM0', 128), (176, 'XMM1', 128), (192, 'XMM2', 128), (208, 'XMM3', 128), (224, 'XMM4', 128), (240, 'XMM5', 128), (256, 'XMM6', 128), (272, 'XMM7', 128), (288, 'XMM8', 128), (304, 'XMM9', 128), (320, 'XMM10', 128), (336, 'XMM11', 128), (352, 'XMM12', 128), (368, 'XMM13', 128), (384, 'XMM14', 128), (400, 'XMM15', 128)]

    def __init__(self, memory: Memory, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Builds a CPU model.\n        :param memory: memory object for this CPU.\n        '
        super().__init__(AMD64RegFile(aliases={'PC': 'RIP', 'STACK': 'RSP', 'FRAME': 'RBP'}), memory, *args, **kwargs)

    def __str__(self):
        if False:
            return 10
        '\n        Returns a string representation of cpu state\n\n        :rtype: str\n        :return: a string containing the name and current value for all the registers.\n        '
        CHEADER = '\x1b[95m'
        CBLUE = '\x1b[94m'
        CGREEN = '\x1b[92m'
        CWARNING = '\x1b[93m'
        CFAIL = '\x1b[91m'
        CEND = '\x1b[0m'
        pos = 0
        result = ''
        try:
            instruction = self.instruction
            result += f'Instruction: 0x{instruction.address:016x}:\t{instruction.mnemonic}\t{instruction.op_str}\n'
        except BaseException:
            result += "{can't decode instruction }\n"
        regs = ('RAX', 'RCX', 'RDX', 'RBX', 'RSP', 'RBP', 'RSI', 'RDI', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'RIP', 'EFLAGS')
        for reg_name in regs:
            value = self.read_register(reg_name)
            if issymbolic(value):
                result += f'{reg_name:3s}: {CFAIL}{visitors.pretty_print(value, depth=10)}{CEND}\n'
            else:
                result += f'{reg_name:3s}: 0x{value:016x}\n'
            pos = 0
        pos = 0
        for reg_name in ('CF', 'SF', 'ZF', 'OF', 'AF', 'PF', 'IF', 'DF'):
            value = self.read_register(reg_name)
            if issymbolic(value):
                result += f'{reg_name}: {CFAIL}{visitors.pretty_print(value, depth=10)}{CEND}\n'
            else:
                result += f'{reg_name}: {value:1x}\n'
            pos = 0
        for reg_name in ['CS', 'DS', 'ES', 'SS', 'FS', 'GS']:
            (base, size, ty) = self.get_descriptor(self.read_register(reg_name))
            result += f'{reg_name}: {base:x}, {size:x} ({ty})\n'
        for reg_name in ['FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7', 'TOP']:
            value = getattr(self, reg_name)
            result += f'{reg_name:3s}: {value!r}\n'
            pos = 0
        return result

    @property
    def canonical_registers(self):
        if False:
            print('Hello World!')
        return self.regfile.canonical_registers

    @instruction
    def XLATB(cpu):
        if False:
            return 10
        "\n        Table look-up translation.\n\n        Locates a byte entry in a table in memory, using the contents of the\n        AL register as a table index, then copies the contents of the table entry\n        back into the AL register. The index in the AL register is treated as\n        an unsigned integer. The XLAT and XLATB instructions get the base address\n        of the table in memory from either the DS:EBX or the DS:BX registers.\n        In 64-bit mode, operation is similar to that in legacy or compatibility mode.\n        AL is used to specify the table index (the operand size is fixed at 8 bits).\n        RBX, however, is used to specify the table's base address::\n\n                IF address_bit_size = 16\n                THEN\n                    AL = (DS:BX + ZeroExtend(AL));\n                ELSE IF (address_bit_size = 32)\n                    AL = (DS:EBX + ZeroExtend(AL)); FI;\n                ELSE (address_bit_size = 64)\n                    AL = (RBX + ZeroExtend(AL));\n                FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        "
        cpu.AL = cpu.read_int(cpu.RBX + Operators.ZEXTEND(cpu.AL, 64), 8)

    @instruction
    def FXSAVE(cpu, dest):
        if False:
            while True:
                i = 10
        return cpu.generic_FXSAVE(dest, AMD64Cpu.FXSAVE_layout)

    @instruction
    def FXRSTOR(cpu, src):
        if False:
            print('Hello World!')
        return cpu.generic_FXRSTOR(src, AMD64Cpu.FXSAVE_layout)

class I386Cpu(X86Cpu):
    max_instr_width = 15
    address_bit_size = 32
    machine = 'i386'
    arch = cs.CS_ARCH_X86
    mode = cs.CS_MODE_32
    FXSAVE_layout = [(0, 'FPCW', 16), (2, 'FPSW', 16), (4, 'FPTAG', 8), (6, 'FOP', 16), (8, 'FIP', 32), (12, 'FCS', 16), (16, 'FDP', 32), (20, 'FDS', 16), (24, 'MXCSR', 32), (28, 'MXCSR_MASK', 32), (32, 'FP0', 80), (48, 'FP1', 80), (64, 'FP2', 80), (80, 'FP3', 80), (96, 'FP4', 80), (112, 'FP5', 80), (128, 'FP6', 80), (144, 'FP7', 80), (160, 'XMM0', 128), (176, 'XMM1', 128), (192, 'XMM2', 128), (208, 'XMM3', 128), (224, 'XMM4', 128), (240, 'XMM5', 128), (256, 'XMM6', 128), (272, 'XMM7', 128)]

    def __init__(self, memory: Memory, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Builds a CPU model.\n        :param memory: memory object for this CPU.\n        '
        super().__init__(AMD64RegFile({'PC': 'EIP', 'STACK': 'ESP', 'FRAME': 'EBP'}), memory, *args, **kwargs)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a string representation of cpu state\n\n        :rtype: str\n        :return: a string containing the name and current value for all the registers.\n        '
        CHEADER = '\x1b[95m'
        CBLUE = '\x1b[94m'
        CGREEN = '\x1b[92m'
        CWARNING = '\x1b[93m'
        CFAIL = '\x1b[91m'
        CEND = '\x1b[0m'
        pos = 0
        result = ''
        try:
            instruction = self.instruction
            result += f'Instruction: 0x{instruction.address:016x}:\t{instruction.mnemonic}\t{instruction.op_str}\n'
        except BaseException:
            result += "{can't decode instruction }\n"
        regs = ('EAX', 'ECX', 'EDX', 'EBX', 'ESP', 'EBP', 'ESI', 'EDI', 'EIP')
        for reg_name in regs:
            value = self.read_register(reg_name)
            if issymbolic(value):
                result += f'{reg_name:3s}: {CFAIL}{visitors.pretty_print(value, depth=10)}{CEND}\n'
            else:
                result += f'{reg_name:3s}: 0x{value:016x}\n'
            pos = 0
        pos = 0
        for reg_name in ['CF', 'SF', 'ZF', 'OF', 'AF', 'PF', 'IF', 'DF']:
            value = self.read_register(reg_name)
            if issymbolic(value):
                result += f'{reg_name}: {CFAIL}{visitors.pretty_print(value, depth=10)}{CEND}\n'
            else:
                result += f'{reg_name}: {value:1x}\n'
            pos = 0
        for reg_name in ['CS', 'DS', 'ES', 'SS', 'FS', 'GS']:
            (base, size, ty) = self.get_descriptor(self.read_register(reg_name))
            result += f'{reg_name}: {base:x}, {size:x} ({ty})\n'
        for reg_name in ['FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7', 'TOP']:
            value = getattr(self, reg_name)
            result += f'{reg_name:3s}: {value!r}\n'
            pos = 0
        return result

    @property
    def canonical_registers(self):
        if False:
            while True:
                i = 10
        regs = ['EAX', 'ECX', 'EDX', 'EBX', 'ESP', 'EBP', 'ESI', 'EDI', 'EIP']
        regs.extend(['CS', 'DS', 'ES', 'SS', 'FS', 'GS'])
        regs.extend(['FP0', 'FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7', 'FPCW', 'FPSW', 'FPTAG', 'FOP', 'FIP', 'FCS', 'FDP', 'FDS', 'MXCSR', 'MXCSR_MASK'])
        regs.extend(['XMM0', 'XMM1', 'XMM10', 'XMM11', 'XMM12', 'XMM13', 'XMM14', 'XMM15', 'XMM2', 'XMM3', 'XMM4', 'XMM5', 'XMM6', 'XMM7', 'XMM8', 'XMM9'])
        regs.extend(['CF', 'PF', 'AF', 'ZF', 'SF', 'IF', 'DF', 'OF'])
        return tuple(regs)

    @instruction
    def XLATB(cpu):
        if False:
            for i in range(10):
                print('nop')
        "\n        Table look-up translation.\n\n        Locates a byte entry in a table in memory, using the contents of the\n        AL register as a table index, then copies the contents of the table entry\n        back into the AL register. The index in the AL register is treated as\n        an unsigned integer. The XLAT and XLATB instructions get the base address\n        of the table in memory from either the DS:EBX or the DS:BX registers.\n        In 64-bit mode, operation is similar to that in legacy or compatibility mode.\n        AL is used to specify the table index (the operand size is fixed at 8 bits).\n        RBX, however, is used to specify the table's base address::\n\n                IF address_bit_size = 16\n                THEN\n                    AL = (DS:BX + ZeroExtend(AL));\n                ELSE IF (address_bit_size = 32)\n                    AL = (DS:EBX + ZeroExtend(AL)); FI;\n                ELSE (address_bit_size = 64)\n                    AL = (RBX + ZeroExtend(AL));\n                FI;\n\n        :param cpu: current CPU.\n        :param dest: destination operand.\n        "
        cpu.AL = cpu.read_int(cpu.EBX + Operators.ZEXTEND(cpu.AL, 32), 8)

    @instruction
    def FXSAVE(cpu, dest):
        if False:
            return 10
        return cpu.generic_FXSAVE(dest, I386Cpu.FXSAVE_layout)

    @instruction
    def FXRSTOR(cpu, src):
        if False:
            return 10
        return cpu.generic_FXRSTOR(src, I386Cpu.FXSAVE_layout)