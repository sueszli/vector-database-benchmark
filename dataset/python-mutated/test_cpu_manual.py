import struct
import unittest
from manticore.native.cpu.x86 import I386Cpu
from manticore.native.cpu.abstractcpu import ConcretizeRegister
from manticore.native.cpu.x86 import AMD64Cpu
from manticore.native.memory import *
from manticore.core.smtlib import BitVecOr, operator, Bool
from manticore.core.smtlib.solver import Z3Solver
from functools import reduce
from typing import List
solver = Z3Solver.instance()
sizes = {'RAX': 64, 'EAX': 32, 'AX': 16, 'AL': 8, 'AH': 8, 'RCX': 64, 'ECX': 32, 'CX': 16, 'CL': 8, 'CH': 8, 'RDX': 64, 'EDX': 32, 'DX': 16, 'DL': 8, 'DH': 8, 'RBX': 64, 'EBX': 32, 'BX': 16, 'BL': 8, 'BH': 8, 'RSP': 64, 'ESP': 32, 'SP': 16, 'SPL': 8, 'RBP': 64, 'EBP': 32, 'BP': 16, 'BPL': 8, 'RSI': 64, 'ESI': 32, 'SI': 16, 'SIL': 8, 'RDI': 64, 'EDI': 32, 'DI': 16, 'DIL': 8, 'R8': 64, 'R8D': 32, 'R8W': 16, 'R8B': 8, 'R9': 64, 'R9D': 32, 'R9W': 16, 'R9B': 8, 'R10': 64, 'R10D': 32, 'R10W': 16, 'R10B': 8, 'R11': 64, 'R11D': 32, 'R11W': 16, 'R11B': 8, 'R12': 64, 'R12D': 32, 'R12W': 16, 'R12B': 8, 'R13': 64, 'R13D': 32, 'R13W': 16, 'R13B': 8, 'R14': 64, 'R14D': 32, 'R14W': 16, 'R14B': 8, 'R15': 64, 'R15D': 32, 'R15W': 16, 'R15B': 8, 'ES': 16, 'CS': 16, 'SS': 16, 'DS': 16, 'FS': 16, 'GS': 16, 'RIP': 64, 'EIP': 32, 'IP': 16, 'RFLAGS': 64, 'EFLAGS': 32, 'FLAGS': 16, 'XMM0': 128, 'XMM1': 128, 'XMM2': 128, 'XMM3': 128, 'XMM4': 128, 'XMM5': 128, 'XMM6': 128, 'XMM7': 128, 'XMM8': 128, 'XMM9': 128, 'XMM10': 128, 'XMM11': 128, 'XMM12': 128, 'XMM13': 128, 'XMM14': 128, 'XMM15': 128, 'YMM0': 256, 'YMM1': 256, 'YMM2': 256, 'YMM3': 256, 'YMM4': 256, 'YMM5': 256, 'YMM6': 256, 'YMM7': 256, 'YMM8': 256, 'YMM9': 256, 'YMM10': 256, 'YMM11': 256, 'YMM12': 256, 'YMM13': 256, 'YMM14': 256, 'YMM15': 256}

def to_bytelist(bs):
    if False:
        print('Hello World!')
    return [bytes([b]) for b in bs]

class SymCPUTest(unittest.TestCase):
    _multiprocess_can_split_ = True
    _flag_offsets = {'CF': 0, 'PF': 2, 'AF': 4, 'ZF': 6, 'SF': 7, 'IF': 9, 'DF': 10, 'OF': 11}
    _flags = {'CF': 1, 'PF': 4, 'AF': 16, 'ZF': 64, 'SF': 128, 'DF': 1024, 'OF': 2048, 'IF': 512}

    def assertItemsEqual(self, a, b):
        if False:
            return 10
        self.assertEqual(sorted(a), sorted(b))

    def assertEqItems(self, a, b):
        if False:
            return 10
        if isinstance(b, bytes):
            b = [bytes([x]) for x in b]
        return self.assertItemsEqual(a, b)

    def testInitialRegState(self):
        if False:
            return 10
        cpu = I386Cpu(Memory32())
        values = {'RFLAGS': 0, 'TOP': 7, 'FP0': (0, 0), 'FP1': (0, 0), 'FP2': (0, 0), 'FP3': (0, 0), 'FP4': (0, 0), 'FP5': (0, 0), 'FP6': (0, 0), 'FP7': (0, 0), 'CS': 0, 'SS': 0, 'DS': 0, 'ES': 0}
        for reg_name in cpu.canonical_registers:
            if len(reg_name) > 2:
                v = values.get(reg_name, 0)
                self.assertEqual(cpu.read_register(reg_name), v)

    def testRegisterCacheAccess(self):
        if False:
            print('Hello World!')
        cpu = I386Cpu(Memory32())
        cpu.ESI = 305419896
        self.assertEqual(cpu.ESI, 305419896)
        cpu.SI = 43690
        self.assertEqual(cpu.SI, 43690)
        cpu.RAX = 1311768467732155613
        self.assertEqual(cpu.ESI, 305441450)
        cpu.SI = 43690
        self.assertEqual(cpu.SI, 43690)

    def testFlagAccess(self) -> None:
        if False:
            print('Hello World!')
        cpu = I386Cpu(Memory32())
        cpu.RFLAGS = 0
        self.assertFalse(cpu.CF)
        self.assertFalse(cpu.PF)
        self.assertFalse(cpu.AF)
        self.assertFalse(cpu.ZF)
        self.assertFalse(cpu.SF)
        self.assertFalse(cpu.DF)
        self.assertFalse(cpu.OF)
        cpu.CF = True
        self.assertTrue(cpu.RFLAGS & self._flags['CF'] != 0)
        cpu.CF = False
        self.assertTrue(cpu.RFLAGS & self._flags['CF'] == 0)
        cpu.RFLAGS |= self._flags['CF']
        self.assertTrue(cpu.CF)
        cpu.RFLAGS &= ~self._flags['CF']
        self.assertFalse(cpu.CF)
        cpu.PF = True
        self.assertTrue(cpu.RFLAGS & self._flags['PF'] != 0)
        cpu.PF = False
        self.assertTrue(cpu.RFLAGS & self._flags['PF'] == 0)
        cpu.RFLAGS |= self._flags['PF']
        self.assertTrue(cpu.PF)
        cpu.RFLAGS &= ~self._flags['PF']
        self.assertFalse(cpu.PF)
        cpu.AF = True
        self.assertTrue(cpu.RFLAGS & self._flags['AF'] != 0)
        cpu.AF = False
        self.assertTrue(cpu.RFLAGS & self._flags['AF'] == 0)
        cpu.RFLAGS |= self._flags['AF']
        self.assertTrue(cpu.AF)
        cpu.RFLAGS &= ~self._flags['AF']
        self.assertFalse(cpu.AF)
        cpu.ZF = True
        self.assertTrue(cpu.RFLAGS & self._flags['ZF'] != 0)
        cpu.ZF = False
        self.assertTrue(cpu.RFLAGS & self._flags['ZF'] == 0)
        cpu.RFLAGS |= self._flags['ZF']
        self.assertTrue(cpu.ZF)
        cpu.RFLAGS &= ~self._flags['ZF']
        self.assertFalse(cpu.ZF)
        cpu.SF = True
        self.assertTrue(cpu.RFLAGS & self._flags['SF'] != 0)
        cpu.SF = False
        self.assertTrue(cpu.RFLAGS & self._flags['SF'] == 0)
        cpu.RFLAGS |= self._flags['SF']
        self.assertTrue(cpu.SF)
        cpu.RFLAGS &= ~self._flags['SF']
        self.assertFalse(cpu.SF)
        cpu.DF = True
        self.assertTrue(cpu.RFLAGS & self._flags['DF'] != 0)
        cpu.DF = False
        self.assertTrue(cpu.RFLAGS & self._flags['DF'] == 0)
        cpu.RFLAGS |= self._flags['DF']
        self.assertTrue(cpu.DF)
        cpu.RFLAGS &= ~self._flags['DF']
        self.assertFalse(cpu.DF)
        cpu.OF = True
        self.assertTrue(cpu.RFLAGS & self._flags['OF'] != 0)
        cpu.OF = False
        self.assertTrue(cpu.RFLAGS & self._flags['OF'] == 0)
        cpu.RFLAGS |= self._flags['OF']
        self.assertTrue(cpu.OF)
        cpu.RFLAGS &= ~self._flags['OF']
        self.assertFalse(cpu.OF)

    def _check_flags_CPAZSIDO(self, cpu, c, p, a, z, s, i, d, o) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(cpu.CF, c)
        self.assertEqual(cpu.PF, p)
        self.assertEqual(cpu.AF, a)
        self.assertEqual(cpu.ZF, z)
        self.assertEqual(cpu.SF, s)
        self.assertEqual(cpu.IF, i)
        self.assertEqual(cpu.DF, d)
        self.assertEqual(cpu.OF, o)

    def _construct_flag_bitfield(self, flags):
        if False:
            print('Hello World!')
        return reduce(operator.or_, (self._flags[f] for f in flags))

    def _construct_sym_flag_bitfield(self, flags):
        if False:
            i = 10
            return i + 15
        return reduce(operator.or_, (BitVecConstant(size=32, value=self._flags[f]) for f in flags))

    def test_set_eflags(self) -> None:
        if False:
            return 10
        cpu = I386Cpu(Memory32())
        self.assertEqual(cpu.EFLAGS, 0)
        flags = ['CF', 'PF', 'AF', 'ZF', 'SF']
        cpu.EFLAGS = self._construct_flag_bitfield(flags)
        self._check_flags_CPAZSIDO(cpu, 1, 1, 1, 1, 1, 0, 0, 0)

    def test_get_eflags(self) -> None:
        if False:
            i = 10
            return i + 15
        cpu = I386Cpu(Memory32())
        self.assertEqual(cpu.EFLAGS, 0)
        flags = ['CF', 'AF', 'SF']
        cpu.CF = 1
        cpu.AF = 1
        cpu.SF = 1
        cpu.DF = 0
        self.assertEqual(cpu.EFLAGS, self._construct_flag_bitfield(flags))

    def test_set_sym_eflags(self):
        if False:
            return 10

        def check_flag(obj, flag):
            if False:
                i = 10
                return i + 15
            equal = obj.operands[0]
            extract = equal.operands[0]
            assert isinstance(obj, Bool)
            assert extract.begining == self._flag_offsets[flag]
            assert extract.end == extract.begining
        flags = ['CF', 'PF', 'AF', 'ZF']
        sym_bitfield = self._construct_sym_flag_bitfield(flags)
        cpu = I386Cpu(Memory32())
        cpu.EFLAGS = sym_bitfield
        check_flag(cpu.CF, 'CF')
        check_flag(cpu.PF, 'PF')
        check_flag(cpu.AF, 'AF')
        check_flag(cpu.ZF, 'ZF')

    def test_get_sym_eflags(self):
        if False:
            i = 10
            return i + 15

        def flatten_ors(x: BitVecOr) -> List:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Retrieve all nodes of a BitVecOr expression tree\n            '
            assert isinstance(x, BitVecOr)
            if any((isinstance(op, BitVecOr) for op in x.operands)):
                ret: List = []
                for op in x.operands:
                    if isinstance(op, BitVecOr):
                        ret += flatten_ors(op)
                    else:
                        ret.append(op)
                return ret
            else:
                return list(x.operands)
        cpu = I386Cpu(Memory32())
        cpu.CF = 1
        cpu.AF = 1
        a = BitVecConstant(size=32, value=1) != 0
        b = BitVecConstant(size=32, value=0) != 0
        cpu.ZF = a
        cpu.SF = b
        flags = flatten_ors(cpu.EFLAGS)
        self.assertTrue(isinstance(cpu.EFLAGS, BitVecOr))
        self.assertEqual(len(flags), 8)
        self.assertEqual(cpu.CF, 1)
        self.assertEqual(cpu.AF, 1)
        self.assertIs(cpu.ZF, a)
        self.assertIs(cpu.SF, b)

    def testRegisterAccess(self):
        if False:
            i = 10
            return i + 15
        cpu = I386Cpu(Memory32())
        self.assertEqual(cpu.EAX, 0)
        cpu.EAX += 1
        self.assertEqual(cpu.EAX, 1)
        cpu.EAX = 134217728
        self.assertEqual(cpu.EAX, 134217728)
        cpu.EAX = 4278190080
        self.assertEqual(cpu.EAX, 4278190080)
        cpu.EAX = 16711680
        self.assertEqual(cpu.EAX, 16711680)
        cpu.EAX = 65280
        self.assertEqual(cpu.EAX, 65280)
        cpu.EAX = 255
        self.assertEqual(cpu.EAX, 255)
        cpu.EAX = 4294967296
        self.assertEqual(cpu.EAX, 0)
        cpu.EAX = 287454020
        self.assertEqual(cpu.EAX, 287454020)
        self.assertEqual(cpu.AX, 13124)
        self.assertEqual(cpu.AH, 51)
        self.assertEqual(cpu.AL, 68)
        cpu.AL = 221
        self.assertEqual(cpu.EAX, 287454173)
        self.assertEqual(cpu.AX, 13277)
        self.assertEqual(cpu.AH, 51)
        self.assertEqual(cpu.AL, 221)
        cpu.AH = 204
        self.assertEqual(cpu.EAX, 287493341)
        self.assertEqual(cpu.AX, 52445)
        self.assertEqual(cpu.AH, 204)
        self.assertEqual(cpu.AL, 221)
        cpu.AL = 221
        self.assertEqual(cpu.EAX, 287493341)
        self.assertEqual(cpu.AX, 52445)
        self.assertEqual(cpu.AH, 204)
        self.assertEqual(cpu.AL, 221)
        cpu.EDX = 134515792
        self.assertEqual(cpu.EDX, 134515792)
        self.assertEqual(cpu.ECX, 0)
        cpu.ECX += 1
        self.assertEqual(cpu.ECX, 1)
        cpu.ECX = 134217728
        self.assertEqual(cpu.ECX, 134217728)
        cpu.ECX = 4278190080
        self.assertEqual(cpu.ECX, 4278190080)
        cpu.ECX = 16711680
        self.assertEqual(cpu.ECX, 16711680)
        cpu.ECX = 65280
        self.assertEqual(cpu.ECX, 65280)
        cpu.ECX = 255
        self.assertEqual(cpu.ECX, 255)
        cpu.ECX = 4294967296
        self.assertEqual(cpu.ECX, 0)
        cpu.ECX = 287454020
        self.assertEqual(cpu.ECX, 287454020)
        self.assertEqual(cpu.CX, 13124)
        self.assertEqual(cpu.CH, 51)
        self.assertEqual(cpu.CL, 68)
        cpu.CL = 221
        self.assertEqual(cpu.ECX, 287454173)
        self.assertEqual(cpu.CX, 13277)
        self.assertEqual(cpu.CH, 51)
        self.assertEqual(cpu.CL, 221)
        cpu.CH = 204
        self.assertEqual(cpu.ECX, 287493341)
        self.assertEqual(cpu.CX, 52445)
        self.assertEqual(cpu.CH, 204)
        self.assertEqual(cpu.CL, 221)
        cpu.CL = 221
        self.assertEqual(cpu.ECX, 287493341)
        self.assertEqual(cpu.CX, 52445)
        self.assertEqual(cpu.CH, 204)
        self.assertEqual(cpu.CL, 221)

    def test_le_or(self):
        if False:
            i = 10
            return i + 15
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4096, 4096, 'rwx')
        cpu.write_int(4096, 4702394921427289928, 64)
        cpu.write_int(4096, cpu.read_int(4096, 32) | 0, 32)
        addr1 = cs.new_bitvec(64)
        cs.add(addr1 == 4100)
        cpu.write_int(addr1, 88, 8)
        self.assertEqual(cpu.read_int(4096, 32), 1162233672)
        addr1 = cs.new_bitvec(64)
        cs.add(addr1 == 4096)
        cpu.write_int(addr1, 89, 8)
        solutions = solver.get_all_values(cs, cpu.read_int(4096, 32))
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0], 1162233689)
        cpu.write_int(4096, cpu.read_int(4096, 32) | 0, 32)
        cpu.write_int(4096, cpu.read_int(4096, 32) | 0, 32)
        cpu.write_int(4096, cpu.read_int(4096, 32) | 0, 32)
        solutions = solver.get_all_values(cs, cpu.read_int(4096, 32))
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0], 1162233689)

    def test_cache_001(self):
        if False:
            i = 10
            return i + 15
        cs = ConstraintSet()
        mem = SMemory64(ConstraintSet())
        cpu = AMD64Cpu(mem)
        mem.mmap(4096, 4096, 'rwx')
        cpu.write_int(4096, 4702394921427289928, 64)
        cpu.write_int(4100, 5859837686836516696, 64)
        cpu.write_int(4104, 7017280452245743464, 64)
        self.assertEqual(cpu.read_int(4096, 32), 1162233672)
        cpu.write_int(4096, 1162233672, 32)
        self.assertEqual(cpu.read_int(4096, 32), 1162233672)
        self.assertEqual(cpu.read_int(4100, 32), 1431721816)
        self.assertEqual(cpu.read_int(4104, 32), 1701209960)
        self.assertEqual(cpu.read_int(4104, 64), 7017280452245743464)
        self.assertEqual(cpu.read_int(4096, 64), 6149198377851963208)
        for i in range(16):
            self.assertEqual(mem[i + 4096], b'HGFEXWVUhgfedcba'[i:i + 1])
        self.assertEqual(mem.read(4096, 16), to_bytelist(b'HGFEXWVUhgfedcba'))

    def test_cache_002(self):
        if False:
            print('Hello World!')
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        addr = mem.mmap(4096, 4096, 'rwx')
        self.assertEqual(addr, 4096)
        cpu.write_int(4096, 4702394921427289928, 64)
        cpu.write_int(4100, 5859837686836516696, 64)
        cpu.write_int(4104, 7017280452245743464, 64)
        self.assertEqual(cpu.read_int(4096, 32), 1162233672)
        self.assertEqual(cpu.read_int(4100, 32), 1431721816)
        self.assertEqual(cpu.read_int(4104, 32), 1701209960)
        self.assertEqual(cpu.read_int(4104, 64), 7017280452245743464)
        self.assertEqual(cpu.read_int(4096, 64), 6149198377851963208)
        for i in range(16):
            self.assertEqual(mem[i + 4096], b'HGFEXWVUhgfedcba'[i:i + 1])

    def test_cache_003(self):
        if False:
            for i in range(10):
                print('nop')
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        addr = mem.mmap(4096, 4096, 'rwx')
        self.assertEqual(addr, 4096)
        cpu.write_int(4096, 4702394921427289928, 64)
        cpu.write_int(4104, 7017280452245743464, 64)
        self.assertEqual(cpu.read_int(4096, 64), 4702394921427289928)
        self.assertEqual(cpu.read_int(4104, 64), 7017280452245743464)
        for i in range(8):
            self.assertEqual(cpu.read_int(4096 + i, 8), ord('HGFEDCBA'[i]))
        for i in range(8):
            self.assertEqual(cpu.read_int(4104 + i, 8), ord('hgfedcba'[i]))
        addr1 = cs.new_bitvec(64)
        cs.add(addr1 == 4100)
        cpu.write_int(addr1, 88, 8)
        value = cpu.read_int(4100, 16)
        self.assertItemsEqual(solver.get_all_values(cs, value), [17240])
        addr2 = cs.new_bitvec(64)
        cs.add(Operators.AND(addr2 >= 4096, addr2 <= 4108))
        cpu.write_int(addr2, 22873, 16)
        solutions = solver.get_all_values(cs, cpu.read_int(addr2, 32))
        self.assertEqual(len(solutions), 4108 - 4096 + 1)
        self.assertEqual(set(solutions), set([1162238297, 1094867289, 1480939865, 1701206361, 1734891865, 1129863513, 1749113177, 1111710041, 1718049113, 1650678105, 1684363609, 1667520857, 1633835353]))

    def test_cache_004(self):
        if False:
            i = 10
            return i + 15
        import random
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        addr = mem.mmap(4096, 4096, 'rwx')
        self.assertEqual(addr, 4096)
        memory = bytearray(4096)
        written = set()
        for _ in range(1000):
            address = random.randint(4096, 8192 - 8)
            [written.add(i) for i in range(address, address + 8)]
            value = random.randint(0, 18446744073709551615)
            memory[address - 4096:address - 4096 + 8] = list(struct.pack('<Q', value))
            cpu.write_int(address, value, 64)
            if random.randint(0, 10) > 5:
                cpu.read_int(random.randint(4096, 8192 - 8), random.choice([8, 16, 32, 64]))
        written = list(written)
        random.shuffle(written)
        for address in written:
            size = random.choice([8, 16, 32, 64])
            if address > 8192 - size // 8:
                continue
            pattern = {8: 'B', 16: '<H', 32: '<L', 64: '<Q'}[size]
            start = address - 4096
            self.assertEqual(cpu.read_int(address, size), struct.unpack(pattern, bytes(memory[start:start + size // 8]))[0])

    def test_cache_005(self):
        if False:
            return 10
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        addr = mem.mmap(4096, 4096, 'rwx')
        self.assertEqual(addr, 4096)
        self.assertRaises(Exception, cpu.write_int, 4096 - 1, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.write_int, 8192 - 7, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.read_int, 4096 - 1, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.read_int, 8192 - 7, 293899682605057864, 64)
        addr = mem.mmap(28672, 4096, 'r')
        self.assertEqual(addr, 28672)
        self.assertRaises(Exception, cpu.write_int, 28672 - 1, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.write_int, 32768 - 7, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.read_int, 28672 - 1, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.read_int, 32768 - 7, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.write_int, 28928, 293899682605057864, 64)
        addr = mem.mmap(61440, 4096, 'w')
        self.assertEqual(addr, 61440)
        self.assertRaises(Exception, cpu.write_int, 61440 - 1, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.write_int, 65536 - 7, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.read_int, 61440 - 1, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.read_int, 65536 - 7, 293899682605057864, 64)
        self.assertRaises(Exception, cpu.read_int, 61696, 293899682605057864, 64)

    def test_IDIV_concrete(self):
        if False:
            while True:
                i = 10
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code:code + 3] = '÷}ô'
        cpu.EIP = code
        cpu.EAX = 116
        cpu.EBP = stack + 1792
        cpu.write_int(cpu.EBP - 12, 100, 32)
        cpu.execute()
        self.assertEqual(cpu.EAX, 1)

    def test_IDIV_symbolic(self):
        if False:
            while True:
                i = 10
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code:code + 3] = '÷}ô'
        cpu.EIP = code
        cpu.EAX = cs.new_bitvec(32, 'EAX')
        cs.add(cpu.EAX == 116)
        cpu.EBP = cs.new_bitvec(32, 'EBP')
        cs.add(cpu.EBP == stack + 1792)
        value = cs.new_bitvec(32, 'VALUE')
        cpu.write_int(cpu.EBP - 12, value, 32)
        cs.add(value == 100)
        cpu.execute()
        cs.add(cpu.EAX == 1)
        self.assertTrue(solver.check(cs))

    def test_IDIV_grr001(self):
        if False:
            print('Hello World!')
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        mem[code:code + 2] = '÷ù'
        cpu.EIP = code
        cpu.EAX = 4294967295
        cpu.EDX = 4294967295
        cpu.ECX = 50
        cpu.execute()
        self.assertEqual(cpu.EAX, 0)

    def test_IDIV_grr001_symbolic(self):
        if False:
            i = 10
            return i + 15
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        mem[code:code + 2] = '÷ù'
        cpu.EIP = code
        cpu.EAX = cs.new_bitvec(32, 'EAX')
        cs.add(cpu.EAX == 4294967295)
        cpu.EDX = cs.new_bitvec(32, 'EDX')
        cs.add(cpu.EDX == 4294967295)
        cpu.ECX = cs.new_bitvec(32, 'ECX')
        cs.add(cpu.ECX == 50)
        cpu.execute()
        cs.add(cpu.EAX == 0)
        self.assertTrue(solver.check(cs))

    def test_ADC_001(self):
        if False:
            for i in range(10):
                print('nop')
        'INSTRUCTION: 0x0000000067756f91:\tadc\tesi, edx'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        mem[code:code + 2] = '\x13ò'
        cpu.EIP = code
        cpu.ESI = 0
        cpu.EDX = 4294967295
        cpu.CF = True
        cpu.execute()
        self.assertEqual(cpu.EDX, 4294967295)
        self.assertEqual(cpu.ESI, 0)
        self.assertEqual(cpu.CF, True)

    def test_ADC_001_symbolic(self):
        if False:
            while True:
                i = 10
        'INSTRUCTION: 0x0000000067756f91:\tadc\tesi, edx'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        mem[code:code + 2] = '\x13ò'
        cpu.EIP = code
        cpu.ESI = cs.new_bitvec(32, 'ESI')
        cs.add(cpu.ESI == 0)
        cpu.EDX = cs.new_bitvec(32, 'EDX')
        cs.add(cpu.EDX == 4294967295)
        cpu.CF = cs.new_bool('CF')
        cs.add(cpu.CF)
        cpu.execute()
        cs.add(cpu.ESI == 0)
        cs.add(cpu.EDX == 4294967295)
        cs.add(cpu.CF)
        self.assertTrue(solver.check(cs))

    def test_AND_1(self):
        if False:
            while True:
                i = 10
        'Instruction AND\n        Groups:\n        0x7ffff7de390a:     and rax, 0xfc000000\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351921664, 4096, 'rwx')
        mem[140737351923978] = 'H'
        mem[140737351923979] = '%'
        mem[140737351923980] = '\x00'
        mem[140737351923981] = '\x00'
        mem[140737351923982] = '\x00'
        mem[140737351923983] = 'ü'
        cpu.PF = True
        cpu.RAX = 140737354102360
        cpu.OF = False
        cpu.ZF = False
        cpu.CF = False
        cpu.RIP = 140737351923978
        cpu.SF = False
        cpu.execute()
        self.assertEqual(mem[140737351923978], b'H')
        self.assertEqual(mem[140737351923979], b'%')
        self.assertEqual(mem[140737351923980], b'\x00')
        self.assertEqual(mem[140737351923981], b'\x00')
        self.assertEqual(mem[140737351923982], b'\x00')
        self.assertEqual(mem[140737351923983], b'\xfc')
        self.assertEqual(cpu.PF, True)
        self.assertEqual(cpu.RAX, 140737287028736)
        self.assertEqual(cpu.OF, False)
        self.assertEqual(cpu.ZF, False)
        self.assertEqual(cpu.CF, False)
        self.assertEqual(cpu.RIP, 140737351923984)
        self.assertEqual(cpu.SF, False)

    def test_CMPXCHG8B_symbolic(self):
        if False:
            print('Hello World!')
        'CMPXCHG8B'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        data = mem.mmap(8192, 4096, 'rwx')
        mem[code:code + 5] = 'ð\x0fÇ\x0f;'
        cpu.EIP = code
        cpu.EDI = cs.new_bitvec(32, 'EDI')
        cs.add(Operators.OR(cpu.EDI == 8192, cpu.EDI == 8448, cpu.EDI == 8704))
        self.assertEqual(sorted(solver.get_all_values(cs, cpu.EDI)), [8192, 8448, 8704])
        self.assertEqual(cpu.read_int(8192, 64), 0)
        self.assertEqual(cpu.read_int(8448, 64), 0)
        self.assertEqual(cpu.read_int(8704, 64), 0)
        self.assertItemsEqual(solver.get_all_values(cs, cpu.read_int(cpu.EDI, 64)), [0])
        cpu.write_int(8448, 4702394921427289928, 64)
        cpu.EAX = cs.new_bitvec(32, 'EAX')
        cs.add(Operators.OR(cpu.EAX == 1094861636, cpu.EAX == 195948557, cpu.EAX == 4160223223))
        cpu.EDX = 1162233672
        cpu.execute()
        self.assertTrue(solver.check(cs))
        self.assertItemsEqual(solver.get_all_values(cs, cpu.read_int(cpu.EDI, 64)), [0, 4702394921427289928])

    def test_POPCNT(self):
        if False:
            return 10
        'POPCNT EAX, EAX\n        CPU Dump\n        Address   Hex dump\n        00333689  F3 0F B8 C0\n        '
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        mem[code:code + 4] = 'ó\x0f¸À'
        cpu.EIP = code
        cpu.EAX = 1968323635
        cpu.execute()
        self.assertEqual(cpu.EAX, 16)
        self.assertEqual(cpu.ZF, False)

    def test_DEC_1(self):
        if False:
            i = 10
            return i + 15
        'Instruction DEC_1\n        Groups: mode64\n        0x41e10a:   dec     ecx\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4317184, 4096, 'rwx')
        mem[4317450] = 'ÿ'
        mem[4317451] = 'É'
        cpu.AF = False
        cpu.OF = False
        cpu.ZF = False
        cpu.RIP = 4317450
        cpu.PF = False
        cpu.SF = False
        cpu.ECX = 13
        cpu.execute()
        self.assertItemsEqual(mem[4317450:4317452], to_bytelist(b'\xff\xc9'))
        self.assertEqual(cpu.AF, False)
        self.assertEqual(cpu.OF, False)
        self.assertEqual(cpu.ZF, False)
        self.assertEqual(cpu.RIP, 4317452)
        self.assertEqual(cpu.PF, True)
        self.assertEqual(cpu.SF, False)
        self.assertEqual(cpu.ECX, 12)

    def test_PUSHFD_1(self):
        if False:
            return 10
        'Instruction PUSHFD_1\n        Groups: not64bitmode\n        0x8065f6f:  pushfd\n        '
        mem = Memory32()
        cpu = I386Cpu(mem)
        mem.mmap(134631424, 4096, 'rwx')
        mem.mmap(4294950912, 4096, 'rwx')
        mem[4294952448:4294952457] = b'\x00\x00\x00\x00\x02\x03\x00\x00\x00'
        mem[134635375] = b'\x9c'
        cpu.EIP = 134635375
        cpu.EBP = 4294948352
        cpu.ESP = 4294952452
        cpu.CF = True
        cpu.OF = True
        cpu.AF = True
        cpu.ZF = True
        cpu.PF = True
        cpu.execute()
        self.assertItemsEqual(mem[4294952448:4294952457], to_bytelist(b'U\x08\x00\x00\x02\x03\x00\x00\x00'))
        self.assertEqual(mem[134635375], b'\x9c')
        self.assertEqual(cpu.EIP, 134635376)
        self.assertEqual(cpu.EBP, 4294948352)
        self.assertEqual(cpu.ESP, 4294952448)

    def test_XLATB_1(self):
        if False:
            i = 10
            return i + 15
        'Instruction XLATB_1\n        Groups:\n        0x8059a8d: xlatb\n        '
        mem = Memory32()
        cpu = I386Cpu(mem)
        mem.mmap(134582272, 4096, 'rwx')
        mem.mmap(4294955008, 4096, 'rwx')
        mem[134584973] = b'\xd7'
        mem[4294955018] = b'A'
        cpu.EBX = 4294955008
        cpu.AL = 10
        cpu.EIP = 134584973
        cpu.execute()
        self.assertEqual(mem[134584973], b'\xd7')
        self.assertEqual(mem[4294955018], b'A')
        self.assertEqual(cpu.AL, 65)
        self.assertEqual(cpu.EIP, 134584974)

    def test_XLATB_1_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction XLATB_1\n        Groups:\n        0x8059a8d: xlatb\n        '
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        mem.mmap(134582272, 4096, 'rwx')
        mem.mmap(4294955008, 4096, 'rwx')
        mem[134584973] = '×'
        mem[4294955018] = 'A'
        cpu.EIP = 134584973
        cpu.AL = 10
        cpu.EBX = 4294955008

    def test_SAR_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction SAR_1\n                    Groups: mode64\n                    0x41e10a:   SAR     cl, EBX\n        Using the SAR instruction to perform a division operation does not produce the same result as the IDIV instruction. The quotient from the IDIV instruction is rounded toward zero, whereas the "quotient" of the SAR instruction is rounded toward negative infinity. This difference is apparent only for negative numbers. For example, when the IDIV instruction is used to divide -9 by 4, the result is -2 with a remainder of -1. If the SAR instruction is used to shift -9 right by two bits, the result is -3 and the "remainder" is +3; however, the SAR instruction stores only the most significant bit of the remainder (in the CF flag).\n\n        '
        mem = Memory32()
        cpu = I386Cpu(mem)
        mem.mmap(4317184, 4096, 'rwx')
        mem[4317450] = 'Á'
        mem[4317451] = 'ø'
        mem[4317452] = '\x02'
        cpu.RIP = 4317450
        cpu.PF = True
        cpu.SF = True
        cpu.ZF = False
        cpu.AF = False
        cpu.OF = False
        cpu.EAX = 4294967287
        cpu.execute()
        self.assertEqual(cpu.CF, True)
        self.assertEqual(cpu.SF, True)
        self.assertEqual(cpu.PF, False)
        self.assertEqual(cpu.ZF, False)
        self.assertEqual(cpu.EAX, 4294967293)

    def test_SAR_1_symbolic(self):
        if False:
            while True:
                i = 10
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        mem.mmap(4317184, 4096, 'rwx')
        mem[4317450] = 'Á'
        mem[4317451] = 'ø'
        mem[4317452] = '\x02'
        cpu.RIP = 4317450
        cpu.PF = cs.new_bool()
        cs.add(cpu.PF == True)
        cpu.SF = cs.new_bool()
        cs.add(cpu.SF == True)
        cpu.ZF = cs.new_bool()
        cs.add(cpu.ZF == False)
        cpu.AF = cs.new_bool()
        cs.add(cpu.AF == False)
        cpu.OF = cs.new_bool()
        cs.add(cpu.OF == False)
        cpu.EAX = cs.new_bitvec(32)
        cs.add(cpu.EAX == 4294967287)
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.EAX == 4294967293)
        condition = Operators.AND(condition, cpu.ZF == False)
        condition = Operators.AND(condition, cpu.CF == True)
        condition = Operators.AND(condition, cpu.SF == True)
        condition = Operators.AND(condition, cpu.PF == False)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_SAR_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction SAR_2'
        mem = Memory32()
        cpu = I386Cpu(mem)
        mem.mmap(4317184, 4096, 'rwx')
        mem[4317450] = 'À'
        mem[4317451] = 'ø'
        mem[4317452] = '\x9f'
        cpu.RIP = 4317450
        cpu.CF = True
        cpu.SF = True
        cpu.ZF = False
        cpu.AF = False
        cpu.OF = False
        cpu.PF = False
        cpu.EAX = 4294967293
        cpu.execute()
        self.assertEqual(cpu.PF, True)
        self.assertEqual(cpu.SF, True)
        self.assertEqual(cpu.ZF, False)
        self.assertEqual(cpu.OF, False)
        self.assertEqual(cpu.AF, False)
        self.assertEqual(cpu.EAX, 4294967295)

    def test_SAR_2_symbolicsa(self):
        if False:
            for i in range(10):
                print('nop')
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        mem.mmap(4317184, 4096, 'rwx')
        mem[4317450] = 'À'
        mem[4317451] = 'ø'
        mem[4317452] = 'ÿ'
        cpu.RIP = 4317450
        cpu.PF = cs.new_bool()
        cs.add(cpu.PF == True)
        cpu.CF = cs.new_bool()
        cs.add(cpu.CF == False)
        cpu.SF = cs.new_bool()
        cs.add(cpu.SF == True)
        cpu.ZF = cs.new_bool()
        cs.add(cpu.ZF == False)
        cpu.AF = cs.new_bool()
        cs.add(cpu.AF == False)
        cpu.OF = cs.new_bool()
        cs.add(cpu.OF == False)
        cpu.EAX = cs.new_bitvec(32)
        cs.add(cpu.EAX == 4294967295)
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.EAX == 4294967295)
        condition = Operators.AND(condition, cpu.ZF == False)
        condition = Operators.AND(condition, cpu.PF == True)
        condition = Operators.AND(condition, cpu.SF == True)
        condition = Operators.AND(condition, cpu.CF == True)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_SAR_3_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction SAR_6\n        eax            0xffffd000   -12288\n        ecx            0x3d1ce0ff   1025302783\n        eip            0x80483f3    0x80483f3\n        eflags         0x287        [ CF PF SF IF ]\n        0xffffd000: 0x8f\n\n        => 0x80483f0 <main+3>:      sarb   %cl,0x0(%eax)\n\n        eax            0xffffd000   -12288\n        ecx            0x3d1ce0ff   1025302783\n        eip            0x80483f4    0x80483f4\n        eflags         0x287        [ CF PF SF IF ]\n        0xffffd000: 0xff\n\n        '
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        mem.mmap(134533120, 4096, 'rwx')
        mem.mmap(4294946816, 4096, 'rwx')
        mem[134534656] = 'Ò'
        mem[134534657] = 'x'
        mem[134534658] = '\x00'
        mem[134534659] = 'ÿ'
        addr = cs.new_bitvec(32)
        cs.add(addr == 4294946816)
        value = cs.new_bitvec(8)
        cs.add(value == 143)
        mem[addr] = value
        cpu.EAX = cs.new_bitvec(32)
        cs.add(cpu.EAX == 4294946816)
        cpu.CL = cs.new_bitvec(8)
        cs.add(cpu.CL == 255)
        cpu.EIP = 134534656
        cpu.CF = cs.new_bool()
        cs.add(cpu.CF == True)
        cpu.PF = cs.new_bool()
        cs.add(cpu.PF == True)
        cpu.SF = cs.new_bool()
        cs.add(cpu.SF == True)
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4294946816, 8) == 255)
        condition = Operators.AND(condition, cpu.CL == 255)
        condition = Operators.AND(condition, cpu.CF == True)
        condition = Operators.AND(condition, cpu.PF == True)
        condition = Operators.AND(condition, cpu.SF == True)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPS_1(self):
        if False:
            print('Hello World!')
        mem = Memory32()
        cpu = I386Cpu(mem)
        mem.mmap(4317184, 4096, 'rwx')
        mem[4317184] = '@'
        mem[4317185] = 'H'
        mem[4317186] = 'õ'
        mem[4317187] = 'Ã'
        mem[4317188] = '@'
        mem[4317189] = 'È'
        mem[4317190] = 'õ'
        mem[4317191] = 'Ã'
        mem[4317450] = '\x0f'
        mem[4317451] = '\x16'
        mem[4317452] = '\x00'
        cpu.RIP = 4317450
        cpu.EAX = 4317184
        cpu.XMM0 = 18446744073709551615
        cpu.execute()
        self.assertEqual(cpu.XMM0, 260475633521568864182627747747107700735)

    def test_MOVHPS_2(self):
        if False:
            print('Hello World!')
        mem = Memory32()
        cpu = I386Cpu(mem)
        mem.mmap(4317184, 4096, 'rwx')
        mem[4317450] = '\x0f'
        mem[4317451] = '\x17'
        mem[4317452] = '\x08'
        cpu.RIP = 4317450
        cpu.EAX = 4317184
        cpu.XMM1 = 85449421763943694783736697154757984255
        cpu.execute()
        self.assertItemsEqual(mem[4317184:4317188], to_bytelist(b'@\xc8\xf5\xc3'))
        self.assertItemsEqual(mem[4317188:4317192], to_bytelist(b'@H\xf5\xc3'))

    def test_symbolic_instruction(self):
        if False:
            i = 10
            return i + 15
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code] = BitVecConstant(size=8, value=144)
        cpu.EIP = code
        cpu.EAX = 116
        cpu.EBP = stack + 1792
        cpu.write_int(cpu.EBP - 12, 100, 32)
        cpu.execute()
        self.assertEqual(cpu.EIP, code + 1)

    def test_AAA_0(self):
        if False:
            return 10
        'ASCII Adjust AL after subtraction.'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code] = BitVecConstant(size=8, value=55)
        cpu.EIP = code
        AL = 10
        AH = 65
        AF = False
        cpu.AL = AL
        cpu.AH = AH
        cpu.AF = False
        cpu.execute()
        self.assertEqual(cpu.AL, 0)
        self.assertEqual(cpu.AH, AH + 1)
        self.assertEqual(cpu.AF, True)
        self.assertEqual(cpu.CF, True)

    def test_AAA_1(self):
        if False:
            while True:
                i = 10
        'ASCII Adjust AL after subtraction.'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code] = BitVecConstant(size=8, value=55)
        cpu.EIP = code
        AL = 18
        AH = 65
        AF = False
        cpu.AL = AL
        cpu.AH = AH
        cpu.AF = False
        cpu.execute()
        self.assertEqual(cpu.AL, AL & 15)
        self.assertEqual(cpu.AF, False)
        self.assertEqual(cpu.CF, False)

    def test_AAS_0(self):
        if False:
            i = 10
            return i + 15
        'ASCII Adjust AL after subtraction.'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code] = BitVecConstant(size=8, value=63)
        cpu.EIP = code
        AL = 10
        AH = 65
        AF = False
        cpu.AL = AL
        cpu.AH = AH
        cpu.AF = False
        cpu.execute()
        self.assertEqual(cpu.AL, AL - 6 & 15)
        self.assertEqual(cpu.AH, AH - 1)
        self.assertEqual(cpu.AF, True)
        self.assertEqual(cpu.CF, True)

    def test_AAS_1(self):
        if False:
            print('Hello World!')
        'ASCII Adjust AL after subtraction.'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code] = BitVecConstant(size=8, value=63)
        cpu.EIP = code
        AL = 18
        AH = 65
        AF = False
        cpu.AL = AL
        cpu.AH = AH
        cpu.AF = False
        cpu.execute()
        self.assertEqual(cpu.AL, AL & 15)
        self.assertEqual(cpu.AF, False)
        self.assertEqual(cpu.CF, False)

    def test_DAA_0(self):
        if False:
            print('Hello World!')
        'Decimal Adjust AL after Addition.'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code] = BitVecConstant(size=8, value=39)
        cpu.EIP = code
        cpu.AL = 174
        cpu.BL = 53
        cpu.OF = True
        cpu.SF = True
        cpu.ZF = False
        cpu.AF = False
        cpu.PF = False
        cpu.CF = False
        cpu.execute()
        self.assertEqual(cpu.AL, 20)
        self.assertEqual(cpu.BL, 53)
        self.assertEqual(cpu.SF, False)
        self.assertEqual(cpu.ZF, False)
        self.assertEqual(cpu.AF, True)
        self.assertEqual(cpu.PF, True)
        self.assertEqual(cpu.CF, True)

    def test_DAS_0(self):
        if False:
            print('Hello World!')
        'Decimal Adjust AL after Subtraction.'
        cs = ConstraintSet()
        mem = SMemory32(cs)
        cpu = I386Cpu(mem)
        code = mem.mmap(4096, 4096, 'rwx')
        stack = mem.mmap(61440, 4096, 'rw')
        mem[code] = BitVecConstant(size=8, value=47)
        cpu.EIP = code
        cpu.AL = 174
        cpu.OF = True
        cpu.SF = True
        cpu.ZF = False
        cpu.AF = False
        cpu.PF = False
        cpu.CF = False
        cpu.execute()
        self.assertEqual(cpu.AL, 72)
        self.assertEqual(cpu.SF, False)
        self.assertEqual(cpu.ZF, False)
        self.assertEqual(cpu.AF, True)
        self.assertEqual(cpu.PF, True)
        self.assertEqual(cpu.CF, True)
if __name__ == '__main__':
    unittest.main()