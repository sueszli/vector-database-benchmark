import binascii
import copy
import unittest
from capstone import CS_MODE_ARM
from functools import wraps
from manticore.core.smtlib import ConstraintSet, Z3Solver, BitVecConstant, BitVecVariable, Operators
from manticore.native.memory import SMemory64, Memory64
from manticore.native.cpu.aarch64 import Aarch64Cpu as Cpu, Aarch64RegisterFile
from manticore.native.cpu.abstractcpu import Interruption, InstructionNotImplementedError, ConcretizeRegister
from manticore.native.cpu.bitwise import LSL, Mask
from manticore.utils.fallback_emulator import UnicornEmulator
from tests.native.test_aarch64rf import MAGIC_64, MAGIC_32
from tests.native.aarch64cpu_asm_cache import assembly_cache
ks = None

def _ks_assemble(asm: str) -> bytes:
    if False:
        return 10
    'Assemble the given string using Keystone.'
    global ks
    from keystone import Ks, KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN
    if ks is None:
        ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
    ords = ks.asm(asm)[0]
    if not ords:
        raise Exception(f'bad assembly: {asm}')
    return binascii.hexlify(bytearray(ords))

def assemble(asm: str) -> bytes:
    if False:
        while True:
            i = 10
    '\n    Assemble the given string.\n\n    An assembly cache is first checked, and if there is no entry there, then Keystone is used.\n    '
    if asm in assembly_cache:
        return binascii.unhexlify(assembly_cache[asm])
    return binascii.unhexlify(_ks_assemble(asm))

def itest_setregs(*preds):
    if False:
        i = 10
        return i + 15

    def instr_dec(custom_func):
        if False:
            i = 10
            return i + 15

        @wraps(custom_func)
        def wrapper(self):
            if False:
                while True:
                    i = 10
            for p in preds:
                (dest, src) = p.split('=')
                try:
                    src = int(src, 0)
                except ValueError:
                    self.fail()
                self._setreg(dest, src)
            custom_func(self)
        return wrapper
    return instr_dec

def skip_sym(msg):
    if False:
        return 10

    def instr_dec(assertions_func):
        if False:
            for i in range(10):
                print('nop')

        @wraps(assertions_func)
        def wrapper(self):
            if False:
                i = 10
                return i + 15
            if self.__class__.__name__ == 'Aarch64SymInstructions':
                self.skipTest(msg)
        return wrapper
    return instr_dec

def itest(asm):
    if False:
        while True:
            i = 10

    def instr_dec(assertions_func):
        if False:
            i = 10
            return i + 15

        @wraps(assertions_func)
        def wrapper(self):
            if False:
                return 10
            self._setupCpu(asm)
            self._execute()
            assertions_func(self)
        return wrapper
    return instr_dec

def itest_custom(asms, multiple_insts=False):
    if False:
        while True:
            i = 10

    def instr_dec(custom_func):
        if False:
            i = 10
            return i + 15

        @wraps(custom_func)
        def wrapper(self):
            if False:
                return 10
            self._setupCpu(asms, mode=CS_MODE_ARM, multiple_insts=multiple_insts)
            custom_func(self)
        return wrapper
    return instr_dec

def itest_multiple(asms, count=None):
    if False:
        for i in range(10):
            print('nop')

    def instr_dec(assertions_func):
        if False:
            for i in range(10):
                print('nop')

        @wraps(assertions_func)
        def wrapper(self):
            if False:
                i = 10
                return i + 15
            self._setupCpu(asms, mode=CS_MODE_ARM, multiple_insts=True)
            for i in range(count if count else len(asms)):
                self._execute()
            assertions_func(self)
        return wrapper
    return instr_dec
NZCV_COND_MAP = {'eq': (1073741824, 2952790016), 'ne': (2952790016, 1073741824), 'cs': (536870912, 3489660928), 'hs': (536870912, 3489660928), 'cc': (3489660928, 536870912), 'lo': (3489660928, 536870912), 'mi': (2147483648, 1879048192), 'pl': (1879048192, 2147483648), 'vs': (268435456, 3758096384), 'vc': (3758096384, 268435456), 'hi': (536870912, 1073741824), 'ls': (1073741824, 536870912), 'ge': (3489660928, 3221225472), 'lt': (3221225472, 3489660928), 'gt': (2415919104, 3489660928), 'le': (3489660928, 2415919104), 'al': (4026531840, None), 'nv': (0, None)}

def testRegisterFileCopy():
    if False:
        i = 10
        return i + 15
    regfile = Aarch64RegisterFile()
    regfile.write('PC', 1234)
    regfile.write('X0', BitVecConstant(size=64, value=24))
    regfile.write('X1', BitVecVariable(size=64, name='b'))
    new_regfile = copy.copy(regfile)
    assert new_regfile.read('PC') == 1234
    assert new_regfile.read('X0') is regfile.read('X0')
    assert new_regfile.read('X0') == regfile.read('X0')
    assert new_regfile.read('X1') is regfile.read('X1')
    assert new_regfile.read('X1') == regfile.read('X1')
    rax_val = regfile.read('X0')
    regfile.write('PC', Operators.ITEBV(64, rax_val == 0, 4321, 1235))
    regfile.write('X0', rax_val * 2)
    assert new_regfile.read('PC') is not regfile.read('PC')
    assert new_regfile.read('PC') != regfile.read('PC')
    assert new_regfile.read('PC') == 1234
    assert new_regfile.read('X0') is not regfile.read('X0')
    assert new_regfile.read('X0') != regfile.read('X0')
    assert new_regfile.read('X0') is rax_val
    assert new_regfile.read('X0') == rax_val

class Aarch64CpuTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        if False:
            return 10
        cs = ConstraintSet()
        self.cpu = Cpu(SMemory64(cs))
        self.rf = self.cpu.regfile
        self._setupStack()

    def _setupStack(self):
        if False:
            i = 10
            return i + 15
        self.stack = self.cpu.memory.mmap(61440, 4096, 'rw')
        self.rf.write('SP', self.stack + 4096)

    def test_read_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)

    def test_read_stack(self):
        if False:
            while True:
                i = 10
        self.cpu.STACK = 4919
        self.assertEqual(self.rf.read('SP'), 4919)

    def test_read_stack2(self):
        if False:
            i = 10
            return i + 15
        self.cpu.STACK = 4919 - 1
        self.assertEqual(self.rf.read('SP'), 4918)

    def test_read_stack3(self):
        if False:
            return 10
        self.cpu.STACK = 4919 + 1
        self.assertEqual(self.rf.read('SP'), 4920)

    def test_read_stack4(self):
        if False:
            return 10
        self.cpu.STACK = 4919
        self.assertEqual(self.cpu.STACK, 4919)

    def test_write_read_int(self):
        if False:
            print('Hello World!')
        self.cpu.STACK -= 8
        self.cpu.write_int(self.cpu.STACK, MAGIC_64, 64)
        self.assertEqual(self.cpu.read_int(self.cpu.STACK), MAGIC_64)

class Aarch64Instructions:
    _multiprocess_can_split_ = True

    def _setupCpu(self, asm, mode=CS_MODE_ARM, multiple_insts=False):
        if False:
            while True:
                i = 10
        if mode != CS_MODE_ARM:
            raise Exception(f"Unsupported mode: '{mode}'")
        self.code = self.mem.mmap(4096, 4096, 'rwx')
        self.data = self.mem.mmap(53248, 4096, 'rw')
        self.stack = self.mem.mmap(9223372036854771712, 4096, 'rw')
        start = self.code
        if multiple_insts:
            offset = 0
            for asm_single in asm:
                asm_inst = assemble(asm_single)
                self.mem.write(start + offset, asm_inst)
                offset += len(asm_inst)
        else:
            self.mem.write(start, assemble(asm))
        self.rf.write('PC', start)
        self.rf.write('SP', self.stack + 4096 - 8)
        self.cpu.mode = mode

    def _setreg(self, reg, val):
        if False:
            return 10
        reg = reg.upper()
        if self.mem.__class__.__name__ == 'Memory64':
            self.rf.write(reg, val)
        elif self.mem.__class__.__name__ == 'SMemory64':
            size = self.rf.size(reg)
            self.rf.write(reg, self.cs.new_bitvec(size, name=reg))
            self.cs.add(self.rf.read(reg) == val)
        else:
            self.fail()

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('add w0, w1, w2, uxtb')
    def test_add_ext_reg_uxtb32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861768)
        self.assertEqual(self.rf.read('W0'), 1094861768)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('add w0, w1, w2, uxtb #0')
    def test_add_ext_reg_uxtb0_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861768)
        self.assertEqual(self.rf.read('W0'), 1094861768)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('add w0, w1, w2, uxtb #4')
    def test_add_ext_reg_uxtb4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094863748)
        self.assertEqual(self.rf.read('W0'), 1094863748)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('add w0, w1, w2, uxth')
    def test_add_ext_reg_uxth32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094895256)
        self.assertEqual(self.rf.read('W0'), 1094895256)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('add w0, w1, w2, uxth #0')
    def test_add_ext_reg_uxth0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094895256)
        self.assertEqual(self.rf.read('W0'), 1094895256)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('add w0, w1, w2, uxth #4')
    def test_add_ext_reg_uxth4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1095399556)
        self.assertEqual(self.rf.read('W0'), 1095399556)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, uxtw')
    def test_add_ext_reg_uxtw32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, uxtw #0')
    def test_add_ext_reg_uxtw0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, uxtw #4')
    def test_add_ext_reg_uxtw4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, uxtx')
    def test_add_ext_reg_uxtx32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, uxtx #0')
    def test_add_ext_reg_uxtx0_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, uxtx #4')
    def test_add_ext_reg_uxtx4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('add w0, w1, w2, sxtb')
    def test_add_ext_reg_sxtb32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861512)
        self.assertEqual(self.rf.read('W0'), 1094861512)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('add w0, w1, w2, sxtb #0')
    def test_add_ext_reg_sxtb0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861512)
        self.assertEqual(self.rf.read('W0'), 1094861512)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('add w0, w1, w2, sxtb #4')
    def test_add_ext_reg_sxtb4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094859652)
        self.assertEqual(self.rf.read('W0'), 1094859652)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('add w0, w1, w2, sxth')
    def test_add_ext_reg_sxth32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094829720)
        self.assertEqual(self.rf.read('W0'), 1094829720)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('add w0, w1, w2, sxth #0')
    def test_add_ext_reg_sxth0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094829720)
        self.assertEqual(self.rf.read('W0'), 1094829720)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('add w0, w1, w2, sxth #4')
    def test_add_ext_reg_sxth4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094350980)
        self.assertEqual(self.rf.read('W0'), 1094350980)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, sxtw')
    def test_add_ext_reg_sxtw32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, sxtw #0')
    def test_add_ext_reg_sxtw0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, sxtw #4')
    def test_add_ext_reg_sxtw4_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, sxtx')
    def test_add_ext_reg_sxtx32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, sxtx #0')
    def test_add_ext_reg_sxtx0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, sxtx #4')
    def test_add_ext_reg_sxtx4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, lsl #0')
    def test_add_ext_reg_lsl0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('add w0, w1, w2, lsl #4')
    def test_add_ext_reg_lsl4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('add x0, x1, w2, uxtb')
    def test_add_ext_reg_uxtb64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427290060)
        self.assertEqual(self.rf.read('W0'), 1162233804)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('add x0, x1, w2, uxtb #0')
    def test_add_ext_reg_uxtb0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427290060)
        self.assertEqual(self.rf.read('W0'), 1162233804)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('add x0, x1, w2, uxtb #4')
    def test_add_ext_reg_uxtb4_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427292040)
        self.assertEqual(self.rf.read('W0'), 1162235784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('add x0, x1, w2, uxth')
    def test_add_ext_reg_uxth64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427323548)
        self.assertEqual(self.rf.read('W0'), 1162267292)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('add x0, x1, w2, uxth #0')
    def test_add_ext_reg_uxth0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427323548)
        self.assertEqual(self.rf.read('W0'), 1162267292)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('add x0, x1, w2, uxth #4')
    def test_add_ext_reg_uxth4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427827848)
        self.assertEqual(self.rf.read('W0'), 1162771592)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('add x0, x1, w2, uxtw')
    def test_add_ext_reg_uxtw64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394923596946076)
        self.assertEqual(self.rf.read('W0'), 3331889820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('add x0, x1, w2, uxtw #0')
    def test_add_ext_reg_uxtw0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394923596946076)
        self.assertEqual(self.rf.read('W0'), 3331889820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('add x0, x1, w2, uxtw #4')
    def test_add_ext_reg_uxtw4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394956141788296)
        self.assertEqual(self.rf.read('W0'), 1516993672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('add x0, x1, x2, uxtx')
    def test_add_ext_reg_uxtx64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('add x0, x1, x2, uxtx #0')
    def test_add_ext_reg_uxtx0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('add x0, x1, x2, uxtx #4')
    def test_add_ext_reg_uxtx4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 6226077542263798984)
        self.assertEqual(self.rf.read('W0'), 2594946248)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('add x0, x1, w2, sxtb')
    def test_add_ext_reg_sxtb64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289804)
        self.assertEqual(self.rf.read('W0'), 1162233548)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('add x0, x1, w2, sxtb #0')
    def test_add_ext_reg_sxtb0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289804)
        self.assertEqual(self.rf.read('W0'), 1162233548)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('add x0, x1, w2, sxtb #4')
    def test_add_ext_reg_sxtb4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427287944)
        self.assertEqual(self.rf.read('W0'), 1162231688)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('add x0, x1, w2, sxth')
    def test_add_ext_reg_sxth64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427258012)
        self.assertEqual(self.rf.read('W0'), 1162201756)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('add x0, x1, w2, sxth #0')
    def test_add_ext_reg_sxth0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427258012)
        self.assertEqual(self.rf.read('W0'), 1162201756)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('add x0, x1, w2, sxth #4')
    def test_add_ext_reg_sxth4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921426779272)
        self.assertEqual(self.rf.read('W0'), 1161723016)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('add x0, x1, w2, sxtw')
    def test_add_ext_reg_sxtw64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394919301978780)
        self.assertEqual(self.rf.read('W0'), 3331889820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('add x0, x1, w2, sxtw #0')
    def test_add_ext_reg_sxtw0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394919301978780)
        self.assertEqual(self.rf.read('W0'), 3331889820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('add x0, x1, w2, sxtw #4')
    def test_add_ext_reg_sxtw4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394887422311560)
        self.assertEqual(self.rf.read('W0'), 1516993672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('add x0, x1, x2, sxtx')
    def test_add_ext_reg_sxtx64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('add x0, x1, x2, sxtx #0')
    def test_add_ext_reg_sxtx0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('add x0, x1, x2, sxtx #4')
    def test_add_ext_reg_sxtx4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 6226077542263798984)
        self.assertEqual(self.rf.read('W0'), 2594946248)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('add x0, x1, x2, lsl #0')
    def test_add_ext_reg_lsl0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('add x0, x1, x2, lsl #4')
    def test_add_ext_reg_lsl4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 6226077542263798984)
        self.assertEqual(self.rf.read('W0'), 2594946248)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('add w0, w1, #0')
    def test_add_imm_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('add w0, w1, #4095')
    def test_add_imm_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094865731)
        self.assertEqual(self.rf.read('W0'), 1094865731)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('add w0, w1, #1')
    def test_add_imm32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('add w0, w1, #1, lsl #0')
    def test_add_imm_lsl0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('add w0, w1, #1, lsl #12')
    def test_add_imm_lsl12_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094865732)
        self.assertEqual(self.rf.read('W0'), 1094865732)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('add x0, x1, #0')
    def test_add_imm_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('add x0, x1, #4095')
    def test_add_imm_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427294023)
        self.assertEqual(self.rf.read('W0'), 1162237767)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('add x0, x1, #1')
    def test_add_imm64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('add x0, x1, #1, lsl #0')
    def test_add_imm_lsl0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('add x0, x1, #1, lsl #12')
    def test_add_imm_lsl12_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427294024)
        self.assertEqual(self.rf.read('W0'), 1162237768)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('add w0, w1, w2')
    def test_add_sft_reg32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 2257095308)
        self.assertEqual(self.rf.read('W0'), 2257095308)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('add w0, w1, w2, lsl #0')
    def test_add_sft_reg_lsl_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2257095308)
        self.assertEqual(self.rf.read('W0'), 2257095308)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=1')
    @itest('add w0, w1, w2, lsl #31')
    def test_add_sft_reg_lsl_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3242345284)
        self.assertEqual(self.rf.read('W0'), 3242345284)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('add w0, w1, w2, lsl #1')
    def test_add_sft_reg_lsl32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3419328980)
        self.assertEqual(self.rf.read('W0'), 3419328980)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('add w0, w1, w2, lsr #0')
    def test_add_sft_reg_lsr_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2257095308)
        self.assertEqual(self.rf.read('W0'), 2257095308)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('add w0, w1, w2, lsr #31')
    def test_add_sft_reg_lsr_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('add w0, w1, w2, lsr #1')
    def test_add_sft_reg_lsr32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 2168603460)
        self.assertEqual(self.rf.read('W0'), 2168603460)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('add w0, w1, w2, asr #0')
    def test_add_sft_reg_asr_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2257095308)
        self.assertEqual(self.rf.read('W0'), 2257095308)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('add w0, w1, w2, asr #31')
    def test_add_sft_reg_asr_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861635)
        self.assertEqual(self.rf.read('W0'), 1094861635)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('add w0, w1, w2, asr #1')
    def test_add_sft_reg_asr32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 21119812)
        self.assertEqual(self.rf.read('W0'), 21119812)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('add x0, x1, x2')
    def test_add_sft_reg64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 10562232608263806624)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('add x0, x1, x2, lsl #0')
    def test_add_sft_reg_lsl_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 10562232608263806624)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=1')
    @itest('add x0, x1, x2, lsl #63')
    def test_add_sft_reg_lsl_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 13925766958282065736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('add x0, x1, x2, lsl #1')
    def test_add_sft_reg_lsl64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 16422070295100323320)
        self.assertEqual(self.rf.read('W0'), 4025677304)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('add x0, x1, x2, lsr #0')
    def test_add_sft_reg_lsr_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 10562232608263806624)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('add x0, x1, x2, lsr #63')
    def test_add_sft_reg_lsr_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('add x0, x1, x2, lsr #1')
    def test_add_sft_reg_lsr64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 9314080939854677832)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('add x0, x1, x2, asr #0')
    def test_add_sft_reg_asr_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 10562232608263806624)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('add x0, x1, x2, asr #63')
    def test_add_sft_reg_asr_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289927)
        self.assertEqual(self.rf.read('W0'), 1162233671)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('add x0, x1, x2, asr #1')
    def test_add_sft_reg_asr64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 90708902999902024)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add d0, d1, d2'], multiple_insts=True)
    def test_add_scalar(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 14034560904491486928)
        self.assertEqual(self.rf.read('Q0'), 14034560904491486928)
        self.assertEqual(self.rf.read('D0'), 14034560904491486928)
        self.assertEqual(self.rf.read('S0'), 3402419920)
        self.assertEqual(self.rf.read('H0'), 52944)
        self.assertEqual(self.rf.read('B0'), 208)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add d0, d1, d2'], multiple_insts=True)
    def test_add_scalar_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744073709551614)
        self.assertEqual(self.rf.read('Q0'), 18446744073709551614)
        self.assertEqual(self.rf.read('D0'), 18446744073709551614)
        self.assertEqual(self.rf.read('S0'), 4294967294)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_add_vector_8b(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 14034560904491486928)
        self.assertEqual(self.rf.read('Q0'), 14034560904491486928)
        self.assertEqual(self.rf.read('D0'), 14034560904491486928)
        self.assertEqual(self.rf.read('S0'), 3402419920)
        self.assertEqual(self.rf.read('H0'), 52944)
        self.assertEqual(self.rf.read('B0'), 208)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_add_vector_8b_max(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18374403900871474942)
        self.assertEqual(self.rf.read('Q0'), 18374403900871474942)
        self.assertEqual(self.rf.read('D0'), 18374403900871474942)
        self.assertEqual(self.rf.read('S0'), 4278124286)
        self.assertEqual(self.rf.read('H0'), 65278)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_add_vector_16b(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 216189852245102803578226751219407048400)
        self.assertEqual(self.rf.read('Q0'), 216189852245102803578226751219407048400)
        self.assertEqual(self.rf.read('D0'), 14034560904491486928)
        self.assertEqual(self.rf.read('S0'), 3402419920)
        self.assertEqual(self.rf.read('H0'), 52944)
        self.assertEqual(self.rf.read('B0'), 208)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_add_vector_16b_max(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 338947926266346547920380981520271081214)
        self.assertEqual(self.rf.read('Q0'), 338947926266346547920380981520271081214)
        self.assertEqual(self.rf.read('D0'), 18374403900871474942)
        self.assertEqual(self.rf.read('S0'), 4278124286)
        self.assertEqual(self.rf.read('H0'), 65278)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.4h, v1.4h, v2.4h'], multiple_insts=True)
    def test_add_vector_4h(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 14034560904491486928)
        self.assertEqual(self.rf.read('Q0'), 14034560904491486928)
        self.assertEqual(self.rf.read('D0'), 14034560904491486928)
        self.assertEqual(self.rf.read('S0'), 3402419920)
        self.assertEqual(self.rf.read('H0'), 52944)
        self.assertEqual(self.rf.read('B0'), 208)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.4h, v1.4h, v2.4h'], multiple_insts=True)
    def test_add_vector_4h_max(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446462594437808126)
        self.assertEqual(self.rf.read('Q0'), 18446462594437808126)
        self.assertEqual(self.rf.read('D0'), 18446462594437808126)
        self.assertEqual(self.rf.read('S0'), 4294901758)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.8h, v1.8h, v2.8h'], multiple_insts=True)
    def test_add_vector_8h(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 216189852245102803578226751219407048400)
        self.assertEqual(self.rf.read('Q0'), 216189852245102803578226751219407048400)
        self.assertEqual(self.rf.read('D0'), 14034560904491486928)
        self.assertEqual(self.rf.read('S0'), 3402419920)
        self.assertEqual(self.rf.read('H0'), 52944)
        self.assertEqual(self.rf.read('B0'), 208)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.8h, v1.8h, v2.8h'], multiple_insts=True)
    def test_add_vector_8h_max(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340277174544850557177215099159739039742)
        self.assertEqual(self.rf.read('Q0'), 340277174544850557177215099159739039742)
        self.assertEqual(self.rf.read('D0'), 18446462594437808126)
        self.assertEqual(self.rf.read('S0'), 4294901758)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.2s, v1.2s, v2.2s'], multiple_insts=True)
    def test_add_vector_2s(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 14034560904491486928)
        self.assertEqual(self.rf.read('Q0'), 14034560904491486928)
        self.assertEqual(self.rf.read('D0'), 14034560904491486928)
        self.assertEqual(self.rf.read('S0'), 3402419920)
        self.assertEqual(self.rf.read('H0'), 52944)
        self.assertEqual(self.rf.read('B0'), 208)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.2s, v1.2s, v2.2s'], multiple_insts=True)
    def test_add_vector_2s_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744069414584318)
        self.assertEqual(self.rf.read('Q0'), 18446744069414584318)
        self.assertEqual(self.rf.read('D0'), 18446744069414584318)
        self.assertEqual(self.rf.read('S0'), 4294967294)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.4s, v1.4s, v2.4s'], multiple_insts=True)
    def test_add_vector_4s(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 216189852245102803578226751219407048400)
        self.assertEqual(self.rf.read('Q0'), 216189852245102803578226751219407048400)
        self.assertEqual(self.rf.read('D0'), 14034560904491486928)
        self.assertEqual(self.rf.read('S0'), 3402419920)
        self.assertEqual(self.rf.read('H0'), 52944)
        self.assertEqual(self.rf.read('B0'), 208)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.4s, v1.4s, v2.4s'], multiple_insts=True)
    def test_add_vector_4s_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366841710300930663525760219742206)
        self.assertEqual(self.rf.read('Q0'), 340282366841710300930663525760219742206)
        self.assertEqual(self.rf.read('D0'), 18446744069414584318)
        self.assertEqual(self.rf.read('S0'), 4294967294)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.2d, v1.2d, v2.2d'], multiple_insts=True)
    def test_add_vector_2d(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 216189852245102803578226751219407048400)
        self.assertEqual(self.rf.read('Q0'), 216189852245102803578226751219407048400)
        self.assertEqual(self.rf.read('D0'), 14034560904491486928)
        self.assertEqual(self.rf.read('S0'), 3402419920)
        self.assertEqual(self.rf.read('H0'), 52944)
        self.assertEqual(self.rf.read('B0'), 208)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'add v0.2d, v1.2d, v2.2d'], multiple_insts=True)
    def test_add_vector_2d_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366920938463444927863358058659838)
        self.assertEqual(self.rf.read('Q0'), 340282366920938463444927863358058659838)
        self.assertEqual(self.rf.read('D0'), 18446744073709551614)
        self.assertEqual(self.rf.read('S0'), 4294967294)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp d0, v1.2d'], multiple_insts=True)
    def test_addp_scalar(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 10562232608263806624)
        self.assertEqual(self.rf.read('Q0'), 10562232608263806624)
        self.assertEqual(self.rf.read('D0'), 10562232608263806624)
        self.assertEqual(self.rf.read('S0'), 2593955488)
        self.assertEqual(self.rf.read('H0'), 40608)
        self.assertEqual(self.rf.read('B0'), 160)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp d0, v1.2d'], multiple_insts=True)
    def test_addp_scalar_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744073709551614)
        self.assertEqual(self.rf.read('Q0'), 18446744073709551614)
        self.assertEqual(self.rf.read('D0'), 18446744073709551614)
        self.assertEqual(self.rf.read('S0'), 4294967294)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_addp_vector_8b(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 16422353980705188783)
        self.assertEqual(self.rf.read('Q0'), 16422353980705188783)
        self.assertEqual(self.rf.read('D0'), 16422353980705188783)
        self.assertEqual(self.rf.read('S0'), 2745674671)
        self.assertEqual(self.rf.read('H0'), 43951)
        self.assertEqual(self.rf.read('B0'), 175)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_addp_vector_8b_max(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18374403900871474942)
        self.assertEqual(self.rf.read('Q0'), 18374403900871474942)
        self.assertEqual(self.rf.read('D0'), 18374403900871474942)
        self.assertEqual(self.rf.read('S0'), 4278124286)
        self.assertEqual(self.rf.read('H0'), 65278)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_addp_vector_16b(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 260236860052819680924283203992748010415)
        self.assertEqual(self.rf.read('Q0'), 260236860052819680924283203992748010415)
        self.assertEqual(self.rf.read('D0'), 9477697389866757039)
        self.assertEqual(self.rf.read('S0'), 2745674671)
        self.assertEqual(self.rf.read('H0'), 43951)
        self.assertEqual(self.rf.read('B0'), 175)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_addp_vector_16b_max(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 338947926266346547920380981520271081214)
        self.assertEqual(self.rf.read('Q0'), 338947926266346547920380981520271081214)
        self.assertEqual(self.rf.read('D0'), 18374403900871474942)
        self.assertEqual(self.rf.read('S0'), 4278124286)
        self.assertEqual(self.rf.read('H0'), 65278)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.4h, v1.4h, v2.4h'], multiple_insts=True)
    def test_addp_vector_4h(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 16494131194999778478)
        self.assertEqual(self.rf.read('Q0'), 16494131194999778478)
        self.assertEqual(self.rf.read('D0'), 16494131194999778478)
        self.assertEqual(self.rf.read('S0'), 2762386606)
        self.assertEqual(self.rf.read('H0'), 44206)
        self.assertEqual(self.rf.read('B0'), 174)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.4h, v1.4h, v2.4h'], multiple_insts=True)
    def test_addp_vector_4h_max(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446462594437808126)
        self.assertEqual(self.rf.read('Q0'), 18446462594437808126)
        self.assertEqual(self.rf.read('D0'), 18446462594437808126)
        self.assertEqual(self.rf.read('S0'), 4294901758)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.8h, v1.8h, v2.8h'], multiple_insts=True)
    def test_addp_vector_8h(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 261560915955235783894957813360186797230)
        self.assertEqual(self.rf.read('Q0'), 261560915955235783894957813360186797230)
        self.assertEqual(self.rf.read('D0'), 9549474604161346734)
        self.assertEqual(self.rf.read('S0'), 2762386606)
        self.assertEqual(self.rf.read('H0'), 44206)
        self.assertEqual(self.rf.read('B0'), 174)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.8h, v1.8h, v2.8h'], multiple_insts=True)
    def test_addp_vector_8h_max(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340277174544850557177215099159739039742)
        self.assertEqual(self.rf.read('Q0'), 340277174544850557177215099159739039742)
        self.assertEqual(self.rf.read('D0'), 18446462594437808126)
        self.assertEqual(self.rf.read('S0'), 4294901758)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.2s, v1.2s, v2.2s'], multiple_insts=True)
    def test_addp_vector_2s(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 16638807125449550508)
        self.assertEqual(self.rf.read('Q0'), 16638807125449550508)
        self.assertEqual(self.rf.read('D0'), 16638807125449550508)
        self.assertEqual(self.rf.read('S0'), 2796071596)
        self.assertEqual(self.rf.read('H0'), 43692)
        self.assertEqual(self.rf.read('B0'), 172)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.2s, v1.2s, v2.2s'], multiple_insts=True)
    def test_addp_vector_2s_max(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744069414584318)
        self.assertEqual(self.rf.read('Q0'), 18446744069414584318)
        self.assertEqual(self.rf.read('D0'), 18446744069414584318)
        self.assertEqual(self.rf.read('S0'), 4294967294)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.4s, v1.4s, v2.4s'], multiple_insts=True)
    def test_addp_vector_4s(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 264229715817868531353953106831354669740)
        self.assertEqual(self.rf.read('Q0'), 264229715817868531353953106831354669740)
        self.assertEqual(self.rf.read('D0'), 9694150534611118764)
        self.assertEqual(self.rf.read('S0'), 2796071596)
        self.assertEqual(self.rf.read('H0'), 43692)
        self.assertEqual(self.rf.read('B0'), 172)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.4s, v1.4s, v2.4s'], multiple_insts=True)
    def test_addp_vector_4s_max(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366841710300930663525760219742206)
        self.assertEqual(self.rf.read('Q0'), 340282366841710300930663525760219742206)
        self.assertEqual(self.rf.read('D0'), 18446744069414584318)
        self.assertEqual(self.rf.read('S0'), 4294967294)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.2d, v1.2d, v2.2d'], multiple_insts=True)
    def test_addp_vector_2d(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 280243003665514749634976138378813939360)
        self.assertEqual(self.rf.read('Q0'), 280243003665514749634976138378813939360)
        self.assertEqual(self.rf.read('D0'), 10562232608263806624)
        self.assertEqual(self.rf.read('S0'), 2593955488)
        self.assertEqual(self.rf.read('H0'), 40608)
        self.assertEqual(self.rf.read('B0'), 160)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'addp v0.2d, v1.2d, v2.2d'], multiple_insts=True)
    def test_addp_vector_2d_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366920938463444927863358058659838)
        self.assertEqual(self.rf.read('Q0'), 340282366920938463444927863358058659838)
        self.assertEqual(self.rf.read('D0'), 18446744073709551614)
        self.assertEqual(self.rf.read('S0'), 4294967294)
        self.assertEqual(self.rf.read('H0'), 65534)
        self.assertEqual(self.rf.read('B0'), 254)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('adds w0, w1, w2, uxtb')
    def test_adds_ext_reg_uxtb32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861768)
        self.assertEqual(self.rf.read('W0'), 1094861768)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('adds w0, w1, w2, uxtb #0')
    def test_adds_ext_reg_uxtb0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861768)
        self.assertEqual(self.rf.read('W0'), 1094861768)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('adds w0, w1, w2, uxtb #4')
    def test_adds_ext_reg_uxtb4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094863748)
        self.assertEqual(self.rf.read('W0'), 1094863748)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('adds w0, w1, w2, uxth')
    def test_adds_ext_reg_uxth32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094895256)
        self.assertEqual(self.rf.read('W0'), 1094895256)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('adds w0, w1, w2, uxth #0')
    def test_adds_ext_reg_uxth0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094895256)
        self.assertEqual(self.rf.read('W0'), 1094895256)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('adds w0, w1, w2, uxth #4')
    def test_adds_ext_reg_uxth4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1095399556)
        self.assertEqual(self.rf.read('W0'), 1095399556)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, uxtw')
    def test_adds_ext_reg_uxtw32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, uxtw #0')
    def test_adds_ext_reg_uxtw0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, uxtw #4')
    def test_adds_ext_reg_uxtw4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, uxtx')
    def test_adds_ext_reg_uxtx32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, uxtx #0')
    def test_adds_ext_reg_uxtx0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, uxtx #4')
    def test_adds_ext_reg_uxtx4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('adds w0, w1, w2, sxtb')
    def test_adds_ext_reg_sxtb32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861512)
        self.assertEqual(self.rf.read('W0'), 1094861512)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('adds w0, w1, w2, sxtb #0')
    def test_adds_ext_reg_sxtb0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861512)
        self.assertEqual(self.rf.read('W0'), 1094861512)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('adds w0, w1, w2, sxtb #4')
    def test_adds_ext_reg_sxtb4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094859652)
        self.assertEqual(self.rf.read('W0'), 1094859652)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('adds w0, w1, w2, sxth')
    def test_adds_ext_reg_sxth32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094829720)
        self.assertEqual(self.rf.read('W0'), 1094829720)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('adds w0, w1, w2, sxth #0')
    def test_adds_ext_reg_sxth0_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094829720)
        self.assertEqual(self.rf.read('W0'), 1094829720)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('adds w0, w1, w2, sxth #4')
    def test_adds_ext_reg_sxth4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094350980)
        self.assertEqual(self.rf.read('W0'), 1094350980)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, sxtw')
    def test_adds_ext_reg_sxtw32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, sxtw #0')
    def test_adds_ext_reg_sxtw0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, sxtw #4')
    def test_adds_ext_reg_sxtw4_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, sxtx')
    def test_adds_ext_reg_sxtx32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, sxtx #0')
    def test_adds_ext_reg_sxtx0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, sxtx #4')
    def test_adds_ext_reg_sxtx4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, lsl #0')
    def test_adds_ext_reg_lsl0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3264517784)
        self.assertEqual(self.rf.read('W0'), 3264517784)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('adds w0, w1, w2, lsl #4')
    def test_adds_ext_reg_lsl4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1449621636)
        self.assertEqual(self.rf.read('W0'), 1449621636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('adds x0, x1, w2, uxtb')
    def test_adds_ext_reg_uxtb64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427290060)
        self.assertEqual(self.rf.read('W0'), 1162233804)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('adds x0, x1, w2, uxtb #0')
    def test_adds_ext_reg_uxtb0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427290060)
        self.assertEqual(self.rf.read('W0'), 1162233804)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('adds x0, x1, w2, uxtb #4')
    def test_adds_ext_reg_uxtb4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427292040)
        self.assertEqual(self.rf.read('W0'), 1162235784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('adds x0, x1, w2, uxth')
    def test_adds_ext_reg_uxth64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427323548)
        self.assertEqual(self.rf.read('W0'), 1162267292)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('adds x0, x1, w2, uxth #0')
    def test_adds_ext_reg_uxth0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427323548)
        self.assertEqual(self.rf.read('W0'), 1162267292)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('adds x0, x1, w2, uxth #4')
    def test_adds_ext_reg_uxth4_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427827848)
        self.assertEqual(self.rf.read('W0'), 1162771592)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('adds x0, x1, w2, uxtw')
    def test_adds_ext_reg_uxtw64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394923596946076)
        self.assertEqual(self.rf.read('W0'), 3331889820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('adds x0, x1, w2, uxtw #0')
    def test_adds_ext_reg_uxtw0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394923596946076)
        self.assertEqual(self.rf.read('W0'), 3331889820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('adds x0, x1, w2, uxtw #4')
    def test_adds_ext_reg_uxtw4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394956141788296)
        self.assertEqual(self.rf.read('W0'), 1516993672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('adds x0, x1, x2, uxtx')
    def test_adds_ext_reg_uxtx64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('adds x0, x1, x2, uxtx #0')
    def test_adds_ext_reg_uxtx0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('adds x0, x1, x2, uxtx #4')
    def test_adds_ext_reg_uxtx4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 6226077542263798984)
        self.assertEqual(self.rf.read('W0'), 2594946248)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('adds x0, x1, w2, sxtb')
    def test_adds_ext_reg_sxtb64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289804)
        self.assertEqual(self.rf.read('W0'), 1162233548)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('adds x0, x1, w2, sxtb #0')
    def test_adds_ext_reg_sxtb0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289804)
        self.assertEqual(self.rf.read('W0'), 1162233548)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('adds x0, x1, w2, sxtb #4')
    def test_adds_ext_reg_sxtb4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427287944)
        self.assertEqual(self.rf.read('W0'), 1162231688)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('adds x0, x1, w2, sxth')
    def test_adds_ext_reg_sxth64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427258012)
        self.assertEqual(self.rf.read('W0'), 1162201756)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('adds x0, x1, w2, sxth #0')
    def test_adds_ext_reg_sxth0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427258012)
        self.assertEqual(self.rf.read('W0'), 1162201756)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('adds x0, x1, w2, sxth #4')
    def test_adds_ext_reg_sxth4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921426779272)
        self.assertEqual(self.rf.read('W0'), 1161723016)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('adds x0, x1, w2, sxtw')
    def test_adds_ext_reg_sxtw64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394919301978780)
        self.assertEqual(self.rf.read('W0'), 3331889820)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('adds x0, x1, w2, sxtw #0')
    def test_adds_ext_reg_sxtw0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394919301978780)
        self.assertEqual(self.rf.read('W0'), 3331889820)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('adds x0, x1, w2, sxtw #4')
    def test_adds_ext_reg_sxtw4_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394887422311560)
        self.assertEqual(self.rf.read('W0'), 1516993672)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('adds x0, x1, x2, sxtx')
    def test_adds_ext_reg_sxtx64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('adds x0, x1, x2, sxtx #0')
    def test_adds_ext_reg_sxtx0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('adds x0, x1, x2, sxtx #4')
    def test_adds_ext_reg_sxtx4_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 6226077542263798984)
        self.assertEqual(self.rf.read('W0'), 2594946248)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('adds x0, x1, x2, lsl #0')
    def test_adds_ext_reg_lsl0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 14020997122084347552)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('adds x0, x1, x2, lsl #4')
    def test_adds_ext_reg_lsl4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 6226077542263798984)
        self.assertEqual(self.rf.read('W0'), 2594946248)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('adds w0, w1, #0')
    def test_adds_imm_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('adds w0, w1, #4095')
    def test_adds_imm_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094865731)
        self.assertEqual(self.rf.read('W0'), 1094865731)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('adds w0, w1, #1')
    def test_adds_imm32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('adds w0, w1, #1, lsl #0')
    def test_adds_imm_lsl0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('adds w0, w1, #1, lsl #12')
    def test_adds_imm_lsl12_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094865732)
        self.assertEqual(self.rf.read('W0'), 1094865732)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('adds x0, x1, #0')
    def test_adds_imm_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('adds x0, x1, #4095')
    def test_adds_imm_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427294023)
        self.assertEqual(self.rf.read('W0'), 1162237767)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('adds x0, x1, #1')
    def test_adds_imm64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('adds x0, x1, #1, lsl #0')
    def test_adds_imm_lsl0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('adds x0, x1, #1, lsl #12')
    def test_adds_imm_lsl12_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427294024)
        self.assertEqual(self.rf.read('W0'), 1162237768)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('adds w0, w1, w2')
    def test_adds_sft_reg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2257095308)
        self.assertEqual(self.rf.read('W0'), 2257095308)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('adds w0, w1, w2, lsl #0')
    def test_adds_sft_reg_lsl_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 2257095308)
        self.assertEqual(self.rf.read('W0'), 2257095308)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=1')
    @itest('adds w0, w1, w2, lsl #31')
    def test_adds_sft_reg_lsl_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3242345284)
        self.assertEqual(self.rf.read('W0'), 3242345284)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('adds w0, w1, w2, lsl #1')
    def test_adds_sft_reg_lsl32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3419328980)
        self.assertEqual(self.rf.read('W0'), 3419328980)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('adds w0, w1, w2, lsr #0')
    def test_adds_sft_reg_lsr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 2257095308)
        self.assertEqual(self.rf.read('W0'), 2257095308)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('adds w0, w1, w2, lsr #31')
    def test_adds_sft_reg_lsr_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('adds w0, w1, w2, lsr #1')
    def test_adds_sft_reg_lsr32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 2168603460)
        self.assertEqual(self.rf.read('W0'), 2168603460)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('adds w0, w1, w2, asr #0')
    def test_adds_sft_reg_asr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 2257095308)
        self.assertEqual(self.rf.read('W0'), 2257095308)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('adds w0, w1, w2, asr #31')
    def test_adds_sft_reg_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861635)
        self.assertEqual(self.rf.read('W0'), 1094861635)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('adds w0, w1, w2, asr #1')
    def test_adds_sft_reg_asr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 21119812)
        self.assertEqual(self.rf.read('W0'), 21119812)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('adds x0, x1, x2')
    def test_adds_sft_reg64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 10562232608263806624)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('adds x0, x1, x2, lsl #0')
    def test_adds_sft_reg_lsl_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 10562232608263806624)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=1')
    @itest('adds x0, x1, x2, lsl #63')
    def test_adds_sft_reg_lsl_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 13925766958282065736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('adds x0, x1, x2, lsl #1')
    def test_adds_sft_reg_lsl64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 16422070295100323320)
        self.assertEqual(self.rf.read('W0'), 4025677304)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('adds x0, x1, x2, lsr #0')
    def test_adds_sft_reg_lsr_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 10562232608263806624)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('adds x0, x1, x2, lsr #63')
    def test_adds_sft_reg_lsr_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('adds x0, x1, x2, lsr #1')
    def test_adds_sft_reg_lsr64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 9314080939854677832)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('adds x0, x1, x2, asr #0')
    def test_adds_sft_reg_asr_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 10562232608263806624)
        self.assertEqual(self.rf.read('W0'), 2593955488)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('adds x0, x1, x2, asr #63')
    def test_adds_sft_reg_asr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289927)
        self.assertEqual(self.rf.read('W0'), 1162233671)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('adds x0, x1, x2, asr #1')
    def test_adds_sft_reg_asr64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 90708902999902024)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_custom('adr x0, .0')
    def test_adr_0(self):
        if False:
            print('Hello World!')
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute()
        self.assertEqual(self.rf.read('X0'), pc)

    @itest_custom('adr x0, .-8')
    def test_adr_neg(self):
        if False:
            print('Hello World!')
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute()
        self.assertEqual(self.rf.read('X0'), pc - 8)

    @itest_custom('adr x0, .+8')
    def test_adr_pos(self):
        if False:
            i = 10
            return i + 15
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute()
        self.assertEqual(self.rf.read('X0'), pc + 8)

    @itest_custom('adrp x0, .0')
    def test_adrp_0(self):
        if False:
            for i in range(10):
                print('nop')
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute()
        self.assertEqual(self.rf.read('X0'), pc)

    @itest_custom('adrp x0, .-0x1000')
    def test_adrp_neg(self):
        if False:
            while True:
                i = 10
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute()
        self.assertEqual(self.rf.read('X0'), pc - 4096)

    @itest_custom('adrp x0, .+0x1000')
    def test_adrp_pos(self):
        if False:
            return 10
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute()
        self.assertEqual(self.rf.read('X0'), pc + 4096)

    @itest_setregs('W1=0x41424344')
    @itest('and w0, w1, #0xffff')
    def test_and_imm32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('and w0, w1, #0xffff0000')
    def test_and_imm2_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094844416)
        self.assertEqual(self.rf.read('W0'), 1094844416)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x44434241')
    @itest('and w0, w1, #1')
    def test_and_imm3_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('and x0, x1, #0xffff0000ffff0000')
    def test_and_imm64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702320962090434560)
        self.assertEqual(self.rf.read('W0'), 1162215424)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('and x0, x1, #0x0000ffff0000ffff')
    def test_and_imm2_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 73959336855368)
        self.assertEqual(self.rf.read('W0'), 18248)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4847464544434241')
    @itest('and x0, x1, #1')
    def test_and_imm3_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('and w0, w1, w2')
    def test_and_sft_reg32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('and w0, w1, w2, lsl #0')
    def test_and_sft_reg_lsl_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('and w0, w1, w2, lsl #31')
    def test_and_sft_reg_lsl_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('and w0, w1, w2, lsl #4')
    def test_and_sft_reg_lsl32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 337916992)
        self.assertEqual(self.rf.read('W0'), 337916992)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('and w0, w1, w2, lsr #0')
    def test_and_sft_reg_lsr_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('and w0, w1, w2, lsr #31')
    def test_and_sft_reg_lsr_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('and w0, w1, w2, lsr #4')
    def test_and_sft_reg_lsr32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 135537716)
        self.assertEqual(self.rf.read('W0'), 135537716)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('and w0, w1, w2, asr #0')
    def test_and_sft_reg_asr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('and w0, w1, w2, asr #31')
    def test_and_sft_reg_asr_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('and w0, w1, w2, asr #4')
    def test_and_sft_reg_asr32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4162069556)
        self.assertEqual(self.rf.read('W0'), 4162069556)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('and w0, w1, w2, ror #0')
    def test_and_sft_reg_ror_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('and w0, w1, w2, ror #31')
    def test_and_sft_reg_ror_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3)
        self.assertEqual(self.rf.read('W0'), 3)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('and w0, w1, w2, ror #4')
    def test_and_sft_reg_ror32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1209279540)
        self.assertEqual(self.rf.read('W0'), 1209279540)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('and x0, x1, x2')
    def test_and_sft_reg64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('and x0, x1, x2, lsl #0')
    def test_and_sft_reg_lsl_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('and x0, x1, x2, lsl #63')
    def test_and_sft_reg_lsl_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('and x0, x1, x2, lsl #4')
    def test_and_sft_reg_lsl64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1451342447998432384)
        self.assertEqual(self.rf.read('W0'), 1415869568)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('and x0, x1, x2, lsr #0')
    def test_and_sft_reg_lsr_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('and x0, x1, x2, lsr #63')
    def test_and_sft_reg_lsr_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('and x0, x1, x2, lsr #4')
    def test_and_sft_reg_lsr64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 582130058740917364)
        self.assertEqual(self.rf.read('W0'), 1146381428)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('and x0, x1, x2, asr #0')
    def test_and_sft_reg_asr_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('and x0, x1, x2, asr #63')
    def test_and_sft_reg_asr_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('and x0, x1, x2, asr #4')
    def test_and_sft_reg_asr64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 17875952627843622004)
        self.assertEqual(self.rf.read('W0'), 1146381428)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('and x0, x1, x2, ror #0')
    def test_and_sft_reg_ror_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('and x0, x1, x2, ror #63')
    def test_and_sft_reg_ror_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3)
        self.assertEqual(self.rf.read('W0'), 3)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('and x0, x1, x2, ror #4')
    def test_and_sft_reg_ror64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 9805502095595693172)
        self.assertEqual(self.rf.read('W0'), 1146381428)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'and v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_and_vector_8b(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 5859837686836516696)
        self.assertEqual(self.rf.read('Q0'), 5859837686836516696)
        self.assertEqual(self.rf.read('D0'), 5859837686836516696)
        self.assertEqual(self.rf.read('S0'), 1431721816)
        self.assertEqual(self.rf.read('H0'), 22360)
        self.assertEqual(self.rf.read('B0'), 88)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'and v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_and_vector_8b_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744073709551615)
        self.assertEqual(self.rf.read('Q0'), 18446744073709551615)
        self.assertEqual(self.rf.read('D0'), 18446744073709551615)
        self.assertEqual(self.rf.read('S0'), 4294967295)
        self.assertEqual(self.rf.read('H0'), 65535)
        self.assertEqual(self.rf.read('B0'), 255)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'and v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_and_vector_16b(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 86743875649080753101215361025749440344)
        self.assertEqual(self.rf.read('Q0'), 86743875649080753101215361025749440344)
        self.assertEqual(self.rf.read('D0'), 5859837686836516696)
        self.assertEqual(self.rf.read('S0'), 1431721816)
        self.assertEqual(self.rf.read('H0'), 22360)
        self.assertEqual(self.rf.read('B0'), 88)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'and v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_and_vector_16b_max(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366920938463463374607431768211455)
        self.assertEqual(self.rf.read('Q0'), 340282366920938463463374607431768211455)
        self.assertEqual(self.rf.read('D0'), 18446744073709551615)
        self.assertEqual(self.rf.read('S0'), 4294967295)
        self.assertEqual(self.rf.read('H0'), 65535)
        self.assertEqual(self.rf.read('B0'), 255)

    @itest_setregs('W1=0x41424344')
    @itest('ands w0, w1, #0xffff')
    def test_ands_imm32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x81424344')
    @itest('ands w0, w1, #0xffff0000')
    def test_ands_imm2_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2168586240)
        self.assertEqual(self.rf.read('W0'), 2168586240)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x44434241')
    @itest('ands w0, w1, #1')
    def test_ands_imm3_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0')
    @itest('ands w0, w1, #1')
    def test_ands_imm4_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('X1=0x8142434445464748')
    @itest('ands x0, x1, #0xffff0000ffff0000')
    def test_ands_imm64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 9314006980517822464)
        self.assertEqual(self.rf.read('W0'), 1162215424)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748')
    @itest('ands x0, x1, #0x0000ffff0000ffff')
    def test_ands_imm2_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 73959336855368)
        self.assertEqual(self.rf.read('W0'), 18248)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4847464544434241')
    @itest('ands x0, x1, #1')
    def test_ands_imm3_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0')
    @itest('ands x0, x1, #1')
    def test_ands_imm4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('ands w0, w1, w2')
    def test_ands_sft_reg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0')
    @itest('ands w0, w1, w2')
    def test_ands_sft_reg_zero32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('ands w0, w1, w2, lsl #0')
    def test_ands_sft_reg_lsl_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('ands w0, w1, w2, lsl #31')
    def test_ands_sft_reg_lsl_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('ands w0, w1, w2, lsl #4')
    def test_ands_sft_reg_lsl32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 337916992)
        self.assertEqual(self.rf.read('W0'), 337916992)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('ands w0, w1, w2, lsr #0')
    def test_ands_sft_reg_lsr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('ands w0, w1, w2, lsr #31')
    def test_ands_sft_reg_lsr_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('ands w0, w1, w2, lsr #4')
    def test_ands_sft_reg_lsr32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 135537716)
        self.assertEqual(self.rf.read('W0'), 135537716)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('ands w0, w1, w2, asr #0')
    def test_ands_sft_reg_asr_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('ands w0, w1, w2, asr #31')
    def test_ands_sft_reg_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('ands w0, w1, w2, asr #4')
    def test_ands_sft_reg_asr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4162069556)
        self.assertEqual(self.rf.read('W0'), 4162069556)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('ands w0, w1, w2, ror #0')
    def test_ands_sft_reg_ror_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('ands w0, w1, w2, ror #31')
    def test_ands_sft_reg_ror_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3)
        self.assertEqual(self.rf.read('W0'), 3)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('ands w0, w1, w2, ror #4')
    def test_ands_sft_reg_ror32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1209279540)
        self.assertEqual(self.rf.read('W0'), 1209279540)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('ands x0, x1, x2')
    def test_ands_sft_reg64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0')
    @itest('ands x0, x1, x2')
    def test_ands_sft_reg_zero64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('ands x0, x1, x2, lsl #0')
    def test_ands_sft_reg_lsl_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('ands x0, x1, x2, lsl #63')
    def test_ands_sft_reg_lsl_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('ands x0, x1, x2, lsl #4')
    def test_ands_sft_reg_lsl64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1451342447998432384)
        self.assertEqual(self.rf.read('W0'), 1415869568)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('ands x0, x1, x2, lsr #0')
    def test_ands_sft_reg_lsr_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('ands x0, x1, x2, lsr #63')
    def test_ands_sft_reg_lsr_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('ands x0, x1, x2, lsr #4')
    def test_ands_sft_reg_lsr64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 582130058740917364)
        self.assertEqual(self.rf.read('W0'), 1146381428)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('ands x0, x1, x2, asr #0')
    def test_ands_sft_reg_asr_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('ands x0, x1, x2, asr #63')
    def test_ands_sft_reg_asr_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('ands x0, x1, x2, asr #4')
    def test_ands_sft_reg_asr64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17875952627843622004)
        self.assertEqual(self.rf.read('W0'), 1146381428)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('ands x0, x1, x2, ror #0')
    def test_ands_sft_reg_ror_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('ands x0, x1, x2, ror #63')
    def test_ands_sft_reg_ror_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3)
        self.assertEqual(self.rf.read('W0'), 3)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('ands x0, x1, x2, ror #4')
    def test_ands_sft_reg_ror64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 9805502095595693172)
        self.assertEqual(self.rf.read('W0'), 1146381428)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x81424344')
    @itest('asr w0, w1, #0')
    def test_asr_imm_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 2168603460)
        self.assertEqual(self.rf.read('W0'), 2168603460)

    @itest_setregs('W1=0x81424344')
    @itest('asr w0, w1, #31')
    def test_asr_imm_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0x81424344')
    @itest('asr w0, w1, #4')
    def test_asr_imm32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4162069556)
        self.assertEqual(self.rf.read('W0'), 4162069556)

    @itest_setregs('X1=0x8142434445464748')
    @itest('asr x0, x1, #0')
    def test_asr_imm_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 9314080939854677832)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x8142434445464748')
    @itest('asr x0, x1, #63')
    def test_asr_imm_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0x8142434445464748')
    @itest('asr x0, x1, #4')
    def test_asr_imm64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 17875952627843622004)
        self.assertEqual(self.rf.read('W0'), 1146381428)

    @itest_setregs('W1=0x81424344', 'W2=0')
    @itest('asr w0, w1, w2')
    def test_asr_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2168603460)
        self.assertEqual(self.rf.read('W0'), 2168603460)

    @itest_setregs('W1=0x81424344', 'W2=0xffffffff')
    @itest('asr w0, w1, w2')
    def test_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0x81424344', 'W2=4')
    @itest('asr w0, w1, w2')
    def test_asr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4162069556)
        self.assertEqual(self.rf.read('W0'), 4162069556)

    @itest_setregs('X1=0x8142434445464748', 'X2=0')
    @itest('asr x0, x1, x2')
    def test_asr_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 9314080939854677832)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x8142434445464748', 'X2=0xffffffffffffffff')
    @itest('asr x0, x1, x2')
    def test_asr_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0x8142434445464748', 'X2=4')
    @itest('asr x0, x1, x2')
    def test_asr64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17875952627843622004)
        self.assertEqual(self.rf.read('W0'), 1146381428)

    @itest_setregs('W1=0x81424344', 'W2=0')
    @itest('asrv w0, w1, w2')
    def test_asrv_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 2168603460)
        self.assertEqual(self.rf.read('W0'), 2168603460)

    @itest_setregs('W1=0x81424344', 'W2=0xffffffff')
    @itest('asrv w0, w1, w2')
    def test_asrv_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0x81424344', 'W2=4')
    @itest('asrv w0, w1, w2')
    def test_asrv32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4162069556)
        self.assertEqual(self.rf.read('W0'), 4162069556)

    @itest_setregs('X1=0x8142434445464748', 'X2=0')
    @itest('asrv x0, x1, x2')
    def test_asrv_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 9314080939854677832)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x8142434445464748', 'X2=0xffffffffffffffff')
    @itest('asrv x0, x1, x2')
    def test_asrv_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0x8142434445464748', 'X2=4')
    @itest('asrv x0, x1, x2')
    def test_asrv64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17875952627843622004)
        self.assertEqual(self.rf.read('W0'), 1146381428)

    def test_b_cond(self):
        if False:
            print('Hello World!')
        for cond in NZCV_COND_MAP:
            (cond_true, cond_false) = NZCV_COND_MAP[cond]
            asms = [f'b.{cond} .+8', 'mov x1, 42', 'mov x2, 43']

            def b_cond(self, add_pc, x1, x2):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                pc = self.cpu.PC
                self._execute(check_pc=False)
                assertEqual(self.rf.read('PC'), pc + add_pc)
                assertEqual(self.rf.read('LR'), 0)
                assertEqual(self.rf.read('X30'), 0)
                self._execute()
                assertEqual(self.rf.read('X1'), x1)
                assertEqual(self.rf.read('X2'), x2)

            @itest_setregs(f'NZCV={cond_true}')
            @itest_custom(asms, multiple_insts=True)
            def b_cond_true(self):
                if False:
                    while True:
                        i = 10
                b_cond(self, add_pc=8, x1=0, x2=43)

            @itest_setregs(f'NZCV={cond_false}')
            @itest_custom(asms, multiple_insts=True)
            def b_cond_false(self):
                if False:
                    return 10
                b_cond(self, add_pc=4, x1=42, x2=0)
            if cond_true:
                self.setUp()
                b_cond_true(self)
            if cond_false:
                self.setUp()
                b_cond_false(self)

    @itest_custom(['b .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_b_pos(self):
        if False:
            while True:
                i = 10
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['mov x1, 42', 'mov x2, 43', 'b .-8'], multiple_insts=True)
    def test_b_neg(self):
        if False:
            return 10
        self.cpu.PC += 8
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc - 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_setregs('W0=0xffffffff')
    @itest('bfc w0, #0, #1')
    def test_bfc_min_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294967294)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('W0=0xffffffff')
    @itest('bfc w0, #0, #32')
    def test_bfc_min_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W0=0xffffffff')
    @itest('bfc w0, #31, #1')
    def test_bfc_max_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2147483647)
        self.assertEqual(self.rf.read('W0'), 2147483647)

    @itest_setregs('W0=0xffffffff')
    @itest('bfc w0, #17, #15')
    def test_bfc32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 131071)
        self.assertEqual(self.rf.read('W0'), 131071)

    @itest_setregs('X0=0xffffffffffffffff')
    @itest('bfc x0, #0, #1')
    def test_bfc_min_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X0=0xffffffffffffffff')
    @itest('bfc x0, #0, #64')
    def test_bfc_min_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X0=0xffffffffffffffff')
    @itest('bfc x0, #63, #1')
    def test_bfc_max_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 9223372036854775807)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X0=0xffffffffffffffff')
    @itest('bfc x0, #33, #31')
    def test_bfc64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 8589934591)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W0=0xffffffff', 'W1=0x4142434e')
    @itest('bfi w0, w1, #0, #1')
    def test_bfi_min_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967294)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('bfi w0, w1, #0, #32')
    def test_bfi_min_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W0=0xffffffff', 'W1=0x4142434e')
    @itest('bfi w0, w1, #31, #1')
    def test_bfi_max_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 2147483647)
        self.assertEqual(self.rf.read('W0'), 2147483647)

    @itest_setregs('W0=0xffffffff', 'W1=0x41428000')
    @itest('bfi w0, w1, #17, #15')
    def test_bfi32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 131071)
        self.assertEqual(self.rf.read('W0'), 131071)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x414243444546474e')
    @itest('bfi x0, x1, #0, #1')
    def test_bfi_min_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464748')
    @itest('bfi x0, x1, #0, #64')
    def test_bfi_min_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x414243444546474e')
    @itest('bfi x0, x1, #63, #1')
    def test_bfi_max_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 9223372036854775807)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434480000000')
    @itest('bfi x0, x1, #33, #31')
    def test_bfi64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 8589934591)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W0=0xffffffff', 'W1=0x414243c7')
    @itest('bfm w0, w1, #3, #5')
    def test_bfm_ge32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967288)
        self.assertEqual(self.rf.read('W0'), 4294967288)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424340')
    @itest('bfm w0, w1, #5, #3')
    def test_bfm_lt32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 2281701375)
        self.assertEqual(self.rf.read('W0'), 2281701375)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('bfm w0, w1, #0, #31')
    def test_bfm_ge_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W0=0xffffffff', 'W1=0x4142434e')
    @itest('bfm w0, w1, #31, #0')
    def test_bfm_lt_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4294967293)
        self.assertEqual(self.rf.read('W0'), 4294967293)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424346')
    @itest('bfm w0, w1, #0, #0')
    def test_bfm_ge_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4294967294)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('W0=0xffffffff', 'W1=0x4142434e')
    @itest('bfm w0, w1, #1, #0')
    def test_bfm_lt_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 2147483647)
        self.assertEqual(self.rf.read('W0'), 2147483647)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x41424344454647c7')
    @itest('bfm x0, x1, #3, #5')
    def test_bfm_ge64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551608)
        self.assertEqual(self.rf.read('W0'), 4294967288)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464740')
    @itest('bfm x0, x1, #5, #3')
    def test_bfm_lt64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 9799832789158199295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464748')
    @itest('bfm x0, x1, #0, #63')
    def test_bfm_ge_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x414243444546474e')
    @itest('bfm x0, x1, #63, #0')
    def test_bfm_lt_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446744073709551613)
        self.assertEqual(self.rf.read('W0'), 4294967293)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464746')
    @itest('bfm x0, x1, #0, #0')
    def test_bfm_ge_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x414243444546474e')
    @itest('bfm x0, x1, #1, #0')
    def test_bfm_lt_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 9223372036854775807)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W0=0xffffffff', 'W1=0x4142434e')
    @itest('bfxil w0, w1, #0, #1')
    def test_bfxil_min_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967294)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('bfxil w0, w1, #0, #32')
    def test_bfxil_min_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W0=0xffffffff', 'W1=0x71424344')
    @itest('bfxil w0, w1, #31, #1')
    def test_bfxil_max_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967294)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('W0=0xffffffff', 'W1=0x4344')
    @itest('bfxil w0, w1, #16, #16')
    def test_bfxil32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294901760)
        self.assertEqual(self.rf.read('W0'), 4294901760)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x414243444546474e')
    @itest('bfxil x0, x1, #0, #1')
    def test_bfxil_min_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464748')
    @itest('bfxil x0, x1, #0, #64')
    def test_bfxil_min_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x7142434445464748')
    @itest('bfxil x0, x1, #63, #1')
    def test_bfxil_max_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x45464748')
    @itest('bfxil x0, x1, #32, #32')
    def test_bfxil64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 18446744069414584320)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bic w0, w1, w2')
    def test_bic32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bic w0, w1, w2, lsl #0')
    def test_bic_lsl_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xf1424344', 'W2=1')
    @itest('bic w0, w1, w2, lsl #31')
    def test_bic_lsl_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1900168004)
        self.assertEqual(self.rf.read('W0'), 1900168004)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff000')
    @itest('bic w0, w1, w2, lsl #4')
    def test_bic_lsl32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bic w0, w1, w2, lsr #0')
    def test_bic_lsr_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142434f', 'W2=0x80000000')
    @itest('bic w0, w1, w2, lsr #31')
    def test_bic_lsr_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861646)
        self.assertEqual(self.rf.read('W0'), 1094861646)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bic w0, w1, w2, lsr #4')
    def test_bic_lsr32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1073742660)
        self.assertEqual(self.rf.read('W0'), 1073742660)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bic w0, w1, w2, asr #0')
    def test_bic_asr_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('bic w0, w1, w2, asr #31')
    def test_bic_asr_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xf0000000')
    @itest('bic w0, w1, w2, asr #4')
    def test_bic_asr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4342596)
        self.assertEqual(self.rf.read('W0'), 4342596)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bic w0, w1, w2, ror #0')
    def test_bic_ror_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142434f', 'W2=0x80000001')
    @itest('bic w0, w1, w2, ror #31')
    def test_bic_ror_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861644)
        self.assertEqual(self.rf.read('W0'), 1094861644)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff000f')
    @itest('bic w0, w1, w2, ror #4')
    def test_bic_ror32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 836)
        self.assertEqual(self.rf.read('W0'), 836)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bic x0, x1, x2')
    def test_bic64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bic x0, x1, x2, lsl #0')
    def test_bic_lsl_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xf142434445464748', 'X2=1')
    @itest('bic x0, x1, x2, lsl #63')
    def test_bic_lsl_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 8161159435247830856)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff0000000')
    @itest('bic x0, x1, x2, lsl #4')
    def test_bic_lsl64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bic x0, x1, x2, lsr #0')
    def test_bic_lsr_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x414243444546474f', 'X2=0x8000000000000000')
    @itest('bic x0, x1, x2, lsr #63')
    def test_bic_lsr_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289934)
        self.assertEqual(self.rf.read('W0'), 1162233678)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bic x0, x1, x2, lsr #4')
    def test_bic_lsr64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4611686018515879752)
        self.assertEqual(self.rf.read('W0'), 88491848)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bic x0, x1, x2, asr #0')
    def test_bic_asr_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('bic x0, x1, x2, asr #63')
    def test_bic_asr_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xf000000000000000')
    @itest('bic x0, x1, x2, asr #4')
    def test_bic_asr64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 18651308961974088)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bic x0, x1, x2, ror #0')
    def test_bic_ror_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x414243444546474f', 'X2=0x8000000000000001')
    @itest('bic x0, x1, x2, ror #63')
    def test_bic_ror_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289932)
        self.assertEqual(self.rf.read('W0'), 1162233676)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff0000000f')
    @itest('bic x0, x1, x2, ror #4')
    def test_bic_ror64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 88491848)
        self.assertEqual(self.rf.read('W0'), 88491848)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bics w0, w1, w2')
    def test_bics32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bics w0, w1, w2, lsl #0')
    def test_bics_lsl_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xf1424344', 'W2=1')
    @itest('bics w0, w1, w2, lsl #31')
    def test_bics_lsl_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1900168004)
        self.assertEqual(self.rf.read('W0'), 1900168004)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff000')
    @itest('bics w0, w1, w2, lsl #4')
    def test_bics_lsl32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bics w0, w1, w2, lsr #0')
    def test_bics_lsr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x9142434f', 'W2=0x80000000')
    @itest('bics w0, w1, w2, lsr #31')
    def test_bics_lsr_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2437038926)
        self.assertEqual(self.rf.read('W0'), 2437038926)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x91424344', 'W2=0xffff0000')
    @itest('bics w0, w1, w2, lsr #4')
    def test_bics_lsr32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2415919940)
        self.assertEqual(self.rf.read('W0'), 2415919940)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bics w0, w1, w2, asr #0')
    def test_bics_asr_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('bics w0, w1, w2, asr #31')
    def test_bics_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('W1=0x41424344', 'W2=0xf0000000')
    @itest('bics w0, w1, w2, asr #4')
    def test_bics_asr32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4342596)
        self.assertEqual(self.rf.read('W0'), 4342596)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('bics w0, w1, w2, ror #0')
    def test_bics_ror_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x9142434f', 'W2=0x80000001')
    @itest('bics w0, w1, w2, ror #31')
    def test_bics_ror_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2437038924)
        self.assertEqual(self.rf.read('W0'), 2437038924)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0xffff000f')
    @itest('bics w0, w1, w2, ror #4')
    def test_bics_ror32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 836)
        self.assertEqual(self.rf.read('W0'), 836)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bics x0, x1, x2')
    def test_bics64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bics x0, x1, x2, lsl #0')
    def test_bics_lsl_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xf142434445464748', 'X2=1')
    @itest('bics x0, x1, x2, lsl #63')
    def test_bics_lsl_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 8161159435247830856)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff0000000')
    @itest('bics x0, x1, x2, lsl #4')
    def test_bics_lsl64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bics x0, x1, x2, lsr #0')
    def test_bics_lsr_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x914243444546474f', 'X2=0x8000000000000000')
    @itest('bics x0, x1, x2, lsr #63')
    def test_bics_lsr_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 10467002444461524814)
        self.assertEqual(self.rf.read('W0'), 1162233678)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x9142434445464748', 'X2=0xffffffff00000000')
    @itest('bics x0, x1, x2, lsr #4')
    def test_bics_lsr64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 10376293541550114632)
        self.assertEqual(self.rf.read('W0'), 88491848)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bics x0, x1, x2, asr #0')
    def test_bics_asr_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('bics x0, x1, x2, asr #63')
    def test_bics_asr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xf000000000000000')
    @itest('bics x0, x1, x2, asr #4')
    def test_bics_asr64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18651308961974088)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('bics x0, x1, x2, ror #0')
    def test_bics_ror_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x914243444546474f', 'X2=0x8000000000000001')
    @itest('bics x0, x1, x2, ror #63')
    def test_bics_ror_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 10467002444461524812)
        self.assertEqual(self.rf.read('W0'), 1162233676)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff0000000f')
    @itest('bics x0, x1, x2, ror #4')
    def test_bics_ror64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 88491848)
        self.assertEqual(self.rf.read('W0'), 88491848)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_custom(['bl .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_bl_pos(self):
        if False:
            i = 10
            return i + 15
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), pc + 4)
        self.assertEqual(self.rf.read('X30'), pc + 4)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['mov x1, 42', 'mov x2, 43', 'bl .-8'], multiple_insts=True)
    def test_bl_neg(self):
        if False:
            i = 10
            return i + 15
        self.cpu.PC += 8
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc - 8)
        self.assertEqual(self.rf.read('LR'), pc + 4)
        self.assertEqual(self.rf.read('X30'), pc + 4)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['blr x0', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_blr_pos(self):
        if False:
            print('Hello World!')
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self.cpu.X0 = pc + 8
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), pc + 4)
        self.assertEqual(self.rf.read('X30'), pc + 4)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['mov x1, 42', 'mov x2, 43', 'blr x0'], multiple_insts=True)
    def test_blr_neg(self):
        if False:
            return 10
        self.cpu.PC += 8
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self.cpu.X0 = pc - 8
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc - 8)
        self.assertEqual(self.rf.read('LR'), pc + 4)
        self.assertEqual(self.rf.read('X30'), pc + 4)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['br x0', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_br_pos(self):
        if False:
            print('Hello World!')
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self.cpu.X0 = pc + 8
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['mov x1, 42', 'mov x2, 43', 'br x0'], multiple_insts=True)
    def test_br_neg(self):
        if False:
            i = 10
            return i + 15
        self.cpu.PC += 8
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self.cpu.X0 = pc - 8
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc - 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['cbnz w0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_cbnz_pos_zero32(self):
        if False:
            return 10
        self._setreg('W0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['cbnz w0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_cbnz_pos_non_zero32(self):
        if False:
            i = 10
            return i + 15
        self._setreg('W0', 1)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['mov x1, 42', 'cbnz w0, .-4', 'mov x2, 43'], multiple_insts=True)
    def test_cbnz_neg_zero32(self):
        if False:
            for i in range(10):
                print('nop')
        self._setreg('W0', 0)
        self.cpu.PC += 4
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['mov x1, 42', 'cbnz w0, .-4', 'mov x2, 43'], multiple_insts=True)
    def test_cbnz_neg_non_zero32(self):
        if False:
            while True:
                i = 10
        self._setreg('W0', 1)
        self.cpu.PC += 4
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc - 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['cbnz x0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_cbnz_pos_zero64(self):
        if False:
            return 10
        self._setreg('X0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['cbnz x0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_cbnz_pos_non_zero64(self):
        if False:
            i = 10
            return i + 15
        self._setreg('X0', 1)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['mov x1, 42', 'cbnz x0, .-4', 'mov x2, 43'], multiple_insts=True)
    def test_cbnz_neg_zero64(self):
        if False:
            while True:
                i = 10
        self._setreg('X0', 0)
        self.cpu.PC += 4
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['mov x1, 42', 'cbnz x0, .-4', 'mov x2, 43'], multiple_insts=True)
    def test_cbnz_neg_non_zero64(self):
        if False:
            return 10
        self._setreg('X0', 1)
        self.cpu.PC += 4
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc - 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['cbz w0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_cbz_pos_zero32(self):
        if False:
            return 10
        self._setreg('W0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['cbz w0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_cbz_pos_non_zero32(self):
        if False:
            print('Hello World!')
        self._setreg('W0', 1)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['mov x1, 42', 'cbz w0, .-4', 'mov x2, 43'], multiple_insts=True)
    def test_cbz_neg_zero32(self):
        if False:
            while True:
                i = 10
        self._setreg('W0', 0)
        self.cpu.PC += 4
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc - 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['mov x1, 42', 'cbz w0, .-4', 'mov x2, 43'], multiple_insts=True)
    def test_cbz_neg_non_zero32(self):
        if False:
            print('Hello World!')
        self._setreg('W0', 1)
        self.cpu.PC += 4
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['cbz x0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_cbz_pos_zero64(self):
        if False:
            return 10
        self._setreg('X0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['cbz x0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_cbz_pos_non_zero64(self):
        if False:
            print('Hello World!')
        self._setreg('X0', 1)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['mov x1, 42', 'cbz x0, .-4', 'mov x2, 43'], multiple_insts=True)
    def test_cbz_neg_zero64(self):
        if False:
            while True:
                i = 10
        self._setreg('X0', 0)
        self.cpu.PC += 4
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc - 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['mov x1, 42', 'cbz x0, .-4', 'mov x2, 43'], multiple_insts=True)
    def test_cbz_neg_non_zero64(self):
        if False:
            print('Hello World!')
        self._setreg('X0', 1)
        self.cpu.PC += 4
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    def test_ccmp_imm(self):
        if False:
            return 10
        for cond in NZCV_COND_MAP:
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}', 'W0=0')
            @itest(f'ccmp w0, #0, #15, {cond}')
            def ccmp_imm_true_zc32(self):
                if False:
                    while True:
                        i = 10

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 1610612736)

            @itest_setregs(f'NZCV={cond_true}', 'W0=0x8fffffff')
            @itest(f'ccmp w0, #31, #15, {cond}')
            def ccmp_imm_true_nc32(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 2684354560)

            @itest_setregs(f'NZCV={cond_false}', 'W0=0xffffffff')
            @itest(f'ccmp w0, #0, #15, {cond}')
            def ccmp_imm_false32(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        print('Hello World!')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 4026531840)

            @itest_setregs(f'NZCV={cond_true}', 'X0=0')
            @itest(f'ccmp x0, #0, #15, {cond}')
            def ccmp_imm_true_zc64(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 1610612736)

            @itest_setregs(f'NZCV={cond_true}', 'X0=0x8fffffffffffffff')
            @itest(f'ccmp x0, #31, #15, {cond}')
            def ccmp_imm_true_nc64(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 2684354560)

            @itest_setregs(f'NZCV={cond_false}', 'X0=0xffffffffffffffff')
            @itest(f'ccmp x0, #0, #15, {cond}')
            def ccmp_imm_false64(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 4026531840)
            if cond_true:
                self.setUp()
                ccmp_imm_true_zc32(self)
                self.setUp()
                ccmp_imm_true_nc32(self)
                self.setUp()
                ccmp_imm_true_zc64(self)
                self.setUp()
                ccmp_imm_true_nc64(self)
            if cond_false:
                self.setUp()
                ccmp_imm_false32(self)
                self.setUp()
                ccmp_imm_false64(self)

    def test_ccmp_reg(self):
        if False:
            i = 10
            return i + 15
        for cond in NZCV_COND_MAP:
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}', 'W0=0xffffffff', 'W1=0xffffffff')
            @itest(f'ccmp w0, w1, #15, {cond}')
            def ccmp_reg_true_zc32(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 1610612736)

            @itest_setregs(f'NZCV={cond_true}', 'W0=0x7fffffff', 'W1=0xffffffff')
            @itest(f'ccmp w0, w1, #15, {cond}')
            def ccmp_reg_true_nv32(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 2415919104)

            @itest_setregs(f'NZCV={cond_false}', 'W0=0xffffffff', 'W1=0xffffffff')
            @itest(f'ccmp w0, w1, #15, {cond}')
            def ccmp_reg_false32(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        print('Hello World!')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 4026531840)

            @itest_setregs(f'NZCV={cond_true}', 'X0=0xffffffffffffffff', 'X1=0xffffffffffffffff')
            @itest(f'ccmp x0, x1, #15, {cond}')
            def ccmp_reg_true_zc64(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 1610612736)

            @itest_setregs(f'NZCV={cond_true}', 'X0=0x7fffffffffffffff', 'X1=0xffffffffffffffff')
            @itest(f'ccmp x0, x1, #15, {cond}')
            def ccmp_reg_true_nv64(self):
                if False:
                    while True:
                        i = 10

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 2415919104)

            @itest_setregs(f'NZCV={cond_false}', 'X0=0xffffffffffffffff', 'X1=0xffffffffffffffff')
            @itest(f'ccmp x0, x1, #15, {cond}')
            def ccmp_reg_false64(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('NZCV'), 4026531840)
            if cond_true:
                self.setUp()
                ccmp_reg_true_zc32(self)
                self.setUp()
                ccmp_reg_true_nv32(self)
                self.setUp()
                ccmp_reg_true_zc64(self)
                self.setUp()
                ccmp_reg_true_nv64(self)
            if cond_false:
                self.setUp()
                ccmp_reg_false32(self)
                self.setUp()
                ccmp_reg_false64(self)

    def test_cinc(self):
        if False:
            return 10
        for cond in NZCV_COND_MAP:
            if cond in ['al', 'nv']:
                continue
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}', 'W1=0x41424344')
            @itest(f'cinc w0, w1, {cond}')
            def cinc_true32(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        print('Hello World!')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1094861637)
                assertEqual(self.rf.read('W0'), 1094861637)

            @itest_setregs(f'NZCV={cond_true}', 'W1=0xffffffff')
            @itest(f'cinc w0, w1, {cond}')
            def cinc_true_of32(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 0)
                assertEqual(self.rf.read('W0'), 0)

            @itest_setregs(f'NZCV={cond_false}', 'W1=0x41424344')
            @itest(f'cinc w0, w1, {cond}')
            def cinc_false32(self):
                if False:
                    while True:
                        i = 10

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1094861636)
                assertEqual(self.rf.read('W0'), 1094861636)

            @itest_setregs(f'NZCV={cond_true}', 'X1=0x4142434445464748')
            @itest(f'cinc x0, x1, {cond}')
            def cinc_true64(self):
                if False:
                    while True:
                        i = 10

                def assertEqual(x, y):
                    if False:
                        i = 10
                        return i + 15
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 4702394921427289929)
                assertEqual(self.rf.read('W0'), 1162233673)

            @itest_setregs(f'NZCV={cond_true}', 'X1=0xffffffffffffffff')
            @itest(f'cinc x0, x1, {cond}')
            def cinc_true_of64(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 0)
                assertEqual(self.rf.read('W0'), 0)

            @itest_setregs(f'NZCV={cond_false}', 'X1=0x4142434445464748')
            @itest(f'cinc x0, x1, {cond}')
            def cinc_false64(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 4702394921427289928)
                assertEqual(self.rf.read('W0'), 1162233672)
            if cond_true:
                self.setUp()
                cinc_true32(self)
                self.setUp()
                cinc_true64(self)
                self.setUp()
                cinc_true_of32(self)
                self.setUp()
                cinc_true_of64(self)
            if cond_false:
                self.setUp()
                cinc_false32(self)
                self.setUp()
                cinc_false64(self)

    def test_cinv(self):
        if False:
            print('Hello World!')
        for cond in NZCV_COND_MAP:
            if cond in ['al', 'nv']:
                continue
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}', 'W1=0x41424344')
            @itest(f'cinv w0, w1, {cond}')
            def cinv_true32(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        i = 10
                        return i + 15
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 3200105659)
                assertEqual(self.rf.read('W0'), 3200105659)

            @itest_setregs(f'NZCV={cond_false}', 'W1=0x41424344')
            @itest(f'cinv w0, w1, {cond}')
            def cinv_false32(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1094861636)
                assertEqual(self.rf.read('W0'), 1094861636)

            @itest_setregs(f'NZCV={cond_true}', 'X1=0x4142434445464748')
            @itest(f'cinv x0, x1, {cond}')
            def cinv_true64(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 13744349152282261687)
                assertEqual(self.rf.read('W0'), 3132733623)

            @itest_setregs(f'NZCV={cond_false}', 'X1=0x4142434445464748')
            @itest(f'cinv x0, x1, {cond}')
            def cinv_false64(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        print('Hello World!')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 4702394921427289928)
                assertEqual(self.rf.read('W0'), 1162233672)
            if cond_true:
                self.setUp()
                cinv_true32(self)
                self.setUp()
                cinv_true64(self)
            if cond_false:
                self.setUp()
                cinv_false32(self)
                self.setUp()
                cinv_false64(self)

    @itest_setregs('W1=0')
    @itest('clz w0, w1')
    def test_clz_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 32)
        self.assertEqual(self.rf.read('W0'), 32)

    @itest_setregs('W1=0xffffffff')
    @itest('clz w0, w1')
    def test_clz_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=0x70f010')
    @itest('clz w0, w1')
    def test_clz32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 9)
        self.assertEqual(self.rf.read('W0'), 9)

    @itest_setregs('X1=0')
    @itest('clz x0, x1')
    def test_clz_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 64)
        self.assertEqual(self.rf.read('W0'), 64)

    @itest_setregs('X1=0xffffffffffffffff')
    @itest('clz x0, x1')
    def test_clz_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0x70f0100000')
    @itest('clz x0, x1')
    def test_clz64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 25)
        self.assertEqual(self.rf.read('W0'), 25)

    @itest_setregs('V1=0x81828384858687889192939495969798', 'V2=0x81828384858687889192939495969798')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq d0, d1, d2'], multiple_insts=True)
    def test_cmeq_reg_scalar_eq(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744073709551615)
        self.assertEqual(self.rf.read('Q0'), 18446744073709551615)
        self.assertEqual(self.rf.read('D0'), 18446744073709551615)
        self.assertEqual(self.rf.read('S0'), 4294967295)
        self.assertEqual(self.rf.read('H0'), 65535)
        self.assertEqual(self.rf.read('B0'), 255)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq d0, d1, d2'], multiple_insts=True)
    def test_cmeq_reg_scalar_neq(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x81428344854687489152935495569758', 'V2=0x81628364856687689172937495769778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_cmeq_reg_vector_8b(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18374966859414961920)
        self.assertEqual(self.rf.read('Q0'), 18374966859414961920)
        self.assertEqual(self.rf.read('D0'), 18374966859414961920)
        self.assertEqual(self.rf.read('S0'), 4278255360)
        self.assertEqual(self.rf.read('H0'), 65280)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x81428344854687489152935495569758', 'V2=0x81628364856687689172937495769778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_cmeq_reg_vector_16b(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 338958311018522360492699998064329424640)
        self.assertEqual(self.rf.read('Q0'), 338958311018522360492699998064329424640)
        self.assertEqual(self.rf.read('D0'), 18374966859414961920)
        self.assertEqual(self.rf.read('S0'), 4278255360)
        self.assertEqual(self.rf.read('H0'), 65280)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x81828344858687489192935495969758', 'V2=0x81828364858687689192937495969778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.4h, v1.4h, v2.4h'], multiple_insts=True)
    def test_cmeq_reg_vector_4h(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446462603027742720)
        self.assertEqual(self.rf.read('Q0'), 18446462603027742720)
        self.assertEqual(self.rf.read('D0'), 18446462603027742720)
        self.assertEqual(self.rf.read('S0'), 4294901760)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x81828344858687489192935495969758', 'V2=0x81828364858687689192937495969778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.8h, v1.8h, v2.8h'], multiple_insts=True)
    def test_cmeq_reg_vector_8h(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340277174703306882242637262502835978240)
        self.assertEqual(self.rf.read('Q0'), 340277174703306882242637262502835978240)
        self.assertEqual(self.rf.read('D0'), 18446462603027742720)
        self.assertEqual(self.rf.read('S0'), 4294901760)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x81828384854687489192939495569758', 'V2=0x81828384856687689192939495769778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.2s, v1.2s, v2.2s'], multiple_insts=True)
    def test_cmeq_reg_vector_2s(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744069414584320)
        self.assertEqual(self.rf.read('Q0'), 18446744069414584320)
        self.assertEqual(self.rf.read('D0'), 18446744069414584320)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x81828384854687489192939495569758', 'V2=0x81828384856687689192939495769778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.4s, v1.4s, v2.4s'], multiple_insts=True)
    def test_cmeq_reg_vector_4s(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366841710300967557013907638845440)
        self.assertEqual(self.rf.read('Q0'), 340282366841710300967557013907638845440)
        self.assertEqual(self.rf.read('D0'), 18446744069414584320)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x81828384858687889152935495569758', 'V2=0x81828384858687889172937495769778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.2d, v1.2d, v2.2d'], multiple_insts=True)
    def test_cmeq_reg_vector_2d(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366920938463444927863358058659840)
        self.assertEqual(self.rf.read('Q0'), 340282366920938463444927863358058659840)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x81828384858687880000000000000000')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq d0, d1, #0'], multiple_insts=True)
    def test_cmeq_zero_scalar_eq(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744073709551615)
        self.assertEqual(self.rf.read('Q0'), 18446744073709551615)
        self.assertEqual(self.rf.read('D0'), 18446744073709551615)
        self.assertEqual(self.rf.read('S0'), 4294967295)
        self.assertEqual(self.rf.read('H0'), 65535)
        self.assertEqual(self.rf.read('B0'), 255)

    @itest_setregs('V1=0x41424344454647485152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq d0, d1, #0'], multiple_insts=True)
    def test_cmeq_zero_scalar_neq(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x00420044004600480052005400560058')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.8b, v1.8b, #0'], multiple_insts=True)
    def test_cmeq_zero_vector_8b(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18374966859414961920)
        self.assertEqual(self.rf.read('Q0'), 18374966859414961920)
        self.assertEqual(self.rf.read('D0'), 18374966859414961920)
        self.assertEqual(self.rf.read('S0'), 4278255360)
        self.assertEqual(self.rf.read('H0'), 65280)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x00420044004600480052005400560058')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.16b, v1.16b, #0'], multiple_insts=True)
    def test_cmeq_zero_vector_16b(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 338958311018522360492699998064329424640)
        self.assertEqual(self.rf.read('Q0'), 338958311018522360492699998064329424640)
        self.assertEqual(self.rf.read('D0'), 18374966859414961920)
        self.assertEqual(self.rf.read('S0'), 4278255360)
        self.assertEqual(self.rf.read('H0'), 65280)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x00008344000087480000935400009758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.4h, v1.4h, #0'], multiple_insts=True)
    def test_cmeq_zero_vector_4h(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446462603027742720)
        self.assertEqual(self.rf.read('Q0'), 18446462603027742720)
        self.assertEqual(self.rf.read('D0'), 18446462603027742720)
        self.assertEqual(self.rf.read('S0'), 4294901760)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x00008344000087480000935400009758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.8h, v1.8h, #0'], multiple_insts=True)
    def test_cmeq_zero_vector_8h(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340277174703306882242637262502835978240)
        self.assertEqual(self.rf.read('Q0'), 340277174703306882242637262502835978240)
        self.assertEqual(self.rf.read('D0'), 18446462603027742720)
        self.assertEqual(self.rf.read('S0'), 4294901760)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x00000000854687480000000095569758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.2s, v1.2s, #0'], multiple_insts=True)
    def test_cmeq_zero_vector_2s(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 18446744069414584320)
        self.assertEqual(self.rf.read('Q0'), 18446744069414584320)
        self.assertEqual(self.rf.read('D0'), 18446744069414584320)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x00000000854687480000000095569758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.4s, v1.4s, #0'], multiple_insts=True)
    def test_cmeq_zero_vector_4s(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366841710300967557013907638845440)
        self.assertEqual(self.rf.read('Q0'), 340282366841710300967557013907638845440)
        self.assertEqual(self.rf.read('D0'), 18446744069414584320)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x00000000000000009152935495569758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'cmeq v0.2d, v1.2d, #0'], multiple_insts=True)
    def test_cmeq_zero_vector_2d(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 340282366920938463444927863358058659840)
        self.assertEqual(self.rf.read('Q0'), 340282366920938463444927863358058659840)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmn w1, w2, uxtb')
    def test_cmn_ext_reg_uxtb32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmn w1, w2, uxtb #0')
    def test_cmn_ext_reg_uxtb0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmn w1, w2, uxtb #4')
    def test_cmn_ext_reg_uxtb4_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmn w1, w2, uxth')
    def test_cmn_ext_reg_uxth32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmn w1, w2, uxth #0')
    def test_cmn_ext_reg_uxth0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmn w1, w2, uxth #4')
    def test_cmn_ext_reg_uxth4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, uxtw')
    def test_cmn_ext_reg_uxtw32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, uxtw #0')
    def test_cmn_ext_reg_uxtw0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, uxtw #4')
    def test_cmn_ext_reg_uxtw4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, uxtx')
    def test_cmn_ext_reg_uxtx32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, uxtx #0')
    def test_cmn_ext_reg_uxtx0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, uxtx #4')
    def test_cmn_ext_reg_uxtx4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmn w1, w2, sxtb')
    def test_cmn_ext_reg_sxtb32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmn w1, w2, sxtb #0')
    def test_cmn_ext_reg_sxtb0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmn w1, w2, sxtb #4')
    def test_cmn_ext_reg_sxtb4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmn w1, w2, sxth')
    def test_cmn_ext_reg_sxth32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmn w1, w2, sxth #0')
    def test_cmn_ext_reg_sxth0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmn w1, w2, sxth #4')
    def test_cmn_ext_reg_sxth4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, sxtw')
    def test_cmn_ext_reg_sxtw32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, sxtw #0')
    def test_cmn_ext_reg_sxtw0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, sxtw #4')
    def test_cmn_ext_reg_sxtw4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, sxtx')
    def test_cmn_ext_reg_sxtx32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, sxtx #0')
    def test_cmn_ext_reg_sxtx0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, sxtx #4')
    def test_cmn_ext_reg_sxtx4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, lsl #0')
    def test_cmn_ext_reg_lsl0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmn w1, w2, lsl #4')
    def test_cmn_ext_reg_lsl4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmn x1, w2, uxtb')
    def test_cmn_ext_reg_uxtb64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmn x1, w2, uxtb #0')
    def test_cmn_ext_reg_uxtb0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmn x1, w2, uxtb #4')
    def test_cmn_ext_reg_uxtb4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmn x1, w2, uxth')
    def test_cmn_ext_reg_uxth64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmn x1, w2, uxth #0')
    def test_cmn_ext_reg_uxth0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmn x1, w2, uxth #4')
    def test_cmn_ext_reg_uxth4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmn x1, w2, uxtw')
    def test_cmn_ext_reg_uxtw64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmn x1, w2, uxtw #0')
    def test_cmn_ext_reg_uxtw0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmn x1, w2, uxtw #4')
    def test_cmn_ext_reg_uxtw4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmn x1, x2, uxtx')
    def test_cmn_ext_reg_uxtx64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmn x1, x2, uxtx #0')
    def test_cmn_ext_reg_uxtx0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmn x1, x2, uxtx #4')
    def test_cmn_ext_reg_uxtx4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmn x1, w2, sxtb')
    def test_cmn_ext_reg_sxtb64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmn x1, w2, sxtb #0')
    def test_cmn_ext_reg_sxtb0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmn x1, w2, sxtb #4')
    def test_cmn_ext_reg_sxtb4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmn x1, w2, sxth')
    def test_cmn_ext_reg_sxth64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmn x1, w2, sxth #0')
    def test_cmn_ext_reg_sxth0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmn x1, w2, sxth #4')
    def test_cmn_ext_reg_sxth4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmn x1, w2, sxtw')
    def test_cmn_ext_reg_sxtw64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmn x1, w2, sxtw #0')
    def test_cmn_ext_reg_sxtw0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmn x1, w2, sxtw #4')
    def test_cmn_ext_reg_sxtw4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmn x1, x2, sxtx')
    def test_cmn_ext_reg_sxtx64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmn x1, x2, sxtx #0')
    def test_cmn_ext_reg_sxtx0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmn x1, x2, sxtx #4')
    def test_cmn_ext_reg_sxtx4_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmn x1, x2, lsl #0')
    def test_cmn_ext_reg_lsl0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmn x1, x2, lsl #4')
    def test_cmn_ext_reg_lsl4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('cmn w1, #0')
    def test_cmn_imm_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('cmn w1, #4095')
    def test_cmn_imm_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('cmn w1, #1')
    def test_cmn_imm32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('cmn w1, #1, lsl #0')
    def test_cmn_imm_lsl0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('cmn w1, #1, lsl #12')
    def test_cmn_imm_lsl12_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmn x1, #0')
    def test_cmn_imm_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmn x1, #4095')
    def test_cmn_imm_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmn x1, #1')
    def test_cmn_imm64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmn x1, #1, lsl #0')
    def test_cmn_imm_lsl0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmn x1, #1, lsl #12')
    def test_cmn_imm_lsl12_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmn w1, w2')
    def test_cmn_sft_reg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmn w1, w2, lsl #0')
    def test_cmn_sft_reg_lsl_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=1')
    @itest('cmn w1, w2, lsl #31')
    def test_cmn_sft_reg_lsl_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmn w1, w2, lsl #1')
    def test_cmn_sft_reg_lsl32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmn w1, w2, lsr #0')
    def test_cmn_sft_reg_lsr_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('cmn w1, w2, lsr #31')
    def test_cmn_sft_reg_lsr_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('cmn w1, w2, lsr #1')
    def test_cmn_sft_reg_lsr32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmn w1, w2, asr #0')
    def test_cmn_sft_reg_asr_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('cmn w1, w2, asr #31')
    def test_cmn_sft_reg_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('cmn w1, w2, asr #1')
    def test_cmn_sft_reg_asr32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmn x1, x2')
    def test_cmn_sft_reg64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmn x1, x2, lsl #0')
    def test_cmn_sft_reg_lsl_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=1')
    @itest('cmn x1, x2, lsl #63')
    def test_cmn_sft_reg_lsl_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmn x1, x2, lsl #1')
    def test_cmn_sft_reg_lsl64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmn x1, x2, lsr #0')
    def test_cmn_sft_reg_lsr_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('cmn x1, x2, lsr #63')
    def test_cmn_sft_reg_lsr_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('cmn x1, x2, lsr #1')
    def test_cmn_sft_reg_lsr64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmn x1, x2, asr #0')
    def test_cmn_sft_reg_asr_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('cmn x1, x2, asr #63')
    def test_cmn_sft_reg_asr_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('cmn x1, x2, asr #1')
    def test_cmn_sft_reg_asr64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmp w1, w2, uxtb')
    def test_cmp_ext_reg_uxtb32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmp w1, w2, uxtb #0')
    def test_cmp_ext_reg_uxtb0_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmp w1, w2, uxtb #4')
    def test_cmp_ext_reg_uxtb4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmp w1, w2, uxth')
    def test_cmp_ext_reg_uxth32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmp w1, w2, uxth #0')
    def test_cmp_ext_reg_uxth0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmp w1, w2, uxth #4')
    def test_cmp_ext_reg_uxth4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, uxtw')
    def test_cmp_ext_reg_uxtw32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, uxtw #0')
    def test_cmp_ext_reg_uxtw0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, uxtw #4')
    def test_cmp_ext_reg_uxtw4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, uxtx')
    def test_cmp_ext_reg_uxtx32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, uxtx #0')
    def test_cmp_ext_reg_uxtx0_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, uxtx #4')
    def test_cmp_ext_reg_uxtx4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmp w1, w2, sxtb')
    def test_cmp_ext_reg_sxtb32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmp w1, w2, sxtb #0')
    def test_cmp_ext_reg_sxtb0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('cmp w1, w2, sxtb #4')
    def test_cmp_ext_reg_sxtb4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmp w1, w2, sxth')
    def test_cmp_ext_reg_sxth32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmp w1, w2, sxth #0')
    def test_cmp_ext_reg_sxth0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('cmp w1, w2, sxth #4')
    def test_cmp_ext_reg_sxth4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, sxtw')
    def test_cmp_ext_reg_sxtw32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, sxtw #0')
    def test_cmp_ext_reg_sxtw0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, sxtw #4')
    def test_cmp_ext_reg_sxtw4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, sxtx')
    def test_cmp_ext_reg_sxtx32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, sxtx #0')
    def test_cmp_ext_reg_sxtx0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, sxtx #4')
    def test_cmp_ext_reg_sxtx4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, lsl #0')
    def test_cmp_ext_reg_lsl0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('cmp w1, w2, lsl #4')
    def test_cmp_ext_reg_lsl4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmp x1, w2, uxtb')
    def test_cmp_ext_reg_uxtb64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmp x1, w2, uxtb #0')
    def test_cmp_ext_reg_uxtb0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmp x1, w2, uxtb #4')
    def test_cmp_ext_reg_uxtb4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmp x1, w2, uxth')
    def test_cmp_ext_reg_uxth64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmp x1, w2, uxth #0')
    def test_cmp_ext_reg_uxth0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmp x1, w2, uxth #4')
    def test_cmp_ext_reg_uxth4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmp x1, w2, uxtw')
    def test_cmp_ext_reg_uxtw64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmp x1, w2, uxtw #0')
    def test_cmp_ext_reg_uxtw0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmp x1, w2, uxtw #4')
    def test_cmp_ext_reg_uxtw4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmp x1, x2, uxtx')
    def test_cmp_ext_reg_uxtx64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmp x1, x2, uxtx #0')
    def test_cmp_ext_reg_uxtx0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmp x1, x2, uxtx #4')
    def test_cmp_ext_reg_uxtx4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmp x1, w2, sxtb')
    def test_cmp_ext_reg_sxtb64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmp x1, w2, sxtb #0')
    def test_cmp_ext_reg_sxtb0_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('cmp x1, w2, sxtb #4')
    def test_cmp_ext_reg_sxtb4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmp x1, w2, sxth')
    def test_cmp_ext_reg_sxth64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmp x1, w2, sxth #0')
    def test_cmp_ext_reg_sxth0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('cmp x1, w2, sxth #4')
    def test_cmp_ext_reg_sxth4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmp x1, w2, sxtw')
    def test_cmp_ext_reg_sxtw64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmp x1, w2, sxtw #0')
    def test_cmp_ext_reg_sxtw0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('cmp x1, w2, sxtw #4')
    def test_cmp_ext_reg_sxtw4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmp x1, x2, sxtx')
    def test_cmp_ext_reg_sxtx64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmp x1, x2, sxtx #0')
    def test_cmp_ext_reg_sxtx0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmp x1, x2, sxtx #4')
    def test_cmp_ext_reg_sxtx4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmp x1, x2, lsl #0')
    def test_cmp_ext_reg_lsl0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('cmp x1, x2, lsl #4')
    def test_cmp_ext_reg_lsl4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('cmp w1, #0')
    def test_cmp_imm_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('cmp w1, #4095')
    def test_cmp_imm_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('cmp w1, #1')
    def test_cmp_imm32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('cmp w1, #1, lsl #0')
    def test_cmp_imm_lsl0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('cmp w1, #1, lsl #12')
    def test_cmp_imm_lsl12_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmp x1, #0')
    def test_cmp_imm_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmp x1, #4095')
    def test_cmp_imm_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmp x1, #1')
    def test_cmp_imm64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmp x1, #1, lsl #0')
    def test_cmp_imm_lsl0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('cmp x1, #1, lsl #12')
    def test_cmp_imm_lsl12_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmp w1, w2')
    def test_cmp_sft_reg32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmp w1, w2, lsl #0')
    def test_cmp_sft_reg_lsl_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=1')
    @itest('cmp w1, w2, lsl #31')
    def test_cmp_sft_reg_lsl_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmp w1, w2, lsl #1')
    def test_cmp_sft_reg_lsl32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmp w1, w2, lsr #0')
    def test_cmp_sft_reg_lsr_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('cmp w1, w2, lsr #31')
    def test_cmp_sft_reg_lsr_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('cmp w1, w2, lsr #1')
    def test_cmp_sft_reg_lsr32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('cmp w1, w2, asr #0')
    def test_cmp_sft_reg_asr_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('cmp w1, w2, asr #31')
    def test_cmp_sft_reg_asr_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('cmp w1, w2, asr #1')
    def test_cmp_sft_reg_asr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmp x1, x2')
    def test_cmp_sft_reg64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmp x1, x2, lsl #0')
    def test_cmp_sft_reg_lsl_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=1')
    @itest('cmp x1, x2, lsl #63')
    def test_cmp_sft_reg_lsl_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmp x1, x2, lsl #1')
    def test_cmp_sft_reg_lsl64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmp x1, x2, lsr #0')
    def test_cmp_sft_reg_lsr_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('cmp x1, x2, lsr #63')
    def test_cmp_sft_reg_lsr_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('cmp x1, x2, lsr #1')
    def test_cmp_sft_reg_lsr64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('cmp x1, x2, asr #0')
    def test_cmp_sft_reg_asr_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('cmp x1, x2, asr #63')
    def test_cmp_sft_reg_asr_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('cmp x1, x2, asr #1')
    def test_cmp_sft_reg_asr64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    def test_csel(self):
        if False:
            while True:
                i = 10
        for cond in NZCV_COND_MAP:
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}', 'W1=0x41424344', 'W2=0x51525354')
            @itest(f'csel w0, w1, w2, {cond}')
            def csel_true32(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1094861636)
                assertEqual(self.rf.read('W0'), 1094861636)

            @itest_setregs(f'NZCV={cond_false}', 'W1=0x41424344', 'W2=0x51525354')
            @itest(f'csel w0, w1, w2, {cond}')
            def csel_false32(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1364349780)
                assertEqual(self.rf.read('W0'), 1364349780)

            @itest_setregs(f'NZCV={cond_true}', 'X1=0x4142434445464748', 'X2=0x5152535455565758')
            @itest(f'csel x0, x1, x2, {cond}')
            def csel_true64(self):
                if False:
                    while True:
                        i = 10

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 4702394921427289928)
                assertEqual(self.rf.read('W0'), 1162233672)

            @itest_setregs(f'NZCV={cond_false}', 'X1=0x4142434445464748', 'X2=0x5152535455565758')
            @itest(f'csel x0, x1, x2, {cond}')
            def csel_false64(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        i = 10
                        return i + 15
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 5859837686836516696)
                assertEqual(self.rf.read('W0'), 1431721816)
            if cond_true:
                self.setUp()
                csel_true32(self)
                self.setUp()
                csel_true64(self)
            if cond_false:
                self.setUp()
                csel_false32(self)
                self.setUp()
                csel_false64(self)

    def test_cset(self):
        if False:
            while True:
                i = 10
        for cond in NZCV_COND_MAP:
            if cond in ['al', 'nv']:
                continue
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}')
            @itest(f'cset w0, {cond}')
            def cset_true32(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        i = 10
                        return i + 15
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1)
                assertEqual(self.rf.read('W0'), 1)

            @itest_setregs(f'NZCV={cond_false}')
            @itest(f'cset w0, {cond}')
            def cset_false32(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 0)
                assertEqual(self.rf.read('W0'), 0)

            @itest_setregs(f'NZCV={cond_true}')
            @itest(f'cset x0, {cond}')
            def cset_true64(self):
                if False:
                    while True:
                        i = 10

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1)
                assertEqual(self.rf.read('W0'), 1)

            @itest_setregs(f'NZCV={cond_false}')
            @itest(f'cset x0, {cond}')
            def cset_false64(self):
                if False:
                    for i in range(10):
                        print('nop')

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 0)
                assertEqual(self.rf.read('W0'), 0)
            if cond_true:
                self.setUp()
                cset_true32(self)
                self.setUp()
                cset_true64(self)
            if cond_false:
                self.setUp()
                cset_false32(self)
                self.setUp()
                cset_false64(self)

    def test_csetm(self):
        if False:
            i = 10
            return i + 15
        for cond in NZCV_COND_MAP:
            if cond in ['al', 'nv']:
                continue
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}')
            @itest(f'csetm w0, {cond}')
            def csetm_true32(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 4294967295)
                assertEqual(self.rf.read('W0'), 4294967295)

            @itest_setregs(f'NZCV={cond_false}')
            @itest(f'csetm w0, {cond}')
            def csetm_false32(self):
                if False:
                    while True:
                        i = 10

                def assertEqual(x, y):
                    if False:
                        print('Hello World!')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 0)
                assertEqual(self.rf.read('W0'), 0)

            @itest_setregs(f'NZCV={cond_true}')
            @itest(f'csetm x0, {cond}')
            def csetm_true64(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 18446744073709551615)
                assertEqual(self.rf.read('W0'), 4294967295)

            @itest_setregs(f'NZCV={cond_false}')
            @itest(f'csetm x0, {cond}')
            def csetm_false64(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 0)
                assertEqual(self.rf.read('W0'), 0)
            if cond_true:
                self.setUp()
                csetm_true32(self)
                self.setUp()
                csetm_true64(self)
            if cond_false:
                self.setUp()
                csetm_false32(self)
                self.setUp()
                csetm_false64(self)

    def test_csinc(self):
        if False:
            while True:
                i = 10
        for cond in NZCV_COND_MAP:
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}', 'W1=0x41424344', 'W2=0x51525354')
            @itest(f'csinc w0, w1, w2, {cond}')
            def csinc_true32(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1094861636)
                assertEqual(self.rf.read('W0'), 1094861636)

            @itest_setregs(f'NZCV={cond_false}', 'W1=0x41424344', 'W2=0x51525354')
            @itest(f'csinc w0, w1, w2, {cond}')
            def csinc_false32(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        print('Hello World!')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1364349781)
                assertEqual(self.rf.read('W0'), 1364349781)

            @itest_setregs(f'NZCV={cond_false}', 'W1=0x41424344', 'W2=0xffffffff')
            @itest(f'csinc w0, w1, w2, {cond}')
            def csinc_false_of32(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 0)
                assertEqual(self.rf.read('W0'), 0)

            @itest_setregs(f'NZCV={cond_true}', 'X1=0x4142434445464748', 'X2=0x5152535455565758')
            @itest(f'csinc x0, x1, x2, {cond}')
            def csinc_true64(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 4702394921427289928)
                assertEqual(self.rf.read('W0'), 1162233672)

            @itest_setregs(f'NZCV={cond_false}', 'X1=0x4142434445464748', 'X2=0x5152535455565758')
            @itest(f'csinc x0, x1, x2, {cond}')
            def csinc_false64(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        return 10
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 5859837686836516697)
                assertEqual(self.rf.read('W0'), 1431721817)

            @itest_setregs(f'NZCV={cond_false}', 'X1=0x4142434445464748', 'X2=0xffffffffffffffff')
            @itest(f'csinc x0, x1, x2, {cond}')
            def csinc_false_of64(self):
                if False:
                    i = 10
                    return i + 15

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 0)
                assertEqual(self.rf.read('W0'), 0)
            if cond_true:
                self.setUp()
                csinc_true32(self)
                self.setUp()
                csinc_true64(self)
            if cond_false:
                self.setUp()
                csinc_false32(self)
                self.setUp()
                csinc_false64(self)
                self.setUp()
                csinc_false_of32(self)
                self.setUp()
                csinc_false_of64(self)

    def test_csinv(self):
        if False:
            print('Hello World!')
        for cond in NZCV_COND_MAP:
            (cond_true, cond_false) = NZCV_COND_MAP[cond]

            @itest_setregs(f'NZCV={cond_true}', 'W1=0x41424344', 'W2=0x51525354')
            @itest(f'csinv w0, w1, w2, {cond}')
            def csinv_true32(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 1094861636)
                assertEqual(self.rf.read('W0'), 1094861636)

            @itest_setregs(f'NZCV={cond_false}', 'W1=0x41424344', 'W2=0x51525354')
            @itest(f'csinv w0, w1, w2, {cond}')
            def csinv_false32(self):
                if False:
                    while True:
                        i = 10

                def assertEqual(x, y):
                    if False:
                        i = 10
                        return i + 15
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 2930617515)
                assertEqual(self.rf.read('W0'), 2930617515)

            @itest_setregs(f'NZCV={cond_true}', 'X1=0x4142434445464748', 'X2=0x5152535455565758')
            @itest(f'csinv x0, x1, x2, {cond}')
            def csinv_true64(self):
                if False:
                    for i in range(10):
                        print('nop')

                def assertEqual(x, y):
                    if False:
                        print('Hello World!')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 4702394921427289928)
                assertEqual(self.rf.read('W0'), 1162233672)

            @itest_setregs(f'NZCV={cond_false}', 'X1=0x4142434445464748', 'X2=0x5152535455565758')
            @itest(f'csinv x0, x1, x2, {cond}')
            def csinv_false64(self):
                if False:
                    print('Hello World!')

                def assertEqual(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(x, y, msg=cond)
                assertEqual(self.rf.read('X0'), 12586906386873034919)
                assertEqual(self.rf.read('W0'), 2863245479)
            if cond_true:
                self.setUp()
                csinv_true32(self)
                self.setUp()
                csinv_true64(self)
            if cond_false:
                self.setUp()
                csinv_false32(self)
                self.setUp()
                csinv_false64(self)

    @skip_sym('dczid_el0 is read-only')
    @itest('mrs x0, dczid_el0')
    def test_dczid_el0(self):
        if False:
            while True:
                i = 10
        if self.__class__.__name__ == 'Aarch64CpuInstructions':
            self.assertEqual(self.rf.read('X0'), 16)
        elif self.__class__.__name__ == 'Aarch64UnicornInstructions':
            self.assertEqual(self.rf.read('X0'), 4)
        else:
            self.fail()

    def test_dmb(self):
        if False:
            i = 10
            return i + 15

        def dmb(x):
            if False:
                while True:
                    i = 10

            @skip_sym('nothing to set')
            @itest(f'dmb {x}')
            def f(self):
                if False:
                    while True:
                        i = 10
                pass
            self.setUp()
            f(self)
        for imm in range(16):
            dmb(f'#{imm}')
        for bar in ('sy', 'st', 'ld', 'ish', 'ishst', 'ishld', 'nsh', 'nshst', 'nshld', 'osh', 'oshst', 'oshld'):
            dmb(f'{bar}')

    @itest_setregs('X1=0x9192939495969798')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'dup v0.8b, w1'], multiple_insts=True)
    def test_dup_gen_8b(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 10995706271387654296)
        self.assertEqual(self.rf.read('Q0'), 10995706271387654296)
        self.assertEqual(self.rf.read('D0'), 10995706271387654296)
        self.assertEqual(self.rf.read('S0'), 2560137368)
        self.assertEqual(self.rf.read('H0'), 39064)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('X1=0x9192939495969798')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'dup v0.16b, w1'], multiple_insts=True)
    def test_dup_gen_16b(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 202834979497971162535031138547563796632)
        self.assertEqual(self.rf.read('Q0'), 202834979497971162535031138547563796632)
        self.assertEqual(self.rf.read('D0'), 10995706271387654296)
        self.assertEqual(self.rf.read('S0'), 2560137368)
        self.assertEqual(self.rf.read('H0'), 39064)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('X1=0x9192939495969798')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'dup v0.4h, w1'], multiple_insts=True)
    def test_dup_gen_4h(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 10923647577821321112)
        self.assertEqual(self.rf.read('Q0'), 10923647577821321112)
        self.assertEqual(self.rf.read('D0'), 10923647577821321112)
        self.assertEqual(self.rf.read('S0'), 2543359896)
        self.assertEqual(self.rf.read('H0'), 38808)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('X1=0x9192939495969798')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'dup v0.8h, w1'], multiple_insts=True)
    def test_dup_gen_8h(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 201505731219467153278197020908095838104)
        self.assertEqual(self.rf.read('Q0'), 201505731219467153278197020908095838104)
        self.assertEqual(self.rf.read('D0'), 10923647577821321112)
        self.assertEqual(self.rf.read('S0'), 2543359896)
        self.assertEqual(self.rf.read('H0'), 38808)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('X1=0x9192939495969798')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'dup v0.2s, w1'], multiple_insts=True)
    def test_dup_gen_2s(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 10778969439758358424)
        self.assertEqual(self.rf.read('Q0'), 10778969439758358424)
        self.assertEqual(self.rf.read('D0'), 10778969439758358424)
        self.assertEqual(self.rf.read('S0'), 2509674392)
        self.assertEqual(self.rf.read('H0'), 38808)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('X1=0x9192939495969798')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'dup v0.4s, w1'], multiple_insts=True)
    def test_dup_gen_4s(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 198836890633558864005705748261014771608)
        self.assertEqual(self.rf.read('Q0'), 198836890633558864005705748261014771608)
        self.assertEqual(self.rf.read('D0'), 10778969439758358424)
        self.assertEqual(self.rf.read('S0'), 2509674392)
        self.assertEqual(self.rf.read('H0'), 38808)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('X1=0x9192939495969798')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'dup v0.2d, x1'], multiple_insts=True)
    def test_dup_gen_2d(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 193499128016433996539547991180110632856)
        self.assertEqual(self.rf.read('Q0'), 193499128016433996539547991180110632856)
        self.assertEqual(self.rf.read('D0'), 10489608748473423768)
        self.assertEqual(self.rf.read('S0'), 2509674392)
        self.assertEqual(self.rf.read('H0'), 38808)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('eor w0, w1, w2')
    def test_eor32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3200074564)
        self.assertEqual(self.rf.read('W0'), 3200074564)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('eor w0, w1, w2, lsl #0')
    def test_eor_lsl_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3200074564)
        self.assertEqual(self.rf.read('W0'), 3200074564)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=1')
    @itest('eor w0, w1, w2, lsl #31')
    def test_eor_lsl_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3242345284)
        self.assertEqual(self.rf.read('W0'), 3242345284)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff000')
    @itest('eor w0, w1, w2, lsl #4')
    def test_eor_lsl32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3200074564)
        self.assertEqual(self.rf.read('W0'), 3200074564)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('eor w0, w1, w2, lsr #0')
    def test_eor_lsr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3200074564)
        self.assertEqual(self.rf.read('W0'), 3200074564)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('eor w0, w1, w2, lsr #31')
    def test_eor_lsr_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('eor w0, w1, w2, lsr #4')
    def test_eor_lsr32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1321055044)
        self.assertEqual(self.rf.read('W0'), 1321055044)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('eor w0, w1, w2, asr #0')
    def test_eor_asr_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3200074564)
        self.assertEqual(self.rf.read('W0'), 3200074564)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('eor w0, w1, w2, asr #31')
    def test_eor_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3200105659)
        self.assertEqual(self.rf.read('W0'), 3200105659)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xf0000000')
    @itest('eor w0, w1, w2, asr #4')
    def test_eor_asr32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3192013636)
        self.assertEqual(self.rf.read('W0'), 3192013636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff0000')
    @itest('eor w0, w1, w2, ror #0')
    def test_eor_ror_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3200074564)
        self.assertEqual(self.rf.read('W0'), 3200074564)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000001')
    @itest('eor w0, w1, w2, ror #31')
    def test_eor_ror_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861639)
        self.assertEqual(self.rf.read('W0'), 1094861639)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0xffff000f')
    @itest('eor w0, w1, w2, ror #4')
    def test_eor_ror32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3200103236)
        self.assertEqual(self.rf.read('W0'), 3200103236)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('eor x0, x1, x2')
    def test_eor64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 13744349150311761736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('eor x0, x1, x2, lsl #0')
    def test_eor_lsl_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13744349150311761736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=1')
    @itest('eor x0, x1, x2, lsl #63')
    def test_eor_lsl_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13925766958282065736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff0000000')
    @itest('eor x0, x1, x2, lsl #4')
    def test_eor_lsl64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 13744349150311761736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('eor x0, x1, x2, lsr #0')
    def test_eor_lsr_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 13744349150311761736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('eor x0, x1, x2, lsr #63')
    def test_eor_lsr_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('eor x0, x1, x2, lsr #4')
    def test_eor_lsr64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 5673898619942881096)
        self.assertEqual(self.rf.read('W0'), 3041281864)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('eor x0, x1, x2, asr #0')
    def test_eor_asr_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 13744349150311761736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('eor x0, x1, x2, asr #63')
    def test_eor_asr_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 13744349152282261687)
        self.assertEqual(self.rf.read('W0'), 3132733623)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xf000000000000000')
    @itest('eor x0, x1, x2, asr #4')
    def test_eor_asr64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 13709594176168281928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff00000000')
    @itest('eor x0, x1, x2, ror #0')
    def test_eor_ror_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13744349150311761736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000001')
    @itest('eor x0, x1, x2, ror #63')
    def test_eor_ror_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289931)
        self.assertEqual(self.rf.read('W0'), 1162233675)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0xffffffff0000000f')
    @itest('eor x0, x1, x2, ror #4')
    def test_eor_ror64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 13744349152190809928)
        self.assertEqual(self.rf.read('W0'), 3041281864)
        self.assertEqual(self.rf.read('NZCV'), 0)

    def _ld1_mlt_structs(self, vess, elem_size, elem_count):
        if False:
            while True:
                i = 10
        for reg_count in range(1, 5):
            for mode in ('no_offset', 'post_indexed_reg', 'post_indexed_imm'):
                val = 4702394921427289928
                step = 1157442765409226768
                size = elem_size * elem_count
                dword_size = 64
                if mode == 'post_indexed_imm':
                    wback = size // 8 * reg_count
                elif mode == 'post_indexed_reg':
                    wback = Mask(64)
                else:
                    wback = 0
                wback_reg = 'x29'
                insn = 'ld1 {'
                for i in range(reg_count):
                    if i != 0:
                        insn += ', '
                    insn += f'v{i}.{elem_count}{vess}'
                insn += '}, [sp]'
                if mode == 'post_indexed_reg':
                    insn += f', {wback_reg}'
                elif mode == 'post_indexed_imm':
                    insn += f', #{wback}'

                @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', insn], multiple_insts=True)
                def f(self):
                    if False:
                        while True:
                            i = 10
                    for i in range(3):
                        self._execute(reset=i == 0, check_cs=False)
                    for i in range(reg_count * (size // dword_size) - 1, -1, -1):
                        self.cpu.push_int(val + i * step)
                    self._setreg('STACK', self.cpu.STACK)
                    stack = self.cpu.STACK
                    if mode == 'post_indexed_reg':
                        self.rf.write(wback_reg.upper(), wback)
                    self._execute(reset=False)

                    def assertEqual(x, y):
                        if False:
                            for i in range(10):
                                print('nop')
                        self.assertEqual(x, y, msg=insn)
                    for i in range(reg_count):
                        j = i * (size // dword_size)
                        res = val + j * step
                        res |= val + (j + 1) * step << dword_size
                        res = res & Mask(size)
                        assertEqual(self.rf.read(f'V{i}'), res & Mask(128))
                        assertEqual(self.rf.read(f'Q{i}'), res & Mask(128))
                        assertEqual(self.rf.read(f'D{i}'), res & Mask(64))
                        assertEqual(self.rf.read(f'S{i}'), res & Mask(32))
                        assertEqual(self.rf.read(f'H{i}'), res & Mask(16))
                        assertEqual(self.rf.read(f'B{i}'), res & Mask(8))
                    if mode in ('post_indexed_reg', 'post_indexed_imm'):
                        assertEqual(self.rf.read('SP'), stack + wback & Mask(64))
                    else:
                        assertEqual(self.rf.read('SP'), stack)
                self.setUp()
                f(self)

    def test_ld1_mlt_structs(self):
        if False:
            print('Hello World!')
        self._ld1_mlt_structs(vess='b', elem_size=8, elem_count=8)
        self._ld1_mlt_structs(vess='b', elem_size=8, elem_count=16)
        self._ld1_mlt_structs(vess='h', elem_size=16, elem_count=4)
        self._ld1_mlt_structs(vess='h', elem_size=16, elem_count=8)
        self._ld1_mlt_structs(vess='s', elem_size=32, elem_count=2)
        self._ld1_mlt_structs(vess='s', elem_size=32, elem_count=4)
        self._ld1_mlt_structs(vess='d', elem_size=64, elem_count=1)
        self._ld1_mlt_structs(vess='d', elem_size=64, elem_count=2)

    @itest_custom('ldaxr w1, [sp]')
    def test_ldaxr32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldaxr w1, [sp, #0]')
    def test_ldaxr_0_32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldaxr x1, [sp]')
    def test_ldaxr64(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldaxr x1, [sp, #0]')
    def test_ldaxr_0_64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp w1, w2, [sp]')
    def test_ldp_base32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp w1, w2, [sp, #8]')
    def test_ldp_base_offset32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp w1, w2, [sp, #252]')
    def test_ldp_base_offset_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 252
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp w1, w2, [sp, #-256]')
    def test_ldp_base_offset_min32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp w1, w2, [sp], #8')
    def test_ldp_post_indexed32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldp w1, w2, [sp], #252')
    def test_ldp_post_indexed_max32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack + 252)

    @itest_custom('ldp w1, w2, [sp], #-256')
    def test_ldp_post_indexed_min32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldp w1, w2, [sp, #8]!')
    def test_ldp_pre_indexed32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldp w1, w2, [sp, #252]!')
    def test_ldp_pre_indexed_max32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 252
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack + 252)

    @itest_custom('ldp w1, w2, [sp, #-256]!')
    def test_ldp_pre_indexed_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 1094861636)
        self.assertEqual(self.rf.read('W2'), 1094861636)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldp x1, x2, [sp]')
    def test_ldp_base64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp x1, x2, [sp, #8]')
    def test_ldp_base_offset64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(7017280452245743464)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp x1, x2, [sp, #504]')
    def test_ldp_base_offset_max64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 504
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp x1, x2, [sp, #-512]')
    def test_ldp_base_offset_min64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 512
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldp x1, x2, [sp], #8')
    def test_ldp_post_indexed64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldp x1, x2, [sp], #504')
    def test_ldp_post_indexed_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack + 504)

    @itest_custom('ldp x1, x2, [sp], #-512')
    def test_ldp_post_indexed_min64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack - 512)

    @itest_custom('ldp x1, x2, [sp, #8]!')
    def test_ldp_pre_indexed64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(7017280452245743464)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldp x1, x2, [sp, #504]!')
    def test_ldp_pre_indexed_max64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 504
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack + 504)

    @itest_custom('ldp x1, x2, [sp, #-512]!')
    def test_ldp_pre_indexed_min64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 512
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('X2'), 5859837686836516696)
        self.assertEqual(self.rf.read('W2'), 1431721816)
        self.assertEqual(self.rf.read('SP'), stack - 512)

    @itest_custom('ldr w1, [sp]')
    def test_ldr_imm_base32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr w1, [sp, #8]')
    def test_ldr_imm_base_offset32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr w1, [sp, #16380]')
    def test_ldr_imm_base_offset_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 16380
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr w1, [sp], #8')
    def test_ldr_imm_post_indexed32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldr w1, [sp], #-256')
    def test_ldr_imm_post_indexed_neg32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldr w1, [sp, #8]!')
    def test_ldr_imm_pre_indexed32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldr w1, [sp, #-256]!')
    def test_ldr_imm_pre_indexed_neg32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldr x1, [sp]')
    def test_ldr_imm_base64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr x1, [sp, #8]')
    def test_ldr_imm_base_offset64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr x1, [sp, #32760]')
    def test_ldr_imm_base_offset_max64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 32760
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr x1, [sp], #8')
    def test_ldr_imm_post_indexed64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldr x1, [sp], #-256')
    def test_ldr_imm_post_indexed_neg64(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldr x1, [sp, #8]!')
    def test_ldr_imm_pre_indexed64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldr x1, [sp, #-256]!')
    def test_ldr_imm_pre_indexed_neg64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldr w0, .+8')
    def test_ldr_lit32(self):
        if False:
            print('Hello World!')
        self._setreg('PC', self.cpu.PC)
        self.cpu.STACK = self.cpu.PC + 16
        self.cpu.push_int(4702394921427289928)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_custom('ldr x0, .+8')
    def test_ldr_lit64(self):
        if False:
            return 10
        self._setreg('PC', self.cpu.PC)
        self.cpu.STACK = self.cpu.PC + 16
        self.cpu.push_int(4702394921427289928)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_custom('ldr w0, .-8')
    def test_ldr_lit_neg32(self):
        if False:
            for i in range(10):
                print('nop')
        insn = self.mem.read(self.code, 4)
        self.mem.write(self.code + 16, insn)
        self.cpu.PC += 16
        self._setreg('PC', self.cpu.PC)
        self.cpu.STACK = self.cpu.PC
        self.cpu.push_int(4702394921427289928)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_custom('ldr x0, .-8')
    def test_ldr_lit_neg64(self):
        if False:
            print('Hello World!')
        insn = self.mem.read(self.code, 4)
        self.mem.write(self.code + 16, insn)
        self.cpu.PC += 16
        self._setreg('PC', self.cpu.PC)
        self.cpu.STACK = self.cpu.PC
        self.cpu.push_int(4702394921427289928)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('W1=-8')
    @itest_custom('ldr w0, [sp, w1, uxtw]')
    def test_ldr_reg_uxtw32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 4294967288
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1431721816)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('W1=-8')
    @itest_custom('ldr w0, [sp, w1, uxtw #2]')
    def test_ldr_reg_uxtw2_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= LSL(4294967288, 2, 64)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1431721816)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('X1=8')
    @itest_custom('ldr w0, [sp, x1]')
    def test_ldr_reg32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=2')
    @itest_custom('ldr w0, [sp, x1, lsl #2]')
    def test_ldr_reg_lsl32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('W1=-8')
    @itest_custom('ldr w0, [sp, w1, sxtw]')
    def test_ldr_reg_sxtw32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1431721816)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('W1=-8')
    @itest_custom('ldr w0, [sp, w1, sxtw #2]')
    def test_ldr_reg_sxtw2_32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += LSL(8, 2, 64)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1431721816)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('X1=-8')
    @itest_custom('ldr w0, [sp, x1, sxtx]')
    def test_ldr_reg_sxtx32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1431721816)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('X1=-2')
    @itest_custom('ldr w0, [sp, x1, sxtx #2]')
    def test_ldr_reg_sxtx2_32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 1431721816)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('W1=-8')
    @itest_custom('ldr x0, [sp, w1, uxtw]')
    def test_ldr_reg_uxtw64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 4294967288
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 5859837686836516696)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('W1=-8')
    @itest_custom('ldr x0, [sp, w1, uxtw #3]')
    def test_ldr_reg_uxtw3_64(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= LSL(4294967288, 3, 64)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 5859837686836516696)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('X1=8')
    @itest_custom('ldr x0, [sp, x1]')
    def test_ldr_reg64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=2')
    @itest_custom('ldr x0, [sp, x1, lsl #3]')
    def test_ldr_reg_lsl64(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('W1=-8')
    @itest_custom('ldr x0, [sp, w1, sxtw]')
    def test_ldr_reg_sxtw64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 5859837686836516696)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('W1=-8')
    @itest_custom('ldr x0, [sp, w1, sxtw #3]')
    def test_ldr_reg_sxtw3_64(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += LSL(8, 3, 64)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 5859837686836516696)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('X1=-8')
    @itest_custom('ldr x0, [sp, x1, sxtx]')
    def test_ldr_reg_sxtx64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 5859837686836516696)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_setregs('X1=-2')
    @itest_custom('ldr x0, [sp, x1, sxtx #3]')
    def test_ldr_reg_sxtx3_64(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 16
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 5859837686836516696)
        self.assertEqual(self.rf.read('W0'), 1431721816)

    @itest_custom('ldrb w1, [sp]')
    def test_ldrb_imm_base32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 72)
        self.assertEqual(self.rf.read('W1'), 72)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrb w1, [sp, #8]')
    def test_ldrb_imm_base_offset32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 72)
        self.assertEqual(self.rf.read('W1'), 72)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrb w1, [sp, #4095]')
    def test_ldrb_imm_base_offset_max32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 4095
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 72)
        self.assertEqual(self.rf.read('W1'), 72)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrb w1, [sp], #8')
    def test_ldrb_imm_post_indexed32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 72)
        self.assertEqual(self.rf.read('W1'), 72)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldrb w1, [sp], #-256')
    def test_ldrb_imm_post_indexed_neg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 72)
        self.assertEqual(self.rf.read('W1'), 72)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldrb w1, [sp, #8]!')
    def test_ldrb_imm_pre_indexed32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 72)
        self.assertEqual(self.rf.read('W1'), 72)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldrb w1, [sp, #-256]!')
    def test_ldrb_imm_pre_indexed_neg32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 72)
        self.assertEqual(self.rf.read('W1'), 72)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W1=-8')
    @itest_custom('ldrb w0, [sp, w1, uxtw]')
    def test_ldrb_reg_uxtw32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 4294967288
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 88)
        self.assertEqual(self.rf.read('W0'), 88)

    @itest_setregs('W1=-8')
    @itest_custom('ldrb w0, [sp, w1, uxtw #0]')
    def test_ldrb_reg_uxtw0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 4294967288
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 88)
        self.assertEqual(self.rf.read('W0'), 88)

    @itest_setregs('X1=8')
    @itest_custom('ldrb w0, [sp, x1]')
    def test_ldrb_reg32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 72)
        self.assertEqual(self.rf.read('W0'), 72)

    @itest_setregs('X1=8')
    @itest_custom('ldrb w0, [sp, x1, lsl #0]')
    def test_ldrb_reg_lsl32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 72)
        self.assertEqual(self.rf.read('W0'), 72)

    @itest_setregs('W1=-8')
    @itest_custom('ldrb w0, [sp, w1, sxtw]')
    def test_ldrb_reg_sxtw32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 88)
        self.assertEqual(self.rf.read('W0'), 88)

    @itest_setregs('W1=-8')
    @itest_custom('ldrb w0, [sp, w1, sxtw #0]')
    def test_ldrb_reg_sxtw0_32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 88)
        self.assertEqual(self.rf.read('W0'), 88)

    @itest_setregs('X1=-8')
    @itest_custom('ldrb w0, [sp, x1, sxtx]')
    def test_ldrb_reg_sxtx32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 88)
        self.assertEqual(self.rf.read('W0'), 88)

    @itest_setregs('X1=-8')
    @itest_custom('ldrb w0, [sp, x1, sxtx #0]')
    def test_ldrb_reg_sxtx0_32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 88)
        self.assertEqual(self.rf.read('W0'), 88)

    @itest_setregs('X0=0x4142434445464749')
    @itest_custom(['strb w0, [sp]', 'ldrb w1, [sp]'], multiple_insts=True)
    def test_strb_ldrb_imm_base32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516681)
        self.assertEqual(self.rf.read('SP'), stack)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.rf.read('X1'), 73)
        self.assertEqual(self.rf.read('W1'), 73)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrh w1, [sp]')
    def test_ldrh_imm_base32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18248)
        self.assertEqual(self.rf.read('W1'), 18248)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrh w1, [sp, #8]')
    def test_ldrh_imm_base_offset32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18248)
        self.assertEqual(self.rf.read('W1'), 18248)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrh w1, [sp, #8190]')
    def test_ldrh_imm_base_offset_max32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 8190
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18248)
        self.assertEqual(self.rf.read('W1'), 18248)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrh w1, [sp], #8')
    def test_ldrh_imm_post_indexed32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18248)
        self.assertEqual(self.rf.read('W1'), 18248)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldrh w1, [sp], #-256')
    def test_ldrh_imm_post_indexed_neg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18248)
        self.assertEqual(self.rf.read('W1'), 18248)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldrh w1, [sp, #8]!')
    def test_ldrh_imm_pre_indexed32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18248)
        self.assertEqual(self.rf.read('W1'), 18248)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldrh w1, [sp, #-256]!')
    def test_ldrh_imm_pre_indexed_neg32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18248)
        self.assertEqual(self.rf.read('W1'), 18248)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W1=-8')
    @itest_custom('ldrh w0, [sp, w1, uxtw]')
    def test_ldrh_reg_uxtw32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 4294967288
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 22360)
        self.assertEqual(self.rf.read('W0'), 22360)

    @itest_setregs('W1=-4')
    @itest_custom('ldrh w0, [sp, w1, uxtw #1]')
    def test_ldrh_reg_uxtw1_32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= LSL(4294967292, 1, 64)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 22360)
        self.assertEqual(self.rf.read('W0'), 22360)

    @itest_setregs('X1=8')
    @itest_custom('ldrh w0, [sp, x1]')
    def test_ldrh_reg32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18248)
        self.assertEqual(self.rf.read('W0'), 18248)

    @itest_setregs('X1=4')
    @itest_custom('ldrh w0, [sp, x1, lsl #1]')
    def test_ldrh_reg_lsl32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18248)
        self.assertEqual(self.rf.read('W0'), 18248)

    @itest_setregs('W1=-8')
    @itest_custom('ldrh w0, [sp, w1, sxtw]')
    def test_ldrh_reg_sxtw32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 22360)
        self.assertEqual(self.rf.read('W0'), 22360)

    @itest_setregs('W1=-4')
    @itest_custom('ldrh w0, [sp, w1, sxtw #1]')
    def test_ldrh_reg_sxtw1_32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += LSL(4, 1, 64)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 22360)
        self.assertEqual(self.rf.read('W0'), 22360)

    @itest_setregs('X1=-8')
    @itest_custom('ldrh w0, [sp, x1, sxtx]')
    def test_ldrh_reg_sxtx32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 22360)
        self.assertEqual(self.rf.read('W0'), 22360)

    @itest_setregs('X1=-4')
    @itest_custom('ldrh w0, [sp, x1, sxtx #1]')
    def test_ldrh_reg_sxtx1_32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 22360)
        self.assertEqual(self.rf.read('W0'), 22360)

    @itest_custom('ldrsw x1, [sp]')
    def test_ldrsw_imm_base64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394922501031752)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18446744071650559816)
        self.assertEqual(self.rf.read('W1'), 2235975496)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrsw x1, [sp, #8]')
    def test_ldrsw_imm_base_offset64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394922501031752)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18446744071650559816)
        self.assertEqual(self.rf.read('W1'), 2235975496)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrsw x1, [sp, #16380]')
    def test_ldrsw_imm_base_offset_max64(self):
        if False:
            return 10
        self.cpu.push_int(4702394922501031752)
        self.cpu.STACK -= 16380
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18446744071650559816)
        self.assertEqual(self.rf.read('W1'), 2235975496)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldrsw x1, [sp], #8')
    def test_ldrsw_imm_post_indexed64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394922501031752)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18446744071650559816)
        self.assertEqual(self.rf.read('W1'), 2235975496)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldrsw x1, [sp], #-256')
    def test_ldrsw_imm_post_indexed_neg64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394922501031752)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18446744071650559816)
        self.assertEqual(self.rf.read('W1'), 2235975496)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldrsw x1, [sp, #8]!')
    def test_ldrsw_imm_pre_indexed64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394922501031752)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18446744071650559816)
        self.assertEqual(self.rf.read('W1'), 2235975496)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_custom('ldrsw x1, [sp, #-256]!')
    def test_ldrsw_imm_pre_indexed_neg64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394922501031752)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 18446744071650559816)
        self.assertEqual(self.rf.read('W1'), 2235975496)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_custom('ldrsw x0, .+8')
    def test_ldrsw_lit64(self):
        if False:
            for i in range(10):
                print('nop')
        self._setreg('PC', self.cpu.PC)
        self.cpu.STACK = self.cpu.PC + 16
        self.cpu.push_int(4702394922501031752)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071650559816)
        self.assertEqual(self.rf.read('W0'), 2235975496)

    @itest_custom('ldrsw x0, .-8')
    def test_ldrsw_lit_neg64(self):
        if False:
            print('Hello World!')
        insn = self.mem.read(self.code, 4)
        self.mem.write(self.code + 16, insn)
        self.cpu.PC += 16
        self._setreg('PC', self.cpu.PC)
        self.cpu.STACK = self.cpu.PC
        self.cpu.push_int(4702394922501031752)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071650559816)
        self.assertEqual(self.rf.read('W0'), 2235975496)

    @itest_setregs('W1=-8')
    @itest_custom('ldrsw x0, [sp, w1, uxtw]')
    def test_ldrsw_reg_uxtw64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837687641823064)
        self.cpu.STACK -= 4294967288
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071651612504)
        self.assertEqual(self.rf.read('W0'), 2237028184)

    @itest_setregs('W1=-8')
    @itest_custom('ldrsw x0, [sp, w1, uxtw #2]')
    def test_ldrsw_reg_uxtw2_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837687641823064)
        self.cpu.STACK -= LSL(4294967288, 2, 64)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071651612504)
        self.assertEqual(self.rf.read('W0'), 2237028184)

    @itest_setregs('X1=8')
    @itest_custom('ldrsw x0, [sp, x1]')
    def test_ldrsw_reg64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394922501031752)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071650559816)
        self.assertEqual(self.rf.read('W0'), 2235975496)

    @itest_setregs('X1=2')
    @itest_custom('ldrsw x0, [sp, x1, lsl #2]')
    def test_ldrsw_reg_lsl64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394922501031752)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071650559816)
        self.assertEqual(self.rf.read('W0'), 2235975496)

    @itest_setregs('W1=-8')
    @itest_custom('ldrsw x0, [sp, w1, sxtw]')
    def test_ldrsw_reg_sxtw64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837687641823064)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071651612504)
        self.assertEqual(self.rf.read('W0'), 2237028184)

    @itest_setregs('W1=-8')
    @itest_custom('ldrsw x0, [sp, w1, sxtw #2]')
    def test_ldrsw_reg_sxtw2_64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837687641823064)
        self.cpu.STACK += LSL(8, 2, 64)
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071651612504)
        self.assertEqual(self.rf.read('W0'), 2237028184)

    @itest_setregs('X1=-8')
    @itest_custom('ldrsw x0, [sp, x1, sxtx]')
    def test_ldrsw_reg_sxtx64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837687641823064)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071651612504)
        self.assertEqual(self.rf.read('W0'), 2237028184)

    @itest_setregs('X1=-2')
    @itest_custom('ldrsw x0, [sp, x1, sxtx #2]')
    def test_ldrsw_reg_sxtx2_64(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837687641823064)
        self.cpu.STACK += 8
        self._setreg('STACK', self.cpu.STACK)
        self._execute()
        self.assertEqual(self.rf.read('X0'), 18446744071651612504)
        self.assertEqual(self.rf.read('W0'), 2237028184)

    @itest_custom('ldr w1, [sp, #1]')
    def test_ldr_ldur32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1145390663)
        self.assertEqual(self.rf.read('W1'), 1145390663)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr w1, [sp, #-256]')
    def test_ldr_ldur_neg32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldur w1, [sp, #-256]')
    def test_ldur_min32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldur w1, [sp, #255]')
    def test_ldur_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 255
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldur w1, [sp, #1]')
    def test_ldur32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1145390663)
        self.assertEqual(self.rf.read('W1'), 1145390663)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr x1, [sp, #4]')
    def test_ldr_ldur64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4991755612914340692)
        self.assertEqual(self.rf.read('W1'), 1364349780)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldr x1, [sp, #-256]')
    def test_ldr_ldur_neg64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldur x1, [sp, #-256]')
    def test_ldur_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK += 256
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldur x1, [sp, #255]')
    def test_ldur_max64(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self.cpu.STACK -= 255
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldur x1, [sp, #4]')
    def test_ldur64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self.cpu.push_int(5859837686836516696)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4991755612914340692)
        self.assertEqual(self.rf.read('W1'), 1364349780)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldxr w1, [sp]')
    def test_ldxr32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldxr w1, [sp, #0]')
    def test_ldxr_0_32(self):
        if False:
            return 10
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 1162233672)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldxr x1, [sp]')
    def test_ldxr64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_custom('ldxr x1, [sp, #0]')
    def test_ldxr_0_64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(4702394921427289928)
        self._setreg('STACK', self.cpu.STACK)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.rf.read('X1'), 4702394921427289928)
        self.assertEqual(self.rf.read('W1'), 1162233672)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344', 'W2=0')
    @itest('lsl w0, w1, w2')
    def test_lsl_reg_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x44434241', 'W2=0xffffffff')
    @itest('lsl w0, w1, w2')
    def test_lsl_reg_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=4')
    @itest('lsl w0, w1, w2')
    def test_lsl_reg32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 337916992)
        self.assertEqual(self.rf.read('W0'), 337916992)

    @itest_setregs('X1=0x4142434445464748', 'X2=0')
    @itest('lsl x0, x1, x2')
    def test_lsl_reg_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4847464544434241', 'X2=0xffffffffffffffff')
    @itest('lsl x0, x1, x2')
    def test_lsl_reg_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=4')
    @itest('lsl x0, x1, x2')
    def test_lsl_reg64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1451342447998432384)
        self.assertEqual(self.rf.read('W0'), 1415869568)

    @itest_setregs('W1=0x41424344')
    @itest('lsl w0, w1, #0')
    def test_lsl_imm_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x44434241')
    @itest('lsl w0, w1, #31')
    def test_lsl_imm_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)

    @itest_setregs('W1=0x41424344')
    @itest('lsl w0, w1, #4')
    def test_lsl_imm32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 337916992)
        self.assertEqual(self.rf.read('W0'), 337916992)

    @itest_setregs('X1=0x4142434445464748')
    @itest('lsl x0, x1, #0')
    def test_lsl_imm_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4847464544434241')
    @itest('lsl x0, x1, #63')
    def test_lsl_imm_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('lsl x0, x1, #4')
    def test_lsl_imm64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1451342447998432384)
        self.assertEqual(self.rf.read('W0'), 1415869568)

    @itest_setregs('W1=0x41424344', 'W2=0')
    @itest('lslv w0, w1, w2')
    def test_lslv_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x44434241', 'W2=0xffffffff')
    @itest('lslv w0, w1, w2')
    def test_lslv_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=4')
    @itest('lslv w0, w1, w2')
    def test_lslv32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 337916992)
        self.assertEqual(self.rf.read('W0'), 337916992)

    @itest_setregs('X1=0x4142434445464748', 'X2=0')
    @itest('lslv x0, x1, x2')
    def test_lslv_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4847464544434241', 'X2=0xffffffffffffffff')
    @itest('lslv x0, x1, x2')
    def test_lslv_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=4')
    @itest('lslv x0, x1, x2')
    def test_lslv64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1451342447998432384)
        self.assertEqual(self.rf.read('W0'), 1415869568)

    @itest_setregs('W1=0x41424344', 'W2=0')
    @itest('lsr w0, w1, w2')
    def test_lsr_reg_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x81424344', 'W2=0xffffffff')
    @itest('lsr w0, w1, w2')
    def test_lsr_reg_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=0x41424344', 'W2=4')
    @itest('lsr w0, w1, w2')
    def test_lsr_reg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 68428852)
        self.assertEqual(self.rf.read('W0'), 68428852)

    @itest_setregs('X1=0x4142434445464748', 'X2=0')
    @itest('lsr x0, x1, x2')
    def test_lsr_reg_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x8142434445464748', 'X2=0xffffffffffffffff')
    @itest('lsr x0, x1, x2')
    def test_lsr_reg_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0x4142434445464748', 'X2=4')
    @itest('lsr x0, x1, x2')
    def test_lsr_reg64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 293899682589205620)
        self.assertEqual(self.rf.read('W0'), 1146381428)

    @itest_setregs('W1=0x41424344')
    @itest('lsr w0, w1, #0')
    def test_lsr_imm_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x81424344')
    @itest('lsr w0, w1, #31')
    def test_lsr_imm_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=0x41424344')
    @itest('lsr w0, w1, #4')
    def test_lsr_imm32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 68428852)
        self.assertEqual(self.rf.read('W0'), 68428852)

    @itest_setregs('X1=0x4142434445464748')
    @itest('lsr x0, x1, #0')
    def test_lsr_imm_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x8142434445464748')
    @itest('lsr x0, x1, #63')
    def test_lsr_imm_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0x4142434445464748')
    @itest('lsr x0, x1, #4')
    def test_lsr_imm64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 293899682589205620)
        self.assertEqual(self.rf.read('W0'), 1146381428)

    @itest_setregs('W1=0x41424344', 'W2=0')
    @itest('lsrv w0, w1, w2')
    def test_lsrv_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x81424344', 'W2=0xffffffff')
    @itest('lsrv w0, w1, w2')
    def test_lsrv_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=0x41424344', 'W2=4')
    @itest('lsrv w0, w1, w2')
    def test_lsrv32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 68428852)
        self.assertEqual(self.rf.read('W0'), 68428852)

    @itest_setregs('X1=0x4142434445464748', 'X2=0')
    @itest('lsrv x0, x1, x2')
    def test_lsrv_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x8142434445464748', 'X2=0xffffffffffffffff')
    @itest('lsrv x0, x1, x2')
    def test_lsrv_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0x4142434445464748', 'X2=4')
    @itest('lsrv x0, x1, x2')
    def test_lsrv64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 293899682589205620)
        self.assertEqual(self.rf.read('W0'), 1146381428)

    @itest_setregs('W1=0xffffffff', 'W2=0xffffffff', 'W3=0xffffffff')
    @itest('madd w0, w1, w2, w3')
    def test_madd_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=-1', 'W2=-1', 'W3=-1')
    @itest('madd w0, w1, w2, w3')
    def test_madd_neg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=1', 'W2=1', 'W3=0xffffffff')
    @itest('madd w0, w1, w2, w3')
    def test_madd_of1_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=2', 'W3=0xffffffff')
    @itest('madd w0, w1, w2, w3')
    def test_madd_of2_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4294967293)
        self.assertEqual(self.rf.read('W0'), 4294967293)

    @itest_setregs('W1=2', 'W2=3', 'W3=4')
    @itest('madd w0, w1, w2, w3')
    def test_madd32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 10)
        self.assertEqual(self.rf.read('W0'), 10)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0xffffffffffffffff', 'X3=0xffffffffffffffff')
    @itest('madd x0, x1, x2, x3')
    def test_madd_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=-1', 'X2=-1', 'X3=-1')
    @itest('madd x0, x1, x2, x3')
    def test_madd_neg64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=1', 'X2=1', 'X3=0xffffffffffffffff')
    @itest('madd x0, x1, x2, x3')
    def test_madd_of1_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=2', 'X3=0xffffffffffffffff')
    @itest('madd x0, x1, x2, x3')
    def test_madd_of2_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709551613)
        self.assertEqual(self.rf.read('W0'), 4294967293)

    @itest_setregs('X1=2', 'X2=3', 'X3=4')
    @itest('madd x0, x1, x2, x3')
    def test_madd64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 10)
        self.assertEqual(self.rf.read('W0'), 10)

    @itest_setregs(f'X0={MAGIC_64}')
    @itest('mov sp, x0')
    def test_mov_to_sp(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('SP'), MAGIC_64)
        self.assertEqual(self.rf.read('WSP'), MAGIC_32)

    @itest_custom('mov x0, sp')
    def test_mov_from_sp(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('SP', MAGIC_64)
        self._setreg('SP', self.cpu.SP)
        self._execute()
        self.assertEqual(self.rf.read('X0'), MAGIC_64)
        self.assertEqual(self.rf.read('W0'), MAGIC_32)

    @itest_setregs(f'W0={MAGIC_32}')
    @itest('mov wsp, w0')
    def test_mov_to_sp32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('SP'), MAGIC_32)
        self.assertEqual(self.rf.read('WSP'), MAGIC_32)

    @itest_custom('mov w0, wsp')
    def test_mov_from_sp32(self):
        if False:
            return 10
        self.rf.write('WSP', MAGIC_32)
        self._setreg('WSP', self.cpu.WSP)
        self._execute()
        self.assertEqual(self.rf.read('X0'), MAGIC_32)
        self.assertEqual(self.rf.read('W0'), MAGIC_32)

    @skip_sym('immediate')
    @itest('mov w0, #0xffffffff')
    def test_mov_inv_wide_imm32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @skip_sym('immediate')
    @itest('mov w0, #-1')
    def test_mov_inv_wide_imm_neg32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @skip_sym('immediate')
    @itest('mov x0, #0xffffffffffffffff')
    def test_mov_inv_wide_imm64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @skip_sym('immediate')
    @itest('mov x0, #-1')
    def test_mov_inv_wide_imm_neg64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @skip_sym('immediate')
    @itest('mov w0, #0')
    def test_mov_wide_imm_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @skip_sym('immediate')
    @itest('mov w0, #0xffff0000')
    def test_mov_wide_imm_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294901760)
        self.assertEqual(self.rf.read('W0'), 4294901760)

    @skip_sym('immediate')
    @itest('mov w0, #1')
    def test_mov_wide_imm32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @skip_sym('immediate')
    @itest('mov x0, #0')
    def test_mov_wide_imm_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @skip_sym('immediate')
    @itest('mov x0, #0xffff000000000000')
    def test_mov_wide_imm_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446462598732840960)
        self.assertEqual(self.rf.read('W0'), 0)

    @skip_sym('immediate')
    @itest('mov x0, #1')
    def test_mov_wide_imm64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @skip_sym('immediate')
    @itest('mov w0, #0x7ffffffe')
    def test_mov_bmask_imm32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 2147483646)
        self.assertEqual(self.rf.read('W0'), 2147483646)

    @skip_sym('immediate')
    @itest('mov x0, #0x7ffffffffffffffe')
    def test_mov_bmask_imm64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 9223372036854775806)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X1=42')
    @itest('mov x0, x1')
    def test_mov_reg(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 42)
        self.assertEqual(self.rf.read('W0'), 42)

    @itest_setregs('W1=42')
    @itest('mov w0, w1')
    def test_mov_reg32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 42)
        self.assertEqual(self.rf.read('W0'), 42)

    def test_mov_to_general(self):
        if False:
            for i in range(10):
                print('nop')
        self._umov(mnem='mov', dst='w', vess='s', elem_size=32, elem_count=4)
        self._umov(mnem='mov', dst='x', vess='d', elem_size=64, elem_count=2)

    @skip_sym('immediate')
    @itest_multiple(['movn x0, #0', 'mov w0, #1'])
    def test_mov_same_reg32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0')
    def test_movk_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094844416)
        self.assertEqual(self.rf.read('W0'), 1094844416)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0xffff')
    def test_movk_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094909951)
        self.assertEqual(self.rf.read('W0'), 1094909951)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0x1001')
    def test_movk32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094848513)
        self.assertEqual(self.rf.read('W0'), 1094848513)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0, lsl #0')
    def test_movk_sft0_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094844416)
        self.assertEqual(self.rf.read('W0'), 1094844416)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0xffff, lsl #0')
    def test_movk_sft0_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094909951)
        self.assertEqual(self.rf.read('W0'), 1094909951)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0x1001, lsl #0')
    def test_movk_sft0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094848513)
        self.assertEqual(self.rf.read('W0'), 1094848513)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0, lsl #16')
    def test_movk_sft16_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0xffff, lsl #16')
    def test_movk_sft16_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294918980)
        self.assertEqual(self.rf.read('W0'), 4294918980)

    @skip_sym('immediate')
    @itest_setregs('W0=0x41424344')
    @itest('movk w0, #0x1001, lsl #16')
    def test_movk_sft16_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 268518212)
        self.assertEqual(self.rf.read('W0'), 268518212)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0')
    def test_movk_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427271680)
        self.assertEqual(self.rf.read('W0'), 1162215424)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0xffff')
    def test_movk_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427337215)
        self.assertEqual(self.rf.read('W0'), 1162280959)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0x1001')
    def test_movk64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427275777)
        self.assertEqual(self.rf.read('W0'), 1162219521)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0, lsl #0')
    def test_movk_sft0_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427271680)
        self.assertEqual(self.rf.read('W0'), 1162215424)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0xffff, lsl #0')
    def test_movk_sft0_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427337215)
        self.assertEqual(self.rf.read('W0'), 1162280959)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0x1001, lsl #0')
    def test_movk_sft0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427275777)
        self.assertEqual(self.rf.read('W0'), 1162219521)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0, lsl #16')
    def test_movk_sft16_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394920265074504)
        self.assertEqual(self.rf.read('W0'), 18248)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0xffff, lsl #16')
    def test_movk_sft16_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394924559976264)
        self.assertEqual(self.rf.read('W0'), 4294920008)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0x1001, lsl #16')
    def test_movk_sft16_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394920533575496)
        self.assertEqual(self.rf.read('W0'), 268519240)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0, lsl #32')
    def test_movk_sft32_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702320962090452808)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0xffff, lsl #32')
    def test_movk_sft32_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702602432772196168)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0x1001, lsl #32')
    def test_movk_sft32_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702338558571464520)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0, lsl #48')
    def test_movk_sft48_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 73960499070792)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0xffff, lsl #48')
    def test_movk_sft48_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446536559231911752)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @skip_sym('immediate')
    @itest_setregs('X0=0x4142434445464748')
    @itest('movk x0, #0x1001, lsl #48')
    def test_movk_sft48_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1153276940082628424)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @skip_sym('immediate')
    @itest('movn w0, #0')
    def test_movn32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @skip_sym('immediate')
    @itest('movn w0, #65535')
    def test_movn_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4294901760)
        self.assertEqual(self.rf.read('W0'), 4294901760)

    @skip_sym('immediate')
    @itest('movn w0, #65535, lsl #16')
    def test_movn_sft16_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 65535)
        self.assertEqual(self.rf.read('W0'), 65535)

    @skip_sym('immediate')
    @itest('movn x0, #0')
    def test_movn64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @skip_sym('immediate')
    @itest('movn x0, #65535')
    def test_movn_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 18446744073709486080)
        self.assertEqual(self.rf.read('W0'), 4294901760)

    @skip_sym('immediate')
    @itest('movn x0, #65535, lsl #16')
    def test_movn_sft16_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446744069414649855)
        self.assertEqual(self.rf.read('W0'), 65535)

    @skip_sym('immediate')
    @itest('movn x0, #65535, lsl #32')
    def test_movn_sft32_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446462603027808255)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @skip_sym('immediate')
    @itest('movn x0, #65535, lsl #48')
    def test_movn_sft48_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 281474976710655)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @skip_sym('immediate')
    @itest('movz w0, #0')
    def test_movz32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @skip_sym('immediate')
    @itest('movz w0, #65535')
    def test_movz_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 65535)
        self.assertEqual(self.rf.read('W0'), 65535)

    @skip_sym('immediate')
    @itest('movz w0, #65535, lsl #16')
    def test_movz_sft16_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294901760)
        self.assertEqual(self.rf.read('W0'), 4294901760)

    @skip_sym('immediate')
    @itest('movz x0, #0')
    def test_movz64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @skip_sym('immediate')
    @itest('movz x0, #65535')
    def test_movz_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 65535)
        self.assertEqual(self.rf.read('W0'), 65535)

    @skip_sym('immediate')
    @itest('movz x0, #65535, lsl #16')
    def test_movz_sft16_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294901760)
        self.assertEqual(self.rf.read('W0'), 4294901760)

    @skip_sym('immediate')
    @itest('movz x0, #65535, lsl #32')
    def test_movz_sft32_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 281470681743360)
        self.assertEqual(self.rf.read('W0'), 0)

    @skip_sym('immediate')
    @itest('movz x0, #65535, lsl #48')
    def test_movz_sft48_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 18446462598732840960)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom(['msr tpidr_el0, x1', 'mrs x0, tpidr_el0'], multiple_insts=True)
    def test_msr_mrs_tpidr_el0(self):
        if False:
            i = 10
            return i + 15
        self._execute()
        self._execute(reset=False)
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom(['msr s3_3_c13_c0_2, x1', 'mrs x0, s3_3_c13_c0_2'], multiple_insts=True)
    def test_msr_mrs_tpidr_el0_s(self):
        if False:
            for i in range(10):
                print('nop')
        self._execute()
        self._execute(reset=False)
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('W1=0xffffffff', 'W2=0xffffffff', 'W3=0xffffffff')
    @itest('msub w0, w1, w2, w3')
    def test_msub_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4294967294)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('W1=-1', 'W2=-1', 'W3=-1')
    @itest('msub w0, w1, w2, w3')
    def test_msub_neg32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967294)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('W1=0xffffffff', 'W2=2', 'W3=1')
    @itest('msub w0, w1, w2, w3')
    def test_msub_of32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3)
        self.assertEqual(self.rf.read('W0'), 3)

    @itest_setregs('W1=3', 'W2=4', 'W3=5')
    @itest('msub w0, w1, w2, w3')
    def test_msub32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967289)
        self.assertEqual(self.rf.read('W0'), 4294967289)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0xffffffffffffffff', 'X3=0xffffffffffffffff')
    @itest('msub x0, x1, x2, x3')
    def test_msub_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X1=-1', 'X2=-1', 'X3=-1')
    @itest('msub x0, x1, x2, x3')
    def test_msub_neg64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=2', 'X3=1')
    @itest('msub x0, x1, x2, x3')
    def test_msub_of64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3)
        self.assertEqual(self.rf.read('W0'), 3)

    @itest_setregs('X1=3', 'X2=4', 'X3=5')
    @itest('msub x0, x1, x2, x3')
    def test_msub64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 18446744073709551609)
        self.assertEqual(self.rf.read('W0'), 4294967289)

    @itest_setregs('W1=0xffffffff', 'W2=0xffffffff')
    @itest('mul w0, w1, w2')
    def test_mul_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=-1', 'W2=-1')
    @itest('mul w0, w1, w2')
    def test_mul_neg32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=0x80000000', 'W2=2')
    @itest('mul w0, w1, w2')
    def test_mul_of32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=2', 'W2=3')
    @itest('mul w0, w1, w2')
    def test_mul32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 6)
        self.assertEqual(self.rf.read('W0'), 6)

    @itest_setregs('W1=2', 'W2=-3')
    @itest('mul w0, w1, w2')
    def test_mul_pos_neg32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294967290)
        self.assertEqual(self.rf.read('W0'), 4294967290)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0xffffffffffffffff')
    @itest('mul x0, x1, x2')
    def test_mul_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=-1', 'X2=-1')
    @itest('mul x0, x1, x2')
    def test_mul_neg64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0x8000000000000000', 'X2=2')
    @itest('mul x0, x1, x2')
    def test_mul_of64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=2', 'X2=3')
    @itest('mul x0, x1, x2')
    def test_mul64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 6)
        self.assertEqual(self.rf.read('W0'), 6)

    @itest_setregs('X1=2', 'X2=-3')
    @itest('mul x0, x1, x2')
    def test_mul_pos_neg64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551610)
        self.assertEqual(self.rf.read('W0'), 4294967290)

    @itest_setregs('W1=0x41424344')
    @itest('neg w0, w1')
    def test_neg_sft_reg32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3200105660)
        self.assertEqual(self.rf.read('W0'), 3200105660)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('neg w0, w1, lsl #0')
    def test_neg_sft_reg_lsl_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3200105660)
        self.assertEqual(self.rf.read('W0'), 3200105660)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=1')
    @itest('neg w0, w1, lsl #31')
    def test_neg_sft_reg_lsl_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('neg w0, w1, lsl #1')
    def test_neg_sft_reg_lsl32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 2105244024)
        self.assertEqual(self.rf.read('W0'), 2105244024)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('neg w0, w1, lsr #0')
    def test_neg_sft_reg_lsr_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3200105660)
        self.assertEqual(self.rf.read('W0'), 3200105660)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x80000000')
    @itest('neg w0, w1, lsr #31')
    def test_neg_sft_reg_lsr_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x80000000')
    @itest('neg w0, w1, lsr #1')
    def test_neg_sft_reg_lsr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3221225472)
        self.assertEqual(self.rf.read('W0'), 3221225472)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('neg w0, w1, asr #0')
    def test_neg_sft_reg_asr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3200105660)
        self.assertEqual(self.rf.read('W0'), 3200105660)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x80000000')
    @itest('neg w0, w1, asr #31')
    def test_neg_sft_reg_asr_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x80000000')
    @itest('neg w0, w1, asr #1')
    def test_neg_sft_reg_asr32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1073741824)
        self.assertEqual(self.rf.read('W0'), 1073741824)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('neg x0, x1')
    def test_neg_sft_reg64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 13744349152282261688)
        self.assertEqual(self.rf.read('W0'), 3132733624)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('neg x0, x1, lsl #0')
    def test_neg_sft_reg_lsl_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13744349152282261688)
        self.assertEqual(self.rf.read('W0'), 3132733624)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=1')
    @itest('neg x0, x1, lsl #63')
    def test_neg_sft_reg_lsl_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('neg x0, x1, lsl #1')
    def test_neg_sft_reg_lsl64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 9041954230854971760)
        self.assertEqual(self.rf.read('W0'), 1970499952)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('neg x0, x1, lsr #0')
    def test_neg_sft_reg_lsr_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 13744349152282261688)
        self.assertEqual(self.rf.read('W0'), 3132733624)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x8000000000000000')
    @itest('neg x0, x1, lsr #63')
    def test_neg_sft_reg_lsr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x8000000000000000')
    @itest('neg x0, x1, lsr #1')
    def test_neg_sft_reg_lsr64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 13835058055282163712)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('neg x0, x1, asr #0')
    def test_neg_sft_reg_asr_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13744349152282261688)
        self.assertEqual(self.rf.read('W0'), 3132733624)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x8000000000000000')
    @itest('neg x0, x1, asr #63')
    def test_neg_sft_reg_asr_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x8000000000000000')
    @itest('neg x0, x1, asr #1')
    def test_neg_sft_reg_asr64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4611686018427387904)
        self.assertEqual(self.rf.read('W0'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_custom('nop')
    def test_nop(self):
        if False:
            return 10
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)

    @itest_setregs('W1=0x41420000')
    @itest('orr w0, w1, #0xffff')
    def test_orr_imm32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094909951)
        self.assertEqual(self.rf.read('W0'), 1094909951)

    @itest_setregs('W1=0x00004344')
    @itest('orr w0, w1, #0xffff0000')
    def test_orr_imm2_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294918980)
        self.assertEqual(self.rf.read('W0'), 4294918980)

    @itest_setregs('W1=0x41424344')
    @itest('orr w0, w1, #1')
    def test_orr_imm3_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)

    @itest_setregs('X1=0x0000414200004344')
    @itest('orr x0, x1, #0xffff0000ffff0000')
    def test_orr_imm64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446534354751406916)
        self.assertEqual(self.rf.read('W0'), 4294918980)

    @itest_setregs('X1=0x4142000043440000')
    @itest('orr x0, x1, #0x0000ffff0000ffff')
    def test_orr_imm2_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702602432738557951)
        self.assertEqual(self.rf.read('W0'), 1128595455)

    @itest_setregs('X1=0x4142434445464748')
    @itest('orr x0, x1, #1')
    def test_orr_imm3_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)

    @itest_setregs('W1=0x41420000', 'W2=0x4344')
    @itest('orr w0, w1, w2')
    def test_orr_sft_reg32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x41420000', 'W2=0x4344')
    @itest('orr w0, w1, w2, lsl #0')
    def test_orr_sft_reg_lsl_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x4142', 'W2=1')
    @itest('orr w0, w1, w2, lsl #31')
    def test_orr_sft_reg_lsl_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 2147500354)
        self.assertEqual(self.rf.read('W0'), 2147500354)

    @itest_setregs('W1=0x41420000', 'W2=0x4344')
    @itest('orr w0, w1, w2, lsl #1')
    def test_orr_sft_reg_lsl32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094878856)
        self.assertEqual(self.rf.read('W0'), 1094878856)

    @itest_setregs('W1=0x41420000', 'W2=0x4344')
    @itest('orr w0, w1, w2, lsr #0')
    def test_orr_sft_reg_lsr_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x41420000', 'W2=0x80000000')
    @itest('orr w0, w1, w2, lsr #31')
    def test_orr_sft_reg_lsr_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094844417)
        self.assertEqual(self.rf.read('W0'), 1094844417)

    @itest_setregs('W1=0x4142', 'W2=0x80000000')
    @itest('orr w0, w1, w2, lsr #1')
    def test_orr_sft_reg_lsr32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1073758530)
        self.assertEqual(self.rf.read('W0'), 1073758530)

    @itest_setregs('W1=0x41420000', 'W2=0x4344')
    @itest('orr w0, w1, w2, asr #0')
    def test_orr_sft_reg_asr_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x41420000', 'W2=0x80000000')
    @itest('orr w0, w1, w2, asr #31')
    def test_orr_sft_reg_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0x4142', 'W2=0x80000000')
    @itest('orr w0, w1, w2, asr #1')
    def test_orr_sft_reg_asr32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3221242178)
        self.assertEqual(self.rf.read('W0'), 3221242178)

    @itest_setregs('W1=0x41420000', 'W2=0x4344')
    @itest('orr w0, w1, w2, ror #0')
    def test_orr_sft_reg_ror_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x4140', 'W2=0x80000001')
    @itest('orr w0, w1, w2, ror #31')
    def test_orr_sft_reg_ror_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 16707)
        self.assertEqual(self.rf.read('W0'), 16707)

    @itest_setregs('W1=0x4142', 'W2=1')
    @itest('orr w0, w1, w2, ror #1')
    def test_orr_sft_reg_ror32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 2147500354)
        self.assertEqual(self.rf.read('W0'), 2147500354)

    @itest_setregs('X1=0x4142434400000000', 'X2=0x45464748')
    @itest('orr x0, x1, x2')
    def test_orr_sft_reg64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4142434400000000', 'X2=0x45464748')
    @itest('orr x0, x1, x2, lsl #0')
    def test_orr_sft_reg_lsl_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x41424344', 'X2=1')
    @itest('orr x0, x1, x2, lsl #63')
    def test_orr_sft_reg_lsl_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 9223372037949637444)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('X1=0x4142434400000000', 'X2=0x45464748')
    @itest('orr x0, x1, x2, lsl #1')
    def test_orr_sft_reg_lsl64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394922589523600)
        self.assertEqual(self.rf.read('W0'), 2324467344)

    @itest_setregs('X1=0x4142434400000000', 'X2=0x45464748')
    @itest('orr x0, x1, x2, lsr #0')
    def test_orr_sft_reg_lsr_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4142434400000000', 'X2=0x8000000000000000')
    @itest('orr x0, x1, x2, lsr #63')
    def test_orr_sft_reg_lsr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394920265056257)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0x41424344', 'X2=0x8000000000000000')
    @itest('orr x0, x1, x2, lsr #1')
    def test_orr_sft_reg_lsr64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4611686019522249540)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('X1=0x4142434400000000', 'X2=0x45464748')
    @itest('orr x0, x1, x2, asr #0')
    def test_orr_sft_reg_asr_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4142434400000000', 'X2=0x8000000000000000')
    @itest('orr x0, x1, x2, asr #63')
    def test_orr_sft_reg_asr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0x41424344', 'X2=0x8000000000000000')
    @itest('orr x0, x1, x2, asr #1')
    def test_orr_sft_reg_asr64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13835058056377025348)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('X1=0x4142434400000000', 'X2=0x45464748')
    @itest('orr x0, x1, x2, ror #0')
    def test_orr_sft_reg_ror_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4142434445464740', 'X2=0x8000000000000001')
    @itest('orr x0, x1, x2, ror #63')
    def test_orr_sft_reg_ror_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289923)
        self.assertEqual(self.rf.read('W0'), 1162233667)

    @itest_setregs('X1=0x41424344', 'X2=1')
    @itest('orr x0, x1, x2, ror #1')
    def test_orr_sft_reg_ror64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 9223372037949637444)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('V1=0x81008300850087009100930095009700', 'V2=0x00820084008600880092009400960098')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'orr v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_orr_vector_8b(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 10489608748473423768)
        self.assertEqual(self.rf.read('Q0'), 10489608748473423768)
        self.assertEqual(self.rf.read('D0'), 10489608748473423768)
        self.assertEqual(self.rf.read('S0'), 2509674392)
        self.assertEqual(self.rf.read('H0'), 38808)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('V1=0x81008300850087009100930095009700', 'V2=0x00820084008600880092009400960098')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'orr v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_orr_vector_16b(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 172148077542963347852807419361565775768)
        self.assertEqual(self.rf.read('Q0'), 172148077542963347852807419361565775768)
        self.assertEqual(self.rf.read('D0'), 10489608748473423768)
        self.assertEqual(self.rf.read('S0'), 2509674392)
        self.assertEqual(self.rf.read('H0'), 38808)
        self.assertEqual(self.rf.read('B0'), 152)

    @itest_setregs('W1=0')
    @itest('rbit w0, w1')
    def test_rbit_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=0xffffffff')
    @itest('rbit w0, w1')
    def test_rbit_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=1')
    @itest('rbit w0, w1')
    def test_rbit_one32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)

    @itest_setregs('W1=0x41424344')
    @itest('rbit w0, w1')
    def test_rbit32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 583156354)
        self.assertEqual(self.rf.read('W0'), 583156354)

    @itest_setregs('X1=0')
    @itest('rbit x0, x1')
    def test_rbit_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0xffffffffffffffff')
    @itest('rbit x0, x1')
    def test_rbit_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=1')
    @itest('rbit x0, x1')
    def test_rbit_one64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('rbit x0, x1')
    def test_rbit64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1360758485926691458)
        self.assertEqual(self.rf.read('W0'), 583156354)

    @itest_custom('ret')
    def test_ret(self):
        if False:
            return 10
        pc = self.cpu.PC
        self.cpu.X30 = pc + 16
        self._setreg('X30', self.cpu.X30)
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 16)

    @itest_custom('ret X0')
    def test_ret_reg(self):
        if False:
            for i in range(10):
                print('nop')
        pc = self.cpu.PC
        self.cpu.X0 = pc + 32
        self._setreg('X0', self.cpu.X0)
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 32)

    @itest_setregs('W1=0')
    @itest('rev w0, w1')
    def test_rev_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=0xffffffff')
    @itest('rev w0, w1')
    def test_rev_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0x41424344')
    @itest('rev w0, w1')
    def test_rev32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1145258561)
        self.assertEqual(self.rf.read('W0'), 1145258561)

    @itest_setregs('X1=0')
    @itest('rev x0, x1')
    def test_rev_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0xffffffffffffffff')
    @itest('rev x0, x1')
    def test_rev_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0x4142434445464748')
    @itest('rev x0, x1')
    def test_rev64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 5208208757389214273)
        self.assertEqual(self.rf.read('W0'), 1145258561)

    @itest_setregs('W1=0x44434241')
    @itest('sbfiz w0, w1, #0, #1')
    def test_sbfiz_min_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0x41424344')
    @itest('sbfiz w0, w1, #0, #32')
    def test_sbfiz_min_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x44434241')
    @itest('sbfiz w0, w1, #31, #1')
    def test_sbfiz_max_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)

    @itest_setregs('W1=0x41427fff')
    @itest('sbfiz w0, w1, #17, #15')
    def test_sbfiz32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294836224)
        self.assertEqual(self.rf.read('W0'), 4294836224)

    @itest_setregs('X1=0x4847464544434241')
    @itest('sbfiz x0, x1, #0, #1')
    def test_sbfiz_min_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sbfiz x0, x1, #0, #64')
    def test_sbfiz_min_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4847464544434241')
    @itest('sbfiz x0, x1, #63, #1')
    def test_sbfiz_max_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0x414243447fffffff')
    @itest('sbfiz x0, x1, #33, #31')
    def test_sbfiz64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 18446744065119617024)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424328')
    @itest('sbfm w0, w1, #3, #5')
    def test_sbfm_ge32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967293)
        self.assertEqual(self.rf.read('W0'), 4294967293)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424349')
    @itest('sbfm w0, w1, #5, #3')
    def test_sbfm_lt32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3355443200)
        self.assertEqual(self.rf.read('W0'), 3355443200)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('sbfm w0, w1, #0, #31')
    def test_sbfm_ge_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W0=0xffffffff', 'W1=0x44434241')
    @itest('sbfm w0, w1, #31, #0')
    def test_sbfm_lt_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967294)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('W0=0xffffffff', 'W1=0x44434241')
    @itest('sbfm w0, w1, #0, #0')
    def test_sbfm_ge_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W0=0xffffffff', 'W1=0x44434241')
    @itest('sbfm w0, w1, #1, #0')
    def test_sbfm_lt_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('sbfm w0, w1, #0, #7')
    def test_sbfm_sxtb_zero32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 68)
        self.assertEqual(self.rf.read('W0'), 68)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424384')
    @itest('sbfm w0, w1, #0, #7')
    def test_sbfm_sxtb_one32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294967172)
        self.assertEqual(self.rf.read('W0'), 4294967172)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('sbfm w0, w1, #0, #15')
    def test_sbfm_sxth_zero32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)

    @itest_setregs('W0=0xffffffff', 'W1=0x41428344')
    @itest('sbfm w0, w1, #0, #15')
    def test_sbfm_sxth_one32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4294935364)
        self.assertEqual(self.rf.read('W0'), 4294935364)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464728')
    @itest('sbfm x0, x1, #3, #5')
    def test_sbfm_ge64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551613)
        self.assertEqual(self.rf.read('W0'), 4294967293)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464749')
    @itest('sbfm x0, x1, #5, #3')
    def test_sbfm_lt64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 14411518807585587200)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464748')
    @itest('sbfm x0, x1, #0, #63')
    def test_sbfm_ge_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4847464544434241')
    @itest('sbfm x0, x1, #63, #0')
    def test_sbfm_lt_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4847464544434241')
    @itest('sbfm x0, x1, #0, #0')
    def test_sbfm_ge_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4847464544434241')
    @itest('sbfm x0, x1, #1, #0')
    def test_sbfm_lt_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464748')
    @itest('sbfm x0, x1, #0, #7')
    def test_sbfm_sxtb_zero64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 72)
        self.assertEqual(self.rf.read('W0'), 72)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464788')
    @itest('sbfm x0, x1, #0, #7')
    def test_sbfm_sxtb_one64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 18446744073709551496)
        self.assertEqual(self.rf.read('W0'), 4294967176)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464748')
    @itest('sbfm x0, x1, #0, #15')
    def test_sbfm_sxth_zero64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18248)
        self.assertEqual(self.rf.read('W0'), 18248)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445468748')
    @itest('sbfm x0, x1, #0, #15')
    def test_sbfm_sxth_one64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709520712)
        self.assertEqual(self.rf.read('W0'), 4294936392)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464748')
    @itest('sbfm x0, x1, #0, #31')
    def test_sbfm_sxtw_zero(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434485464748')
    @itest('sbfm x0, x1, #0, #31')
    def test_sbfm_sxtw_one(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744071650559816)
        self.assertEqual(self.rf.read('W0'), 2235975496)

    @itest_setregs('W1=0x44434241')
    @itest('sbfx w0, w1, #0, #1')
    def test_sbfx_min_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0x41424344')
    @itest('sbfx w0, w1, #0, #32')
    def test_sbfx_min_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x81424344')
    @itest('sbfx w0, w1, #31, #1')
    def test_sbfx_max_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0xffff4344')
    @itest('sbfx w0, w1, #16, #16')
    def test_sbfx32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0x4847464544434241')
    @itest('sbfx x0, x1, #0, #1')
    def test_sbfx_min_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sbfx x0, x1, #0, #64')
    def test_sbfx_min_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x8142434445464748')
    @itest('sbfx x0, x1, #63, #1')
    def test_sbfx_max_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=0xffffffff45464748')
    @itest('sbfx x0, x1, #32, #32')
    def test_sbfx64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709551615)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp]')
    def test_stp_base32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp, #8]')
    def test_stp_base_offset32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp, #252]')
    def test_stp_base_offset_max32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 252
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 252), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp, #-256]')
    def test_stp_base_offset_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 256), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp], #8')
    def test_stp_post_indexed32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp], #252')
    def test_stp_post_indexed_max32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 252)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp], #-256')
    def test_stp_post_indexed_min32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp, #8]!')
    def test_stp_pre_indexed32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp, #252]!')
    def test_stp_pre_indexed_max32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 252
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 252), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 252)

    @itest_setregs('W1=0x45464748', 'W2=0x41424344')
    @itest_custom('stp w1, w2, [sp, #-256]!')
    def test_stp_pre_indexed_min32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 256), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp]')
    def test_stp_base64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp, #8]')
    def test_stp_base_offset64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        self.cpu.push_int(9332165983064197000)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp, #504]')
    def test_stp_base_offset_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        self.cpu.STACK -= 504
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 504), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 504 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp, #-512]')
    def test_stp_base_offset_min64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        self.cpu.STACK += 512
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 512), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack - 512 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp], #8')
    def test_stp_post_indexed64(self):
        if False:
            return 10
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp], #504')
    def test_stp_post_indexed_max64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack + 504)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp], #-512')
    def test_stp_post_indexed_min64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack - 512)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp, #8]!')
    def test_stp_pre_indexed64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp, #504]!')
    def test_stp_pre_indexed_max64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        self.cpu.STACK -= 504
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 504), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 504 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack + 504)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest_custom('stp x1, x2, [sp, #-512]!')
    def test_stp_pre_indexed_min64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        self.cpu.STACK += 512
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 512), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack - 512 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack - 512)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp]'], multiple_insts=True)
    def test_stp_simd_fp_base32(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp, #8]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset32(self):
        if False:
            i = 10
            return i + 15
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp, #252]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset_max32(self):
        if False:
            return 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK -= 252
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 252), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp, #-256]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset_min32(self):
        if False:
            return 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack - 256), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp], #8'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed32(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp], #252'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed_max32(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 252)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp], #-256'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed_min32(self):
        if False:
            while True:
                i = 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp, #8]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed32(self):
        if False:
            return 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp, #252]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed_max32(self):
        if False:
            return 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK -= 252
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 252), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 252)

    @itest_setregs('S1=0x45464748', 'S2=0x41424344')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp s1, s2, [sp, #-256]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed_min32(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack - 256), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp]'], multiple_insts=True)
    def test_stp_simd_fp_base64(self):
        if False:
            return 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp, #8]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset64(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp, #504]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset_max64(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK -= 504
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 504), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 504 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp, #-512]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset_min64(self):
        if False:
            while True:
                i = 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK += 512
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack - 512), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack - 512 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp], #8'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed64(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp], #504'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed_max64(self):
        if False:
            while True:
                i = 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack + 504)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp], #-512'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed_min64(self):
        if False:
            while True:
                i = 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack - 512)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp, #8]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed64(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 8 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp, #504]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed_max64(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK -= 504
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 504), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 504 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack + 504)

    @itest_setregs('D1=0x4142434445464748', 'D2=0x5152535455565758')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp d1, d2, [sp, #-512]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed_min64(self):
        if False:
            i = 10
            return i + 15
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK += 512
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack - 512), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack - 512 + 8), 5859837686836516696)
        self.assertEqual(self.rf.read('SP'), stack - 512)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp]'], multiple_insts=True)
    def test_stp_simd_fp_base128(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp, #16]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset128(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 16), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack + 16 + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 16 + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack + 16 + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp, #1008]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset_max128(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK -= 1008
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 1008), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack + 1008 + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 1008 + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack + 1008 + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp, #-1024]'], multiple_insts=True)
    def test_stp_simd_fp_base_offset_min128(self):
        if False:
            while True:
                i = 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK += 1024
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack - 1024), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack - 1024 + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack - 1024 + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack - 1024 + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp], #16'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed128(self):
        if False:
            print('Hello World!')
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack + 16)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp], #1008'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed_max128(self):
        if False:
            i = 10
            return i + 15
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack + 1008)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp], #-1024'], multiple_insts=True)
    def test_stp_simd_fp_post_indexed_min128(self):
        if False:
            i = 10
            return i + 15
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack - 1024)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp, #16]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed128(self):
        if False:
            while True:
                i = 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 16), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack + 16 + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 16 + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack + 16 + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack + 16)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp, #1008]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed_max128(self):
        if False:
            while True:
                i = 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK -= 1008
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack + 1008), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack + 1008 + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack + 1008 + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack + 1008 + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack + 1008)

    @itest_setregs('Q1=0x41424344454647485152535455565758', 'Q2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'stp q1, q2, [sp, #-1024]!'], multiple_insts=True)
    def test_stp_simd_fp_pre_indexed_min128(self):
        if False:
            return 10
        for i in range(3):
            self._execute(reset=i == 0)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.push_int(18446744073709551615)
        self.cpu.STACK += 1024
        stack = self.cpu.STACK
        self._execute(reset=False)
        self.assertEqual(self.cpu.read_int(stack - 1024), 5859837686836516696)
        self.assertEqual(self.cpu.read_int(stack - 1024 + 8), 4702394921427289928)
        self.assertEqual(self.cpu.read_int(stack - 1024 + 16), 8174723217654970232)
        self.assertEqual(self.cpu.read_int(stack - 1024 + 24), 7017280452245743464)
        self.assertEqual(self.rf.read('SP'), stack - 1024)

    @itest_setregs('W1=0x41424344')
    @itest_custom('str w1, [sp]')
    def test_str_imm_base32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('str w1, [sp, #8]')
    def test_str_imm_base_offset32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('str w1, [sp, #16380]')
    def test_str_imm_base_offset_max32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 16380
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 16380), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('str w1, [sp], #8')
    def test_str_imm_post_indexed32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('W1=0x41424344')
    @itest_custom('str w1, [sp], #-256')
    def test_str_imm_post_indexed_neg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W1=0x41424344')
    @itest_custom('str w1, [sp, #8]!')
    def test_str_imm_pre_indexed32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('W1=0x41424344')
    @itest_custom('str w1, [sp, #-256]!')
    def test_str_imm_pre_indexed_neg32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 256), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('str x1, [sp]')
    def test_str_imm_base64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('str x1, [sp, #8]')
    def test_str_imm_base_offset64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('str x1, [sp, #32760]')
    def test_str_imm_base_offset_max64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 32760
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 32760), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('str x1, [sp], #8')
    def test_str_imm_post_indexed64(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('str x1, [sp], #-256')
    def test_str_imm_post_indexed_neg64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('str x1, [sp, #8]!')
    def test_str_imm_pre_indexed64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('str x1, [sp, #-256]!')
    def test_str_imm_pre_indexed_neg64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 256), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('str w0, [sp, w1, uxtw]')
    def test_str_reg_uxtw32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK -= 4294967288
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('str w0, [sp, w1, uxtw #2]')
    def test_str_reg_uxtw2_32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK -= LSL(4294967288, 2, 64)
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)

    @itest_setregs('W0=0x41424344', 'X1=8')
    @itest_custom('str w0, [sp, x1]')
    def test_str_reg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        self._execute()
        self.assertEqual(self.cpu.read_int(self.cpu.STACK + 8), 5859837686499656516)

    @itest_setregs('W0=0x41424344', 'X1=2')
    @itest_custom('str w0, [sp, x1, lsl #2]')
    def test_str_reg_lsl32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        self._execute()
        self.assertEqual(self.cpu.read_int(self.cpu.STACK + 8), 5859837686499656516)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('str w0, [sp, w1, sxtw]')
    def test_str_reg_sxtw32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('str w0, [sp, w1, sxtw #2]')
    def test_str_reg_sxtw2_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += LSL(8, 2, 64)
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)

    @itest_setregs('W0=0x41424344', 'X1=-8')
    @itest_custom('str w0, [sp, x1, sxtx]')
    def test_str_reg_sxtx32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)

    @itest_setregs('W0=0x41424344', 'X1=-2')
    @itest_custom('str w0, [sp, x1, sxtx #2]')
    def test_str_reg_sxtx2_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)

    @itest_setregs('X0=0x4142434445464748', 'W1=-8')
    @itest_custom('str x0, [sp, w1, uxtw]')
    def test_str_reg_uxtw64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK -= 4294967288
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)

    @itest_setregs('X0=0x4142434445464748', 'W1=-8')
    @itest_custom('str x0, [sp, w1, uxtw #3]')
    def test_str_reg_uxtw3_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK -= LSL(4294967288, 3, 64)
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)

    @itest_setregs('X0=0x4142434445464748', 'X1=8')
    @itest_custom('str x0, [sp, x1]')
    def test_str_reg64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        self._execute()
        self.assertEqual(self.cpu.read_int(self.cpu.STACK + 8), 4702394921427289928)

    @itest_setregs('X0=0x4142434445464748', 'X1=2')
    @itest_custom('str x0, [sp, x1, lsl #3]')
    def test_str_reg_lsl64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        self.cpu.push_int(8174723217654970232)
        self._execute()
        self.assertEqual(self.cpu.read_int(self.cpu.STACK + 16), 4702394921427289928)

    @itest_setregs('X0=0x4142434445464748', 'W1=-8')
    @itest_custom('str x0, [sp, w1, sxtw]')
    def test_str_reg_sxtw64(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)

    @itest_setregs('X0=0x4142434445464748', 'W1=-8')
    @itest_custom('str x0, [sp, w1, sxtw #3]')
    def test_str_reg_sxtw3_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += LSL(8, 3, 64)
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)

    @itest_setregs('X0=0x4142434445464748', 'X1=-8')
    @itest_custom('str x0, [sp, x1, sxtx]')
    def test_str_reg_sxtx64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)

    @itest_setregs('X0=0x4142434445464748', 'X1=-2')
    @itest_custom('str x0, [sp, x1, sxtx #3]')
    def test_str_reg_sxtx3_64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 16
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strb w1, [sp]')
    def test_strb_imm_base32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strb w1, [sp, #8]')
    def test_strb_imm_base_offset32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516676)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strb w1, [sp, #4095]')
    def test_strb_imm_base_offset_max32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 4095
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 4095), 5859837686836516676)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strb w1, [sp], #8')
    def test_strb_imm_post_indexed32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strb w1, [sp], #-256')
    def test_strb_imm_post_indexed_neg32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strb w1, [sp, #8]!')
    def test_strb_imm_pre_indexed32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836516676)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strb w1, [sp, #-256]!')
    def test_strb_imm_pre_indexed_neg32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 256), 5859837686836516676)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('strb w0, [sp, w1, uxtw]')
    def test_strb_reg_uxtw32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK -= 4294967288
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('strb w0, [sp, w1, uxtw #0]')
    def test_strb_reg_uxtw0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK -= 4294967288
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)

    @itest_setregs('W0=0x41424344', 'X1=8')
    @itest_custom('strb w0, [sp, x1]')
    def test_strb_reg32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        self._execute()
        self.assertEqual(self.cpu.read_int(self.cpu.STACK + 8), 5859837686836516676)

    @itest_setregs('W0=0x41424344', 'X1=8')
    @itest_custom('strb w0, [sp, x1, lsl #0]')
    def test_strb_reg_lsl32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        self._execute()
        self.assertEqual(self.cpu.read_int(self.cpu.STACK + 8), 5859837686836516676)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('strb w0, [sp, w1, sxtw]')
    def test_strb_reg_sxtw32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('strb w0, [sp, w1, sxtw #0]')
    def test_strb_reg_sxtw0_32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)

    @itest_setregs('W0=0x41424344', 'X1=-8')
    @itest_custom('strb w0, [sp, x1, sxtx]')
    def test_strb_reg_sxtx32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)

    @itest_setregs('W0=0x41424344', 'X1=-8')
    @itest_custom('strb w0, [sp, x1, sxtx #0]')
    def test_strb_reg_sxtx0_32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836516676)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strh w1, [sp]')
    def test_strh_imm_base32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strh w1, [sp, #8]')
    def test_strh_imm_base_offset32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836511556)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strh w1, [sp, #8190]')
    def test_strh_imm_base_offset_max32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 8190
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8190), 5859837686836511556)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strh w1, [sp], #8')
    def test_strh_imm_post_indexed32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strh w1, [sp], #-256')
    def test_strh_imm_post_indexed_neg32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strh w1, [sp, #8]!')
    def test_strh_imm_pre_indexed32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 8), 5859837686836511556)
        self.assertEqual(self.rf.read('SP'), stack + 8)

    @itest_setregs('W1=0x41424344')
    @itest_custom('strh w1, [sp, #-256]!')
    def test_strh_imm_pre_indexed_neg32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 256), 5859837686836511556)
        self.assertEqual(self.rf.read('SP'), stack - 256)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('strh w0, [sp, w1, uxtw]')
    def test_strh_reg_uxtw32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK -= 4294967288
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)

    @itest_setregs('W0=0x41424344', 'W1=-4')
    @itest_custom('strh w0, [sp, w1, uxtw #1]')
    def test_strh_reg_uxtw1_32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK -= LSL(4294967292, 1, 64)
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)

    @itest_setregs('W0=0x41424344', 'X1=8')
    @itest_custom('strh w0, [sp, x1]')
    def test_strh_reg32(self):
        if False:
            print('Hello World!')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        self._execute()
        self.assertEqual(self.cpu.read_int(self.cpu.STACK + 8), 5859837686836511556)

    @itest_setregs('W0=0x41424344', 'X1=4')
    @itest_custom('strh w0, [sp, x1, lsl #1]')
    def test_strh_reg_lsl32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        self._execute()
        self.assertEqual(self.cpu.read_int(self.cpu.STACK + 8), 5859837686836511556)

    @itest_setregs('W0=0x41424344', 'W1=-8')
    @itest_custom('strh w0, [sp, w1, sxtw]')
    def test_strh_reg_sxtw32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)

    @itest_setregs('W0=0x41424344', 'W1=-4')
    @itest_custom('strh w0, [sp, w1, sxtw #1]')
    def test_strh_reg_sxtw1_32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += LSL(4, 1, 64)
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)

    @itest_setregs('W0=0x41424344', 'X1=-8')
    @itest_custom('strh w0, [sp, x1, sxtx]')
    def test_strh_reg_sxtx32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)

    @itest_setregs('W0=0x41424344', 'X1=-4')
    @itest_custom('strh w0, [sp, x1, sxtx #1]')
    def test_strh_reg_sxtx1_32(self):
        if False:
            return 10
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self.cpu.STACK += 8
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686836511556)

    @itest_setregs('W1=0x41424344')
    @itest_custom('stur w1, [sp, #-256]')
    def test_stur_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 256), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('stur w1, [sp, #255]')
    def test_stur_max32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 255
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 255), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('stur w1, [sp, #1]')
    def test_stur_one32(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 1), 6368479526514737988)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344')
    @itest_custom('stur w1, [sp]')
    def test_stur32(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 5859837686499656516)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('stur x1, [sp, #-256]')
    def test_stur_min64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK += 256
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack - 256), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('stur x1, [sp, #255]')
    def test_stur_max64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        self.cpu.STACK -= 255
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 255), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('stur x1, [sp, #1]')
    def test_stur_one64(self):
        if False:
            while True:
                i = 10
        self.cpu.push_int(5859837686836516696)
        self.cpu.push_int(7017280452245743464)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack + 1), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('X1=0x4142434445464748')
    @itest_custom('stur x1, [sp]')
    def test_stur64(self):
        if False:
            i = 10
            return i + 15
        self.cpu.push_int(5859837686836516696)
        stack = self.cpu.STACK
        self._execute()
        self.assertEqual(self.cpu.read_int(stack), 4702394921427289928)
        self.assertEqual(self.rf.read('SP'), stack)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('sub w0, w1, w2, uxtb')
    def test_sub_ext_reg_uxtb32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861504)
        self.assertEqual(self.rf.read('W0'), 1094861504)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('sub w0, w1, w2, uxtb #0')
    def test_sub_ext_reg_uxtb0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861504)
        self.assertEqual(self.rf.read('W0'), 1094861504)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('sub w0, w1, w2, uxtb #4')
    def test_sub_ext_reg_uxtb4_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094859524)
        self.assertEqual(self.rf.read('W0'), 1094859524)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('sub w0, w1, w2, uxth')
    def test_sub_ext_reg_uxth32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094828016)
        self.assertEqual(self.rf.read('W0'), 1094828016)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('sub w0, w1, w2, uxth #0')
    def test_sub_ext_reg_uxth0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094828016)
        self.assertEqual(self.rf.read('W0'), 1094828016)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('sub w0, w1, w2, uxth #4')
    def test_sub_ext_reg_uxth4_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094323716)
        self.assertEqual(self.rf.read('W0'), 1094323716)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, uxtw')
    def test_sub_ext_reg_uxtw32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, uxtw #0')
    def test_sub_ext_reg_uxtw0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, uxtw #4')
    def test_sub_ext_reg_uxtw4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, uxtx')
    def test_sub_ext_reg_uxtx32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, uxtx #0')
    def test_sub_ext_reg_uxtx0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, uxtx #4')
    def test_sub_ext_reg_uxtx4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('sub w0, w1, w2, sxtb')
    def test_sub_ext_reg_sxtb32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861760)
        self.assertEqual(self.rf.read('W0'), 1094861760)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('sub w0, w1, w2, sxtb #0')
    def test_sub_ext_reg_sxtb0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861760)
        self.assertEqual(self.rf.read('W0'), 1094861760)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('sub w0, w1, w2, sxtb #4')
    def test_sub_ext_reg_sxtb4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094863620)
        self.assertEqual(self.rf.read('W0'), 1094863620)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('sub w0, w1, w2, sxth')
    def test_sub_ext_reg_sxth32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094893552)
        self.assertEqual(self.rf.read('W0'), 1094893552)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('sub w0, w1, w2, sxth #0')
    def test_sub_ext_reg_sxth0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094893552)
        self.assertEqual(self.rf.read('W0'), 1094893552)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('sub w0, w1, w2, sxth #4')
    def test_sub_ext_reg_sxth4_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1095372292)
        self.assertEqual(self.rf.read('W0'), 1095372292)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, sxtw')
    def test_sub_ext_reg_sxtw32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, sxtw #0')
    def test_sub_ext_reg_sxtw0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, sxtw #4')
    def test_sub_ext_reg_sxtw4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, sxtx')
    def test_sub_ext_reg_sxtx32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, sxtx #0')
    def test_sub_ext_reg_sxtx0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, sxtx #4')
    def test_sub_ext_reg_sxtx4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, lsl #0')
    def test_sub_ext_reg_lsl0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('sub w0, w1, w2, lsl #4')
    def test_sub_ext_reg_lsl4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('sub x0, x1, w2, uxtb')
    def test_sub_ext_reg_uxtb64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289796)
        self.assertEqual(self.rf.read('W0'), 1162233540)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('sub x0, x1, w2, uxtb #0')
    def test_sub_ext_reg_uxtb0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289796)
        self.assertEqual(self.rf.read('W0'), 1162233540)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('sub x0, x1, w2, uxtb #4')
    def test_sub_ext_reg_uxtb4_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427287816)
        self.assertEqual(self.rf.read('W0'), 1162231560)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('sub x0, x1, w2, uxth')
    def test_sub_ext_reg_uxth64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427256308)
        self.assertEqual(self.rf.read('W0'), 1162200052)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('sub x0, x1, w2, uxth #0')
    def test_sub_ext_reg_uxth0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427256308)
        self.assertEqual(self.rf.read('W0'), 1162200052)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('sub x0, x1, w2, uxth #4')
    def test_sub_ext_reg_uxth4_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921426752008)
        self.assertEqual(self.rf.read('W0'), 1161695752)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('sub x0, x1, w2, uxtw')
    def test_sub_ext_reg_uxtw64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394919257633780)
        self.assertEqual(self.rf.read('W0'), 3287544820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('sub x0, x1, w2, uxtw #0')
    def test_sub_ext_reg_uxtw0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394919257633780)
        self.assertEqual(self.rf.read('W0'), 3287544820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('sub x0, x1, w2, uxtw #4')
    def test_sub_ext_reg_uxtw4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394886712791560)
        self.assertEqual(self.rf.read('W0'), 807473672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('sub x0, x1, x2, uxtx')
    def test_sub_ext_reg_uxtx64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('sub x0, x1, x2, uxtx #0')
    def test_sub_ext_reg_uxtx0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('sub x0, x1, x2, uxtx #4')
    def test_sub_ext_reg_uxtx4_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3178712300590780872)
        self.assertEqual(self.rf.read('W0'), 4024488392)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('sub x0, x1, w2, sxtb')
    def test_sub_ext_reg_sxtb64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427290052)
        self.assertEqual(self.rf.read('W0'), 1162233796)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('sub x0, x1, w2, sxtb #0')
    def test_sub_ext_reg_sxtb0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427290052)
        self.assertEqual(self.rf.read('W0'), 1162233796)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('sub x0, x1, w2, sxtb #4')
    def test_sub_ext_reg_sxtb4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427291912)
        self.assertEqual(self.rf.read('W0'), 1162235656)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('sub x0, x1, w2, sxth')
    def test_sub_ext_reg_sxth64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427321844)
        self.assertEqual(self.rf.read('W0'), 1162265588)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('sub x0, x1, w2, sxth #0')
    def test_sub_ext_reg_sxth0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427321844)
        self.assertEqual(self.rf.read('W0'), 1162265588)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('sub x0, x1, w2, sxth #4')
    def test_sub_ext_reg_sxth4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427800584)
        self.assertEqual(self.rf.read('W0'), 1162744328)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('sub x0, x1, w2, sxtw')
    def test_sub_ext_reg_sxtw64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394923552601076)
        self.assertEqual(self.rf.read('W0'), 3287544820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('sub x0, x1, w2, sxtw #0')
    def test_sub_ext_reg_sxtw0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394923552601076)
        self.assertEqual(self.rf.read('W0'), 3287544820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('sub x0, x1, w2, sxtw #4')
    def test_sub_ext_reg_sxtw4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394955432268296)
        self.assertEqual(self.rf.read('W0'), 807473672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('sub x0, x1, x2, sxtx')
    def test_sub_ext_reg_sxtx64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('sub x0, x1, x2, sxtx #0')
    def test_sub_ext_reg_sxtx0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('sub x0, x1, x2, sxtx #4')
    def test_sub_ext_reg_sxtx4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3178712300590780872)
        self.assertEqual(self.rf.read('W0'), 4024488392)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('sub x0, x1, x2, lsl #0')
    def test_sub_ext_reg_lsl0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('sub x0, x1, x2, lsl #4')
    def test_sub_ext_reg_lsl4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3178712300590780872)
        self.assertEqual(self.rf.read('W0'), 4024488392)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('sub w0, w1, #0')
    def test_sub_imm_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('sub w0, w1, #4095')
    def test_sub_imm_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094857541)
        self.assertEqual(self.rf.read('W0'), 1094857541)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('sub w0, w1, #1')
    def test_sub_imm32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094861635)
        self.assertEqual(self.rf.read('W0'), 1094861635)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('sub w0, w1, #1, lsl #0')
    def test_sub_imm_lsl0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861635)
        self.assertEqual(self.rf.read('W0'), 1094861635)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('sub w0, w1, #1, lsl #12')
    def test_sub_imm_lsl12_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094857540)
        self.assertEqual(self.rf.read('W0'), 1094857540)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sub x0, x1, #0')
    def test_sub_imm_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sub x0, x1, #4095')
    def test_sub_imm_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427285833)
        self.assertEqual(self.rf.read('W0'), 1162229577)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sub x0, x1, #1')
    def test_sub_imm64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289927)
        self.assertEqual(self.rf.read('W0'), 1162233671)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sub x0, x1, #1, lsl #0')
    def test_sub_imm_lsl0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289927)
        self.assertEqual(self.rf.read('W0'), 1162233671)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sub x0, x1, #1, lsl #12')
    def test_sub_imm_lsl12_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427285832)
        self.assertEqual(self.rf.read('W0'), 1162229576)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('sub w0, w1, w2')
    def test_sub_sft_reg32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4227595260)
        self.assertEqual(self.rf.read('W0'), 4227595260)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('sub w0, w1, w2, lsl #0')
    def test_sub_sft_reg_lsl_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4227595260)
        self.assertEqual(self.rf.read('W0'), 4227595260)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=1')
    @itest('sub w0, w1, w2, lsl #31')
    def test_sub_sft_reg_lsl_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3242345284)
        self.assertEqual(self.rf.read('W0'), 3242345284)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('sub w0, w1, w2, lsl #1')
    def test_sub_sft_reg_lsl32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3065361588)
        self.assertEqual(self.rf.read('W0'), 3065361588)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('sub w0, w1, w2, lsr #0')
    def test_sub_sft_reg_lsr_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4227595260)
        self.assertEqual(self.rf.read('W0'), 4227595260)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('sub w0, w1, w2, lsr #31')
    def test_sub_sft_reg_lsr_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861635)
        self.assertEqual(self.rf.read('W0'), 1094861635)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('sub w0, w1, w2, lsr #1')
    def test_sub_sft_reg_lsr32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 21119812)
        self.assertEqual(self.rf.read('W0'), 21119812)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('sub w0, w1, w2, asr #0')
    def test_sub_sft_reg_asr_min32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4227595260)
        self.assertEqual(self.rf.read('W0'), 4227595260)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('sub w0, w1, w2, asr #31')
    def test_sub_sft_reg_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('sub w0, w1, w2, asr #1')
    def test_sub_sft_reg_asr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 2168603460)
        self.assertEqual(self.rf.read('W0'), 2168603460)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('sub x0, x1, x2')
    def test_sub_sft_reg64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 17289301308300324848)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('sub x0, x1, x2, lsl #0')
    def test_sub_sft_reg_lsl_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 17289301308300324848)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=1')
    @itest('sub x0, x1, x2, lsl #63')
    def test_sub_sft_reg_lsl_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13925766958282065736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('sub x0, x1, x2, lsl #1')
    def test_sub_sft_reg_lsl64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 11429463621463808152)
        self.assertEqual(self.rf.read('W0'), 2593757336)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('sub x0, x1, x2, lsr #0')
    def test_sub_sft_reg_lsr_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 17289301308300324848)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('sub x0, x1, x2, lsr #63')
    def test_sub_sft_reg_lsr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289927)
        self.assertEqual(self.rf.read('W0'), 1162233671)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('sub x0, x1, x2, lsr #1')
    def test_sub_sft_reg_lsr64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 90708902999902024)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('sub x0, x1, x2, asr #0')
    def test_sub_sft_reg_asr_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 17289301308300324848)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('sub x0, x1, x2, asr #63')
    def test_sub_sft_reg_asr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('sub x0, x1, x2, asr #1')
    def test_sub_sft_reg_asr64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 9314080939854677832)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub d0, d1, d2'], multiple_insts=True)
    def test_sub_scalar(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 16131858542891098080)
        self.assertEqual(self.rf.read('Q0'), 16131858542891098080)
        self.assertEqual(self.rf.read('D0'), 16131858542891098080)
        self.assertEqual(self.rf.read('S0'), 3755991008)
        self.assertEqual(self.rf.read('H0'), 57312)
        self.assertEqual(self.rf.read('B0'), 224)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub d0, d1, d2'], multiple_insts=True)
    def test_sub_scalar_max(self):
        if False:
            i = 10
            return i + 15
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_sub_vector_8b(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 16204198715729174752)
        self.assertEqual(self.rf.read('Q0'), 16204198715729174752)
        self.assertEqual(self.rf.read('D0'), 16204198715729174752)
        self.assertEqual(self.rf.read('S0'), 3772834016)
        self.assertEqual(self.rf.read('H0'), 57568)
        self.assertEqual(self.rf.read('B0'), 224)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.8b, v1.8b, v2.8b'], multiple_insts=True)
    def test_sub_vector_8b_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_sub_vector_16b(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 298914706628589081630572204175357173984)
        self.assertEqual(self.rf.read('Q0'), 298914706628589081630572204175357173984)
        self.assertEqual(self.rf.read('D0'), 16204198715729174752)
        self.assertEqual(self.rf.read('S0'), 3772834016)
        self.assertEqual(self.rf.read('H0'), 57568)
        self.assertEqual(self.rf.read('B0'), 224)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.16b, v1.16b, v2.16b'], multiple_insts=True)
    def test_sub_vector_16b_max(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.4h, v1.4h, v2.4h'], multiple_insts=True)
    def test_sub_vector_4h(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 16132140022162841568)
        self.assertEqual(self.rf.read('Q0'), 16132140022162841568)
        self.assertEqual(self.rf.read('D0'), 16132140022162841568)
        self.assertEqual(self.rf.read('S0'), 3756056544)
        self.assertEqual(self.rf.read('H0'), 57312)
        self.assertEqual(self.rf.read('B0'), 224)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.4h, v1.4h, v2.4h'], multiple_insts=True)
    def test_sub_vector_4h_max(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.8h, v1.8h, v2.8h'], multiple_insts=True)
    def test_sub_vector_8h(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 297585458350085072373738086535889215456)
        self.assertEqual(self.rf.read('Q0'), 297585458350085072373738086535889215456)
        self.assertEqual(self.rf.read('D0'), 16132140022162841568)
        self.assertEqual(self.rf.read('S0'), 3756056544)
        self.assertEqual(self.rf.read('H0'), 57312)
        self.assertEqual(self.rf.read('B0'), 224)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.8h, v1.8h, v2.8h'], multiple_insts=True)
    def test_sub_vector_8h_max(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.2s, v1.2s, v2.2s'], multiple_insts=True)
    def test_sub_vector_2s(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 16131858547186065376)
        self.assertEqual(self.rf.read('Q0'), 16131858547186065376)
        self.assertEqual(self.rf.read('D0'), 16131858547186065376)
        self.assertEqual(self.rf.read('S0'), 3755991008)
        self.assertEqual(self.rf.read('H0'), 57312)
        self.assertEqual(self.rf.read('B0'), 224)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.2s, v1.2s, v2.2s'], multiple_insts=True)
    def test_sub_vector_2s_max(self):
        if False:
            return 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.4s, v1.4s, v2.4s'], multiple_insts=True)
    def test_sub_vector_4s(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 297580266053225328620289659935408512992)
        self.assertEqual(self.rf.read('Q0'), 297580266053225328620289659935408512992)
        self.assertEqual(self.rf.read('D0'), 16131858547186065376)
        self.assertEqual(self.rf.read('S0'), 3755991008)
        self.assertEqual(self.rf.read('H0'), 57312)
        self.assertEqual(self.rf.read('B0'), 224)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.4s, v1.4s, v2.4s'], multiple_insts=True)
    def test_sub_vector_4s_max(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('V1=0x41424344454647485152535455565758', 'V2=0x61626364656667687172737475767778')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.2d, v1.2d, v2.2d'], multiple_insts=True)
    def test_sub_vector_2d(self):
        if False:
            print('Hello World!')
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 297580265973997166106025322337569595360)
        self.assertEqual(self.rf.read('Q0'), 297580265973997166106025322337569595360)
        self.assertEqual(self.rf.read('D0'), 16131858542891098080)
        self.assertEqual(self.rf.read('S0'), 3755991008)
        self.assertEqual(self.rf.read('H0'), 57312)
        self.assertEqual(self.rf.read('B0'), 224)

    @itest_setregs('V1=0xffffffffffffffffffffffffffffffff', 'V2=0xffffffffffffffffffffffffffffffff')
    @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', 'sub v0.2d, v1.2d, v2.2d'], multiple_insts=True)
    def test_sub_vector_2d_max(self):
        if False:
            while True:
                i = 10
        for i in range(4):
            self._execute(reset=i == 0)
        self.assertEqual(self.rf.read('V0'), 0)
        self.assertEqual(self.rf.read('Q0'), 0)
        self.assertEqual(self.rf.read('D0'), 0)
        self.assertEqual(self.rf.read('S0'), 0)
        self.assertEqual(self.rf.read('H0'), 0)
        self.assertEqual(self.rf.read('B0'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('subs w0, w1, w2, uxtb')
    def test_subs_ext_reg_uxtb32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094861504)
        self.assertEqual(self.rf.read('W0'), 1094861504)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('subs w0, w1, w2, uxtb #0')
    def test_subs_ext_reg_uxtb0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861504)
        self.assertEqual(self.rf.read('W0'), 1094861504)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('subs w0, w1, w2, uxtb #4')
    def test_subs_ext_reg_uxtb4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094859524)
        self.assertEqual(self.rf.read('W0'), 1094859524)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('subs w0, w1, w2, uxth')
    def test_subs_ext_reg_uxth32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094828016)
        self.assertEqual(self.rf.read('W0'), 1094828016)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('subs w0, w1, w2, uxth #0')
    def test_subs_ext_reg_uxth0_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1094828016)
        self.assertEqual(self.rf.read('W0'), 1094828016)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('subs w0, w1, w2, uxth #4')
    def test_subs_ext_reg_uxth4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094323716)
        self.assertEqual(self.rf.read('W0'), 1094323716)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, uxtw')
    def test_subs_ext_reg_uxtw32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, uxtw #0')
    def test_subs_ext_reg_uxtw0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, uxtw #4')
    def test_subs_ext_reg_uxtw4_32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, uxtx')
    def test_subs_ext_reg_uxtx32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, uxtx #0')
    def test_subs_ext_reg_uxtx0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, uxtx #4')
    def test_subs_ext_reg_uxtx4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('subs w0, w1, w2, sxtb')
    def test_subs_ext_reg_sxtb32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861760)
        self.assertEqual(self.rf.read('W0'), 1094861760)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('subs w0, w1, w2, sxtb #0')
    def test_subs_ext_reg_sxtb0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861760)
        self.assertEqual(self.rf.read('W0'), 1094861760)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51525384')
    @itest('subs w0, w1, w2, sxtb #4')
    def test_subs_ext_reg_sxtb4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094863620)
        self.assertEqual(self.rf.read('W0'), 1094863620)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('subs w0, w1, w2, sxth')
    def test_subs_ext_reg_sxth32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094893552)
        self.assertEqual(self.rf.read('W0'), 1094893552)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('subs w0, w1, w2, sxth #0')
    def test_subs_ext_reg_sxth0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094893552)
        self.assertEqual(self.rf.read('W0'), 1094893552)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x51528354')
    @itest('subs w0, w1, w2, sxth #4')
    def test_subs_ext_reg_sxth4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1095372292)
        self.assertEqual(self.rf.read('W0'), 1095372292)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, sxtw')
    def test_subs_ext_reg_sxtw32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, sxtw #0')
    def test_subs_ext_reg_sxtw0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, sxtw #4')
    def test_subs_ext_reg_sxtw4_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, sxtx')
    def test_subs_ext_reg_sxtx32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, sxtx #0')
    def test_subs_ext_reg_sxtx0_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, sxtx #4')
    def test_subs_ext_reg_sxtx4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, lsl #0')
    def test_subs_ext_reg_lsl0_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 3220172784)
        self.assertEqual(self.rf.read('W0'), 3220172784)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x81525354')
    @itest('subs w0, w1, w2, lsl #4')
    def test_subs_ext_reg_lsl4_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 740101636)
        self.assertEqual(self.rf.read('W0'), 740101636)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('subs x0, x1, w2, uxtb')
    def test_subs_ext_reg_uxtb64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289796)
        self.assertEqual(self.rf.read('W0'), 1162233540)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('subs x0, x1, w2, uxtb #0')
    def test_subs_ext_reg_uxtb0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289796)
        self.assertEqual(self.rf.read('W0'), 1162233540)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('subs x0, x1, w2, uxtb #4')
    def test_subs_ext_reg_uxtb4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427287816)
        self.assertEqual(self.rf.read('W0'), 1162231560)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('subs x0, x1, w2, uxth')
    def test_subs_ext_reg_uxth64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427256308)
        self.assertEqual(self.rf.read('W0'), 1162200052)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('subs x0, x1, w2, uxth #0')
    def test_subs_ext_reg_uxth0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427256308)
        self.assertEqual(self.rf.read('W0'), 1162200052)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('subs x0, x1, w2, uxth #4')
    def test_subs_ext_reg_uxth4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921426752008)
        self.assertEqual(self.rf.read('W0'), 1161695752)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('subs x0, x1, w2, uxtw')
    def test_subs_ext_reg_uxtw64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394919257633780)
        self.assertEqual(self.rf.read('W0'), 3287544820)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('subs x0, x1, w2, uxtw #0')
    def test_subs_ext_reg_uxtw0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394919257633780)
        self.assertEqual(self.rf.read('W0'), 3287544820)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('subs x0, x1, w2, uxtw #4')
    def test_subs_ext_reg_uxtw4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394886712791560)
        self.assertEqual(self.rf.read('W0'), 807473672)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('subs x0, x1, x2, uxtx')
    def test_subs_ext_reg_uxtx64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('subs x0, x1, x2, uxtx #0')
    def test_subs_ext_reg_uxtx0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('subs x0, x1, x2, uxtx #4')
    def test_subs_ext_reg_uxtx4_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3178712300590780872)
        self.assertEqual(self.rf.read('W0'), 4024488392)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('subs x0, x1, w2, sxtb')
    def test_subs_ext_reg_sxtb64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427290052)
        self.assertEqual(self.rf.read('W0'), 1162233796)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('subs x0, x1, w2, sxtb #0')
    def test_subs_ext_reg_sxtb0_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427290052)
        self.assertEqual(self.rf.read('W0'), 1162233796)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51525384')
    @itest('subs x0, x1, w2, sxtb #4')
    def test_subs_ext_reg_sxtb4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427291912)
        self.assertEqual(self.rf.read('W0'), 1162235656)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('subs x0, x1, w2, sxth')
    def test_subs_ext_reg_sxth64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427321844)
        self.assertEqual(self.rf.read('W0'), 1162265588)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('subs x0, x1, w2, sxth #0')
    def test_subs_ext_reg_sxth0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427321844)
        self.assertEqual(self.rf.read('W0'), 1162265588)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x51528354')
    @itest('subs x0, x1, w2, sxth #4')
    def test_subs_ext_reg_sxth4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427800584)
        self.assertEqual(self.rf.read('W0'), 1162744328)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('subs x0, x1, w2, sxtw')
    def test_subs_ext_reg_sxtw64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394923552601076)
        self.assertEqual(self.rf.read('W0'), 3287544820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('subs x0, x1, w2, sxtw #0')
    def test_subs_ext_reg_sxtw0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394923552601076)
        self.assertEqual(self.rf.read('W0'), 3287544820)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'W2=0x81525354')
    @itest('subs x0, x1, w2, sxtw #4')
    def test_subs_ext_reg_sxtw4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394955432268296)
        self.assertEqual(self.rf.read('W0'), 807473672)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('subs x0, x1, x2, sxtx')
    def test_subs_ext_reg_sxtx64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('subs x0, x1, x2, sxtx #0')
    def test_subs_ext_reg_sxtx0_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('subs x0, x1, x2, sxtx #4')
    def test_subs_ext_reg_sxtx4_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 3178712300590780872)
        self.assertEqual(self.rf.read('W0'), 4024488392)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('subs x0, x1, x2, lsl #0')
    def test_subs_ext_reg_lsl0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 13830536794479783920)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8152535455565758')
    @itest('subs x0, x1, x2, lsl #4')
    def test_subs_ext_reg_lsl4_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3178712300590780872)
        self.assertEqual(self.rf.read('W0'), 4024488392)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('subs w0, w1, #0')
    def test_subs_imm_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('subs w0, w1, #4095')
    def test_subs_imm_max32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1094857541)
        self.assertEqual(self.rf.read('W0'), 1094857541)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('subs w0, w1, #1')
    def test_subs_imm32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861635)
        self.assertEqual(self.rf.read('W0'), 1094861635)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('subs w0, w1, #1, lsl #0')
    def test_subs_imm_lsl0_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861635)
        self.assertEqual(self.rf.read('W0'), 1094861635)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344')
    @itest('subs w0, w1, #1, lsl #12')
    def test_subs_imm_lsl12_32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094857540)
        self.assertEqual(self.rf.read('W0'), 1094857540)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('subs x0, x1, #0')
    def test_subs_imm_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('subs x0, x1, #4095')
    def test_subs_imm_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427285833)
        self.assertEqual(self.rf.read('W0'), 1162229577)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('subs x0, x1, #1')
    def test_subs_imm64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289927)
        self.assertEqual(self.rf.read('W0'), 1162233671)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('subs x0, x1, #1, lsl #0')
    def test_subs_imm_lsl0_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289927)
        self.assertEqual(self.rf.read('W0'), 1162233671)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748')
    @itest('subs x0, x1, #1, lsl #12')
    def test_subs_imm_lsl12_64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427285832)
        self.assertEqual(self.rf.read('W0'), 1162229576)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('subs w0, w1, w2')
    def test_subs_sft_reg32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 4227595260)
        self.assertEqual(self.rf.read('W0'), 4227595260)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('subs w0, w1, w2, lsl #0')
    def test_subs_sft_reg_lsl_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4227595260)
        self.assertEqual(self.rf.read('W0'), 4227595260)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=1')
    @itest('subs w0, w1, w2, lsl #31')
    def test_subs_sft_reg_lsl_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 3242345284)
        self.assertEqual(self.rf.read('W0'), 3242345284)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('subs w0, w1, w2, lsl #1')
    def test_subs_sft_reg_lsl32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 3065361588)
        self.assertEqual(self.rf.read('W0'), 3065361588)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('subs w0, w1, w2, lsr #0')
    def test_subs_sft_reg_lsr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4227595260)
        self.assertEqual(self.rf.read('W0'), 4227595260)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('subs w0, w1, w2, lsr #31')
    def test_subs_sft_reg_lsr_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861635)
        self.assertEqual(self.rf.read('W0'), 1094861635)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('subs w0, w1, w2, lsr #1')
    def test_subs_sft_reg_lsr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 21119812)
        self.assertEqual(self.rf.read('W0'), 21119812)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('W1=0x41424344', 'W2=0x45464748')
    @itest('subs w0, w1, w2, asr #0')
    def test_subs_sft_reg_asr_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4227595260)
        self.assertEqual(self.rf.read('W0'), 4227595260)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('subs w0, w1, w2, asr #31')
    def test_subs_sft_reg_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861637)
        self.assertEqual(self.rf.read('W0'), 1094861637)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x41424344', 'W2=0x80000000')
    @itest('subs w0, w1, w2, asr #1')
    def test_subs_sft_reg_asr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 2168603460)
        self.assertEqual(self.rf.read('W0'), 2168603460)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('subs x0, x1, x2')
    def test_subs_sft_reg64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 17289301308300324848)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('subs x0, x1, x2, lsl #0')
    def test_subs_sft_reg_lsl_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17289301308300324848)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=1')
    @itest('subs x0, x1, x2, lsl #63')
    def test_subs_sft_reg_lsl_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 13925766958282065736)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('subs x0, x1, x2, lsl #1')
    def test_subs_sft_reg_lsl64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 11429463621463808152)
        self.assertEqual(self.rf.read('W0'), 2593757336)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('subs x0, x1, x2, lsr #0')
    def test_subs_sft_reg_lsr_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 17289301308300324848)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('subs x0, x1, x2, lsr #63')
    def test_subs_sft_reg_lsr_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289927)
        self.assertEqual(self.rf.read('W0'), 1162233671)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('subs x0, x1, x2, lsr #1')
    def test_subs_sft_reg_lsr64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 90708902999902024)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 536870912)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x5152535455565758')
    @itest('subs x0, x1, x2, asr #0')
    def test_subs_sft_reg_asr_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 17289301308300324848)
        self.assertEqual(self.rf.read('W0'), 4025479152)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('subs x0, x1, x2, asr #63')
    def test_subs_sft_reg_asr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4702394921427289929)
        self.assertEqual(self.rf.read('W0'), 1162233673)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x8000000000000000')
    @itest('subs x0, x1, x2, asr #1')
    def test_subs_sft_reg_asr64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 9314080939854677832)
        self.assertEqual(self.rf.read('W0'), 1162233672)
        self.assertEqual(self.rf.read('NZCV'), 2415919104)

    @skip_sym('immediate')
    def test_svc0(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(Interruption):
            self._setupCpu('svc #0')
            self._execute()

    @skip_sym('immediate')
    def test_svc1(self):
        if False:
            for i in range(10):
                print('nop')
        if self.__class__.__name__ == 'Aarch64CpuInstructions':
            e = InstructionNotImplementedError
        elif self.__class__.__name__ == 'Aarch64UnicornInstructions':
            e = Interruption
        else:
            self.fail()
        with self.assertRaises(e):
            self._setupCpu('svc #1')
            self._execute()

    @itest_setregs('W1=0x41424344')
    @itest('sxtb w0, w1')
    def test_sxtb_zero32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 68)
        self.assertEqual(self.rf.read('W0'), 68)

    @itest_setregs('W1=0x41424384')
    @itest('sxtb w0, w1')
    def test_sxtb_one32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4294967172)
        self.assertEqual(self.rf.read('W0'), 4294967172)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sxtb x0, x1')
    def test_sxtb_zero64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 72)
        self.assertEqual(self.rf.read('W0'), 72)

    @itest_setregs('X1=0x4142434445464788')
    @itest('sxtb x0, x1')
    def test_sxtb_one64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 18446744073709551496)
        self.assertEqual(self.rf.read('W0'), 4294967176)

    @itest_setregs('W1=0x41424344')
    @itest('sxth w0, w1')
    def test_sxth_zero32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)

    @itest_setregs('W1=0x41428344')
    @itest('sxth w0, w1')
    def test_sxth_one32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4294935364)
        self.assertEqual(self.rf.read('W0'), 4294935364)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sxth x0, x1')
    def test_sxth_zero64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 18248)
        self.assertEqual(self.rf.read('W0'), 18248)

    @itest_setregs('X1=0x4142434445468748')
    @itest('sxth x0, x1')
    def test_sxth_one64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709520712)
        self.assertEqual(self.rf.read('W0'), 4294936392)

    @itest_setregs('X1=0x4142434445464748')
    @itest('sxtw x0, x1')
    def test_sxtw_zero(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1162233672)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4142434485464748')
    @itest('sxtw x0, x1')
    def test_sxtw_one(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744071650559816)
        self.assertEqual(self.rf.read('W0'), 2235975496)

    @itest_custom(['tbnz w0, 0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_min_zero32(self):
        if False:
            print('Hello World!')
        self._setreg('W0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbnz w0, 0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_min_one32(self):
        if False:
            print('Hello World!')
        self._setreg('W0', 1)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbnz w0, 31, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_max_zero32(self):
        if False:
            return 10
        self._setreg('W0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbnz w0, 31, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_max_one32(self):
        if False:
            while True:
                i = 10
        self._setreg('W0', 2147483648)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbnz w0, 3, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_zero32(self):
        if False:
            print('Hello World!')
        self._setreg('W0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbnz w0, 3, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_one32(self):
        if False:
            print('Hello World!')
        self._setreg('W0', 8)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbnz x0, 0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_min_zero64(self):
        if False:
            for i in range(10):
                print('nop')
        self._setreg('X0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbnz x0, 0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_min_one64(self):
        if False:
            while True:
                i = 10
        self._setreg('X0', 1)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbnz x0, 63, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_max_zero64(self):
        if False:
            i = 10
            return i + 15
        self._setreg('X0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbnz x0, 63, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_max_one64(self):
        if False:
            i = 10
            return i + 15
        self._setreg('X0', 9223372036854775808)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbnz x0, 3, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_zero64(self):
        if False:
            i = 10
            return i + 15
        self._setreg('X0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbnz x0, 3, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbnz_one64(self):
        if False:
            i = 10
            return i + 15
        self._setreg('X0', 8)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbz w0, 0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_min_zero32(self):
        if False:
            i = 10
            return i + 15
        self._setreg('W0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbz w0, 0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_min_one32(self):
        if False:
            i = 10
            return i + 15
        self._setreg('W0', 1)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbz w0, 31, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_max_zero32(self):
        if False:
            print('Hello World!')
        self._setreg('W0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbz w0, 31, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_max_one32(self):
        if False:
            print('Hello World!')
        self._setreg('W0', 2147483648)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbz w0, 3, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_zero32(self):
        if False:
            while True:
                i = 10
        self._setreg('W0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbz w0, 3, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_one32(self):
        if False:
            for i in range(10):
                print('nop')
        self._setreg('W0', 8)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbz x0, 0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_min_zero64(self):
        if False:
            return 10
        self._setreg('X0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbz x0, 0, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_min_one64(self):
        if False:
            i = 10
            return i + 15
        self._setreg('X0', 1)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbz x0, 63, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_max_zero64(self):
        if False:
            i = 10
            return i + 15
        self._setreg('X0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbz x0, 63, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_max_one64(self):
        if False:
            return 10
        self._setreg('X0', 9223372036854775808)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_custom(['tbz x0, 3, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_zero64(self):
        if False:
            while True:
                i = 10
        self._setreg('X0', 0)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 8)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 0)
        self.assertEqual(self.rf.read('X2'), 43)

    @itest_custom(['tbz x0, 3, .+8', 'mov x1, 42', 'mov x2, 43'], multiple_insts=True)
    def test_tbz_one64(self):
        if False:
            while True:
                i = 10
        self._setreg('X0', 8)
        self._setreg('PC', self.cpu.PC)
        pc = self.cpu.PC
        self._execute(check_pc=False)
        self.assertEqual(self.rf.read('PC'), pc + 4)
        self.assertEqual(self.rf.read('LR'), 0)
        self.assertEqual(self.rf.read('X30'), 0)
        self._execute()
        self.assertEqual(self.rf.read('X1'), 42)
        self.assertEqual(self.rf.read('X2'), 0)

    @itest_setregs('W1=0x41424344')
    @itest('tst w1, #0xffff')
    def test_tst_imm32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x81424344')
    @itest('tst w1, #0xffff0000')
    def test_tst_imm2_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x44434241')
    @itest('tst w1, #1')
    def test_tst_imm3_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0')
    @itest('tst w1, #1')
    def test_tst_imm4_32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('X1=0x8142434445464748')
    @itest('tst x1, #0xffff0000ffff0000')
    def test_tst_imm64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x4142434445464748')
    @itest('tst x1, #0x0000ffff0000ffff')
    def test_tst_imm2_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x4847464544434241')
    @itest('tst x1, #1')
    def test_tst_imm3_64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0')
    @itest('tst x1, #1')
    def test_tst_imm4_64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('tst w1, w2')
    def test_tst_sft_reg32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0')
    @itest('tst w1, w2')
    def test_tst_sft_reg_zero32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('tst w1, w2, lsl #0')
    def test_tst_sft_reg_lsl_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('tst w1, w2, lsl #31')
    def test_tst_sft_reg_lsl_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('tst w1, w2, lsl #4')
    def test_tst_sft_reg_lsl32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('tst w1, w2, lsr #0')
    def test_tst_sft_reg_lsr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('tst w1, w2, lsr #31')
    def test_tst_sft_reg_lsr_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('tst w1, w2, lsr #4')
    def test_tst_sft_reg_lsr32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('tst w1, w2, asr #0')
    def test_tst_sft_reg_asr_min32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('tst w1, w2, asr #31')
    def test_tst_sft_reg_asr_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('tst w1, w2, asr #4')
    def test_tst_sft_reg_asr32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x4142ffff', 'W2=0xffff4344')
    @itest('tst w1, w2, ror #0')
    def test_tst_sft_reg_ror_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x80000001')
    @itest('tst w1, w2, ror #31')
    def test_tst_sft_reg_ror_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0x81424344')
    @itest('tst w1, w2, ror #4')
    def test_tst_sft_reg_ror32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('tst x1, x2')
    def test_tst_sft_reg64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0')
    @itest('tst x1, x2')
    def test_tst_sft_reg_zero64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 1073741824)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('tst x1, x2, lsl #0')
    def test_tst_sft_reg_lsl_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('tst x1, x2, lsl #63')
    def test_tst_sft_reg_lsl_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('tst x1, x2, lsl #4')
    def test_tst_sft_reg_lsl64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('tst x1, x2, lsr #0')
    def test_tst_sft_reg_lsr_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('tst x1, x2, lsr #63')
    def test_tst_sft_reg_lsr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('tst x1, x2, lsr #4')
    def test_tst_sft_reg_lsr64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('tst x1, x2, asr #0')
    def test_tst_sft_reg_asr_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('tst x1, x2, asr #63')
    def test_tst_sft_reg_asr_max64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('tst x1, x2, asr #4')
    def test_tst_sft_reg_asr64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('X1=0x41424344ffffffff', 'X2=0xffffffff45464748')
    @itest('tst x1, x2, ror #0')
    def test_tst_sft_reg_ror_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8000000000000001')
    @itest('tst x1, x2, ror #63')
    def test_tst_sft_reg_ror_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0x8142434445464748')
    @itest('tst x1, x2, ror #4')
    def test_tst_sft_reg_ror64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('XZR'), 0)
        self.assertEqual(self.rf.read('WZR'), 0)
        self.assertEqual(self.rf.read('NZCV'), 2147483648)

    @itest_setregs('W1=0x44434241')
    @itest('ubfiz w0, w1, #0, #1')
    def test_ubfiz_min_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=0x41424344')
    @itest('ubfiz w0, w1, #0, #32')
    def test_ubfiz_min_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x44434241')
    @itest('ubfiz w0, w1, #31, #1')
    def test_ubfiz_max_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)

    @itest_setregs('W1=0x41427fff')
    @itest('ubfiz w0, w1, #17, #15')
    def test_ubfiz32(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 4294836224)
        self.assertEqual(self.rf.read('W0'), 4294836224)

    @itest_setregs('X1=0x4847464544434241')
    @itest('ubfiz x0, x1, #0, #1')
    def test_ubfiz_min_min64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0x4142434445464748')
    @itest('ubfiz x0, x1, #0, #64')
    def test_ubfiz_min_max64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x4847464544434241')
    @itest('ubfiz x0, x1, #63, #1')
    def test_ubfiz_max_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0x414243447fffffff')
    @itest('ubfiz x0, x1, #33, #31')
    def test_ubfiz64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744065119617024)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424328')
    @itest('ubfm w0, w1, #3, #5')
    def test_ubfm_ge32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 5)
        self.assertEqual(self.rf.read('W0'), 5)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424349')
    @itest('ubfm w0, w1, #5, #3')
    def test_ubfm_lt32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1207959552)
        self.assertEqual(self.rf.read('W0'), 1207959552)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('ubfm w0, w1, #0, #31')
    def test_ubfm_ge_max32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W0=0xffffffff', 'W1=0x44434241')
    @itest('ubfm w0, w1, #31, #0')
    def test_ubfm_lt_max32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 2)
        self.assertEqual(self.rf.read('W0'), 2)

    @itest_setregs('W0=0xffffffff', 'W1=0x44434241')
    @itest('ubfm w0, w1, #0, #0')
    def test_ubfm_ge_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W0=0xffffffff', 'W1=0x44434241')
    @itest('ubfm w0, w1, #1, #0')
    def test_ubfm_lt_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 2147483648)
        self.assertEqual(self.rf.read('W0'), 2147483648)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('ubfm w0, w1, #0, #7')
    def test_ubfm_uxtb(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 68)
        self.assertEqual(self.rf.read('W0'), 68)

    @itest_setregs('W0=0xffffffff', 'W1=0x41424344')
    @itest('ubfm w0, w1, #0, #15')
    def test_ubfm_uxth(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 17220)
        self.assertEqual(self.rf.read('W0'), 17220)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464728')
    @itest('ubfm x0, x1, #3, #5')
    def test_ubfm_ge64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 5)
        self.assertEqual(self.rf.read('W0'), 5)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464749')
    @itest('ubfm x0, x1, #5, #3')
    def test_ubfm_lt64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 5188146770730811392)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4142434445464748')
    @itest('ubfm x0, x1, #0, #63')
    def test_ubfm_ge_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4847464544434241')
    @itest('ubfm x0, x1, #63, #0')
    def test_ubfm_lt_max64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 2)
        self.assertEqual(self.rf.read('W0'), 2)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4847464544434241')
    @itest('ubfm x0, x1, #0, #0')
    def test_ubfm_ge_min64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X0=0xffffffffffffffff', 'X1=0x4847464544434241')
    @itest('ubfm x0, x1, #1, #0')
    def test_ubfm_lt_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 9223372036854775808)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=0x44434241')
    @itest('ubfx w0, w1, #0, #1')
    def test_ubfx_min_min32(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=0x41424344')
    @itest('ubfx w0, w1, #0, #32')
    def test_ubfx_min_max32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1094861636)
        self.assertEqual(self.rf.read('W0'), 1094861636)

    @itest_setregs('W1=0x81424344')
    @itest('ubfx w0, w1, #31, #1')
    def test_ubfx_max_min32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=0xffff4344')
    @itest('ubfx w0, w1, #16, #16')
    def test_ubfx32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 65535)
        self.assertEqual(self.rf.read('W0'), 65535)

    @itest_setregs('X1=0x4847464544434241')
    @itest('ubfx x0, x1, #0, #1')
    def test_ubfx_min_min64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0x4142434445464748')
    @itest('ubfx x0, x1, #0, #64')
    def test_ubfx_min_max64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 4702394921427289928)
        self.assertEqual(self.rf.read('W0'), 1162233672)

    @itest_setregs('X1=0x8142434445464748')
    @itest('ubfx x0, x1, #63, #1')
    def test_ubfx_max_min64(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0xffffffff45464748')
    @itest('ubfx x0, x1, #32, #32')
    def test_ubfx64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 4294967295)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('W1=0', 'W2=0')
    @itest('udiv w0, w1, w2')
    def test_udiv_min32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=0xffffffff')
    @itest('udiv w0, w1, w2')
    def test_udiv_max32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('W1=0xffffffff', 'W2=0')
    @itest('udiv w0, w1, w2')
    def test_udiv0_32(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('W1=0xffffffff', 'W2=2')
    @itest('udiv w0, w1, w2')
    def test_udiv_neg32(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 2147483647)
        self.assertEqual(self.rf.read('W0'), 2147483647)

    @itest_setregs('W1=1', 'W2=2')
    @itest('udiv w0, w1, w2')
    def test_udiv_pos32(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0', 'X2=0')
    @itest('udiv x0, x1, x2')
    def test_udiv_min64(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0xffffffffffffffff')
    @itest('udiv x0, x1, x2')
    def test_udiv_max64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 1)
        self.assertEqual(self.rf.read('W0'), 1)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0')
    @itest('udiv x0, x1, x2')
    def test_udiv0_64(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=2')
    @itest('udiv x0, x1, x2')
    def test_udiv_neg64(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('X0'), 9223372036854775807)
        self.assertEqual(self.rf.read('W0'), 4294967295)

    @itest_setregs('X1=1', 'X2=2')
    @itest('udiv x0, x1, x2')
    def test_udiv_pos64(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    def _umov(self, mnem, dst, vess, elem_size, elem_count):
        if False:
            return 10
        for i in range(elem_count):
            val = 172148077542963347852807419361565775768
            sft = i * elem_size
            res = val >> sft & Mask(elem_size)
            insn = f'{mnem} {dst}0, v1.{vess}[{i}]'

            @itest_setregs(f'V1={val}')
            @itest_custom(['mrs x30, cpacr_el1', 'orr x30, x30, #0x300000', 'msr cpacr_el1, x30', insn], multiple_insts=True)
            def f(self):
                if False:
                    return 10

                def assertEqual(x, y):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(x, y, msg=insn)
                for i in range(4):
                    self._execute(reset=i == 0)
                assertEqual(self.rf.read('X0'), res & Mask(64))
                assertEqual(self.rf.read('W0'), res & Mask(32))
            self.setUp()
            f(self)

    def test_umov(self):
        if False:
            print('Hello World!')
        self._umov(mnem='umov', dst='w', vess='b', elem_size=8, elem_count=16)
        self._umov(mnem='umov', dst='w', vess='h', elem_size=16, elem_count=8)
        self._umov(mnem='umov', dst='w', vess='s', elem_size=32, elem_count=4)
        self._umov(mnem='umov', dst='x', vess='d', elem_size=64, elem_count=2)

    @itest_setregs('X1=0', 'X2=0')
    @itest('umulh x0, x1, x2')
    def test_umulh_min(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 0)
        self.assertEqual(self.rf.read('W0'), 0)

    @itest_setregs('X1=0xffffffffffffffff', 'X2=0xffffffffffffffff')
    @itest('umulh x0, x1, x2')
    def test_umulh_max(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 18446744073709551614)
        self.assertEqual(self.rf.read('W0'), 4294967294)

    @itest_setregs('X1=0x4142434445464748', 'X2=0x4142434445464748')
    @itest('umulh x0, x1, x2')
    def test_umulh(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('X0'), 1198722002577143526)
        self.assertEqual(self.rf.read('W0'), 1812870886)

    @itest_setregs('W1=0x41424381')
    @itest('uxtb w0, w1')
    def test_uxtb(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('X0'), 129)
        self.assertEqual(self.rf.read('W0'), 129)

    @itest_setregs('W1=0x41428561')
    @itest('uxth w0, w1')
    def test_uxth(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('X0'), 34145)
        self.assertEqual(self.rf.read('W0'), 34145)

class Aarch64CpuInstructions(unittest.TestCase, Aarch64Instructions):

    def setUp(self):
        if False:
            print('Hello World!')
        self.cpu = Cpu(Memory64())
        self.mem = self.cpu.memory
        self.rf = self.cpu.regfile

    def _execute(self, check_pc=True, **kwargs):
        if False:
            while True:
                i = 10
        pc = self.cpu.PC
        self.cpu.execute()
        if check_pc:
            self.assertEqual(self.cpu.PC, pc + 4)

class Aarch64UnicornInstructions(unittest.TestCase, Aarch64Instructions):

    def setUp(self):
        if False:
            return 10
        self.cpu = Cpu(Memory64())
        self.mem = self.cpu.memory
        self.rf = self.cpu.regfile

    def _setupCpu(self, asm, mode=CS_MODE_ARM, multiple_insts=False):
        if False:
            i = 10
            return i + 15
        super()._setupCpu(asm, mode, multiple_insts)
        self.backup_emu = UnicornEmulator(self.cpu)
        self.backup_emu.reset()
        self.backup_emu._create_emulated_mapping(self.backup_emu._emu, self.cpu.STACK)

    def _execute(self, check_pc=True, reset=True, **kwargs):
        if False:
            return 10
        pc = self.cpu.PC
        insn = self.cpu.decode_instruction(pc)
        self.backup_emu.emulate(insn, reset=reset)
        if check_pc:
            self.assertEqual(self.cpu.PC, pc + 4)

class Aarch64SymInstructions(unittest.TestCase, Aarch64Instructions):

    def setUp(self):
        if False:
            print('Hello World!')
        self.cs = ConstraintSet()
        self.cpu = Cpu(SMemory64(self.cs))
        self.mem = self.cpu.memory
        self.rf = self.cpu.regfile

    def _get_all_values1(self, expr):
        if False:
            return 10
        values = Z3Solver.instance().get_all_values(self.cs, expr)
        self.assertEqual(len(values), 1)
        return values[0]

    def _execute(self, check_pc=True, check_cs=True, **kwargs):
        if False:
            i = 10
            return i + 15
        if check_cs:
            self.assertTrue(len(self.cs) > 0)
        pc = self.cpu.PC
        done = False
        while not done:
            try:
                self.cpu.execute()
                done = True
            except ConcretizeRegister as e:
                expr = getattr(self.cpu, e.reg_name)
                value = self._get_all_values1(expr)
                setattr(self.cpu, e.reg_name, value)
        if check_pc:
            self.assertEqual(self.cpu.PC, pc + 4)

    def assertEqual(self, actual, expected, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if isinstance(actual, int):
            pass
        else:
            actual = self._get_all_values1(actual)
        if isinstance(expected, int):
            pass
        else:
            expected = self._get_all_values1(expected)
        super().assertEqual(actual, expected, *args, **kwargs)