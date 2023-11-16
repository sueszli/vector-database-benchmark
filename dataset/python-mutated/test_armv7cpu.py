import unittest
import struct
import binascii
from capstone import CS_MODE_THUMB, CS_MODE_ARM
from functools import wraps
from manticore.native.cpu.abstractcpu import ConcretizeRegister
from manticore.native.cpu.arm import Armv7Cpu as Cpu, Mask, Interruption, Armv7RegisterFile
from manticore.core.smtlib import *
from manticore.core.state import Concretize
from manticore.core.smtlib.solver import Z3Solver
from manticore.native.memory import SMemory32
from manticore.utils.helpers import pickle_dumps
ks = None
ks_thumb = None
import logging
logger = logging.getLogger('ARM_TESTS')
solver = Z3Solver.instance()
assembly_cache = {CS_MODE_ARM: {'adc r3, r1, r2': b'0230a1e0', 'adc r3, r1, #0x18000': b'0639a1e2', 'adc r3, r1, #24, 20': b'183aa1e2', 'adc r3, r1, r2, ror #3': b'e231a1e0', 'add r3, r1, 0x1000000': b'013481e2', 'add r3, r1, 0x18000': b'063981e2', 'add r3, r1, 24, 20': b'183a81e2', 'add r3, r1, 0xff000000': b'ff3481e2', 'add r3, r1, 0x100': b'013c81e2', 'add r3, r1, 55': b'373081e2', 'add r3, r1, 0x1': b'013081e2', 'add r3, r1, r2': b'023081e0', 'add r3, r1, r2, asr #3': b'c23181e0', 'add r3, r1, r2, asr r4': b'523481e0', 'add r3, r1, r2, lsl #3': b'823181e0', 'add r3, r1, r2, lsl r4': b'123481e0', 'add r3, r1, r2, lsr #3': b'a23181e0', 'add r3, r1, r2, lsr r4': b'323481e0', 'add r3, r1, r2, ror #3': b'e23181e0', 'add r3, r1, r2, ror r4': b'723481e0', 'add r3, r1, r2, rrx': b'623081e0', 'add pc, pc, r1': b'01f08fe0', 'adds r3, r1, 0x1000000': b'013491e2', 'adds r3, r1, 0x80000000': b'023191e2', 'adds r3, r1, 0xff000000': b'ff3491e2', 'adds r3, r1, 0x100': b'013c91e2', 'adds r3, r1, 55': b'373091e2', 'adds r3, r1, 0x1': b'013091e2', 'adds r3, r3, 0x0': b'003093e2', 'adds r3, r1, r2': b'023091e0', 'adds r3, r1, r2, asr #3': b'c23191e0', 'adds r3, r1, r2, rrx': b'623091e0', 'adr r0, #16': b'10008fe2', 'add r0, PC, #0x10': b'10008fe2', 'add r0, PC, #1, 28': b'10008fe2', 'and r2, r2, #1': b'012002e2', 'and r2, r2, #0x18000': b'062902e2', 'and r2, r2, #24, 20': b'182a02e2', 'and r1, r1, r2': b'021001e0', 'BIC R2, R1, #0x10': b'1020c1e3', 'BIC R2, R1, #0x18000': b'0629c1e3', 'BIC R2, R1, #24, 20': b'182ac1e3', 'bl 0x170': b'5a0000eb', 'bl #-4': b'fdffffeb', 'BLX R1': b'31ff2fe1', 'blx  r1': b'31ff2fe1', 'bx r1': b'11ff2fe1', 'clz r1, r2': b'121f6fe1', 'cmn r0, #0x18000': b'060970e3', 'cmn r0, #24, 20': b'180a70e3', 'cmp r0, 0': b'000050e3', 'cmp r0, 0x40000000': b'010150e3', 'cmp r0, 3': b'030050e3', 'cmp r0, #0x18000': b'060950e3', 'cmp r0, #24, 20': b'180a50e3', 'cmp r0, 2': b'020050e3', 'cmp r0, 5': b'050050e3', 'cmp r0, 0xa0000000': b'0a0250e3', 'dmb ish': b'5bf07ff5', 'eor r2, r3, #5': b'052023e2', 'eor r2, r3, #0x18000': b'062923e2', 'eor r2, r3, #24, 20': b'182a23e2', 'eor r2, r3, r4': b'042023e0', 'eor r2, r3, r4, LSL #4': b'042223e0', 'eors r2, r3': b'032032e0', 'adds r2, r1, #0x1': b'012091e2', 'tst r3, r1': b'010013e1', 'ldm sp, {r1, r2, r3}': b'0e009de8', 'ldm sp!, {r1, r2, r3}': b'0e00bde8', 'ldmda r0!, {r1, r2, r3}': b'0e0030e8', 'ldmdb r0!, {r1, r2, r3}': b'0e0030e9', 'ldmia r0!, {r1, r2, r3}': b'0e00b0e8', 'ldmib r0!, {r1, r2, r3}': b'0e00b0e9', 'ldr r1, [sp, #-4]': b'04101de5', 'ldr r1, [sp]': b'00109de5', 'ldr pc, [sp]': b'00f09de5', 'ldr r1, [sp, #4]': b'04109de5', 'ldr r1, [sp], #-5': b'05101de4', 'ldr r1, [sp], #5': b'05109de4', 'ldr r1, [sp, #-4]!': b'04103de5', 'ldr r1, [sp, #4]!': b'0410bde5', 'ldr r1, [sp, r2]': b'02109de7', 'ldr r1, [sp, -r2]': b'02101de7', 'ldr r1, [sp, -r2, lsl #3]': b'82111de7', 'ldr r1, [sp, r2, lsl #3]': b'82119de7', 'ldr r1, [sp], r2': b'02109de6', 'ldr r1, [sp], -r2, lsl #3': b'82111de6', 'ldr r1, [sp, r2]!': b'0210bde7', 'ldr r1, [sp, -r2, lsl #3]!': b'82113de7', 'ldrb r1, [sp]': b'0010dde5', 'ldrb r1, [sp, r2]': b'0210dde7', 'ldrd r2, [sp]': b'd020cde1', 'ldrh r1, [sp]': b'b010dde1', 'ldrh r1, [sp, r2]': b'b2109de1', 'ldrsb r1, [sp]': b'd010dde1', 'ldrsb r1, [sp, r2]': b'd2109de1', 'ldrsh r1, [sp]': b'f010dde1', 'ldrsh r1, [sp, r2]': b'f2109de1', 'lsls r2, r2, #0x1f': b'822fb0e1', 'lsls r4, r3, 31': b'834fb0e1', 'lsls r4, r3, 1': b'8340b0e1', 'lsls r4, r3, r2': b'1342b0e1', 'lsr r0, r0, r2': b'3002a0e1', 'lsr r0, r0, #3': b'a001a0e1', 'MLA R1, R2, R3, R4': b'924321e0', 'mov r0, 0x0': b'0000a0e3', 'mov r0, 0xff000000': b'ff04a0e3', 'mov r0, 0x100': b'010ca0e3', 'mov r0, 42': b'2a00a0e3', 'mov r0, r1': b'0100a0e1', 'mov r0, #0x18000': b'0609a0e3', 'mov r0, #24, 20': b'180aa0e3', 'movs r0, 0': b'0000b0e3', 'movs r0, 0xff000000': b'ff04b0e3', 'movs r0, 0x100': b'010cb0e3', 'movs r0, 0x0e000000': b'0e04b0e3', 'movs r0, 42': b'2a00b0e3', 'movs r0, r1': b'0100b0e1', 'movt R3, #9': b'093040e3', 'movw r0, 0xffff': b'ff0f0fe3', 'movw r0, 0': b'000000e3', 'mrc p15, #0, r2, c13, c0, #3': b'702f1dee', 'MUL R1, R2': b'910201e0', 'MUL R3, R1, R2': b'910203e0', 'mvn r0, #0xFFFFFFFF': b'0000a0e3', 'mvn r0, #0x0': b'0000e0e3', 'mvn r0, #0x18000': b'0609e0e3', 'mvn r0, #24, 20': b'180ae0e3', 'orr r2, r3, #5': b'052083e3', 'orr r2, r3, #0x18000': b'062983e3', 'orr r2, r3, #24, 20': b'182a83e3', 'orr r2, r3, r4': b'042083e1', 'orr r2, r3, r4, LSL #4': b'042283e1', 'orr r2, r3': b'032082e1', 'orrs r2, r3': b'032092e1', 'pop {r1, r2, r3}': b'0e00bde8', 'pop {r1}': b'04109de4', 'push {r1, r2, r3}': b'0e002de9', 'push {r1}': b'04102de5', 'rev r2, r1': b'312fbfe6', 'RSB r2, r2, #31': b'1f2062e2', 'RSB r2, r2, #0x18000': b'062962e2', 'RSB r2, r2, #24, 20': b'182a62e2', 'RSBS r8, r6, #0': b'008076e2', 'rsc r3, r1, #0x18000': b'0639e1e2', 'rsc r3, r1, #24, 20': b'183ae1e2', 'sbc r3, r1, #5': b'0530c1e2', 'sbc r3, r1, #0x18000': b'0639c1e2', 'sbc r3, r1, #24, 20': b'183ac1e2', 'stm sp, {r1, r2, r3}': b'0e008de8', 'stm sp!, {r1, r2, r3}': b'0e00ade8', 'stmda r0!, {r1, r2, r3}': b'0e0020e8', 'stmdb r0!, {r1, r2, r3}': b'0e0020e9', 'stmia r0!, {r1, r2, r3}': b'0e00a0e8', 'stmib r0!, {r1, r2, r3}': b'0e00a0e9', 'str R2, [R1]': b'002081e5', 'str SP, [R1]': b'00d081e5', 'str R1, [R2, R3]': b'031082e7', 'str R1, [R2, R3, LSL #3]': b'831182e7', 'str R1, [R2, #3]!': b'0310a2e5', 'str R1, [R2], #3': b'031082e4', 'strd R2, [R1]': b'f020c1e1', 'sub r3, r1, r2': b'023041e0', 'sub r3, r1, #5': b'053041e2', 'sub r3, r1, #0x18000': b'063941e2', 'sub r3, r1, #24, 20': b'183a41e2', 'svc #0': b'000000ef', 'sxth r1, r2': b'7210bfe6', 'sxth r3, r4': b'7430bfe6', 'sxth r5, r4, ROR #8': b'7454bfe6', 'teq r3, r1': b'010033e1', 'teq r3, #0x18000': b'060933e3', 'teq r3, #24, 20': b'180a33e3', 'BIC R1, #0x10': b'1010c1e3', 'tst r3, #0x18000': b'060913e3', 'tst r3, #24, 20': b'180a13e3', 'UMULLS R1, R2, R1, R2': b'911292e0', 'uqsub8 r3, r1, r2': b'f23f61e6', 'uxtb r1, r2': b'7210efe6', 'uxth r1, r2': b'7210ffe6', 'vldmia  r1, {d8, d9, d10}': b'068b91ec', 'vldmia  r1!, {d8, d9, d10}': b'068bb1ec'}, CS_MODE_THUMB: {'adds r0, #4': b'0430', 'addw r0, r1, #0x2a': b'01f22a00', 'addw r0, pc, #0x2a': b'0ff22a00', 'adr r0, #16': b'04a0', 'asr.w R5, R6, #3': b'4feae605', 'cbnz r0, #0x2a': b'98b9', 'cbz r0, #0x2a': b'98b1', 'cmp r1, #1': b'0129', 'ite ne': b'14bf', 'mov r2, r12': b'6246', 'mov r3, r12': b'6346', 'mov r4, r12': b'6446', 'itete ne': b'15bf', 'mov r1, #1': b'4ff00101', 'mov r2, #1': b'4ff00102', 'mov r3, #1': b'4ff00103', 'mov r4, #4': b'4ff00404', 'itt ne': b'1cbf', 'lsl.w r5, r6, #3': b'4feac605', 'lsr.w R5, R6, #3': b'4fead605', 'lsr.w R0, R0, R2': b'20fa02f0', 'orn r2, r2, r5': b'62ea0502', 'sbcs r0, r3': b'9841', 'sel r4, r5, r6': b'a5fa86f4', 'subw r0, r1, #0x2a': b'a1f22a00', 'subw r0, pc, #0x2a': b'aff22a00', '  tst r0, r0\n  beq label\n  bne label\nlabel:\n  nop': b'004200d0ffd100bf', 'tbb [r0, r1]': b'd0e801f0', 'tbb [pc, r1]': b'dfe801f0', 'tbh [r0, r1, lsl #1]': b'd0e811f0', 'tbh [pc, r1, lsl #1]': b'dfe811f0', 'adcs r3, r4': b'6341', 'eor r3, #5': b'83f00503', 'lsrs r1, r2': b'd140', 'orr r3, #5': b'43f00503', 'sub r3, #12': b'a3f10c03', 'uadd8 r2, r2, r3': b'82fa43f2'}}

def _ks_assemble(asm: str, mode=CS_MODE_ARM) -> bytes:
    if False:
        while True:
            i = 10
    'Assemble the given string using Keystone using the specified CPU mode.'
    global ks, ks_thumb
    from keystone import Ks, KS_ARCH_ARM, KS_MODE_ARM, KS_MODE_THUMB
    if ks is None:
        ks = Ks(KS_ARCH_ARM, KS_MODE_ARM)
    if ks_thumb is None:
        ks_thumb = Ks(KS_ARCH_ARM, KS_MODE_THUMB)
    if CS_MODE_ARM == mode:
        ords = ks.asm(asm)[0]
    elif CS_MODE_THUMB == mode:
        ords = ks_thumb.asm(asm)[0]
    else:
        raise Exception(f'bad processor mode for assembly: {mode}')
    if not ords:
        raise Exception(f'bad assembly: {asm}')
    return binascii.hexlify(bytearray(ords))

def assemble(asm: str, mode=CS_MODE_ARM) -> bytes:
    if False:
        return 10
    '\n    Assemble the given string.\n\n    An assembly cache is first checked, and if there is no entry there, then Keystone is used.\n    '
    if asm in assembly_cache[mode]:
        return binascii.unhexlify(assembly_cache[mode][asm])
    return binascii.unhexlify(_ks_assemble(asm, mode=mode))

def testRegisterFileCopy():
    if False:
        return 10
    regfile = Armv7RegisterFile()
    regfile.write('PC', 1234)
    regfile.write('R0', BitVecConstant(size=64, value=24))
    regfile.write('R1', BitVecVariable(size=64, name='b'))
    new_regfile = copy.copy(regfile)
    assert new_regfile.read('PC') == 1234
    assert new_regfile.read('R0') is regfile.read('R0')
    assert new_regfile.read('R0') == regfile.read('R0')
    assert new_regfile.read('R1') is regfile.read('R1')
    assert new_regfile.read('R1') == regfile.read('R1')
    rax_val = regfile.read('R0')
    regfile.write('PC', Operators.ITEBV(64, rax_val == 0, 4321, 1235))
    regfile.write('R0', rax_val * 2)
    assert new_regfile.read('PC') is not regfile.read('PC')
    assert new_regfile.read('PC') != regfile.read('PC')
    assert new_regfile.read('PC') == 1234
    assert new_regfile.read('R0') is not regfile.read('R0')
    assert new_regfile.read('R0') != regfile.read('R0')
    assert new_regfile.read('R0') is rax_val
    assert new_regfile.read('R0') == rax_val

class Armv7CpuTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        if False:
            while True:
                i = 10
        cs = ConstraintSet()
        self.c = Cpu(SMemory32(cs))
        self.rf = self.c.regfile
        self._setupStack()

    def _setupStack(self):
        if False:
            return 10
        self.stack = self.c.memory.mmap(61440, 4096, 'rw')
        self.rf.write('SP', self.stack + 4096)

    def test_rd(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R0'), 0)

    def test_rd2(self):
        if False:
            while True:
                i = 10
        self.c.STACK = 4919
        self.assertEqual(self.rf.read('SP'), 4919)

    def test_stack_set_get(self):
        if False:
            while True:
                i = 10
        self.c.STACK = 4919
        self.assertEqual(self.c.STACK, 4919)

    def test_rd3(self):
        if False:
            print('Hello World!')
        self.c.STACK = 4919 - 1
        self.assertEqual(self.rf.read('SP'), 4918)

    def test_rd4(self):
        if False:
            for i in range(10):
                print('nop')
        self.c.STACK = 4919 + 1
        self.assertEqual(self.rf.read('SP'), 4920)

    def test_stack_push(self):
        if False:
            return 10
        self.c.stack_push(42)
        self.c.stack_push(44)
        self.assertEqual(b''.join(self.c.read(self.c.STACK, 4)), b',\x00\x00\x00')
        self.assertEqual(b''.join(self.c.read(self.c.STACK + 4, 4)), b'*\x00\x00\x00')

    def test_stack_pop(self):
        if False:
            return 10
        v = 85
        v_bytes = struct.pack('<I', v)
        self.c.stack_push(v)
        val = self.c.stack_pop()
        self.assertEqual(b''.join(self.c.read(self.c.STACK - 4, 4)), v_bytes)

    def test_stack_peek(self):
        if False:
            for i in range(10):
                print('nop')
        self.c.stack_push(42)
        self.assertEqual(b''.join(self.c.stack_peek()), b'*\x00\x00\x00')

    def test_readwrite_int(self):
        if False:
            for i in range(10):
                print('nop')
        self.c.STACK -= 4
        self.c.write_int(self.c.STACK, 16962, 32)
        self.assertEqual(self.c.read_int(self.c.STACK), 16962)

def itest_failing(asm):
    if False:
        print('Hello World!')

    def instr_dec(assertions_func):
        if False:
            for i in range(10):
                print('nop')

        @wraps(assertions_func)
        def wrapper(self):
            if False:
                for i in range(10):
                    print('nop')
            self._setupCpu(asm)
            self.cpu.instruction = '\x00\x00\x00\x00'
            assertions_func(self)
        return wrapper
    return instr_dec

def itest(asm):
    if False:
        for i in range(10):
            print('nop')

    def instr_dec(assertions_func):
        if False:
            while True:
                i = 10

        @wraps(assertions_func)
        def wrapper(self):
            if False:
                for i in range(10):
                    print('nop')
            self._setupCpu(asm)
            self.cpu.execute()
            assertions_func(self)
        return wrapper
    return instr_dec

def itest_setregs(*preds):
    if False:
        print('Hello World!')

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
                except Exception:
                    pass
                self.rf.write(dest.upper(), src)
            custom_func(self)
        return wrapper
    return instr_dec

def itest_custom(asm, mode=CS_MODE_ARM):
    if False:
        for i in range(10):
            print('nop')

    def instr_dec(custom_func):
        if False:
            while True:
                i = 10

        @wraps(custom_func)
        def wrapper(self):
            if False:
                i = 10
                return i + 15
            self._setupCpu(asm, mode)
            custom_func(self)
        return wrapper
    return instr_dec

def itest_thumb(asm):
    if False:
        for i in range(10):
            print('nop')

    def instr_dec(assertions_func):
        if False:
            print('Hello World!')

        @wraps(assertions_func)
        def wrapper(self):
            if False:
                while True:
                    i = 10
            self._setupCpu(asm, mode=CS_MODE_THUMB)
            self.cpu.execute()
            assertions_func(self)
        return wrapper
    return instr_dec

def itest_multiple(asms):
    if False:
        i = 10
        return i + 15

    def instr_dec(assertions_func):
        if False:
            return 10

        @wraps(assertions_func)
        def wrapper(self):
            if False:
                i = 10
                return i + 15
            self._setupCpu(asms, mode=CS_MODE_ARM, multiple_insts=True)
            for i in range(len(asms)):
                self.cpu.execute()
            assertions_func(self)
        return wrapper
    return instr_dec

def itest_thumb_multiple(asms):
    if False:
        return 10

    def instr_dec(assertions_func):
        if False:
            return 10

        @wraps(assertions_func)
        def wrapper(self):
            if False:
                return 10
            self._setupCpu(asms, mode=CS_MODE_THUMB, multiple_insts=True)
            for i in range(len(asms)):
                self.cpu.execute()
            assertions_func(self)
        return wrapper
    return instr_dec

class Armv7CpuInstructions(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        cs = ConstraintSet()
        self.cpu = Cpu(SMemory32(cs))
        self.mem = self.cpu.memory
        self.rf = self.cpu.regfile

    def _setupCpu(self, asm, mode=CS_MODE_ARM, multiple_insts=False):
        if False:
            while True:
                i = 10
        self.code = self.mem.mmap(4096, 4096, 'rwx')
        self.data = self.mem.mmap(53248, 4096, 'rw')
        self.stack = self.mem.mmap(61440, 4096, 'rw')
        start = self.code + 4
        if multiple_insts:
            offset = 0
            for asm_single in asm:
                asm_inst = assemble(asm_single, mode)
                self.mem.write(start + offset, asm_inst)
                offset += len(asm_inst)
        else:
            self.mem.write(start, assemble(asm, mode))
        self.rf.write('PC', start)
        self.rf.write('SP', self.stack + 4096)
        self.cpu.mode = mode

    def _checkFlagsNZCV(self, n, z, c, v):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('APSR_N'), n)
        self.assertEqual(self.rf.read('APSR_Z'), z)
        self.assertEqual(self.rf.read('APSR_C'), c)
        self.assertEqual(self.rf.read('APSR_V'), v)

    @itest('mvn r0, #0x0')
    def test_mvn_imm_min(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R0'), 4294967295)

    @itest('mvn r0, #0xFFFFFFFF')
    def test_mvn_imm_max(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('R0'), 0)

    @itest('mvn r0, #0x18000')
    def test_mvn_mod_imm_1(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R0'), 4294868991)

    @itest('mvn r0, #24, 20')
    def test_mvn_mod_imm_2(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R0'), 4294868991)

    @itest('mov r0, 0x0')
    def test_mov_imm_min(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R0'), 0)

    @itest('mov r0, 42')
    def test_mov_imm_norm(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R0'), 42)

    @itest('mov r0, 0x100')
    def test_mov_imm_modified_imm_min(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('R0'), 256)

    @itest('mov r0, 0xff000000')
    def test_mov_imm_modified_imm_max(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R0'), 4278190080)

    @itest('mov r0, #0x18000')
    def test_mov_mod_imm_1(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('R0'), 98304)

    @itest('mov r0, #24, 20')
    def test_mov_mod_imm_2(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R0'), 98304)

    @itest_custom('mov r0, r1')
    def test_mov_immreg(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 0)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 0)

    @itest_custom('mov r0, r1')
    def test_mov_immreg1(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 2 ** 32)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 0)

    @itest_custom('mov r0, r1')
    def test_mov_immreg2(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 4294967295)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 4294967295)

    @itest_custom('mov r0, r1')
    def test_mov_immreg3(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 42)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 42)

    @itest('movw r0, 0')
    def test_movw_imm_min(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R0'), 0)

    @itest('movw r0, 0xffff')
    def test_movw_imm_max(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R0'), 65535)

    @itest_custom('movs r0, 0')
    def test_movs_imm_min(self):
        if False:
            i = 10
            return i + 15
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 0)
        self._checkFlagsNZCV(0, 1, pre_c, pre_v)

    @itest_custom('movs r0, 42')
    def test_movs_imm_norm(self):
        if False:
            for i in range(10):
                print('nop')
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 42)
        self._checkFlagsNZCV(0, 0, pre_c, pre_v)

    @itest_custom('movs r0, 0x100')
    def test_movs_imm_modified_imm_min(self):
        if False:
            while True:
                i = 10
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 256)
        self._checkFlagsNZCV(0, 0, pre_c, pre_v)

    @itest_custom('movs r0, 0xff000000')
    def test_movs_imm_modified_imm_max(self):
        if False:
            return 10
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 4278190080)
        self._checkFlagsNZCV(1, 0, 1, pre_v)

    @itest_custom('movs r0, 0x0e000000')
    def test_movs_imm_modified_imm_sans_carry(self):
        if False:
            while True:
                i = 10
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 234881024)
        self._checkFlagsNZCV(0, 0, 0, pre_v)

    @itest_custom('movs r0, r1')
    def test_movs_reg(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 0)
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 0)
        self._checkFlagsNZCV(0, 1, pre_c, pre_v)

    @itest_custom('movs r0, r1')
    def test_movs_reg1(self):
        if False:
            return 10
        self.rf.write('R1', 2 ** 32)
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 0)
        self._checkFlagsNZCV(0, 1, pre_c, pre_v)

    @itest_custom('movs r0, r1')
    def test_movs_reg2(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 2 ** 32 - 1)
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 2 ** 32 - 1)
        self._checkFlagsNZCV(1, 0, pre_c, pre_v)

    @itest_custom('movs r0, r1')
    def test_movs_reg3(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 42)
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), 42)
        self._checkFlagsNZCV(0, 0, pre_c, pre_v)

    @itest_custom('add r3, r1, 55')
    def test_add_imm_norm(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 44)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 99)

    @itest_custom('add r3, r1, 0x100')
    def test_add_imm_mod_imm_min(self):
        if False:
            print('Hello World!')
        self.rf.write('R1', 44)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 256)

    @itest_custom('add r3, r1, 0x18000')
    def test_add_imm_mod_imm_case1(self):
        if False:
            return 10
        self.rf.write('R1', 44)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 98304)

    @itest_custom('add r3, r1, 24, 20')
    def test_add_imm_mod_imm_case2(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 44)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 98304)

    @itest_custom('add r3, r1, 0xff000000')
    def test_add_imm_mod_imm_max(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 44)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 4278190080)

    @itest_custom('add r3, r1, 0x1000000')
    def test_add_imm_carry(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 4278190081)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1)

    @itest_custom('add r3, r1, 0x1')
    def test_add_imm_overflow(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 2 ** 31 - 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2147483648)

    @itest_custom('add r3, r1, r2')
    def test_add_reg_norm(self):
        if False:
            return 10
        self.rf.write('R1', 44)
        self.rf.write('R2', 55)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 99)

    @itest_custom('add r3, r1, r2')
    def test_add_reg_mod_imm_min(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 44)
        self.rf.write('R2', 256)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 256)

    @itest_custom('add r3, r1, r2')
    def test_add_reg_mod_imm_max(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 44)
        self.rf.write('R2', 4278190080)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 4278190080)

    @itest_custom('add r3, r1, r2')
    def test_add_reg_carry(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 16777216)
        self.rf.write('R2', 4278190081)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1)

    @itest_custom('add r3, r1, r2')
    def test_add_reg_overflow(self):
        if False:
            return 10
        self.rf.write('R1', 2 ** 31 - 1)
        self.rf.write('R2', 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1 << 31)

    @itest_custom('add r3, r1, r2, lsl #3')
    def test_add_reg_sft_lsl(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 0)
        self.rf.write('R2', 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1 << 3)

    @itest_custom('add r3, r1, r2, lsr #3')
    def test_add_reg_sft_lsr(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 0)
        self.rf.write('R2', 8)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 8 >> 3)

    @itest_custom('add r3, r1, r2, asr #3')
    def test_add_reg_sft_asr(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 0)
        self.rf.write('R2', 2147483648)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 4026531840)

    @itest_custom('add r3, r1, r2, asr #3')
    def test_add_reg_sft_asr2(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 0)
        self.rf.write('R2', 1073741824)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1073741824 >> 3)

    @itest_custom('add r3, r1, r2, ror #3')
    def test_add_reg_sft_ror_norm(self):
        if False:
            print('Hello World!')
        self.rf.write('R1', 0)
        self.rf.write('R2', 8)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1)

    @itest_custom('add r3, r1, r2, ror #3')
    def test_add_reg_sft_ror(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 0)
        self.rf.write('R2', 3)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1610612736)

    @itest_setregs('R3=0xfffffff6', 'R4=10')
    @itest_thumb('adcs r3, r4')
    def test_thumb_adc_basic(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R3'), 0)

    @itest_custom('adc r3, r1, r2')
    @itest_setregs('R1=1', 'R2=2', 'APSR_C=1')
    def test_adc_basic(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 4)

    @itest_custom('adc r3, r1, r2, ror #3')
    @itest_setregs('R1=1', 'R2=2', 'APSR_C=1')
    def test_adc_reg_sft_ror(self):
        if False:
            print('Hello World!')
        self.rf.write('R1', 0)
        self.rf.write('R2', 3)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1610612737)

    @itest_custom('adc r3, r1, #0x18000')
    @itest_setregs('R1=1', 'APSR_C=1')
    def test_adc_mod_imm_1(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 98306)

    @itest_custom('adc r3, r1, #24, 20')
    @itest_setregs('R1=1', 'APSR_C=1')
    def test_adc_mod_imm_2(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 98306)

    @itest_custom('add r3, r1, r2, rrx')
    def test_add_reg_sft_rrx(self):
        if False:
            print('Hello World!')
        self.rf.write('APSR_C', 0)
        self.rf.write('R1', 0)
        self.rf.write('R2', 2 ** 32 - 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2 ** 31 - 1)

    @itest_custom('add r3, r1, r2, rrx')
    def test_add_reg_sft_rrx2(self):
        if False:
            while True:
                i = 10
        self.rf.write('APSR_C', 1)
        self.rf.write('R1', 0)
        self.rf.write('R2', 2 ** 32 - 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2 ** 32 - 1)

    @itest_custom('add r3, r1, r2, lsl r4')
    def test_add_reg_sft_lsl_reg(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 0)
        self.rf.write('R4', 3)
        self.rf.write('R2', 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1 << 3)

    @itest_custom('add r3, r1, r2, lsr r4')
    def test_add_reg_sft_lsr_reg(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 0)
        self.rf.write('R4', 3)
        self.rf.write('R2', 8)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 8 >> 3)

    @itest_custom('add r3, r1, r2, asr r4')
    def test_add_reg_sft_asr_reg(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 0)
        self.rf.write('R4', 3)
        self.rf.write('R2', 2147483648)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 4026531840)

    @itest_custom('add r3, r1, r2, asr r4')
    def test_add_reg_sft_asr2_reg(self):
        if False:
            print('Hello World!')
        self.rf.write('R1', 0)
        self.rf.write('R4', 3)
        self.rf.write('R2', 1073741824)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1073741824 >> 3)

    @itest_custom('add r3, r1, r2, ror r4')
    def test_add_reg_sft_ror_norm_reg(self):
        if False:
            return 10
        self.rf.write('R1', 0)
        self.rf.write('R4', 3)
        self.rf.write('R2', 8)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1)

    @itest_custom('add r3, r1, r2, ror r4')
    def test_add_reg_sft_ror_reg(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 0)
        self.rf.write('R4', 3)
        self.rf.write('R2', 3)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1610612736)

    @itest_custom('add r3, r1, r2, rrx')
    def test_add_reg_sft_rrx_reg(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 0)
        self.rf.write('APSR_C', 0)
        self.rf.write('R2', 2 ** 32 - 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2 ** 31 - 1)

    @itest_custom('add r3, r1, r2, rrx')
    def test_add_reg_sft_rrx2_reg(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 0)
        self.rf.write('APSR_C', 1)
        self.rf.write('R2', 2 ** 32 - 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2 ** 32 - 1)

    @itest_custom('adds r3, r1, 55')
    def test_adds_imm_norm(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 44)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 99)
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_custom('adds r3, r1, 0x100')
    def test_adds_imm_mod_imm_min(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 44)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 256)
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_custom('adds r3, r1, 0xff000000')
    def test_adds_imm_mod_imm_max(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 44)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 4278190080)
        self._checkFlagsNZCV(1, 0, 0, 0)

    @itest_custom('adds r3, r1, 0x1000000')
    def test_adds_imm_carry(self):
        if False:
            return 10
        self.rf.write('R1', 4278190081)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1)
        self._checkFlagsNZCV(0, 0, 1, 0)

    @itest_custom('adds r3, r1, 0x80000000')
    def test_adds_imm_carry_overflow(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 2147483649)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1)
        self._checkFlagsNZCV(0, 0, 1, 1)

    @itest_custom('adds r3, r1, 0x1')
    def test_adds_imm_overflow(self):
        if False:
            while True:
                i = 10
        self.rf.write('R1', 2 ** 31 - 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2147483648)
        self._checkFlagsNZCV(1, 0, 0, 1)

    @itest_custom('adds r3, r3, 0x0')
    def test_adds_imm_zf(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R3', 0)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 0)
        self._checkFlagsNZCV(0, 1, 0, 0)

    @itest_custom('adds r3, r1, r2')
    def test_adds_reg_norm(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 44)
        self.rf.write('R2', 55)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 99)
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_custom('adds r3, r1, r2')
    def test_adds_reg_mod_imm_min(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 44)
        self.rf.write('R2', 256)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 256)
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_custom('adds r3, r1, r2')
    def test_adds_reg_mod_imm_max(self):
        if False:
            return 10
        self.rf.write('R1', 44)
        self.rf.write('R2', 4278190080)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 44 + 4278190080)
        self._checkFlagsNZCV(1, 0, 0, 0)

    @itest_custom('adds r3, r1, r2')
    def test_adds_reg_carry(self):
        if False:
            print('Hello World!')
        self.rf.write('R1', 16777216)
        self.rf.write('R2', 4278190081)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1)
        self._checkFlagsNZCV(0, 0, 1, 0)

    @itest_custom('adds r3, r1, r2')
    def test_adds_reg_overflow(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 2 ** 31 - 1)
        self.rf.write('R2', 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1 << 31)
        self._checkFlagsNZCV(1, 0, 0, 1)

    @itest_custom('adds r3, r1, r2')
    def test_adds_reg_carry_overflow(self):
        if False:
            return 10
        self.rf.write('R1', 2147483649)
        self.rf.write('R2', 2147483648)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1)
        self._checkFlagsNZCV(0, 0, 1, 1)

    @itest_custom('adds r3, r1, r2')
    def test_adds_reg_zf(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 0)
        self.rf.write('R2', 0)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 0)
        self._checkFlagsNZCV(0, 1, 0, 0)

    @itest_custom('adds r3, r1, r2, asr #3')
    def test_adds_reg_sft_asr(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf.write('R1', 0)
        self.rf.write('R2', 2147483648)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 4026531840)
        self._checkFlagsNZCV(1, 0, 0, 0)

    @itest_custom('adds r3, r1, r2, asr #3')
    def test_adds_reg_sft_asr2(self):
        if False:
            i = 10
            return i + 15
        self.rf.write('R1', 0)
        self.rf.write('R2', 1073741824)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 1073741824 >> 3)
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_custom('adds r3, r1, r2, rrx')
    def test_adds_reg_sft_rrx(self):
        if False:
            while True:
                i = 10
        self.rf.write('APSR_C', 0)
        self.rf.write('R1', 0)
        self.rf.write('R2', 2 ** 32 - 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2 ** 31 - 1)
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_custom('adds r3, r1, r2, rrx')
    def test_adds_reg_sft_rrx2(self):
        if False:
            print('Hello World!')
        self.rf.write('APSR_C', 1)
        self.rf.write('R1', 0)
        self.rf.write('R2', 2 ** 32 - 1)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2 ** 32 - 1)
        self._checkFlagsNZCV(1, 0, 0, 0)

    @itest_setregs('R0=0')
    @itest_thumb('adds r0, #4')
    def test_adds_thumb_two_op(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R0'), 4)

    @itest_setregs('R2=0x00FF00FF', 'R3=0x00010002')
    @itest_thumb('uadd8 r2, r2, r3')
    def test_uadd8(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R2'), 1)
        self.assertEqual(self.rf.read('APSR_GE'), 5)

    @itest_custom('ldr r1, [sp]')
    def test_ldr_imm_off_none(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(42)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.cpu.mode, CS_MODE_ARM)

    @itest_custom('ldr pc, [sp]')
    def test_ldr_imm_off_none_to_thumb(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(43)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R15'), 42)
        self.assertEqual(self.cpu.mode, CS_MODE_THUMB)

    @itest_custom('ldr r1, [sp, #4]')
    def test_ldr_imm_off_pos(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(42)
        self.cpu.stack_push(41)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)

    @itest_custom('ldr r1, [sp, #-4]')
    def test_ldr_imm_off_neg(self):
        if False:
            while True:
                i = 10
        self.cpu.stack_push(42)
        self.cpu.stack_push(41)
        self.cpu.STACK += 4
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 41)

    @itest_custom('ldr r1, [sp, #4]!')
    def test_ldr_imm_preind_pos(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(42)
        self.cpu.stack_push(41)
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.rf.read('SP'), pre_stack + 4)

    @itest_custom('ldr r1, [sp, #-4]!')
    def test_ldr_imm_preind_neg(self):
        if False:
            while True:
                i = 10
        self.cpu.stack_push(42)
        self.cpu.stack_push(41)
        self.cpu.STACK += 4
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 41)
        self.assertEqual(self.rf.read('SP'), pre_stack - 4)

    @itest_custom('ldr r1, [sp], #5')
    def test_ldr_imm_postind_pos(self):
        if False:
            print('Hello World!')
        self.cpu.stack_push(42)
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.rf.read('SP'), pre_stack + 5)

    @itest_custom('ldr r1, [sp], #-5')
    def test_ldr_imm_postind_neg(self):
        if False:
            return 10
        self.cpu.stack_push(42)
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.rf.read('SP'), pre_stack - 5)

    @itest_custom('ldr r1, [sp, r2]')
    def test_ldr_reg_off(self):
        if False:
            return 10
        self.cpu.regfile.write('R2', 4)
        self.cpu.stack_push(42)
        self.cpu.stack_push(48)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)

    @itest_custom('ldr r1, [sp, -r2]')
    def test_ldr_reg_off_neg(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.regfile.write('R2', 4)
        self.cpu.stack_push(42)
        self.cpu.stack_push(48)
        self.cpu.STACK += 4
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 48)

    @itest_custom('ldr r1, [sp, r2, lsl #3]')
    def test_ldr_reg_off_shift(self):
        if False:
            i = 10
            return i + 15
        self.cpu.regfile.write('R2', 1)
        self.cpu.stack_push(42)
        self.cpu.stack_push(48)
        self.cpu.stack_push(40)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)

    @itest_custom('ldr r1, [sp, -r2, lsl #3]')
    def test_ldr_reg_off_neg_shift(self):
        if False:
            print('Hello World!')
        self.cpu.regfile.write('R2', 1)
        self.cpu.stack_push(42)
        self.cpu.stack_push(48)
        self.cpu.STACK += 8
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 48)

    @itest_custom('ldr r1, [sp, r2]!')
    def test_ldr_reg_preind(self):
        if False:
            return 10
        self.cpu.regfile.write('R2', 4)
        self.cpu.stack_push(42)
        self.cpu.stack_push(48)
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.rf.read('SP'), pre_stack + 4)

    @itest_custom('ldr r1, [sp, -r2, lsl #3]!')
    def test_ldr_reg_preind_shift(self):
        if False:
            print('Hello World!')
        self.cpu.regfile.write('R2', 1)
        self.cpu.stack_push(42)
        self.cpu.stack_push(48)
        self.cpu.STACK += 8
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 48)
        self.assertEqual(self.rf.read('SP'), pre_stack - 8)

    @itest_custom('ldr r1, [sp], r2')
    def test_ldr_reg_postind(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.regfile.write('R2', 4)
        self.cpu.stack_push(42)
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.rf.read('SP'), pre_stack + 4)

    @itest_custom('ldr r1, [sp], -r2, lsl #3')
    def test_ldr_reg_postind_neg_shift(self):
        if False:
            print('Hello World!')
        self.cpu.regfile.write('R2', 1)
        self.cpu.stack_push(42)
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.rf.read('SP'), pre_stack - 8)

    @itest_custom('ldrd r2, [sp]')
    def test_ldrd(self):
        if False:
            print('Hello World!')
        r2 = 65
        r3 = 66
        self.cpu.stack_push(r3)
        self.cpu.stack_push(r2)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), r2)
        self.assertEqual(self.rf.read('R3'), r3)

    @itest_custom('pop {r1}')
    def test_pop_one_reg(self):
        if False:
            return 10
        self.cpu.stack_push(85)
        pre_stack = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 85)
        self.assertEqual(self.rf.read('SP'), pre_stack + 4)

    @itest_custom('pop {r1, r2, r3}')
    def test_pop_multops(self):
        if False:
            while True:
                i = 10
        vals = [1, 85, 170]
        for v in vals:
            self.cpu.stack_push(v)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 170)
        self.assertEqual(self.rf.read('R2'), 85)
        self.assertEqual(self.rf.read('R3'), 1)

    @itest_custom('push {r1}')
    @itest_setregs('R1=3')
    def test_push_one_reg(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(b''.join(self.cpu.stack_peek()), struct.pack('<I', 3))

    @itest_custom('push {r1, r2, r3}')
    @itest_setregs('R1=3', 'R2=0x55', 'R3=0xffffffff')
    def test_push_multi_reg(self):
        if False:
            for i in range(10):
                print('nop')
        pre_sp = self.cpu.STACK
        self.cpu.execute()
        sp = self.cpu.STACK
        self.assertEqual(self.rf.read('SP'), pre_sp - 3 * 4)
        self.assertEqual(b''.join(self.cpu.stack_peek()), struct.pack('<I', 3))
        self.assertEqual(self.cpu.read_int(sp + 4, self.cpu.address_bit_size), 85)
        self.assertEqual(self.cpu.read_int(sp + 8, self.cpu.address_bit_size), 4294967295)

    @itest_custom('str SP, [R1]')
    @itest_setregs('R1=0xd000')
    def test_str_basic(self):
        if False:
            while True:
                i = 10
        r1 = self.rf.read('R1')
        sp = self.rf.read('SP')
        self.cpu.execute()
        dr1 = self.cpu.read_int(r1, self.cpu.address_bit_size)
        self.assertEqual(sp, dr1)

    @itest_custom('str R1, [R2, R3]')
    @itest_setregs('R1=34', 'R2=0xD000', 'R3=8')
    def test_str_index(self):
        if False:
            i = 10
            return i + 15
        r1 = self.rf.read('R1')
        r2 = self.rf.read('R2')
        r3 = self.rf.read('R3')
        self.cpu.execute()
        retrieved = self.cpu.read_int(r2 + r3, self.cpu.address_bit_size)
        self.assertEqual(retrieved, r1)

    @itest_custom('str R1, [R2, R3, LSL #3]')
    @itest_setregs('R1=34', 'R2=0xD000', 'R3=1')
    def test_str_index_w_shift(self):
        if False:
            print('Hello World!')
        r1 = self.rf.read('R1')
        r2 = self.rf.read('R2')
        r3 = self.rf.read('R3')
        r3 = r3 << 3
        self.cpu.execute()
        retrieved = self.cpu.read_int(r2 + r3, self.cpu.address_bit_size)
        self.assertEqual(retrieved, r1)

    @itest_custom('str R1, [R2], #3')
    @itest_setregs('R1=34', 'R2=0xD000')
    def test_str_postindex(self):
        if False:
            for i in range(10):
                print('nop')
        r1 = self.rf.read('R1')
        r2 = self.rf.read('R2')
        self.cpu.execute()
        data = self.cpu.read_int(r2, self.cpu.address_bit_size)
        self.assertEqual(data, r1)
        new_r2 = self.rf.read('R2')
        self.assertEqual(new_r2, r2 + 3)

    @itest_custom('str R1, [R2, #3]!')
    @itest_setregs('R1=34', 'R2=0xD000')
    def test_str_index_writeback(self):
        if False:
            i = 10
            return i + 15
        r1 = self.rf.read('R1')
        r2 = self.rf.read('R2')
        self.cpu.execute()
        data = self.cpu.read_int(r2 + 3, self.cpu.address_bit_size)
        self.assertEqual(data, r1)
        new_r2 = self.rf.read('R2')
        self.assertEqual(new_r2, r2 + 3)

    @itest_custom('strd R2, [R1]')
    @itest_setregs('R1=0xD000', 'R2=34', 'R3=35')
    def test_strd(self):
        if False:
            while True:
                i = 10
        r1 = self.rf.read('R1')
        r2 = self.rf.read('R2')
        r3 = self.rf.read('R3')
        self.cpu.execute()
        dr2 = self.cpu.read_int(r1, self.cpu.address_bit_size)
        dr3 = self.cpu.read_int(r1 + 4, self.cpu.address_bit_size)
        self.assertEqual(dr2, r2)
        self.assertEqual(dr3, r3)

    @itest_custom('str R2, [R1]')
    @itest_setregs('R1=0xD000', 'R2=34')
    def test_str(self):
        if False:
            return 10
        r1 = self.rf.read('R1')
        r2 = self.rf.read('R2')
        self.cpu.execute()
        dr2 = self.cpu.read_int(r1, self.cpu.address_bit_size)
        self.assertEqual(dr2, r2)

    @itest_custom('adr r0, #16')
    def test_adr(self):
        if False:
            print('Hello World!')
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), pre_pc + 8 + 16)

    @itest_custom('add r0, PC, #0x10')
    def test_adr_mod_imm_1(self):
        if False:
            i = 10
            return i + 15
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), pre_pc + 8 + 16)

    @itest_custom('add r0, PC, #1, 28')
    def test_adr_mod_imm_2(self):
        if False:
            for i in range(10):
                print('nop')
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), pre_pc + 8 + 16)

    @itest_custom('adr r0, #16', mode=CS_MODE_THUMB)
    def test_adr_thumb(self):
        if False:
            i = 10
            return i + 15
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), pre_pc + 4 + 16)

    @itest_setregs('R1=0x1234')
    @itest_thumb('addw r0, r1, #0x2a')
    def test_addw(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('R0'), 4660 + 42)

    @itest_custom('addw r0, pc, #0x2a', mode=CS_MODE_THUMB)
    def test_addw_pc_relative(self):
        if False:
            print('Hello World!')
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), pre_pc + 4 + 42)

    @itest_setregs('R1=0x1234')
    @itest_thumb('subw r0, r1, #0x2a')
    def test_subw(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R0'), 4660 - 42)

    @itest_custom('subw r0, pc, #0x2a', mode=CS_MODE_THUMB)
    def test_subw_pc_relative(self):
        if False:
            for i in range(10):
                print('nop')
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R0'), pre_pc + 4 - 42)

    @itest_custom('bl 0x170')
    def test_bl(self):
        if False:
            while True:
                i = 10
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 368)
        self.assertEqual(self.rf.read('LR'), pre_pc + 4)

    @itest_custom('bl #-4')
    def test_bl_neg(self):
        if False:
            i = 10
            return i + 15
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc - 4)
        self.assertEqual(self.rf.read('LR'), pre_pc + 4)

    @itest_setregs('R0=0')
    @itest_custom('cbz r0, #0x2a', mode=CS_MODE_THUMB)
    def test_cbz_taken(self):
        if False:
            i = 10
            return i + 15
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 42)

    @itest_setregs('R0=1')
    @itest_custom('cbz r0, #0x2a', mode=CS_MODE_THUMB)
    def test_cbz_not_taken(self):
        if False:
            while True:
                i = 10
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 2)

    @itest_setregs('R0=1')
    @itest_custom('cbnz r0, #0x2a', mode=CS_MODE_THUMB)
    def test_cbnz_taken(self):
        if False:
            i = 10
            return i + 15
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 42)

    @itest_setregs('R0=0')
    @itest_custom('cbnz r0, #0x2a', mode=CS_MODE_THUMB)
    def test_cbnz_not_taken(self):
        if False:
            return 10
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 2)

    @itest_setregs('R0=0xd000', 'R1=1')
    @itest_custom('tbb [r0, r1]', mode=CS_MODE_THUMB)
    def test_tbb(self):
        if False:
            for i in range(10):
                print('nop')
        for (i, offset) in enumerate([11, 21, 31]):
            self.mem.write(53248 + i, struct.pack('<B', offset))
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 4 + 42)

    @itest_setregs('R1=1')
    @itest_custom('tbb [pc, r1]', mode=CS_MODE_THUMB)
    def test_tbb_pc_relative(self):
        if False:
            while True:
                i = 10
        for (i, offset) in enumerate([11, 21, 31]):
            self.mem.write(self.cpu.PC + 4 + i, struct.pack('<B', offset))
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 4 + 42)

    @itest_setregs('R0=0xd000', 'R1=1')
    @itest_custom('tbh [r0, r1, lsl #1]', mode=CS_MODE_THUMB)
    def test_tbh(self):
        if False:
            return 10
        for (i, offset) in enumerate([11, 21, 31]):
            self.mem.write(53248 + i * 2, struct.pack('<H', offset))
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 4 + 42)

    @itest_setregs('R1=1')
    @itest_custom('tbh [pc, r1, lsl #1]', mode=CS_MODE_THUMB)
    def test_tbh_pc_relative(self):
        if False:
            print('Hello World!')
        for (i, offset) in enumerate([11, 21, 31]):
            self.mem.write(self.cpu.PC + 4 + i * 2, struct.pack('<H', offset))
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 4 + 42)

    @itest_setregs('R0=-0x18000')
    @itest('cmn r0, #0x18000')
    def test_cmn_eq_mod_imm_1(self):
        if False:
            i = 10
            return i + 15
        self._checkFlagsNZCV(0, 1, 1, 0)

    @itest_setregs('R0=-0x18000')
    @itest('cmn r0, #24, 20')
    def test_cmn_eq_mod_imm_2(self):
        if False:
            for i in range(10):
                print('nop')
        self._checkFlagsNZCV(0, 1, 1, 0)

    @itest_setregs('R0=3')
    @itest('cmp r0, 3')
    def test_cmp_eq(self):
        if False:
            for i in range(10):
                print('nop')
        self._checkFlagsNZCV(0, 1, 1, 0)

    @itest_setregs('R0=0x18000')
    @itest('cmp r0, #0x18000')
    def test_cmp_eq_mod_imm_1(self):
        if False:
            for i in range(10):
                print('nop')
        self._checkFlagsNZCV(0, 1, 1, 0)

    @itest_setregs('R0=0x18000')
    @itest('cmp r0, #24, 20')
    def test_cmp_eq_mod_imm_2(self):
        if False:
            while True:
                i = 10
        self._checkFlagsNZCV(0, 1, 1, 0)

    @itest_setregs('R0=3')
    @itest('cmp r0, 5')
    def test_cmp_lt(self):
        if False:
            while True:
                i = 10
        self._checkFlagsNZCV(1, 0, 0, 0)

    @itest_setregs('R0=3')
    @itest('cmp r0, 2')
    def test_cmp_gt(self):
        if False:
            return 10
        self._checkFlagsNZCV(0, 0, 1, 0)

    @itest_setregs('R0=0')
    @itest('cmp r0, 0')
    def test_cmp_carry(self):
        if False:
            for i in range(10):
                print('nop')
        self._checkFlagsNZCV(0, 1, 1, 0)

    @itest_setregs('R0=0x40000000')
    @itest('cmp r0, 0xa0000000')
    def test_cmp_ovf(self):
        if False:
            return 10
        self._checkFlagsNZCV(1, 0, 0, 1)

    @itest_setregs('R0=0x80000000')
    @itest('cmp r0, 0x40000000')
    def test_cmp_carry_ovf(self):
        if False:
            return 10
        self._checkFlagsNZCV(0, 0, 1, 1)

    @itest_custom('clz r1, r2')
    @itest_setregs('R2=0xFFFF')
    def test_clz_sixteen_zeroes(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 16)

    @itest_custom('clz r1, r2')
    @itest_setregs('R2=0')
    def test_clz_all_zero(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), self.cpu.address_bit_size)

    @itest_custom('clz r1, r2')
    @itest_setregs('R2=0xffffffff')
    def test_clz_no_leading_zeroes(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 0)

    @itest_custom('clz r1, r2')
    @itest_setregs('R2=0x7fffffff')
    def test_clz_one_leading_zero(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 1)

    @itest_custom('clz r1, r2')
    @itest_setregs('R2=0x7f7fffff')
    def test_clz_lead_zero_then_more_zeroes(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 1)

    @itest_custom('sub r3, r1, r2')
    @itest_setregs('R1=4', 'R2=2')
    def test_sub_basic(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 2)

    @itest_setregs('R3=0xE')
    @itest_thumb('sub r3, #12')
    def test_thumb_sub_basic(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R3'), 2)

    @itest_custom('sub r3, r1, #5')
    @itest_setregs('R1=10')
    def test_sub_imm(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 5)

    @itest_custom('sub r3, r1, #0x18000')
    @itest_setregs('R1=0x18000')
    def test_sub_mod_imm_1(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 0)

    @itest_custom('sub r3, r1, #24, 20')
    @itest_setregs('R1=0x18000')
    def test_sub_mod_imm_2(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 0)

    @itest_custom('uqsub8 r3, r1, r2')
    @itest_setregs('R1=0x04030201', 'R2=0x01010101')
    def test_uqsub8_concrete(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 50462976)

    @itest_custom('uqsub8 r3, r1, r2')
    @itest_setregs('R1=0x05040302', 'R2=0x07050101')
    def test_uqsub8_concrete_saturated(self):
        if False:
            i = 10
            return i + 15
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 513)

    @itest_custom('uqsub8 r3, r1, r2')
    @itest_setregs('R2=0x01010101')
    def test_uqsub8_sym(self):
        if False:
            while True:
                i = 10
        op1 = self.cpu.memory.constraints.new_bitvec(32, 'op1')
        self.cpu.memory.constraints.add(op1 >= 67305985)
        self.cpu.memory.constraints.add(op1 < 67305988)
        self.cpu.R1 = op1
        self.cpu.execute()
        all_vals = solver.get_all_values(self.cpu.memory.constraints, self.cpu.R3)
        self.assertIn(50462976, all_vals)

    @itest_custom('rsc r3, r1, #0x18000')
    @itest_setregs('R1=0x18000', 'APSR_C=1')
    def test_rsc_mod_imm_1(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 0)

    @itest_custom('rsc r3, r1, #0x18000')
    @itest_setregs('R1=0x17fff', 'APSR_C=0')
    def test_rsc_mod_imm_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 0)

    @itest_custom('rsc r3, r1, #24, 20')
    @itest_setregs('R1=0x18000', 'APSR_C=1')
    def test_rsc_mod_imm_3(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 0)

    @itest_custom('rsc r3, r1, #24, 20')
    @itest_setregs('R1=0x17fff', 'APSR_C=0')
    def test_rsc_mod_imm_4(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 0)

    @itest_custom('sbc r3, r1, #5')
    @itest_setregs('R1=10')
    def test_sbc_imm(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 4)

    @itest_setregs('R0=0', 'R3=0xffffffff')
    @itest_thumb('sbcs r0, r3')
    def test_sbc_thumb(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('R0'), 0)

    @itest_custom('sbc r3, r1, #0x18000')
    @itest_setregs('R1=0x18010', 'APSR_C=1')
    def test_sbc_mod_imm_1(self):
        if False:
            while True:
                i = 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 16)

    @itest_custom('sbc r3, r1, #0x18000')
    @itest_setregs('R1=0x18010', 'APSR_C=0')
    def test_sbc_mod_imm_2(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 15)

    @itest_custom('sbc r3, r1, #24, 20')
    @itest_setregs('R1=0x18010', 'APSR_C=1')
    def test_sbc_mod_imm_3(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 16)

    @itest_custom('sbc r3, r1, #24, 20')
    @itest_setregs('R1=0x18010', 'APSR_C=0')
    def test_sbc_mod_imm_4(self):
        if False:
            i = 10
            return i + 15
        self.cpu.execute()
        self.assertEqual(self.rf.read('R3'), 15)

    @itest_custom('ldm sp, {r1, r2, r3}')
    def test_ldm(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.stack_push(1094795585)
        self.cpu.stack_push(2)
        self.cpu.stack_push(42)
        pre_sp = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.rf.read('R2'), 2)
        self.assertEqual(self.rf.read('R3'), 1094795585)
        self.assertEqual(self.cpu.STACK, pre_sp)

    @itest_custom('ldm sp!, {r1, r2, r3}')
    def test_ldm_wb(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(1094795585)
        self.cpu.stack_push(2)
        self.cpu.stack_push(42)
        pre_sp = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 42)
        self.assertEqual(self.rf.read('R2'), 2)
        self.assertEqual(self.rf.read('R3'), 1094795585)
        self.assertEqual(self.cpu.STACK, pre_sp + 12)

    @itest_setregs('R0=0xd100')
    @itest_custom('ldmia r0!, {r1, r2, r3}')
    def test_ldmia(self):
        if False:
            print('Hello World!')
        self.cpu.write_int(53504 + 0, 1, self.cpu.address_bit_size)
        self.cpu.write_int(53504 + 4, 2, self.cpu.address_bit_size)
        self.cpu.write_int(53504 + 8, 3, self.cpu.address_bit_size)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 1)
        self.assertEqual(self.rf.read('R2'), 2)
        self.assertEqual(self.rf.read('R3'), 3)
        self.assertEqual(self.rf.read('R0'), 53504 + 12)

    @itest_setregs('R0=0xd100')
    @itest_custom('ldmib r0!, {r1, r2, r3}')
    def test_ldmib(self):
        if False:
            i = 10
            return i + 15
        self.cpu.write_int(53504 + 4, 1, self.cpu.address_bit_size)
        self.cpu.write_int(53504 + 8, 2, self.cpu.address_bit_size)
        self.cpu.write_int(53504 + 12, 3, self.cpu.address_bit_size)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 1)
        self.assertEqual(self.rf.read('R2'), 2)
        self.assertEqual(self.rf.read('R3'), 3)
        self.assertEqual(self.rf.read('R0'), 53504 + 12)

    @itest_setregs('R0=0xd100')
    @itest_custom('ldmda r0!, {r1, r2, r3}')
    def test_ldmda(self):
        if False:
            while True:
                i = 10
        self.cpu.write_int(53504 - 0, 1, self.cpu.address_bit_size)
        self.cpu.write_int(53504 - 4, 2, self.cpu.address_bit_size)
        self.cpu.write_int(53504 - 8, 3, self.cpu.address_bit_size)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 1)
        self.assertEqual(self.rf.read('R2'), 2)
        self.assertEqual(self.rf.read('R3'), 3)
        self.assertEqual(self.rf.read('R0'), 53504 - 12)

    @itest_setregs('R0=0xd100')
    @itest_custom('ldmdb r0!, {r1, r2, r3}')
    def test_ldmdb(self):
        if False:
            i = 10
            return i + 15
        self.cpu.write_int(53504 - 4, 1, self.cpu.address_bit_size)
        self.cpu.write_int(53504 - 8, 2, self.cpu.address_bit_size)
        self.cpu.write_int(53504 - 12, 3, self.cpu.address_bit_size)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 1)
        self.assertEqual(self.rf.read('R2'), 2)
        self.assertEqual(self.rf.read('R3'), 3)
        self.assertEqual(self.rf.read('R0'), 53504 - 12)

    @itest_setregs('R1=42', 'R2=2', 'R3=0x42424242')
    @itest_custom('stm sp, {r1, r2, r3}')
    def test_stm(self):
        if False:
            print('Hello World!')
        self.cpu.STACK -= 12
        pre_sp = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.cpu.read_int(pre_sp, self.cpu.address_bit_size), 42)
        self.assertEqual(self.cpu.read_int(pre_sp + 4, self.cpu.address_bit_size), 2)
        self.assertEqual(self.cpu.read_int(pre_sp + 8, self.cpu.address_bit_size), 1111638594)
        self.assertEqual(self.cpu.STACK, pre_sp)

    @itest_setregs('R1=42', 'R2=2', 'R3=0x42424242')
    @itest_custom('stm sp!, {r1, r2, r3}')
    def test_stm_wb(self):
        if False:
            print('Hello World!')
        self.cpu.STACK -= 12
        pre_sp = self.cpu.STACK
        self.cpu.execute()
        self.assertEqual(self.cpu.read_int(pre_sp, self.cpu.address_bit_size), 42)
        self.assertEqual(self.cpu.read_int(pre_sp + 4, self.cpu.address_bit_size), 2)
        self.assertEqual(self.cpu.read_int(pre_sp + 8, self.cpu.address_bit_size), 1111638594)
        self.assertEqual(self.cpu.STACK, pre_sp + 12)

    @itest_setregs('R0=0xd100', 'R1=1', 'R2=2', 'R3=3')
    @itest_custom('stmia r0!, {r1, r2, r3}')
    def test_stmia(self):
        if False:
            i = 10
            return i + 15
        self.cpu.execute()
        self.assertEqual(self.cpu.read_int(53504 + 0, self.cpu.address_bit_size), 1)
        self.assertEqual(self.cpu.read_int(53504 + 4, self.cpu.address_bit_size), 2)
        self.assertEqual(self.cpu.read_int(53504 + 8, self.cpu.address_bit_size), 3)
        self.assertEqual(self.rf.read('R0'), 53504 + 12)

    @itest_setregs('R0=0xd100', 'R1=1', 'R2=2', 'R3=3')
    @itest_custom('stmib r0!, {r1, r2, r3}')
    def test_stmib(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.cpu.read_int(53504 + 4, self.cpu.address_bit_size), 1)
        self.assertEqual(self.cpu.read_int(53504 + 8, self.cpu.address_bit_size), 2)
        self.assertEqual(self.cpu.read_int(53504 + 12, self.cpu.address_bit_size), 3)
        self.assertEqual(self.rf.read('R0'), 53504 + 12)

    @itest_setregs('R0=0xd100', 'R1=1', 'R2=2', 'R3=3')
    @itest_custom('stmda r0!, {r1, r2, r3}')
    def test_stmda(self):
        if False:
            i = 10
            return i + 15
        self.cpu.execute()
        self.assertEqual(self.cpu.read_int(53504 - 0, self.cpu.address_bit_size), 1)
        self.assertEqual(self.cpu.read_int(53504 - 4, self.cpu.address_bit_size), 2)
        self.assertEqual(self.cpu.read_int(53504 - 8, self.cpu.address_bit_size), 3)
        self.assertEqual(self.rf.read('R0'), 53504 - 12)

    @itest_setregs('R0=0xd100', 'R1=1', 'R2=2', 'R3=3')
    @itest_custom('stmdb r0!, {r1, r2, r3}')
    def test_stmdb(self):
        if False:
            while True:
                i = 10
        self.cpu.execute()
        self.assertEqual(self.cpu.read_int(53504 - 4, self.cpu.address_bit_size), 1)
        self.assertEqual(self.cpu.read_int(53504 - 8, self.cpu.address_bit_size), 2)
        self.assertEqual(self.cpu.read_int(53504 - 12, self.cpu.address_bit_size), 3)
        self.assertEqual(self.rf.read('R0'), 53504 - 12)

    @itest_custom('bx r1')
    @itest_setregs('R1=0x1008')
    def test_bx_basic(self):
        if False:
            i = 10
            return i + 15
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), 4104)
        self.assertEqual(self.cpu.mode, CS_MODE_ARM)

    @itest_custom('bx r1')
    @itest_setregs('R1=0x1009')
    def test_bx_thumb(self):
        if False:
            while True:
                i = 10
        pre_pc = self.rf.read('PC')
        self.cpu.execute()
        self.assertEqual(self.rf.read('PC'), pre_pc + 4)
        self.assertEqual(self.cpu.mode, CS_MODE_THUMB)

    @itest_custom('orr r2, r3, #5')
    @itest_setregs('R3=0x1000')
    def test_orr_imm(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 4101)

    @itest_setregs('R3=0x1000')
    @itest_thumb('orr r3, #5')
    def test_thumb_orr_imm(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R3'), 4101)

    @itest_custom('orr r2, r3, #0x18000')
    @itest_setregs('R3=0x1000')
    def test_orr_mod_imm_1(self):
        if False:
            i = 10
            return i + 15
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 102400)

    @itest_custom('orr r2, r3, #24, 20')
    @itest_setregs('R3=0x1000')
    def test_orr_mod_imm_2(self):
        if False:
            while True:
                i = 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 102400)

    @itest_custom('orrs r2, r3')
    @itest_setregs('R2=0x5', 'R3=0x80000000')
    def test_orrs_imm_flags(self):
        if False:
            i = 10
            return i + 15
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 2147483653)
        self.assertEqual(self.rf.read('APSR_N'), True)

    @itest_custom('orr r2, r3')
    @itest_setregs('R2=0x5', 'R3=0x80000000')
    def test_orr_reg_w_flags(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 2147483653)

    @itest_custom('orr r2, r3, r4')
    @itest_setregs('R3=0x5', 'R4=0x80000000')
    def test_orr_reg_two_op(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 2147483653)

    @itest_custom('orr r2, r3, r4, LSL #4')
    @itest_setregs('R3=0x5', 'R4=0xF')
    def test_orr_reg_two_op_shifted(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 245)

    @itest_setregs('R2=0x0', 'R5=0xFFFFFFFA')
    @itest_thumb('orn r2, r2, r5')
    def test_orn(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('R2'), 5)

    @itest_custom('eor r2, r3, #5')
    @itest_setregs('R3=0xA')
    def test_eor_imm(self):
        if False:
            return 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 15)

    @itest_setregs('R3=0xA')
    @itest_thumb('eor r3, #5')
    def test_thumb_eor_imm(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R3'), 15)

    @itest_custom('eors r2, r3')
    @itest_setregs('R2=0xAA', 'R3=0x80000000')
    def test_eors_imm_flags(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 2147483818)
        self.assertEqual(self.rf.read('APSR_N'), True)

    @itest_custom('eors r2, r3')
    @itest_setregs('R2=0x5', 'R3=0x80000005')
    def test_eor_reg_w_flags(self):
        if False:
            print('Hello World!')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 2147483648)
        self.assertEqual(self.rf.read('APSR_N'), 1)

    @itest_custom('eor r2, r3, r4')
    @itest_setregs('R3=0x80000005', 'R4=0x80000005')
    def test_eor_reg_two_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 0)

    @itest_custom('eor r2, r3, r4, LSL #4')
    @itest_setregs('R3=0x55', 'R4=0x5')
    def test_eor_reg_two_op_shifted(self):
        if False:
            i = 10
            return i + 15
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 5)

    @itest_custom('eor r2, r3, #0x18000')
    @itest_setregs('R3=0xA')
    def test_eor_mod_imm_1(self):
        if False:
            while True:
                i = 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 98314)

    @itest_custom('eor r2, r3, #24, 20')
    @itest_setregs('R3=0xA')
    def test_eor_mod_imm_2(self):
        if False:
            while True:
                i = 10
        self.cpu.execute()
        self.assertEqual(self.rf.read('R2'), 98314)

    @itest_custom('ldrh r1, [sp]')
    def test_ldrh_imm_off_none(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(1094778945)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 65)

    @itest_custom('ldrh r1, [sp, r2]')
    @itest_setregs('R2=4')
    def test_ldrh_reg_off(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.stack_push(1094778945)
        self.cpu.stack_push(48)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 65)

    @itest_custom('ldrsh r1, [sp]')
    def test_ldrsh_imm_off_none_neg(self):
        if False:
            print('Hello World!')
        self.cpu.stack_push(196367)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 4294967055)

    @itest_custom('ldrsh r1, [sp]')
    def test_ldrsh_imm_off_none_pos(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.stack_push(16715775)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 4095)

    @itest_custom('ldrsh r1, [sp, r2]')
    @itest_setregs('R2=4')
    def test_ldrsh_reg_off_neg(self):
        if False:
            print('Hello World!')
        self.cpu.stack_push(196367)
        self.cpu.stack_push(48)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 4294967055)

    @itest_custom('ldrsh r1, [sp, r2]')
    @itest_setregs('R2=4')
    def test_ldrsh_reg_off_pos(self):
        if False:
            print('Hello World!')
        self.cpu.stack_push(16715775)
        self.cpu.stack_push(48)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 4095)

    @itest_custom('ldrb r1, [sp]')
    def test_ldrb_imm_off_none(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(65)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 65)

    @itest_custom('ldrb r1, [sp, r2]')
    @itest_setregs('R2=4')
    def test_ldrb_reg_off(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(65)
        self.cpu.stack_push(48)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 65)

    @itest_custom('ldrsb r1, [sp]')
    def test_ldrsb_imm_off_none_neg(self):
        if False:
            while True:
                i = 10
        self.cpu.stack_push(767)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), Mask(32))

    @itest_custom('ldrsb r1, [sp]')
    def test_ldrsb_imm_off_none_pos(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.stack_push(65295)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 15)

    @itest_custom('ldrsb r1, [sp, r2]')
    @itest_setregs('R2=4')
    def test_ldrsb_reg_off_neg(self):
        if False:
            while True:
                i = 10
        self.cpu.stack_push(767)
        self.cpu.stack_push(48)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), Mask(32))

    @itest_custom('ldrsb r1, [sp, r2]')
    @itest_setregs('R2=4')
    def test_ldrsb_reg_off_pos(self):
        if False:
            for i in range(10):
                print('nop')
        self.cpu.stack_push(65295)
        self.cpu.stack_push(48)
        self.cpu.execute()
        self.assertEqual(self.rf.read('R1'), 15)

    @itest_setregs('R1=1', 'R3=0')
    @itest('tst r3, r1')
    def test_tst_1(self):
        if False:
            i = 10
            return i + 15
        self._checkFlagsNZCV(0, 1, 0, 0)

    @itest_setregs('R1=1', 'R3=1')
    @itest('tst r3, r1')
    def test_tst_2(self):
        if False:
            i = 10
            return i + 15
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_setregs('R1=1', 'R3=3')
    @itest('tst r3, r1')
    def test_tst_3(self):
        if False:
            while True:
                i = 10
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_setregs('R3=0')
    @itest('tst r3, #0x18000')
    def test_tst_mod_imm_1(self):
        if False:
            i = 10
            return i + 15
        self._checkFlagsNZCV(0, 1, 0, 0)

    @itest_setregs('R3=0')
    @itest('tst r3, #24, 20')
    def test_tst_mod_imm_2(self):
        if False:
            print('Hello World!')
        self._checkFlagsNZCV(0, 1, 0, 0)

    @itest_setregs('R1=1', 'R3=0')
    @itest('teq r3, r1')
    def test_teq_1(self):
        if False:
            return 10
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_setregs('R1=1', 'R3=1')
    @itest('teq r3, r1')
    def test_teq_2(self):
        if False:
            return 10
        self._checkFlagsNZCV(0, 1, 0, 0)

    @itest_setregs('R3=0')
    @itest('teq r3, #0x18000')
    def test_teq_mod_imm_1(self):
        if False:
            for i in range(10):
                print('nop')
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_setregs('R3=0x18000')
    @itest('teq r3, #0x18000')
    def test_teq_mod_imm_2(self):
        if False:
            print('Hello World!')
        self._checkFlagsNZCV(0, 1, 0, 0)

    @itest_setregs('R3=0')
    @itest('teq r3, #24, 20')
    def test_teq_mod_imm_3(self):
        if False:
            print('Hello World!')
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_setregs('R3=0x18000')
    @itest('teq r3, #24, 20')
    def test_teq_mod_imm_4(self):
        if False:
            for i in range(10):
                print('nop')
        self._checkFlagsNZCV(0, 1, 0, 0)

    @itest_setregs('R2=5')
    @itest('and r2, r2, #1')
    def test_and_imm(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R2'), 1)

    @itest_setregs('R1=5', 'R2=3')
    @itest('and r1, r1, r2')
    def test_and_reg(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R1'), 3 & 5)

    @itest_setregs('R1=5', 'R2=3', 'APSR_C=1')
    @itest('and r1, r1, r2')
    def test_and_reg_carry(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R1'), 3 & 5)
        self.assertEqual(self.rf.read('APSR_C'), 1)

    @itest_setregs('R2=5')
    @itest('and r2, r2, #0x18000')
    def test_and_mod_imm_1(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R2'), 0)

    @itest_setregs('R2=5')
    @itest('and r2, r2, #24, 20')
    def test_and_mod_imm_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R2'), 0)

    def test_svc(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(Interruption):
            self._setupCpu('svc #0')
            self.cpu.execute()

    @itest_setregs('R3=0x11')
    @itest('lsls r4, r3, 1')
    def test_lsl_imm_min(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R4'), 17 << 1)
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_setregs('R3=0x11')
    @itest('lsls r4, r3, 31')
    def test_lsl_imm_max(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R4'), 1 << 31)
        self._checkFlagsNZCV(1, 0, 0, 0)

    @itest_setregs('R3=0x11', 'R2=0xff01')
    @itest('lsls r4, r3, r2')
    def test_lsl_reg_min(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R4'), 17 << 1)
        self._checkFlagsNZCV(0, 0, 0, 0)

    @itest_setregs('R3=0x11', 'R2=0xff1f')
    @itest('lsls r4, r3, r2')
    def test_lsl_reg_max(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R4'), 1 << 31)
        self._checkFlagsNZCV(1, 0, 0, 0)

    @itest_setregs('R2=0xffffffff')
    @itest('lsls r2, r2, #0x1f')
    def test_lsl_imm_carry(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.cpu.R2, 1 << 31)
        self._checkFlagsNZCV(1, 0, 1, 0)

    @itest_setregs('R5=1', 'R6=2')
    @itest_thumb('lsl.w r5, r6, #3')
    def test_lslw_thumb(self):
        if False:
            print('Hello World!')
        'thumb mode specific behavior'
        self.assertEqual(self.cpu.R5, 2 << 3)

    @itest_setregs('R0=0x1000', 'R2=3')
    @itest('lsr r0, r0, r2')
    def test_lsr_reg(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('R0'), 4096 >> 3)

    @itest_setregs('R0=0x1000')
    @itest('lsr r0, r0, #3')
    def test_lsr_reg_imm(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R0'), 4096 >> 3)

    @itest_setregs('R1=0', 'R2=3')
    @itest_thumb('lsrs r1, r2')
    def test_thumb_lsrs(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.cpu.R1, 0)

    @itest_setregs('R5=0', 'R6=16')
    @itest_thumb('lsr.w R5, R6, #3')
    def test_lsrw_thumb(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.cpu.R5, 16 >> 3)

    @itest_setregs('R0=11', 'R2=2')
    @itest_thumb('lsr.w R0, R0, R2')
    def test_lsrw_thumb_reg(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.cpu.R0, 11 >> 2)

    @itest_setregs('R5=0', 'R6=16')
    @itest_thumb('asr.w R5, R6, #3')
    def test_asrw_thumb(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.cpu.R5, 16 >> 3)

    @itest_setregs('R2=29')
    @itest('RSB r2, r2, #31')
    def test_rsb_imm(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R2'), 2)

    @itest_setregs('R2=0x17000')
    @itest('RSB r2, r2, #0x18000')
    def test_rsb_mod_imm_1(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('R2'), 4096)

    @itest_setregs('R2=0x17000')
    @itest('RSB r2, r2, #24, 20')
    def test_rsb_mod_imm_2(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('R2'), 4096)

    @itest_setregs('R6=2', 'R8=0xfffffffe')
    @itest('RSBS r8, r6, #0')
    def test_rsbs_carry(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R8'), 4294967294)
        self._checkFlagsNZCV(1, 0, 0, 0)

    def test_flag_state_continuity(self):
        if False:
            for i in range(10):
                print('nop')
        'If an instruction only partially updates flags, cpu.set_flags should\n        ensure unupdated flags are preserved.\n\n        For example:\n        r1 = 2**31 - 1\n        add r2, r1, 0x1 // overflow = 1\n        mov r1, 1\n        mov r3, 0\n        tst r3, r1 // does not change overflow flag\n        // ovf should still be 1\n        '
        self.rf.write('R1', 2 ** 31 - 1)
        self._setupCpu('adds r2, r1, #0x1')
        self.cpu.execute()
        self.rf.write('R1', 1)
        self.rf.write('R3', 0)
        self.mem.write(self.cpu.PC, assemble('tst r3, r1'))
        self.cpu.execute()
        self._checkFlagsNZCV(0, 1, 0, 1)

    @itest_setregs('R1=30', 'R2=10')
    @itest('MUL R1, R2')
    def test_mul_reg(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('R1'), 300)

    @itest_setregs('R1=30', 'R2=10')
    @itest('MUL R3, R1, R2')
    def test_mul_reg_w_dest(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R3'), 300)

    @itest_setregs('R2=10', 'R3=15', 'R4=7')
    @itest('MLA R1, R2, R3, R4')
    def test_mla_reg(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R1'), 157)

    @itest_setregs('R1=0xFF')
    @itest('BIC R2, R1, #0x10')
    def test_bic_reg_imm(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R2'), 239)

    @itest_setregs('R1=0xFF')
    @itest('BIC R1, #0x10')
    def test_thumb_bic_reg_imm(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('R1'), 239)

    @itest_setregs('R1=0x18002')
    @itest('BIC R2, R1, #0x18000')
    def test_bic_reg_mod_imm_1(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R2'), 2)

    @itest_setregs('R1=0x18002')
    @itest('BIC R2, R1, #24, 20')
    def test_bic_reg_mod_imm_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R2'), 2)

    @itest_setregs('R1=0x1008')
    @itest('BLX R1')
    def test_blx_reg(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('PC'), 4104)
        self.assertEqual(self.rf.read('LR'), 4104)
        self.assertEqual(self.cpu.mode, CS_MODE_ARM)

    @itest_setregs('R1=0x1009')
    @itest('BLX R1')
    def test_blx_reg_thumb(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('PC'), 4104)
        self.assertEqual(self.rf.read('LR'), 4104)
        self.assertEqual(self.cpu.mode, CS_MODE_THUMB)

    @itest_setregs('R1=0xffffffff', 'R2=2')
    @itest('UMULLS R1, R2, R1, R2')
    def test_umull(self):
        if False:
            return 10
        mul = 4294967295 * 2
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.assertEqual(self.rf.read('R1'), mul & Mask(32))
        self.assertEqual(self.rf.read('R2'), mul >> 32)
        self._checkFlagsNZCV(0, 0, pre_c, pre_v)

    @itest_setregs('R1=2', 'R2=2')
    @itest('UMULLS R1, R2, R1, R2')
    def test_umull_still32(self):
        if False:
            return 10
        mul = 2 * 2
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.assertEqual(self.rf.read('R1'), mul & Mask(32))
        self.assertEqual(self.rf.read('R2'), mul >> 32)
        self._checkFlagsNZCV(0, 0, pre_c, pre_v)

    @itest_setregs('R1=0xfffffffe', 'R2=0xfffffffe')
    @itest('UMULLS R1, R2, R1, R2')
    def test_umull_max(self):
        if False:
            return 10
        mul = 4294967294 ** 2
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.assertEqual(self.rf.read('R1'), mul & Mask(32))
        self.assertEqual(self.rf.read('R2'), mul >> 32)
        self._checkFlagsNZCV(1, 0, pre_c, pre_v)

    @itest_setregs('R1=3', 'R2=0')
    @itest('UMULLS R1, R2, R1, R2')
    def test_umull_z(self):
        if False:
            for i in range(10):
                print('nop')
        mul = 3 * 0
        pre_c = self.rf.read('APSR_C')
        pre_v = self.rf.read('APSR_V')
        self.assertEqual(self.rf.read('R1'), mul & Mask(32))
        self.assertEqual(self.rf.read('R2'), mul >> 32 & Mask(32))
        self._checkFlagsNZCV(0, 1, pre_c, pre_v)

    @itest('dmb ish')
    def test_dmb(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(True)

    @itest_custom('vldmia  r1, {d8, d9, d10}')
    def test_vldmia(self):
        if False:
            i = 10
            return i + 15
        self.cpu.stack_push(20, 8)
        self.cpu.stack_push(21, 8)
        self.cpu.stack_push(22, 8)
        self.cpu.R1 = self.cpu.SP
        pre = self.cpu.R1
        self.cpu.execute()
        self.assertEqual(self.cpu.D8, 22)
        self.assertEqual(self.cpu.D9, 21)
        self.assertEqual(self.cpu.D10, 20)
        self.assertEqual(self.cpu.R1, pre)

    @itest_custom('vldmia  r1!, {d8, d9, d10}')
    def test_vldmia_wb(self):
        if False:
            for i in range(10):
                print('nop')
        pre = self.cpu.SP
        self.cpu.stack_push(20, 8)
        self.cpu.stack_push(21, 8)
        self.cpu.stack_push(22, 8)
        self.cpu.R1 = self.cpu.SP
        self.cpu.execute()
        self.assertEqual(self.cpu.D8, 22)
        self.assertEqual(self.cpu.D9, 21)
        self.assertEqual(self.cpu.D10, 20)
        self.assertEqual(self.cpu.R1, pre)

    @itest_setregs('R3=3')
    @itest('movt R3, #9')
    def test_movt(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.cpu.R3, 589827)

    @itest_custom('mrc p15, #0, r2, c13, c0, #3')
    def test_mrc(self):
        if False:
            return 10
        self.cpu.set_arm_tls(349525)
        self.cpu.write_register('R2', 0)
        self.cpu.execute()
        self.assertEqual(self.cpu.R2, 349525)

    @itest_setregs('R1=0x45', 'R2=0x55555555')
    @itest('uxtb r1, r2')
    def test_uxtb(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.cpu.R2, 1431655765)
        self.assertEqual(self.cpu.R1, 85)

    @itest_setregs('R1=0x45', 'R2=0x55555555')
    @itest('uxth r1, r2')
    def test_uxth(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.cpu.R2, 1431655765)
        self.assertEqual(self.cpu.R1, 21845)

    @itest_setregs('R1=1', 'R2=0', 'R3=0', 'R4=0', 'R12=0x4141')
    @itest_thumb_multiple(['cmp r1, #1', 'itt ne', 'mov r2, r12', 'mov r3, r12', 'mov r4, r12'])
    def test_itt_ne_noexec(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rf.read('R2'), 0)
        self.assertEqual(self.rf.read('R3'), 0)
        self.assertEqual(self.rf.read('R4'), 16705)

    @itest_setregs('R1=0', 'R2=0', 'R3=0', 'R4=0', 'R12=0x4141')
    @itest_thumb_multiple(['cmp r1, #1', 'itt ne', 'mov r2, r12', 'mov r3, r12', 'mov r4, r12'])
    def test_itt_ne_exec(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R2'), 16705)
        self.assertEqual(self.rf.read('R3'), 16705)
        self.assertEqual(self.rf.read('R4'), 16705)

    @itest_setregs('R1=0', 'R2=0', 'R3=0', 'R4=0', 'R12=0x4141')
    @itest_thumb_multiple(['cmp r1, #1', 'ite ne', 'mov r2, r12', 'mov r3, r12', 'mov r4, r12'])
    def test_ite_ne_exec(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R2'), 16705)
        self.assertEqual(self.rf.read('R3'), 0)
        self.assertEqual(self.rf.read('R4'), 16705)

    @itest_setregs('R1=0', 'R2=0', 'R3=0', 'R4=0')
    @itest_thumb_multiple(['cmp r1, #1', 'itete ne', 'mov r1, #1', 'mov r2, #1', 'mov r3, #1', 'mov r4, #4'])
    def test_itete_exec(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rf.read('R1'), 1)
        self.assertEqual(self.rf.read('R2'), 0)
        self.assertEqual(self.rf.read('R3'), 1)
        self.assertEqual(self.rf.read('R4'), 0)

    @itest_setregs('APSR_GE=3', 'R4=0', 'R5=0x01020304', 'R6=0x05060708')
    @itest_thumb('sel r4, r5, r6')
    def test_sel(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('R4'), 84280068)

    @itest_setregs('R2=0', 'R1=0x01020304')
    @itest('rev r2, r1')
    def test_rev(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rf.read('R1'), 16909060)
        self.assertEqual(self.rf.read('R2'), 67305985)

    @itest_setregs('R1=0x01020304', 'R2=0x05060708', 'R3=0', 'R4=0xF001')
    @itest_multiple(['sxth r1, r2', 'sxth r3, r4', 'sxth r5, r4, ROR #8'])
    def test_sxth(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rf.read('R1'), 1800)
        self.assertEqual(self.rf.read('R3'), 4294963201)
        self.assertEqual(self.rf.read('R5'), 240)

    @itest_custom('blx  r1')
    def test_blx_reg_sym(self):
        if False:
            for i in range(10):
                print('nop')
        dest = self.cpu.memory.constraints.new_bitvec(32, 'dest')
        self.cpu.memory.constraints.add(dest >= 4096)
        self.cpu.memory.constraints.add(dest <= 4097)
        self.cpu.R1 = dest
        with self.assertRaises(Concretize) as cm:
            self.cpu.execute()
        e = cm.exception
        all_modes = solver.get_all_values(self.cpu.memory.constraints, e.expression)
        self.assertIn(CS_MODE_THUMB, all_modes)
        self.assertIn(CS_MODE_ARM, all_modes)
        self.assertEqual(self.cpu.mode, CS_MODE_ARM)
        e.setstate(self, CS_MODE_THUMB)
        self.assertEqual(self.cpu.mode, CS_MODE_THUMB)

    @itest_setregs('R1=0x00000008')
    @itest('add pc, pc, r1')
    def test_add_to_pc(self):
        if False:
            return 10
        self.assertEqual(self.rf.read('R15'), 4116)

    def test_arm_save_restore_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        import pickle
        dumped_s = pickle_dumps(self.cpu)
        self.cpu = pickle.loads(dumped_s)

    def test_symbolic_conditional(self):
        if False:
            while True:
                i = 10
        asm = ''
        asm += '  tst r0, r0\n'
        asm += '  beq label\n'
        asm += '  bne label\n'
        asm += 'label:\n'
        asm += '  nop'
        self._setupCpu(asm, mode=CS_MODE_THUMB)
        self.cpu.R0 = self.cpu.memory.constraints.new_bitvec(32, 'val')
        self.cpu.execute()
        self.cpu.execute()
        with self.assertRaises(ConcretizeRegister) as cm:
            self.cpu.execute()
        expression = self.cpu.read_register(cm.exception.reg_name)
        all_values = solver.get_all_values(self.cpu.memory.constraints, expression)
        self.assertEqual(sorted(all_values), [4102, 4104])
        self.cpu.PC = 4104
        self.cpu.execute()
        with self.assertRaises(ConcretizeRegister) as cm:
            self.cpu.execute()
        expression = self.cpu.read_register(cm.exception.reg_name)
        all_values = solver.get_all_values(self.cpu.memory.constraints, expression)
        self.assertEqual(sorted(all_values), [4104, 4106])