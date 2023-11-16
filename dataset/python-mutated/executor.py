import struct
from ctypes import c_int32
from .types import ConcretizeStack, CurGrowMemImm, F32, F32ConstImm, F64, F64ConstImm, GlobalVarXsImm, I32, I32ConstImm, I64, I64ConstImm, InvalidConversionTrap, LocalVarXsImm, MemoryImm, OutOfBoundsMemoryTrap, OverflowDivisionTrap, UnreachableInstructionTrap, Value_t, ZeroDivisionTrap
from ..core.smtlib import Operators, BitVec, issymbolic
from ..utils.event import Eventful
from decimal import Decimal, InvalidOperation
import operator
import math
MASK_64 = (1 << 64) - 1
MASK_32 = (1 << 32) - 1

class Executor(Eventful):
    """
    Contains execution semantics for all WASM instructions that don't involve control flow (and thus only need access
    to the store and the stack).

    In lieu of annotating every single instruction with the relevant link to the docs, we direct you here:
    https://www.w3.org/TR/wasm-core-1/#a7-index-of-instructions
    """
    _published_events = {'set_global', 'get_global', 'set_local', 'get_local'}

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._mapping = {0: self.unreachable, 1: self.nop, 2: self.nop, 3: self.nop, 4: self.nop, 5: self.nop, 11: self.nop, 12: self.nop, 13: self.nop, 14: self.nop, 15: self.nop, 16: self.nop, 17: self.nop, 26: self.drop, 27: self.select, 32: self.get_local, 33: self.set_local, 34: self.tee_local, 35: self.get_global, 36: self.set_global, 40: self.i32_load, 41: self.i64_load, 42: self.f32_load, 43: self.f64_load, 44: self.i32_load8_s, 45: self.i32_load8_u, 46: self.i32_load16_s, 47: self.i32_load16_u, 48: self.i64_load8_s, 49: self.i64_load8_u, 50: self.i64_load16_s, 51: self.i64_load16_u, 52: self.i64_load32_s, 53: self.i64_load32_u, 54: self.i32_store, 55: self.i64_store, 56: self.f32_store, 57: self.f64_store, 58: self.i32_store8, 59: self.i32_store16, 60: self.i64_store8, 61: self.i64_store16, 62: self.i64_store32, 63: self.current_memory, 64: self.grow_memory, 65: self.i32_const, 66: self.i64_const, 67: self.f32_const, 68: self.f64_const, 69: self.i32_eqz, 70: self.i32_eq, 71: self.i32_ne, 72: self.i32_lt_s, 73: self.i32_lt_u, 74: self.i32_gt_s, 75: self.i32_gt_u, 76: self.i32_le_s, 77: self.i32_le_u, 78: self.i32_ge_s, 79: self.i32_ge_u, 80: self.i64_eqz, 81: self.i64_eq, 82: self.i64_ne, 83: self.i64_lt_s, 84: self.i64_lt_u, 85: self.i64_gt_s, 86: self.i64_gt_u, 87: self.i64_le_s, 88: self.i64_le_u, 89: self.i64_ge_s, 90: self.i64_ge_u, 91: self.f32_eq, 92: self.f32_ne, 93: self.f32_lt, 94: self.f32_gt, 95: self.f32_le, 96: self.f32_ge, 97: self.f64_eq, 98: self.f64_ne, 99: self.f64_lt, 100: self.f64_gt, 101: self.f64_le, 102: self.f64_ge, 103: self.i32_clz, 104: self.i32_ctz, 105: self.i32_popcnt, 106: self.i32_add, 107: self.i32_sub, 108: self.i32_mul, 109: self.i32_div_s, 110: self.i32_div_u, 111: self.i32_rem_s, 112: self.i32_rem_u, 113: self.i32_and, 114: self.i32_or, 115: self.i32_xor, 116: self.i32_shl, 117: self.i32_shr_s, 118: self.i32_shr_u, 119: self.i32_rotl, 120: self.i32_rotr, 121: self.i64_clz, 122: self.i64_ctz, 123: self.i64_popcnt, 124: self.i64_add, 125: self.i64_sub, 126: self.i64_mul, 127: self.i64_div_s, 128: self.i64_div_u, 129: self.i64_rem_s, 130: self.i64_rem_u, 131: self.i64_and, 132: self.i64_or, 133: self.i64_xor, 134: self.i64_shl, 135: self.i64_shr_s, 136: self.i64_shr_u, 137: self.i64_rotl, 138: self.i64_rotr, 139: self.f32_abs, 140: self.f32_neg, 141: self.f32_ceil, 142: self.f32_floor, 143: self.f32_trunc, 144: self.f32_nearest, 145: self.f32_sqrt, 146: self.f32_add, 147: self.f32_sub, 148: self.f32_mul, 149: self.f32_div, 150: self.f32_min, 151: self.f32_max, 152: self.f32_copysign, 153: self.f64_abs, 154: self.f64_neg, 155: self.f64_ceil, 156: self.f64_floor, 157: self.f64_trunc, 158: self.f64_nearest, 159: self.f64_sqrt, 160: self.f64_add, 161: self.f64_sub, 162: self.f64_mul, 163: self.f64_div, 164: self.f64_min, 165: self.f64_max, 166: self.f64_copysign, 167: self.i32_wrap_i64, 168: self.i32_trunc_s_f32, 169: self.i32_trunc_u_f32, 170: self.i32_trunc_s_f64, 171: self.i32_trunc_u_f64, 172: self.i64_extend_s_i32, 173: self.i64_extend_u_i32, 174: self.i64_trunc_s_f32, 175: self.i64_trunc_u_f32, 176: self.i64_trunc_s_f64, 177: self.i64_trunc_u_f64, 178: self.f32_convert_s_i32, 179: self.f32_convert_u_i32, 180: self.f32_convert_s_i64, 181: self.f32_convert_u_i64, 182: self.f32_demote_f64, 183: self.f64_convert_s_i32, 184: self.f64_convert_u_i32, 185: self.f64_convert_s_i64, 186: self.f64_convert_u_i64, 187: self.f64_promote_f32, 188: self.i32_reinterpret_f32, 189: self.i64_reinterpret_f64, 190: self.f32_reinterpret_i32, 191: self.f64_reinterpret_i64}
        self.zero_div = False
        self.overflow = False
        super().__init__()

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = super().__getstate__()
        state['mapping'] = self._mapping
        state['zero_div'] = self.zero_div
        state['overflow'] = self.overflow
        return state

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self._mapping = state['mapping']
        self.zero_div = state['zero_div']
        self.overflow = state['overflow']
        super().__setstate__(state)

    def check_overflow(self, expression) -> bool:
        if False:
            while True:
                i = 10
        if issymbolic(expression):
            self.overflow = Operators.OR(self.overflow, expression)
            return False
        return expression

    def check_zero_div(self, expression) -> bool:
        if False:
            while True:
                i = 10
        if issymbolic(expression):
            self.zero_div = Operators.OR(self.zero_div, expression)
            return False
        return expression

    def dispatch(self, inst, store, stack):
        if False:
            while True:
                i = 10
        '\n        Selects the correct semantics for the given instruction, and executes them\n\n        :param inst: the Instruction to execute\n        :param store: the current Store\n        :param stack: the current Stack\n        :return: the result of the semantic function, which is (probably) always None\n        '
        opcode = inst.opcode
        assert opcode in self._mapping
        func = self._mapping[opcode]
        try:
            if inst.imm:
                return func(store, stack, inst.imm)
            else:
                return func(store, stack)
        except (ZeroDivisionError, InvalidOperation):
            raise ZeroDivisionTrap()

    def unreachable(self, store, stack):
        if False:
            i = 10
            return i + 15
        raise UnreachableInstructionTrap()

    def nop(self, store, stack):
        if False:
            i = 10
            return i + 15
        pass

    def drop(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(Value_t, 1)
        stack.pop()

    def select(self, store, stack):
        if False:
            i = 10
            return i + 15
        c = stack.pop()
        v2 = stack.pop()
        v1 = stack.pop()
        assert isinstance(c, (I32, BitVec)), f'{type(c)} is not I32'
        if not issymbolic(v2) and (not issymbolic(v1)):
            assert type(v2) == type(v1), f'{type(v2)} is not the same as {type(v1)}'
        if issymbolic(c):
            stack.push(Operators.ITEBV(getattr(v1, 'size', 32), c != 0, v1, v2))
        elif c != 0:
            stack.push(v1)
        else:
            stack.push(v2)

    def get_local(self, store, stack, imm: LocalVarXsImm):
        if False:
            print('Hello World!')
        f = stack.get_frame().frame
        assert imm.local_index in range(len(f.locals))
        self._publish('will_get_local', imm.local_index)
        stack.push(f.locals[imm.local_index])
        self._publish('did_get_local', imm.local_index, stack.peek())

    def set_local(self, store, stack, imm: LocalVarXsImm):
        if False:
            while True:
                i = 10
        f = stack.get_frame().frame
        assert imm.local_index in range(len(f.locals))
        stack.has_type_on_top(Value_t, 1)
        self._publish('will_set_local', imm.local_index, stack.peek())
        f.locals[imm.local_index] = stack.pop()
        self._publish('did_set_local', imm.local_index, f.locals[imm.local_index])

    def tee_local(self, store, stack, imm: LocalVarXsImm):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(Value_t, 1)
        v = stack.pop()
        stack.push(v)
        stack.push(v)
        self.set_local(store, stack, imm)

    def get_global(self, store, stack, imm: GlobalVarXsImm):
        if False:
            while True:
                i = 10
        f = stack.get_frame().frame
        assert imm.global_index in range(len(f.module.globaladdrs))
        a = f.module.globaladdrs[imm.global_index]
        assert a in range(len(store.globals))
        glob = store.globals[a]
        self._publish('will_get_global', imm.global_index, glob.value)
        stack.push(glob.value)
        self._publish('did_get_global', imm.global_index, stack.peek())

    def set_global(self, store, stack, imm: GlobalVarXsImm):
        if False:
            i = 10
            return i + 15
        f = stack.get_frame().frame
        assert imm.global_index in range(len(f.module.globaladdrs))
        a = f.module.globaladdrs[imm.global_index]
        assert a in range(len(store.globals))
        stack.has_type_on_top(Value_t, 1)
        self._publish('will_set_global', imm.global_index, stack.peek())
        store.globals[a].value = stack.pop()
        self._publish('did_set_global', imm.global_index, store.globals[a].value)

    def i32_load(self, store, stack, imm: MemoryImm):
        if False:
            print('Hello World!')
        f = stack.get_frame().frame
        assert f.module.memaddrs
        a = f.module.memaddrs[0]
        assert a in range(len(store.mems))
        mem = store.mems[a]
        stack.has_type_on_top(I32, 1)
        i = stack.pop()
        if issymbolic(i):
            raise ConcretizeStack(-1, I32, 'Concretizing memory read', i)
        ea = i + imm.offset
        if ea + 4 - 1 not in mem:
            raise OutOfBoundsMemoryTrap(ea + 4)
        c = mem.read_int(ea, 32)
        stack.push(I32.cast(c))

    def i64_load(self, store, stack, imm: MemoryImm):
        if False:
            print('Hello World!')
        f = stack.get_frame().frame
        assert f.module.memaddrs
        a = f.module.memaddrs[0]
        assert a in range(len(store.mems))
        mem = store.mems[a]
        stack.has_type_on_top(I32, 1)
        i = stack.pop()
        if issymbolic(i):
            raise ConcretizeStack(-1, I32, 'Concretizing memory read', i)
        ea = i + imm.offset
        if ea + 8 - 1 not in mem:
            raise OutOfBoundsMemoryTrap(ea + 8)
        c = mem.read_int(ea, 64)
        stack.push(I64.cast(c))

    def int_load(self, store, stack, imm: MemoryImm, ty: type, size: int, signed: bool):
        if False:
            i = 10
            return i + 15
        assert ty in {I32, I64}, f'{type(ty)} is not an I32 or I64'
        f = stack.get_frame().frame
        assert f.module.memaddrs
        a = f.module.memaddrs[0]
        assert a in range(len(store.mems))
        mem = store.mems[a]
        stack.has_type_on_top(I32, 1)
        i = stack.pop()
        if issymbolic(i):
            raise ConcretizeStack(-1, I32, 'Concretizing memory read', i)
        ea = i + imm.offset
        if ea not in mem:
            raise OutOfBoundsMemoryTrap(ea)
        if ea + size // 8 - 1 not in mem:
            raise OutOfBoundsMemoryTrap(ea + size // 8)
        c = mem.read_int(ea, size)
        width = 32 if ty is I32 else 64
        if signed:
            c = Operators.SEXTEND(c, size, width)
        else:
            c = Operators.ZEXTEND(c, width)
        stack.push(ty.cast(c))

    def i32_load8_s(self, store, stack, imm: MemoryImm):
        if False:
            i = 10
            return i + 15
        self.int_load(store, stack, imm, I32, 8, True)

    def i32_load8_u(self, store, stack, imm: MemoryImm):
        if False:
            while True:
                i = 10
        self.int_load(store, stack, imm, I32, 8, False)

    def i32_load16_s(self, store, stack, imm: MemoryImm):
        if False:
            print('Hello World!')
        self.int_load(store, stack, imm, I32, 16, True)

    def i32_load16_u(self, store, stack, imm: MemoryImm):
        if False:
            while True:
                i = 10
        self.int_load(store, stack, imm, I32, 16, False)

    def i64_load8_s(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        self.int_load(store, stack, imm, I64, 8, True)

    def i64_load8_u(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        self.int_load(store, stack, imm, I64, 8, False)

    def i64_load16_s(self, store, stack, imm: MemoryImm):
        if False:
            while True:
                i = 10
        self.int_load(store, stack, imm, I64, 16, True)

    def i64_load16_u(self, store, stack, imm: MemoryImm):
        if False:
            i = 10
            return i + 15
        self.int_load(store, stack, imm, I64, 16, False)

    def i64_load32_s(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        self.int_load(store, stack, imm, I64, 32, True)

    def i64_load32_u(self, store, stack, imm: MemoryImm):
        if False:
            print('Hello World!')
        self.int_load(store, stack, imm, I64, 32, False)

    def int_store(self, store, stack, imm: MemoryImm, ty: type, n=None):
        if False:
            while True:
                i = 10
        assert ty in {I32, I64}, f'{type(ty)} is not an I32 or I64'
        f = stack.get_frame().frame
        assert f.module.memaddrs
        a = f.module.memaddrs[0]
        assert a in range(len(store.mems))
        mem = store.mems[a]
        stack.has_type_on_top(ty, 1)
        c = stack.pop()
        stack.has_type_on_top(I32, 1)
        i = stack.pop()
        if issymbolic(i):
            raise ConcretizeStack(-2, I32, 'Concretizing integer memory write', i)
        ea = i + imm.offset
        N = n if n else 32 if ty is I32 else 64
        mask = (1 << N) - 1
        if ea not in mem:
            raise OutOfBoundsMemoryTrap(ea)
        if ea + N // 8 - 1 not in mem:
            raise OutOfBoundsMemoryTrap(ea + N // 8)
        if n:
            b = [Operators.CHR(Operators.EXTRACT(c & mask, offset, 8)) for offset in range(0, N, 8)]
        else:
            b = [Operators.CHR(Operators.EXTRACT(c, offset, 8)) for offset in range(0, N, 8)]
        mem.write_bytes(ea, b)

    def i32_store(self, store, stack, imm: MemoryImm):
        if False:
            while True:
                i = 10
        self.int_store(store, stack, imm, I32)

    def i64_store(self, store, stack, imm: MemoryImm):
        if False:
            while True:
                i = 10
        self.int_store(store, stack, imm, I64)

    def i32_store8(self, store, stack, imm: MemoryImm):
        if False:
            print('Hello World!')
        self.int_store(store, stack, imm, I32, 8)

    def i32_store16(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        self.int_store(store, stack, imm, I32, 16)

    def i64_store8(self, store, stack, imm: MemoryImm):
        if False:
            while True:
                i = 10
        self.int_store(store, stack, imm, I64, 8)

    def i64_store16(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        self.int_store(store, stack, imm, I64, 16)

    def i64_store32(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        self.int_store(store, stack, imm, I64, 32)

    def current_memory(self, store, stack, imm: CurGrowMemImm):
        if False:
            print('Hello World!')
        f = stack.get_frame().frame
        assert f.module.memaddrs
        a = f.module.memaddrs[0]
        assert a in range(len(store.mems))
        mem = store.mems[a]
        stack.push(I32(mem.npages))

    def grow_memory(self, store, stack, imm: CurGrowMemImm):
        if False:
            return 10
        f = stack.get_frame().frame
        assert f.module.memaddrs
        a = f.module.memaddrs[0]
        assert a in range(len(store.mems))
        mem = store.mems[a]
        sz = mem.npages
        stack.has_type_on_top(I32, 1)
        if issymbolic(stack.peek()):
            raise ConcretizeStack(-1, I32, 'Concretizing memory grow operand', stack.peek())
        if mem.grow(stack.pop()):
            stack.push(I32(sz))
        else:
            stack.push(I32(-1))

    def i32_const(self, store, stack, imm: I32ConstImm):
        if False:
            for i in range(10):
                print('nop')
        stack.push(I32.cast(imm.value))

    def i64_const(self, store, stack, imm: I64ConstImm):
        if False:
            print('Hello World!')
        stack.push(I64.cast(imm.value))

    def i32_eqz(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I32, 1)
        c1 = stack.pop()
        v = c1 == 0
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_eq(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c2 == c1
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_ne(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c2 != c1
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_lt_s(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c1 < c2
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_lt_u(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = Operators.ULT(c1, c2)
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_gt_s(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c1 > c2
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_gt_u(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = Operators.UGT(c1, c2)
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_le_s(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c1 <= c2
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_le_u(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = Operators.ULE(c1, c2)
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_ge_s(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c1 >= c2
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_ge_u(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = Operators.UGE(c1, c2)
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_eqz(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I64, 1)
        c1 = stack.pop()
        v = c1 == 0
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_eq(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c2 == c1
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_ne(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c2 != c1
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_lt_s(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c1 < c2
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_lt_u(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = Operators.ULT(c1, c2)
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_gt_s(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c1 > c2
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_gt_u(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = Operators.UGT(c1, c2)
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_le_s(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c1 <= c2
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_le_u(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = Operators.ULE(c1, c2)
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_ge_s(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = c1 >= c2
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i64_ge_u(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        v = Operators.UGE(c1, c2)
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(I32.cast(I32(1) if v else I32(0)))

    def i32_clz(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I32, 1)
        c1 = stack.pop()
        flag = Operators.EXTRACT(c1, 31, 1) == 1
        res = 0
        for pos in range(1, 32):
            res = Operators.ITEBV(32, flag, res, pos)
            flag = Operators.OR(flag, Operators.EXTRACT(c1, 31 - pos, 1) == 1)
        res = Operators.ITEBV(32, flag, res, 32)
        stack.push(I32.cast(res))

    def i32_ctz(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I32, 1)
        c1 = stack.pop()
        flag = Operators.EXTRACT(c1, 0, 1) == 1
        res = 0
        for pos in range(1, 32):
            res = Operators.ITEBV(32, flag, res, pos)
            flag = Operators.OR(flag, Operators.EXTRACT(c1, pos, 1) == 1)
        res = Operators.ITEBV(32, flag, res, 32)
        stack.push(I32.cast(res))

    def i32_popcnt(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I32, 1)
        c1 = stack.pop()
        flag = Operators.EXTRACT(c1, 0, 1) != 0
        res = 0
        for pos in range(1, 32):
            res = Operators.ITEBV(32, flag, res + 1, res)
            flag = Operators.EXTRACT(c1, pos, 1) != 0
        res = Operators.ITEBV(32, flag, res + 1, res)
        stack.push(I32.cast(res))

    def i32_add(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I32.cast(c2 + c1 & MASK_32))

    def i32_sub(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I32.cast(c1 - c2 + 2 ** 32 & MASK_32))

    def i32_mul(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I32.cast(c2 * c1 & MASK_32))

    def i32_div_s(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        can_div_0 = c2 == 0
        if self.check_zero_div(can_div_0):
            raise ZeroDivisionTrap()
        res = Operators.SDIV(c1, c2)
        can_overflow = res == 2 ** 31
        if self.check_overflow(can_overflow):
            raise OverflowDivisionTrap()
        stack.push(I32.cast(res))

    def i32_div_u(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        can_div_0 = c2 == 0
        if self.check_zero_div(can_div_0):
            raise ZeroDivisionTrap()
        if not issymbolic(c2):
            c2 = I32.to_unsigned(c2)
        if not issymbolic(c1):
            c1 = I32.to_unsigned(c1)
        stack.push(I32.cast(Operators.UDIV(c1, c2)))

    def i32_rem_s(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if self.check_zero_div(c2 == 0):
            raise ZeroDivisionTrap()
        stack.push(I32.cast(Operators.SREM(c1, c2)))

    def i32_rem_u(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if not issymbolic(c2):
            c2 = I32.to_unsigned(c2)
        if not issymbolic(c1):
            c1 = I32.to_unsigned(c1)
        if self.check_zero_div(c2 == 0):
            raise ZeroDivisionTrap()
        stack.push(I32.cast(Operators.UREM(c1, c2)))

    def i32_and(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I32.cast(c2 & c1))

    def i32_or(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I32.cast(c2 | c1))

    def i32_xor(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I32.cast(c2 ^ c1))

    def i32_shl(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I32.cast(c1 << c2 % 32 & MASK_32))

    def i32_shr_s(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        k = c2 % 32
        stack.push(I32.cast(Operators.SAR(32, c1, k)))

    def i32_shr_u(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if not issymbolic(c2):
            c2 = I32.to_unsigned(c2)
        if not issymbolic(c1):
            c1 = I32.to_unsigned(c1)
        stack.push(I32.cast(c1 >> c2 % 32))

    def i32_rotl(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if not issymbolic(c1):
            c1 = I32.to_unsigned(c1)
        k = c2 % 32
        stack.push(I32.cast(c1 << k | c1 >> 32 - k))

    def i32_rotr(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if not issymbolic(c1):
            c1 = I32.to_unsigned(c1)
        k = c2 % 32
        stack.push(I32.cast(c1 >> k | c1 << 32 - k))

    def i64_clz(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I64, 1)
        c1 = stack.pop()
        flag = Operators.EXTRACT(c1, 63, 1) == 1
        res = 0
        for pos in range(1, 64):
            res = Operators.ITEBV(64, flag, res, pos)
            flag = Operators.OR(flag, Operators.EXTRACT(c1, 63 - pos, 1) == 1)
        res = Operators.ITEBV(64, flag, res, 64)
        stack.push(I64.cast(res))

    def i64_ctz(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I64, 1)
        c1 = stack.pop()
        flag = Operators.EXTRACT(c1, 0, 1) == 1
        res = 0
        for pos in range(1, 64):
            res = Operators.ITEBV(64, flag, res, pos)
            flag = Operators.OR(flag, Operators.EXTRACT(c1, pos, 1) == 1)
        res = Operators.ITEBV(64, flag, res, 64)
        stack.push(I64.cast(res))

    def i64_popcnt(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 1)
        c1 = stack.pop()
        flag = Operators.EXTRACT(c1, 0, 1) != 0
        res = 0
        for pos in range(1, 64):
            res = Operators.ITEBV(64, flag, res + 1, res)
            flag = Operators.EXTRACT(c1, pos, 1) != 0
        res = Operators.ITEBV(64, flag, res + 1, res)
        stack.push(I64.cast(res))

    def i64_add(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I64.cast(c2 + c1 & MASK_64))

    def i64_sub(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I64.cast(c1 - c2 + 2 ** 64 & MASK_64))

    def i64_mul(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I64.cast(c2 * c1 & MASK_64))

    def i64_div_s(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        can_div_0 = c2 == 0
        if self.check_zero_div(can_div_0):
            raise ZeroDivisionTrap()
        if issymbolic(c1) or issymbolic(c2):
            res = Operators.SDIV(c1, c2)
        else:
            res = int(math.trunc(Decimal(c1) / Decimal(c2)))
        can_overflow = res == 2 ** 63
        if self.check_overflow(can_overflow):
            raise OverflowDivisionTrap()
        stack.push(I64.cast(res))

    def i64_div_u(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        can_div_0 = c2 == 0
        if self.check_zero_div(can_div_0):
            raise ZeroDivisionTrap()
        if not issymbolic(c2):
            c2 = I64.to_unsigned(c2)
        if not issymbolic(c1):
            c1 = I64.to_unsigned(c1)
        stack.push(I64.cast(Operators.UDIV(c1, c2)))

    def i64_rem_s(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if self.check_zero_div(c2 == 0):
            raise ZeroDivisionTrap()
        if issymbolic(c1) or issymbolic(c2):
            res = Operators.SREM(c1, c2)
        else:
            res = c1 - int(Decimal(c1) / Decimal(c2)) * c2
        stack.push(I64.cast(res))

    def i64_rem_u(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if not issymbolic(c2):
            c2 = I64.to_unsigned(c2)
        if not issymbolic(c1):
            c1 = I64.to_unsigned(c1)
        if self.check_zero_div(c2 == 0):
            raise ZeroDivisionTrap()
        stack.push(I64.cast(Operators.UREM(c1, c2)))

    def i64_and(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I64.cast(c2 & c1))

    def i64_or(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I64.cast(c2 | c1))

    def i64_xor(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I64.cast(c2 ^ c1))

    def i64_shl(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        stack.push(I64.cast(c1 << c2 % 64 & MASK_64))

    def i64_shr_s(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        k = c2 % 64
        stack.push(I64.cast(Operators.SAR(64, c1, k)))

    def i64_shr_u(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if not issymbolic(c2):
            c2 = I64.to_unsigned(c2)
        if not issymbolic(c1):
            c1 = I64.to_unsigned(c1)
        stack.push(I64.cast(c1 >> c2 % 64))

    def i64_rotl(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if not issymbolic(c1):
            c1 = I64.to_unsigned(c1)
        k = c2 % 64
        stack.push(I64.cast(c1 << k | c1 >> 64 - k))

    def i64_rotr(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 2)
        c2 = stack.pop()
        c1 = stack.pop()
        if not issymbolic(c1):
            c1 = I64.to_unsigned(c1)
        k = c2 % 64
        stack.push(I64.cast(c1 >> k | c1 << 64 - k))

    def i32_wrap_i64(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I64, 1)
        c1: I64 = stack.pop()
        c1 &= MASK_32
        c1 = Operators.EXTRACT(c1, 0, 32)
        stack.push(I32.cast(c1))

    def i32_trunc_s_f32(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(F32, 1)
        c1: F32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F32, 'Concretizing for float->int conversion', c1)
        if math.isnan(c1):
            raise InvalidConversionTrap(I32, 'NaN')
        if math.isinf(c1):
            raise InvalidConversionTrap(I32, 'infinity')
        if c1 >= 2 ** 31 or c1 <= -2 ** 31 - 1:
            raise InvalidConversionTrap(I32, c1)
        stack.push(I32.cast(c1))

    def i32_trunc_u_f32(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(F32, 1)
        c1: F32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F32, 'Concretizing for float->int conversion', c1)
        if math.isnan(c1):
            raise InvalidConversionTrap(I32, 'NaN')
        if math.isinf(c1):
            raise InvalidConversionTrap(I32, 'infinity')
        if c1 >= 2 ** 32 or c1 <= -1:
            raise InvalidConversionTrap(I32, c1)
        stack.push(I32.cast(c1))

    def i32_trunc_s_f64(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(F64, 1)
        c1: F64 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F64, 'Concretizing for float->int conversion', c1)
        if math.isnan(c1):
            raise InvalidConversionTrap(I32, 'NaN')
        if math.isinf(c1):
            raise InvalidConversionTrap(I32, 'infinity')
        if c1 >= 2 ** 31 or c1 <= -2 ** 31 - 1:
            raise InvalidConversionTrap(I32, c1)
        stack.push(I32.cast(c1))

    def i32_trunc_u_f64(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(F64, 1)
        c1: F64 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F64, 'Concretizing for float->int conversion', c1)
        if math.isnan(c1):
            raise InvalidConversionTrap(I32, 'NaN')
        if math.isinf(c1):
            raise InvalidConversionTrap(I32, 'infinity')
        if c1 >= 2 ** 32 or c1 <= -1:
            raise InvalidConversionTrap(I32, c1)
        stack.push(I32.cast(c1))

    def i64_extend_s_i32(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I32, 1)
        c1: I32 = stack.pop()
        stack.push(I64.cast(Operators.SEXTEND(c1, 32, 64)))

    def i64_extend_u_i32(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I32, 1)
        c1: I32 = stack.pop()
        if issymbolic(c1):
            stack.push(I64.cast(Operators.ZEXTEND(c1, 64)))
        else:
            stack.push(I64.cast(struct.unpack('q', bytes(c_int32(c1)) + b'\x00' * 4)[0]))

    def i64_trunc_s_f32(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(F32, 1)
        c1: F32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F32, 'Concretizing float', c1)
        if math.isnan(c1):
            raise InvalidConversionTrap(I64, 'NaN')
        if math.isinf(c1):
            raise InvalidConversionTrap(I64, 'infinity')
        if c1 >= 2 ** 63 or c1 <= -2 ** 63 - 1:
            raise InvalidConversionTrap(I64, c1)
        stack.push(I64.cast(c1))

    def i64_trunc_u_f32(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(F32, 1)
        c1: F32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F32, 'Concretizing float', c1)
        if math.isnan(c1):
            raise InvalidConversionTrap(I64, 'NaN')
        if math.isinf(c1):
            raise InvalidConversionTrap(I64, 'infinity')
        if c1 >= 2 ** 64 or c1 <= -1:
            raise InvalidConversionTrap(I64, c1)
        stack.push(I64.cast(c1))

    def i64_trunc_s_f64(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(F64, 1)
        c1: F64 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F64, 'Concretizing float', c1)
        if math.isnan(c1):
            raise InvalidConversionTrap(I64, 'NaN')
        if math.isinf(c1):
            raise InvalidConversionTrap(I64, 'infinity')
        if c1 >= 2 ** 63 or c1 <= -2 ** 63 - 1:
            raise InvalidConversionTrap(I64, c1)
        stack.push(I64.cast(c1))

    def i64_trunc_u_f64(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(F64, 1)
        c1: F64 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F64, 'Concretizing float', c1)
        if math.isnan(c1):
            raise InvalidConversionTrap(I64, 'NaN')
        if math.isinf(c1):
            raise InvalidConversionTrap(I64, 'infinity')
        if c1 >= 2 ** 64 or c1 <= -1:
            raise InvalidConversionTrap(I64, c1)
        stack.push(I64.cast(c1))

    def i32_reinterpret_f32(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(F32, 1)
        c1: F32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F32, 'Concretizing float', c1)
        c1 = struct.unpack('i', struct.pack('f', c1))[0]
        stack.push(I32.cast(c1))

    def i64_reinterpret_f64(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(F64, 1)
        c1: F64 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F64, 'Concretizing float', c1)
        c1 = struct.unpack('q', struct.pack('d', c1))[0]
        stack.push(I64.cast(c1))

    def float_load(self, store, stack, imm: MemoryImm, ty: type):
        if False:
            return 10
        assert ty in {F32, F64}, f'{type(ty)} is not an F32 or F64'
        size = 32 if ty == F32 else 64
        f = stack.get_frame().frame
        a = f.module.memaddrs[0]
        mem = store.mems[a]
        stack.has_type_on_top(I32, 1)
        i = stack.pop()
        if issymbolic(i):
            raise ConcretizeStack(-1, I32, 'Concretizing float memory read', i)
        ea = i + imm.offset
        if ea not in mem:
            raise OutOfBoundsMemoryTrap(ea)
        if ea + size // 8 - 1 not in mem:
            raise OutOfBoundsMemoryTrap(ea + size // 8)
        c = mem.read_int(ea, size)
        ret = ty.cast(c)
        stack.push(ret)

    def f32_load(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        return self.float_load(store, stack, imm, F32)

    def f64_load(self, store, stack, imm: MemoryImm):
        if False:
            print('Hello World!')
        return self.float_load(store, stack, imm, F64)

    def float_store(self, store, stack, imm: MemoryImm, ty: type, n=None):
        if False:
            print('Hello World!')
        f = stack.get_frame().frame
        a = f.module.memaddrs[0]
        mem = store.mems[a]
        c = stack.pop()
        i = stack.pop()
        if issymbolic(i):
            raise ConcretizeStack(-2, I32, 'Concretizing memory address for float_store', i)
        ea = i + imm.offset
        if ty == F32:
            size = 32
        else:
            size = 64
        if ea not in mem:
            raise OutOfBoundsMemoryTrap(ea)
        if ea + size // 8 - 1 not in mem:
            raise OutOfBoundsMemoryTrap(ea + size // 8)
        if not issymbolic(c):
            c = struct.unpack('i' if size == 32 else 'q', struct.pack('f' if size == 32 else 'd', c))[0]
        b = [Operators.CHR(Operators.EXTRACT(c, offset, 8)) for offset in range(0, size, 8)]
        mem.write_bytes(ea, b)

    def float_push_compare_return(self, stack, v, rettype=I32):
        if False:
            i = 10
            return i + 15
        if issymbolic(v):
            stack.push(Operators.ITEBV(32, v, I32(1), I32(0)))
        else:
            stack.push(rettype(v))

    def f32_store(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        self.float_store(store, stack, imm, F32)

    def f64_store(self, store, stack, imm: MemoryImm):
        if False:
            return 10
        self.float_store(store, stack, imm, F64)

    def f32_const(self, store, stack, imm: F32ConstImm):
        if False:
            return 10
        stack.push(F32.cast(imm.value))

    def f64_const(self, store, stack, imm: F64ConstImm):
        if False:
            for i in range(10):
                print('nop')
        stack.push(F64.cast(imm.value))

    def f32_unary(self, store, stack, op, rettype: type=I32):
        if False:
            print('Hello World!')
        stack.has_type_on_top(F32, 1)
        if issymbolic(stack.peek()):
            raise ConcretizeStack(-1, F32, 'Concretizing before float op', stack.peek())
        v1 = stack.pop()
        v = op(v1)
        self.float_push_compare_return(stack, v, rettype)

    def f32_binary(self, store, stack, op, rettype: type=I32):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(F32, 2)
        if issymbolic(stack.peek()):
            raise ConcretizeStack(-1, F32, 'Concretizing before float op', stack.peek())
        v2 = stack.pop()
        if issymbolic(stack.peek()):
            raise ConcretizeStack(-2, F32, 'Concretizing before float op', stack.peek())
        v1 = stack.pop()
        v = op(v1, v2)
        self.float_push_compare_return(stack, v, rettype)

    def f64_unary(self, store, stack, op, rettype: type=F64):
        if False:
            return 10
        stack.has_type_on_top(F64, 1)
        if issymbolic(stack.peek()):
            raise ConcretizeStack(-1, F64, 'Concretizing before float op', stack.peek())
        v1 = stack.pop()
        v = op(v1)
        self.float_push_compare_return(stack, v, rettype)

    def f64_binary(self, store, stack, op, rettype: type=I32):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(F64, 2)
        if issymbolic(stack.peek()):
            raise ConcretizeStack(-1, F64, 'Concretizing before float op', stack.peek())
        v2 = stack.pop()
        if issymbolic(stack.peek()):
            raise ConcretizeStack(-2, F64, 'Concretizing before float op', stack.peek())
        v1 = stack.pop()
        v = op(v1, v2)
        self.float_push_compare_return(stack, v, rettype)

    def f32_eq(self, store, stack):
        if False:
            while True:
                i = 10
        return self.f32_binary(store, stack, operator.eq)

    def f32_ne(self, store, stack):
        if False:
            print('Hello World!')
        return self.f32_binary(store, stack, operator.ne)

    def f32_lt(self, store, stack):
        if False:
            return 10
        return self.f32_binary(store, stack, operator.lt)

    def f32_gt(self, store, stack):
        if False:
            return 10
        return self.f32_binary(store, stack, operator.gt)

    def f32_le(self, store, stack):
        if False:
            i = 10
            return i + 15
        return self.f32_binary(store, stack, operator.le)

    def f32_ge(self, store, stack):
        if False:
            i = 10
            return i + 15
        return self.f32_binary(store, stack, operator.ge)

    def f64_eq(self, store, stack):
        if False:
            print('Hello World!')
        return self.f64_binary(store, stack, operator.eq)

    def f64_ne(self, store, stack):
        if False:
            print('Hello World!')
        return self.f64_binary(store, stack, operator.ne)

    def f64_lt(self, store, stack):
        if False:
            return 10
        return self.f64_binary(store, stack, operator.lt)

    def f64_gt(self, store, stack):
        if False:
            while True:
                i = 10
        return self.f64_binary(store, stack, operator.gt)

    def f64_le(self, store, stack):
        if False:
            print('Hello World!')
        return self.f64_binary(store, stack, operator.le)

    def f64_ge(self, store, stack):
        if False:
            return 10
        return self.f64_binary(store, stack, operator.ge)

    def f32_abs(self, store, stack):
        if False:
            return 10
        return self.f32_unary(store, stack, operator.abs, F32)

    def f32_neg(self, store, stack):
        if False:
            return 10
        return self.f32_unary(store, stack, operator.neg, F32)

    def f32_ceil(self, store, stack):
        if False:
            print('Hello World!')
        return self.f32_unary(store, stack, operator_ceil, F32)

    def f32_floor(self, store, stack):
        if False:
            print('Hello World!')
        return self.f32_unary(store, stack, operator_floor, F32)

    def f32_trunc(self, store, stack):
        if False:
            print('Hello World!')
        return self.f32_unary(store, stack, operator_trunc, F32)

    def f32_nearest(self, store, stack):
        if False:
            return 10
        return self.f32_unary(store, stack, operator_nearest, F32)

    def f32_sqrt(self, store, stack):
        if False:
            print('Hello World!')
        return self.f32_unary(store, stack, math.sqrt, F32)

    def f32_add(self, store, stack):
        if False:
            print('Hello World!')
        return self.f32_binary(store, stack, operator.add, F32)

    def f32_sub(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        return self.f32_binary(store, stack, operator.sub, F32)

    def f32_mul(self, store, stack):
        if False:
            print('Hello World!')
        return self.f32_binary(store, stack, operator.mul, F32)

    def f32_div(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        return self.f32_binary(store, stack, operator_div, F32)

    def f32_min(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        return self.f32_binary(store, stack, operator_min, F32)

    def f32_max(self, store, stack):
        if False:
            while True:
                i = 10
        return self.f32_binary(store, stack, operator_max, F32)

    def f32_copysign(self, store, stack):
        if False:
            return 10
        return self.f32_binary(store, stack, math.copysign, F32)

    def f64_abs(self, store, stack):
        if False:
            while True:
                i = 10
        return self.f64_unary(store, stack, operator.abs, F64)

    def f64_neg(self, store, stack):
        if False:
            print('Hello World!')
        return self.f64_unary(store, stack, operator.neg, F64)

    def f64_ceil(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        return self.f64_unary(store, stack, operator_ceil, F64)

    def f64_floor(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        return self.f64_unary(store, stack, operator_floor, F64)

    def f64_trunc(self, store, stack):
        if False:
            print('Hello World!')
        return self.f64_unary(store, stack, operator_trunc, F64)

    def f64_nearest(self, store, stack):
        if False:
            i = 10
            return i + 15
        return self.f32_unary(store, stack, operator_nearest, F64)

    def f64_sqrt(self, store, stack):
        if False:
            print('Hello World!')
        return self.f64_unary(store, stack, math.sqrt, F64)

    def f64_add(self, store, stack):
        if False:
            while True:
                i = 10
        return self.f64_binary(store, stack, operator.add, F64)

    def f64_sub(self, store, stack):
        if False:
            while True:
                i = 10
        return self.f64_binary(store, stack, operator.sub, F64)

    def f64_mul(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        return self.f64_binary(store, stack, operator.mul, F64)

    def f64_div(self, store, stack):
        if False:
            return 10
        return self.f64_binary(store, stack, operator_div, F64)

    def f64_min(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        return self.f64_binary(store, stack, operator_min, F64)

    def f64_max(self, store, stack):
        if False:
            return 10
        return self.f64_binary(store, stack, operator_max, F64)

    def f64_copysign(self, store, stack):
        if False:
            print('Hello World!')
        return self.f64_binary(store, stack, math.copysign, F64)

    def f32_convert_s_i32(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I32, 1)
        c1: I32 = stack.pop()
        stack.push(F32.cast(float(c1)))

    def f32_convert_u_i32(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I32, 1)
        c1: I32 = stack.pop()
        stack.push(F32.cast(float(I32.to_unsigned(c1))))

    def f32_convert_s_i64(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 1)
        c1: I64 = stack.pop()
        stack.push(F32.cast(float(c1)))

    def f32_convert_u_i64(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I64, 1)
        c1: I64 = stack.pop()
        stack.push(F32.cast(float(I64.to_unsigned(c1))))

    def f32_demote_f64(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(F64, 1)
        c1: F64 = stack.pop()
        if math.isnan(c1) or math.isinf(c1) or c1 == 0.0 or (c1 == -0.0):
            stack.push(F32.cast(c1))
            return
        raise NotImplementedError('f32_demote_f64')

    def f64_convert_s_i32(self, store, stack):
        if False:
            for i in range(10):
                print('nop')
        stack.has_type_on_top(I32, 1)
        c1: I32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, I32, 'Concretizing int for float conversion', c1)
        stack.push(F64.cast(float(c1)))

    def f64_convert_u_i32(self, store, stack):
        if False:
            i = 10
            return i + 15
        stack.has_type_on_top(I32, 1)
        c1: I32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, I32, 'Concretizing int for float conversion', c1)
        stack.push(F64.cast(float(I32.to_unsigned(c1))))

    def f64_convert_s_i64(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I64, 1)
        c1: I64 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, I64, 'Concretizing int for float conversion', c1)
        stack.push(F64.cast(float(c1)))

    def f64_convert_u_i64(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I64, 1)
        c1: I64 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, I64, 'Concretizing int for float conversion', c1)
        stack.push(F64.cast(float(I64.to_unsigned(c1))))

    def f64_promote_f32(self, store, stack):
        if False:
            return 10
        stack.has_type_on_top(F32, 1)
        c1: F32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, F32, 'Concretizing F32 for F64 promotion', c1)
        stack.push(F64.cast(c1))

    def f32_reinterpret_i32(self, store, stack):
        if False:
            while True:
                i = 10
        stack.has_type_on_top(I32, 1)
        c1: I32 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, I32, 'Concretizing int for float conversion', c1)
        c1 = struct.unpack('f', struct.pack('i', c1))[0]
        stack.push(F32.cast(c1))

    def f64_reinterpret_i64(self, store, stack):
        if False:
            print('Hello World!')
        stack.has_type_on_top(I64, 1)
        c1: I64 = stack.pop()
        if issymbolic(c1):
            raise ConcretizeStack(-1, I64, 'Concretizing int for float conversion', c1)
        c1 = struct.unpack('d', struct.pack('q', c1))[0]
        stack.push(F64.cast(c1))

def operator_nearest(a):
    if False:
        return 10
    if math.isnan(a) or math.isinf(a):
        return a.integer
    else:
        return round(a)

def operator_trunc(a):
    if False:
        i = 10
        return i + 15
    if math.isnan(a) or math.isinf(a):
        return a.integer
    else:
        return math.trunc(a)

def operator_ceil(a):
    if False:
        for i in range(10):
            print('nop')
    if math.isnan(a) or math.isinf(a):
        return a.integer
    else:
        return math.ceil(a)

def operator_floor(a):
    if False:
        while True:
            i = 10
    if math.isnan(a) or math.isinf(a):
        return a.integer
    else:
        return math.floor(a)

def operator_div(a, b):
    if False:
        return 10
    if b == 0:
        return math.inf
    else:
        return operator.truediv(a, b)

def operator_min(a, b):
    if False:
        while True:
            i = 10
    return a if a < b else b

def operator_max(a, b):
    if False:
        return 10
    return a if a > b else b