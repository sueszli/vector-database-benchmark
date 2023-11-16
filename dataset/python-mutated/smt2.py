from builtins import map
from builtins import range
import logging
from miasm.ir.translators.translator import Translator
from miasm.expression.smt2_helper import *
from miasm.expression.expression import ExprCond, ExprInt
log = logging.getLogger('translator_smt2')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(levelname)-8s]: %(message)s'))
log.addHandler(console_handler)
log.setLevel(logging.WARNING)

class SMT2Mem(object):
    """
    Memory abstraction for TranslatorSMT2. Memory elements are only accessed,
    never written. To give a concrete value for a given memory cell in a solver,
    add "mem32.get(address, size) == <value>" constraints to your equation.
    The endianness of memory accesses is handled accordingly to the "endianness"
    attribute.
    Note: Will have one memory space for each addressing size used.
    For example, if memory is accessed via 32 bits values and 16 bits values,
    these access will not occur in the same address space.

    Adapted from Z3Mem
    """

    def __init__(self, endianness='<', name='mem'):
        if False:
            return 10
        "Initializes an SMT2Mem object with a given @name and @endianness.\n        @endianness: Endianness of memory representation. '<' for little endian,\n            '>' for big endian.\n        @name: name of memory Arrays generated. They will be named\n            name+str(address size) (for example mem32, mem16...).\n        "
        if endianness not in ['<', '>']:
            raise ValueError("Endianness should be '>' (big) or '<' (little)")
        self.endianness = endianness
        self.mems = {}
        self.name = name
        self.addr_size = 0

    def get_mem_array(self, size):
        if False:
            for i in range(10):
                print('nop')
        'Returns an SMT Array used internally to represent memory for addresses\n        of size @size.\n        @size: integer, size in bit of addresses in the memory to get.\n        Return an string with the name of the SMT array..\n        '
        try:
            mem = self.mems[size]
        except KeyError:
            self.mems[size] = self.name + str(size)
            mem = self.mems[size]
        return mem

    def __getitem__(self, addr):
        if False:
            return 10
        'One byte memory access. Different address sizes with the same value\n        will result in different memory accesses.\n        @addr: an SMT2 expression, the address to read.\n        Return an SMT2 expression of size 8 bits representing a memory access.\n        '
        size = self.addr_size
        mem = self.get_mem_array(size)
        return array_select(mem, addr)

    def get(self, addr, size, addr_size):
        if False:
            while True:
                i = 10
        ' Memory access at address @addr of size @size with\n        address size @addr_size.\n        @addr: an SMT2 expression, the address to read.\n        @size: int, size of the read in bits.\n        @addr_size: int, size of the address\n        Return a SMT2 expression representing a memory access.\n        '
        self.addr_size = addr_size
        original_size = size
        if original_size % 8 != 0:
            size = (original_size // 8 + 1) * 8
        res = self[addr]
        if self.is_little_endian():
            for i in range(1, size // 8):
                index = bvadd(addr, bit_vec_val(i, addr_size))
                res = bv_concat(self[index], res)
        else:
            for i in range(1, size // 8):
                res = bv_concat(res, self[index])
        if size == original_size:
            return res
        else:
            return bv_extract(original_size - 1, 0, res)

    def is_little_endian(self):
        if False:
            return 10
        'True if this memory is little endian.'
        return self.endianness == '<'

    def is_big_endian(self):
        if False:
            for i in range(10):
                print('nop')
        'True if this memory is big endian.'
        return not self.is_little_endian()

class TranslatorSMT2(Translator):
    """Translate a Miasm expression into an equivalent SMT2
    expression. Memory is abstracted via SMT2Mem.
    The result of from_expr will be an SMT2 expression.

    If you want to interact with the memory abstraction after the translation,
    you can instantiate your own SMT2Mem that will be equivalent to the one
    used by TranslatorSMT2.

    TranslatorSMT2 provides the creation of a valid SMT2 file. For this,
    it keeps track of the translated bit vectors.

    Adapted from TranslatorZ3
    """
    __LANG__ = 'smt2'

    def __init__(self, endianness='<', loc_db=None, **kwargs):
        if False:
            return 10
        'Instance a SMT2 translator\n        @endianness: (optional) memory endianness\n        '
        super(TranslatorSMT2, self).__init__(**kwargs)
        self._mem = SMT2Mem(endianness)
        self._bitvectors = dict()
        self.loc_db = loc_db

    def from_ExprInt(self, expr):
        if False:
            i = 10
            return i + 15
        return bit_vec_val(int(expr), expr.size)

    def from_ExprId(self, expr):
        if False:
            while True:
                i = 10
        if str(expr) not in self._bitvectors:
            self._bitvectors[str(expr)] = expr.size
        return str(expr)

    def from_ExprLoc(self, expr):
        if False:
            return 10
        loc_key = expr.loc_key
        if self.loc_db is None or self.loc_db.get_location_offset(loc_key) is None:
            if str(loc_key) not in self._bitvectors:
                self._bitvectors[str(loc_key)] = expr.size
            return str(loc_key)
        offset = self.loc_db.get_location_offset(loc_key)
        return bit_vec_val(str(offset), expr.size)

    def from_ExprMem(self, expr):
        if False:
            while True:
                i = 10
        addr = self.from_expr(expr.ptr)
        size = expr.size
        addr_size = expr.ptr.size
        return self._mem.get(addr, size, addr_size)

    def from_ExprSlice(self, expr):
        if False:
            i = 10
            return i + 15
        res = self.from_expr(expr.arg)
        res = bv_extract(expr.stop - 1, expr.start, res)
        return res

    def from_ExprCompose(self, expr):
        if False:
            print('Hello World!')
        res = None
        for arg in expr.args:
            e = bv_extract(arg.size - 1, 0, self.from_expr(arg))
            if res:
                res = bv_concat(e, res)
            else:
                res = e
        return res

    def from_ExprCond(self, expr):
        if False:
            i = 10
            return i + 15
        cond = self.from_expr(expr.cond)
        src1 = self.from_expr(expr.src1)
        src2 = self.from_expr(expr.src2)
        zero = bit_vec_val(0, expr.cond.size)
        distinct = smt2_distinct(cond, zero)
        distinct_and = smt2_and(distinct, 'true')
        return smt2_ite(distinct_and, src1, src2)

    def from_ExprOp(self, expr):
        if False:
            while True:
                i = 10
        args = list(map(self.from_expr, expr.args))
        res = args[0]
        if len(args) > 1:
            for arg in args[1:]:
                if expr.op == '+':
                    res = bvadd(res, arg)
                elif expr.op == '-':
                    res = bvsub(res, arg)
                elif expr.op == '*':
                    res = bvmul(res, arg)
                elif expr.op == '/':
                    res = bvsdiv(res, arg)
                elif expr.op == 'sdiv':
                    res = bvsdiv(res, arg)
                elif expr.op == 'udiv':
                    res = bvudiv(res, arg)
                elif expr.op == '%':
                    res = bvsmod(res, arg)
                elif expr.op == 'smod':
                    res = bvsmod(res, arg)
                elif expr.op == 'umod':
                    res = bvurem(res, arg)
                elif expr.op == '&':
                    res = bvand(res, arg)
                elif expr.op == '^':
                    res = bvxor(res, arg)
                elif expr.op == '|':
                    res = bvor(res, arg)
                elif expr.op == '<<':
                    res = bvshl(res, arg)
                elif expr.op == '>>':
                    res = bvlshr(res, arg)
                elif expr.op == 'a>>':
                    res = bvashr(res, arg)
                elif expr.op == '<<<':
                    res = bv_rotate_left(res, arg, expr.size)
                elif expr.op == '>>>':
                    res = bv_rotate_right(res, arg, expr.size)
                elif expr.op == '==':
                    res = self.from_expr(ExprCond(expr.args[0] - expr.args[1], ExprInt(0, 1), ExprInt(1, 1)))
                else:
                    raise NotImplementedError('Unsupported OP yet: %s' % expr.op)
        elif expr.op == 'parity':
            arg = bv_extract(7, 0, res)
            res = bit_vec_val(1, 1)
            for i in range(8):
                res = bvxor(res, bv_extract(i, i, arg))
        elif expr.op == '-':
            res = bvneg(res)
        elif expr.op == 'cnttrailzeros':
            src = res
            size = expr.size
            size_smt2 = bit_vec_val(size, size)
            one_smt2 = bit_vec_val(1, size)
            zero_smt2 = bit_vec_val(0, size)
            op = bvand(src, bvshl(one_smt2, bvsub(size_smt2, one_smt2)))
            cond = smt2_distinct(op, zero_smt2)
            res = smt2_ite(cond, bvsub(size_smt2, one_smt2), src)
            for i in range(size - 2, -1, -1):
                i_smt2 = bit_vec_val(i, size)
                op = bvand(src, bvshl(one_smt2, i_smt2))
                cond = smt2_distinct(op, zero_smt2)
                res = smt2_ite(cond, i_smt2, res)
        elif expr.op == 'cntleadzeros':
            src = res
            size = expr.size
            one_smt2 = bit_vec_val(1, size)
            zero_smt2 = bit_vec_val(0, size)
            cond = smt2_distinct(bvand(src, one_smt2), zero_smt2)
            res = smt2_ite(cond, zero_smt2, src)
            for i in range(size - 1, 0, -1):
                index = -i % size
                index_smt2 = bit_vec_val(index, size)
                op = bvand(src, bvshl(one_smt2, index_smt2))
                cond = smt2_distinct(op, zero_smt2)
                value_smt2 = bit_vec_val(size - (index + 1), size)
                res = smt2_ite(cond, value_smt2, res)
        else:
            raise NotImplementedError('Unsupported OP yet: %s' % expr.op)
        return res

    def from_ExprAssign(self, expr):
        if False:
            for i in range(10):
                print('nop')
        src = self.from_expr(expr.src)
        dst = self.from_expr(expr.dst)
        return smt2_assert(smt2_eq(src, dst))

    def to_smt2(self, exprs, logic='QF_ABV', model=False):
        if False:
            print('Hello World!')
        '\n        Converts a valid SMT2 file for a given list of\n        SMT2 expressions.\n\n        :param exprs: list of SMT2 expressions\n        :param logic: SMT2 logic\n        :param model: model generation flag\n        :return: String of the SMT2 file\n        '
        ret = ''
        ret += '(set-logic {})\n'.format(logic)
        for bv in self._bitvectors:
            size = self._bitvectors[bv]
            ret += '{}\n'.format(declare_bv(bv, size))
        for size in self._mem.mems:
            mem = self._mem.mems[size]
            ret += '{}\n'.format(declare_array(mem, bit_vec(size), bit_vec(8)))
        for expr in exprs:
            ret += expr + '\n'
        ret += '(check-sat)\n'
        if model:
            ret += '(get-model)\n'
        return ret
Translator.register(TranslatorSMT2)