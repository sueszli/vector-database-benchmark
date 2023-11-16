def smt2_eq(a, b):
    if False:
        print('Hello World!')
    '\n    Assignment: a = b\n    '
    return '(= {} {})'.format(a, b)

def smt2_implies(a, b):
    if False:
        i = 10
        return i + 15
    '\n    Implication: a => b\n    '
    return '(=> {} {})'.format(a, b)

def smt2_and(*args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Conjunction: a and b and c ...\n    '
    args = [str(arg) for arg in args]
    return '(and {})'.format(' '.join(args))

def smt2_or(*args):
    if False:
        i = 10
        return i + 15
    '\n    Disjunction: a or b or c ...\n    '
    args = [str(arg) for arg in args]
    return '(or {})'.format(' '.join(args))

def smt2_ite(cond, a, b):
    if False:
        print('Hello World!')
    '\n    If-then-else: cond ? a : b\n    '
    return '(ite {} {} {})'.format(cond, a, b)

def smt2_distinct(*args):
    if False:
        i = 10
        return i + 15
    '\n    Distinction: a != b != c != ...\n    '
    args = [str(arg) for arg in args]
    return '(distinct {})'.format(' '.join(args))

def smt2_assert(expr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Assertion that @expr holds\n    '
    return '(assert {})'.format(expr)

def declare_bv(bv, size):
    if False:
        return 10
    '\n    Declares an bit vector @bv of size @size\n    '
    return '(declare-fun {} () {})'.format(bv, bit_vec(size))

def declare_array(a, bv1, bv2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Declares an SMT2 array represented as a map\n    from a bit vector to another bit vector.\n    :param a: array name\n    :param bv1: SMT2 bit vector\n    :param bv2: SMT2 bit vector\n    '
    return '(declare-fun {} () (Array {} {}))'.format(a, bv1, bv2)

def bit_vec_val(v, size):
    if False:
        for i in range(10):
            print('nop')
    '\n    Declares a bit vector value\n    :param v: int, value of the bit vector\n    :param size: size of the bit vector\n    '
    return '(_ bv{} {})'.format(v, size)

def bit_vec(size):
    if False:
        return 10
    '\n    Returns a bit vector of size @size\n    '
    return '(_ BitVec {})'.format(size)

def bvadd(a, b):
    if False:
        while True:
            i = 10
    '\n    Addition: a + b\n    '
    return '(bvadd {} {})'.format(a, b)

def bvsub(a, b):
    if False:
        return 10
    '\n    Subtraction: a - b\n    '
    return '(bvsub {} {})'.format(a, b)

def bvmul(a, b):
    if False:
        print('Hello World!')
    '\n    Multiplication: a * b\n    '
    return '(bvmul {} {})'.format(a, b)

def bvand(a, b):
    if False:
        return 10
    '\n    Bitwise AND: a & b\n    '
    return '(bvand {} {})'.format(a, b)

def bvor(a, b):
    if False:
        return 10
    '\n    Bitwise OR: a | b\n    '
    return '(bvor {} {})'.format(a, b)

def bvxor(a, b):
    if False:
        return 10
    '\n    Bitwise XOR: a ^ b\n    '
    return '(bvxor {} {})'.format(a, b)

def bvneg(bv):
    if False:
        i = 10
        return i + 15
    '\n    Unary minus: - bv\n    '
    return '(bvneg {})'.format(bv)

def bvsdiv(a, b):
    if False:
        i = 10
        return i + 15
    '\n    Signed division: a / b\n    '
    return '(bvsdiv {} {})'.format(a, b)

def bvudiv(a, b):
    if False:
        return 10
    '\n    Unsigned division: a / b\n    '
    return '(bvudiv {} {})'.format(a, b)

def bvsmod(a, b):
    if False:
        while True:
            i = 10
    '\n    Signed modulo: a mod b\n    '
    return '(bvsmod {} {})'.format(a, b)

def bvurem(a, b):
    if False:
        return 10
    '\n    Unsigned modulo: a mod b\n    '
    return '(bvurem {} {})'.format(a, b)

def bvshl(a, b):
    if False:
        print('Hello World!')
    '\n    Shift left: a << b\n    '
    return '(bvshl {} {})'.format(a, b)

def bvlshr(a, b):
    if False:
        return 10
    '\n    Logical shift right: a >> b\n    '
    return '(bvlshr {} {})'.format(a, b)

def bvashr(a, b):
    if False:
        i = 10
        return i + 15
    '\n    Arithmetic shift right: a a>> b\n    '
    return '(bvashr {} {})'.format(a, b)

def bv_rotate_left(a, b, size):
    if False:
        print('Hello World!')
    '\n    Rotates bits of a to the left b times: a <<< b\n\n    Since ((_ rotate_left b) a) does not support\n    symbolic values for b, the implementation is\n    based on a C implementation.\n\n    Therefore, the rotation will be computed as\n    a << (b & (size - 1))) | (a >> (size - (b & (size - 1))))\n\n    :param a: bit vector\n    :param b: bit vector\n    :param size: size of a\n    '
    s = bit_vec_val(size, size)
    shift = bvand(b, bvsub(s, bit_vec_val(1, size)))
    rotate = bvor(bvshl(a, shift), bvlshr(a, bvsub(s, shift)))
    return rotate

def bv_rotate_right(a, b, size):
    if False:
        while True:
            i = 10
    '\n    Rotates bits of a to the right b times: a >>> b\n\n    Since ((_ rotate_right b) a) does not support\n    symbolic values for b, the implementation is\n    based on a C implementation.\n\n    Therefore, the rotation will be computed as\n    a >> (b & (size - 1))) | (a << (size - (b & (size - 1))))\n\n    :param a: bit vector\n    :param b: bit vector\n    :param size: size of a\n    '
    s = bit_vec_val(size, size)
    shift = bvand(b, bvsub(s, bit_vec_val(1, size)))
    rotate = bvor(bvlshr(a, shift), bvshl(a, bvsub(s, shift)))
    return rotate

def bv_extract(high, low, bv):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extracts bits from a bit vector\n    :param high: end bit\n    :param low: start bit\n    :param bv: bit vector\n    '
    return '((_ extract {} {}) {})'.format(high, low, bv)

def bv_concat(a, b):
    if False:
        i = 10
        return i + 15
    '\n    Concatenation of two SMT2 expressions\n    '
    return '(concat {} {})'.format(a, b)

def array_select(array, index):
    if False:
        while True:
            i = 10
    '\n    Reads from an SMT2 array at index @index\n    :param array: SMT2 array\n    :param index: SMT2 expression, index of the array\n    '
    return '(select {} {})'.format(array, index)

def array_store(array, index, value):
    if False:
        return 10
    '\n    Writes an value into an SMT2 array at address @index\n    :param array: SMT array\n    :param index: SMT2 expression, index of the array\n    :param value: SMT2 expression, value to write\n    '
    return '(store {} {} {})'.format(array, index, value)