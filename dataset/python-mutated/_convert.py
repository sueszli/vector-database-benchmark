import decimal
import functools
import math
from vyper import ast as vy_ast
from vyper.codegen.core import LOAD, IRnode, bytes_clamp, bytes_data_ptr, clamp, clamp_basetype, get_bytearray_length, int_clamp, is_bytes_m_type, is_decimal_type, is_enum_type, is_integer_type, sar, shl, shr, unwrap_location
from vyper.codegen.expr import Expr
from vyper.exceptions import CompilerPanic, InvalidLiteral, InvalidType, StructureException, TypeMismatch
from vyper.semantics.types import AddressT, BoolT, BytesM_T, BytesT, DecimalT, EnumT, IntegerT, StringT
from vyper.semantics.types.bytestrings import _BytestringT
from vyper.semantics.types.shortcuts import INT256_T, UINT160_T, UINT256_T
from vyper.utils import DECIMAL_DIVISOR, round_towards_zero, unsigned_to_signed

def _FAIL(ityp, otyp, source_expr=None):
    if False:
        print('Hello World!')
    raise TypeMismatch(f"Can't convert {ityp} to {otyp}", source_expr)

def _input_types(*allowed_types):
    if False:
        print('Hello World!')

    def decorator(f):
        if False:
            while True:
                i = 10

        @functools.wraps(f)
        def check_input_type(expr, arg, out_typ):
            if False:
                return 10
            ok = isinstance(arg.typ, allowed_types)
            if not ok:
                _FAIL(arg.typ, out_typ, expr)
            if arg.typ == out_typ and arg.typ not in (UINT256_T, INT256_T):
                raise InvalidType(f'value and target are both {out_typ}', expr)
            return f(expr, arg, out_typ)
        return check_input_type
    return decorator

def _bytes_to_num(arg, out_typ, signed):
    if False:
        while True:
            i = 10
    if isinstance(arg.typ, _BytestringT):
        _len = get_bytearray_length(arg)
        arg = LOAD(bytes_data_ptr(arg))
        num_zero_bits = ['mul', 8, ['sub', 32, _len]]
    elif is_bytes_m_type(arg.typ):
        num_zero_bits = 8 * (32 - arg.typ.m)
    else:
        raise CompilerPanic('unreachable')
    if signed:
        ret = sar(num_zero_bits, arg)
    else:
        ret = shr(num_zero_bits, arg)
    annotation = (f'__intrinsic__byte_array_to_num({out_typ})',)
    return IRnode.from_list(ret, annotation=annotation)

def _clamp_numeric_convert(arg, arg_bounds, out_bounds, arg_is_signed):
    if False:
        for i in range(10):
            print('nop')
    (arg_lo, arg_hi) = arg_bounds
    (out_lo, out_hi) = out_bounds
    if arg_lo < out_lo:
        assert arg_is_signed, 'bad assumption in numeric convert'
        arg = clamp('sge', arg, out_lo)
    if arg_hi > out_hi:
        assert out_hi < 2 ** 256 - 1, 'bad assumption in numeric convert'
        CLAMP_OP = 'sle' if arg_is_signed else 'le'
        arg = clamp(CLAMP_OP, arg, out_hi)
    return arg

def _fixed_to_int(arg, out_typ):
    if False:
        while True:
            i = 10
    DIVISOR = arg.typ.divisor
    (out_lo, out_hi) = out_typ.int_bounds
    out_lo *= DIVISOR
    out_hi *= DIVISOR
    arg_bounds = arg.typ.int_bounds
    clamped_arg = _clamp_numeric_convert(arg, arg_bounds, (out_lo, out_hi), arg.typ.is_signed)
    assert arg.typ.is_signed, 'should use unsigned div'
    return IRnode.from_list(['sdiv', clamped_arg, DIVISOR], typ=out_typ)

def _int_to_fixed(arg, out_typ):
    if False:
        while True:
            i = 10
    DIVISOR = out_typ.divisor
    (out_lo, out_hi) = out_typ.int_bounds
    out_lo = round_towards_zero(out_lo / decimal.Decimal(DIVISOR))
    out_hi = round_towards_zero(out_hi / decimal.Decimal(DIVISOR))
    arg_bounds = arg.typ.int_bounds
    clamped_arg = _clamp_numeric_convert(arg, arg_bounds, (out_lo, out_hi), arg.typ.is_signed)
    return IRnode.from_list(['mul', clamped_arg, DIVISOR], typ=out_typ)

def _int_to_int(arg, out_typ):
    if False:
        print('Hello World!')
    if arg.typ.is_signed and (not out_typ.is_signed):
        if out_typ.bits < arg.typ.bits:
            assert out_typ.bits < 256, 'unreachable'
            arg = int_clamp(arg, out_typ.bits, signed=False)
        else:
            arg = clamp('sge', arg, 0)
    elif not arg.typ.is_signed and out_typ.is_signed:
        arg = int_clamp(arg, out_typ.bits - 1, signed=False)
    elif out_typ.bits < arg.typ.bits:
        assert out_typ.bits < 256, 'unreachable'
        arg = int_clamp(arg, out_typ.bits, out_typ.is_signed)
    else:
        assert arg.typ.is_signed == out_typ.is_signed and out_typ.bits >= arg.typ.bits
    return IRnode.from_list(arg, typ=out_typ)

def _check_bytes(expr, arg, output_type, max_bytes_allowed):
    if False:
        while True:
            i = 10
    if isinstance(arg.typ, _BytestringT):
        if arg.typ.maxlen > max_bytes_allowed:
            _FAIL(arg.typ, output_type, expr)
    else:
        assert output_type.memory_bytes_required == 32

def _signextend(expr, val, arg_typ):
    if False:
        print('Hello World!')
    if isinstance(expr, vy_ast.Hex):
        assert len(expr.value[2:]) // 2 == arg_typ.m
        n_bits = arg_typ.m_bits
    else:
        assert len(expr.value) == arg_typ.maxlen
        n_bits = arg_typ.maxlen * 8
    return unsigned_to_signed(val, n_bits)

def _literal_int(expr, arg_typ, out_typ):
    if False:
        print('Hello World!')
    if isinstance(expr, vy_ast.Hex):
        val = int(expr.value, 16)
    elif isinstance(expr, vy_ast.Bytes):
        val = int.from_bytes(expr.value, 'big')
    elif isinstance(expr, (vy_ast.Int, vy_ast.Decimal, vy_ast.NameConstant)):
        val = expr.value
    else:
        raise CompilerPanic('unreachable')
    if isinstance(expr, (vy_ast.Hex, vy_ast.Bytes)) and out_typ.is_signed:
        val = _signextend(expr, val, arg_typ)
    (lo, hi) = out_typ.int_bounds
    if not lo <= val <= hi:
        raise InvalidLiteral('Number out of range', expr)
    val = int(val)
    return IRnode.from_list(val, typ=out_typ)

def _literal_decimal(expr, arg_typ, out_typ):
    if False:
        print('Hello World!')
    if isinstance(expr, vy_ast.Hex):
        val = decimal.Decimal(int(expr.value, 16))
    else:
        val = decimal.Decimal(expr.value)
        val *= DECIMAL_DIVISOR
    assert math.ceil(val) == math.floor(val)
    val = int(val)
    if isinstance(expr, (vy_ast.Hex, vy_ast.Bytes)) and out_typ.is_signed:
        val = _signextend(expr, val, arg_typ)
    (lo, hi) = out_typ.int_bounds
    if not lo <= val <= hi:
        raise InvalidLiteral('Number out of range', expr)
    return IRnode.from_list(val, typ=out_typ)

@_input_types(IntegerT, DecimalT, BytesM_T, AddressT, BoolT, BytesT, StringT)
def to_bool(expr, arg, out_typ):
    if False:
        for i in range(10):
            print('nop')
    _check_bytes(expr, arg, out_typ, 32)
    if isinstance(arg.typ, _BytestringT):
        arg = _bytes_to_num(arg, out_typ, signed=False)
    return IRnode.from_list(['iszero', ['iszero', arg]], typ=out_typ)

@_input_types(IntegerT, DecimalT, BytesM_T, AddressT, BoolT, EnumT, BytesT)
def to_int(expr, arg, out_typ):
    if False:
        return 10
    return _to_int(expr, arg, out_typ)

def _to_int(expr, arg, out_typ):
    if False:
        print('Hello World!')
    assert out_typ.bits % 8 == 0
    _check_bytes(expr, arg, out_typ, 32)
    if isinstance(expr, vy_ast.Constant):
        return _literal_int(expr, arg.typ, out_typ)
    elif isinstance(arg.typ, BytesT):
        arg_typ = arg.typ
        arg = _bytes_to_num(arg, out_typ, signed=out_typ.is_signed)
        if arg_typ.maxlen * 8 > out_typ.bits:
            arg = int_clamp(arg, out_typ.bits, signed=out_typ.is_signed)
    elif is_bytes_m_type(arg.typ):
        arg_typ = arg.typ
        arg = _bytes_to_num(arg, out_typ, signed=out_typ.is_signed)
        if arg_typ.m_bits > out_typ.bits:
            arg = int_clamp(arg, out_typ.bits, signed=out_typ.is_signed)
    elif is_decimal_type(arg.typ):
        arg = _fixed_to_int(arg, out_typ)
    elif is_enum_type(arg.typ):
        if out_typ != UINT256_T:
            _FAIL(arg.typ, out_typ, expr)
        arg = IRnode.from_list(arg, typ=UINT256_T)
        arg = _int_to_int(arg, out_typ)
    elif is_integer_type(arg.typ):
        arg = _int_to_int(arg, out_typ)
    elif arg.typ == AddressT():
        if out_typ.is_signed:
            _FAIL(arg.typ, out_typ, expr)
        if out_typ.bits < 160:
            arg = int_clamp(arg, out_typ.bits, signed=False)
    return IRnode.from_list(arg, typ=out_typ)

@_input_types(IntegerT, BoolT, BytesM_T, BytesT)
def to_decimal(expr, arg, out_typ):
    if False:
        return 10
    _check_bytes(expr, arg, out_typ, 32)
    if isinstance(expr, vy_ast.Constant):
        return _literal_decimal(expr, arg.typ, out_typ)
    if isinstance(arg.typ, BytesT):
        arg_typ = arg.typ
        arg = _bytes_to_num(arg, out_typ, signed=True)
        if arg_typ.maxlen * 8 > 168:
            arg = IRnode.from_list(arg, typ=out_typ)
            arg = clamp_basetype(arg)
        return IRnode.from_list(arg, typ=out_typ)
    elif is_bytes_m_type(arg.typ):
        arg_typ = arg.typ
        arg = _bytes_to_num(arg, out_typ, signed=True)
        if arg_typ.m_bits > 168:
            arg = IRnode.from_list(arg, typ=out_typ)
            arg = clamp_basetype(arg)
        return IRnode.from_list(arg, typ=out_typ)
    elif is_integer_type(arg.typ):
        arg = _int_to_fixed(arg, out_typ)
        return IRnode.from_list(arg, typ=out_typ)
    elif arg.typ == BoolT():
        arg = ['mul', arg, 10 ** out_typ.decimals]
        return IRnode.from_list(arg, typ=out_typ)
    else:
        raise CompilerPanic('unreachable')

@_input_types(IntegerT, DecimalT, BytesM_T, AddressT, BytesT, BoolT)
def to_bytes_m(expr, arg, out_typ):
    if False:
        i = 10
        return i + 15
    _check_bytes(expr, arg, out_typ, max_bytes_allowed=out_typ.m)
    if isinstance(arg.typ, BytesT):
        bytes_val = LOAD(bytes_data_ptr(arg))
        len_ = get_bytearray_length(arg)
        num_zero_bits = IRnode.from_list(['mul', ['sub', 32, len_], 8])
        with num_zero_bits.cache_when_complex('bits') as (b, num_zero_bits):
            arg = shl(num_zero_bits, shr(num_zero_bits, bytes_val))
            arg = b.resolve(arg)
    elif is_bytes_m_type(arg.typ):
        if arg.typ.m > out_typ.m:
            arg = bytes_clamp(arg, out_typ.m)
    elif is_integer_type(arg.typ) or arg.typ == AddressT():
        if arg.typ == AddressT():
            int_bits = 160
        else:
            int_bits = arg.typ.bits
        if out_typ.m_bits < int_bits:
            _FAIL(arg.typ, out_typ, expr)
        arg = shl(256 - out_typ.m_bits, arg)
    elif is_decimal_type(arg.typ):
        if out_typ.m_bits < arg.typ.bits:
            _FAIL(arg.typ, out_typ, expr)
        arg = shl(256 - out_typ.m_bits, arg)
    else:
        arg = shl(256 - out_typ.m_bits, arg)
    return IRnode.from_list(arg, typ=out_typ)

@_input_types(BytesM_T, IntegerT, BytesT)
def to_address(expr, arg, out_typ):
    if False:
        while True:
            i = 10
    if is_integer_type(arg.typ):
        if arg.typ.is_signed:
            _FAIL(arg.typ, out_typ, expr)
    ret = _to_int(expr, arg, UINT160_T)
    return IRnode.from_list(ret, out_typ)

@_input_types(BytesT)
def to_string(expr, arg, out_typ):
    if False:
        i = 10
        return i + 15
    _check_bytes(expr, arg, out_typ, out_typ.maxlen)
    return IRnode.from_list(arg, typ=out_typ)

@_input_types(StringT)
def to_bytes(expr, arg, out_typ):
    if False:
        while True:
            i = 10
    _check_bytes(expr, arg, out_typ, out_typ.maxlen)
    return IRnode.from_list(arg, typ=out_typ)

@_input_types(IntegerT)
def to_enum(expr, arg, out_typ):
    if False:
        for i in range(10):
            print('nop')
    if arg.typ != UINT256_T:
        _FAIL(arg.typ, out_typ, expr)
    if len(out_typ._enum_members) < 256:
        arg = int_clamp(arg, bits=len(out_typ._enum_members), signed=False)
    return IRnode.from_list(arg, typ=out_typ)

def convert(expr, context):
    if False:
        print('Hello World!')
    assert len(expr.args) == 2, 'bad typecheck: convert'
    arg_ast = expr.args[0]
    arg = Expr(arg_ast, context).ir_node
    original_arg = arg
    out_typ = expr.args[1]._metadata['type'].typedef
    if arg.typ._is_prim_word:
        arg = unwrap_location(arg)
    with arg.cache_when_complex('arg') as (b, arg):
        if out_typ == BoolT():
            ret = to_bool(arg_ast, arg, out_typ)
        elif out_typ == AddressT():
            ret = to_address(arg_ast, arg, out_typ)
        elif is_enum_type(out_typ):
            ret = to_enum(arg_ast, arg, out_typ)
        elif is_integer_type(out_typ):
            ret = to_int(arg_ast, arg, out_typ)
        elif is_bytes_m_type(out_typ):
            ret = to_bytes_m(arg_ast, arg, out_typ)
        elif is_decimal_type(out_typ):
            ret = to_decimal(arg_ast, arg, out_typ)
        elif isinstance(out_typ, BytesT):
            ret = to_bytes(arg_ast, arg, out_typ)
        elif isinstance(out_typ, StringT):
            ret = to_string(arg_ast, arg, out_typ)
        else:
            raise StructureException(f'Conversion to {out_typ} is invalid.', arg_ast)
        test_arg = IRnode.from_list(arg, typ=out_typ)
        if test_arg == ret:
            original_arg.typ = out_typ
            return original_arg
        return IRnode.from_list(b.resolve(ret))