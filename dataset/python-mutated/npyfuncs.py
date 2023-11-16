"""Codegen for functions used as kernels in NumPy functions

Typically, the kernels of several ufuncs that can't map directly to
Python builtins
"""
import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
_NPY_LOG2E = 1.4426950408889634
_NPY_LOG10E = 0.4342944819032518
_NPY_LOGE2 = 0.6931471805599453

def _check_arity_and_homogeneity(sig, args, arity, return_type=None):
    if False:
        i = 10
        return i + 15
    "checks that the following are true:\n    - args and sig.args have arg_count elements\n    - all input types are homogeneous\n    - return type is 'return_type' if provided, otherwise it must be\n      homogeneous with the input types.\n    "
    assert len(args) == arity
    assert len(sig.args) == arity
    ty = sig.args[0]
    if return_type is None:
        return_type = ty
    if not (all((arg == ty for arg in sig.args)) and sig.return_type == return_type):
        import inspect
        fname = inspect.currentframe().f_back.f_code.co_name
        msg = '{0} called with invalid types: {1}'.format(fname, sig)
        assert False, msg

def _call_func_by_name_with_cast(context, builder, sig, args, func_name, ty=types.float64):
    if False:
        print('Hello World!')
    mod = builder.module
    lty = context.get_argument_type(ty)
    fnty = llvmlite.ir.FunctionType(lty, [lty] * len(sig.args))
    fn = cgutils.insert_pure_function(mod, fnty, name=func_name)
    cast_args = [context.cast(builder, arg, argty, ty) for (arg, argty) in zip(args, sig.args)]
    result = builder.call(fn, cast_args)
    return context.cast(builder, result, types.float64, sig.return_type)

def _dispatch_func_by_name_type(context, builder, sig, args, table, user_name):
    if False:
        while True:
            i = 10
    ty = sig.args[0]
    try:
        func_name = table[ty]
    except KeyError as e:
        msg = 'No {0} function for real type {1}'.format(user_name, str(e))
        raise errors.LoweringError(msg)
    mod = builder.module
    if ty in types.complex_domain:
        out = context.make_complex(builder, ty)
        ptrargs = [cgutils.alloca_once_value(builder, arg) for arg in args]
        call_args = [out._getpointer()] + ptrargs
        call_argtys = [ty] + list(sig.args)
        call_argltys = [context.get_value_type(ty).as_pointer() for ty in call_argtys]
        fnty = llvmlite.ir.FunctionType(llvmlite.ir.VoidType(), call_argltys)
        fn = cgutils.get_or_insert_function(mod, fnty, func_name)
        builder.call(fn, call_args)
        retval = builder.load(call_args[0])
    else:
        argtypes = [context.get_argument_type(aty) for aty in sig.args]
        restype = context.get_argument_type(sig.return_type)
        fnty = llvmlite.ir.FunctionType(restype, argtypes)
        fn = cgutils.insert_pure_function(mod, fnty, name=func_name)
        retval = context.call_external_function(builder, fn, sig.args, args)
    return retval

def np_int_sdiv_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 2)
    (num, den) = args
    ty = sig.args[0]
    ZERO = context.get_constant(ty, 0)
    MINUS_ONE = context.get_constant(ty, -1)
    MIN_INT = context.get_constant(ty, 1 << den.type.width - 1)
    den_is_zero = builder.icmp_unsigned('==', ZERO, den)
    den_is_minus_one = builder.icmp_unsigned('==', MINUS_ONE, den)
    num_is_min_int = builder.icmp_unsigned('==', MIN_INT, num)
    could_cause_sigfpe = builder.and_(den_is_minus_one, num_is_min_int)
    force_zero = builder.or_(den_is_zero, could_cause_sigfpe)
    with builder.if_else(force_zero, likely=False) as (then, otherwise):
        with then:
            bb_then = builder.basic_block
        with otherwise:
            bb_otherwise = builder.basic_block
            div = builder.sdiv(num, den)
            mod = builder.srem(num, den)
            num_gt_zero = builder.icmp_signed('>', num, ZERO)
            den_gt_zero = builder.icmp_signed('>', den, ZERO)
            not_same_sign = builder.xor(num_gt_zero, den_gt_zero)
            mod_not_zero = builder.icmp_unsigned('!=', mod, ZERO)
            needs_fixing = builder.and_(not_same_sign, mod_not_zero)
            fix_value = builder.select(needs_fixing, MINUS_ONE, ZERO)
            result_otherwise = builder.add(div, fix_value)
    result = builder.phi(ZERO.type)
    result.add_incoming(ZERO, bb_then)
    result.add_incoming(result_otherwise, bb_otherwise)
    return result

def np_int_srem_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    (num, den) = args
    ty = sig.args[0]
    ZERO = context.get_constant(ty, 0)
    den_not_zero = builder.icmp_unsigned('!=', ZERO, den)
    bb_no_if = builder.basic_block
    with cgutils.if_unlikely(builder, den_not_zero):
        bb_if = builder.basic_block
        mod = builder.srem(num, den)
        num_gt_zero = builder.icmp_signed('>', num, ZERO)
        den_gt_zero = builder.icmp_signed('>', den, ZERO)
        not_same_sign = builder.xor(num_gt_zero, den_gt_zero)
        mod_not_zero = builder.icmp_unsigned('!=', mod, ZERO)
        needs_fixing = builder.and_(not_same_sign, mod_not_zero)
        fix_value = builder.select(needs_fixing, den, ZERO)
        final_mod = builder.add(fix_value, mod)
    result = builder.phi(ZERO.type)
    result.add_incoming(ZERO, bb_no_if)
    result.add_incoming(final_mod, bb_if)
    return result

def np_int_sdivrem_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    div = np_int_sdiv_impl(context, builder, sig.return_type[0](*sig.args), args)
    rem = np_int_srem_impl(context, builder, sig.return_type[1](*sig.args), args)
    return context.make_tuple(builder, sig.return_type, [div, rem])

def np_int_udiv_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 2)
    (num, den) = args
    ty = sig.args[0]
    ZERO = context.get_constant(ty, 0)
    div_by_zero = builder.icmp_unsigned('==', ZERO, den)
    with builder.if_else(div_by_zero, likely=False) as (then, otherwise):
        with then:
            bb_then = builder.basic_block
        with otherwise:
            div = builder.udiv(num, den)
            bb_otherwise = builder.basic_block
    result = builder.phi(ZERO.type)
    result.add_incoming(ZERO, bb_then)
    result.add_incoming(div, bb_otherwise)
    return result

def np_int_urem_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    (num, den) = args
    ty = sig.args[0]
    ZERO = context.get_constant(ty, 0)
    den_not_zero = builder.icmp_unsigned('!=', ZERO, den)
    bb_no_if = builder.basic_block
    with cgutils.if_unlikely(builder, den_not_zero):
        bb_if = builder.basic_block
        mod = builder.urem(num, den)
    result = builder.phi(ZERO.type)
    result.add_incoming(ZERO, bb_no_if)
    result.add_incoming(mod, bb_if)
    return result

def np_int_udivrem_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    div = np_int_udiv_impl(context, builder, sig.return_type[0](*sig.args), args)
    rem = np_int_urem_impl(context, builder, sig.return_type[1](*sig.args), args)
    return context.make_tuple(builder, sig.return_type, [div, rem])
np_int_fmod_impl = np_int_urem_impl

def np_real_div_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 2)
    return builder.fdiv(*args)

def np_real_mod_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    (in1, in2) = args
    ty = sig.args[0]
    ZERO = context.get_constant(ty, 0.0)
    res = builder.frem(in1, in2)
    res_ne_zero = builder.fcmp_ordered('!=', res, ZERO)
    den_lt_zero = builder.fcmp_ordered('<', in2, ZERO)
    res_lt_zero = builder.fcmp_ordered('<', res, ZERO)
    needs_fixing = builder.and_(res_ne_zero, builder.xor(den_lt_zero, res_lt_zero))
    fix_value = builder.select(needs_fixing, in2, ZERO)
    return builder.fadd(res, fix_value)

def np_real_fmod_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    return builder.frem(*args)

def _fabs(context, builder, arg):
    if False:
        for i in range(10):
            print('nop')
    ZERO = llvmlite.ir.Constant(arg.type, 0.0)
    arg_negated = builder.fsub(ZERO, arg)
    arg_is_negative = builder.fcmp_ordered('<', arg, ZERO)
    return builder.select(arg_is_negative, arg_negated, arg)

def np_complex_div_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    (in1, in2) = [context.make_complex(builder, sig.args[0], value=arg) for arg in args]
    in1r = in1.real
    in1i = in1.imag
    in2r = in2.real
    in2i = in2.imag
    ftype = in1r.type
    assert all([i.type == ftype for i in [in1r, in1i, in2r, in2i]]), 'mismatched types'
    out = context.make_helper(builder, sig.return_type)
    ZERO = llvmlite.ir.Constant(ftype, 0.0)
    ONE = llvmlite.ir.Constant(ftype, 1.0)
    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)
    in2r_abs_ge_in2i_abs = builder.fcmp_ordered('>=', in2r_abs, in2i_abs)
    with builder.if_else(in2r_abs_ge_in2i_abs) as (then, otherwise):
        with then:
            in2r_is_zero = builder.fcmp_ordered('==', in2r_abs, ZERO)
            in2i_is_zero = builder.fcmp_ordered('==', in2i_abs, ZERO)
            in2_is_zero = builder.and_(in2r_is_zero, in2i_is_zero)
            with builder.if_else(in2_is_zero) as (inn_then, inn_otherwise):
                with inn_then:
                    out.real = builder.fdiv(in1r, in2r_abs)
                    out.imag = builder.fdiv(in1i, in2i_abs)
                with inn_otherwise:
                    rat = builder.fdiv(in2i, in2r)
                    tmp1 = builder.fmul(in2i, rat)
                    tmp2 = builder.fadd(in2r, tmp1)
                    scl = builder.fdiv(ONE, tmp2)
                    tmp3 = builder.fmul(in1i, rat)
                    tmp4 = builder.fmul(in1r, rat)
                    tmp5 = builder.fadd(in1r, tmp3)
                    tmp6 = builder.fsub(in1i, tmp4)
                    out.real = builder.fmul(tmp5, scl)
                    out.imag = builder.fmul(tmp6, scl)
        with otherwise:
            rat = builder.fdiv(in2r, in2i)
            tmp1 = builder.fmul(in2r, rat)
            tmp2 = builder.fadd(in2i, tmp1)
            scl = builder.fdiv(ONE, tmp2)
            tmp3 = builder.fmul(in1r, rat)
            tmp4 = builder.fmul(in1i, rat)
            tmp5 = builder.fadd(tmp3, in1i)
            tmp6 = builder.fsub(tmp4, in1r)
            out.real = builder.fmul(tmp5, scl)
            out.imag = builder.fmul(tmp6, scl)
    return out._getvalue()

def _npy_logaddexp(x1, x2):
    if False:
        for i in range(10):
            print('nop')
    pass

def _generate_logaddexp(fnoverload, const, log1pfn, expfn):
    if False:
        return 10

    @overload(fnoverload, target='generic')
    def ol_npy_logaddexp(x1, x2):
        if False:
            return 10
        if x1 != x2:
            return
        shift = x1(const)

        def impl(x1, x2):
            if False:
                for i in range(10):
                    print('nop')
            (x, y) = (x1, x2)
            if x == y:
                return x + shift
            else:
                tmp = x - y
                if tmp > 0:
                    return x + log1pfn(expfn(-tmp))
                elif tmp <= 0:
                    return y + log1pfn(expfn(tmp))
                else:
                    return tmp
        return impl

def _npy_logaddexp(x1, x2):
    if False:
        print('Hello World!')
    pass
_generate_logaddexp(_npy_logaddexp, _NPY_LOGE2, np.log1p, np.exp)

def np_real_logaddexp_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 2)
    fnty = context.typing_context.resolve_value_type(_npy_logaddexp)
    sig = fnty.get_call_type(context.typing_context, (*sig.args,), {})
    impl = context.get_function(fnty, sig)
    return impl(builder, args)

def _npy_logaddexp2(x1, x2):
    if False:
        for i in range(10):
            print('nop')
    pass

def npy_log2_1p(x):
    if False:
        i = 10
        return i + 15
    pass

@overload(npy_log2_1p, target='generic')
def ol_npy_log2_1p(x):
    if False:
        while True:
            i = 10
    LOG2E = x(_NPY_LOG2E)

    def impl(x):
        if False:
            return 10
        return LOG2E * np.log1p(x)
    return impl
_generate_logaddexp(_npy_logaddexp2, 1.0, npy_log2_1p, np.exp2)

def np_real_logaddexp2_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    fnty = context.typing_context.resolve_value_type(_npy_logaddexp2)
    sig = fnty.get_call_type(context.typing_context, (*sig.args,), {})
    impl = context.get_function(fnty, sig)
    return impl(builder, args)

def np_int_truediv_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    (num, den) = args
    lltype = num.type
    assert all((i.type == lltype for i in args)), 'must have homogeneous types'
    (numty, denty) = sig.args
    num = context.cast(builder, num, numty, types.float64)
    den = context.cast(builder, den, denty, types.float64)
    return builder.fdiv(num, den)

def np_real_floor_div_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    res = np_real_div_impl(context, builder, sig, args)
    s = typing.signature(sig.return_type, sig.return_type)
    return np_real_floor_impl(context, builder, s, (res,))

def np_real_divmod_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    div = np_real_floor_div_impl(context, builder, sig.return_type[0](*sig.args), args)
    rem = np_real_mod_impl(context, builder, sig.return_type[1](*sig.args), args)
    return context.make_tuple(builder, sig.return_type, [div, rem])

def np_complex_floor_div_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    float_kind = sig.args[0].underlying_float
    floor_sig = typing.signature(float_kind, float_kind)
    (in1, in2) = [context.make_complex(builder, sig.args[0], value=arg) for arg in args]
    in1r = in1.real
    in1i = in1.imag
    in2r = in2.real
    in2i = in2.imag
    ftype = in1r.type
    assert all([i.type == ftype for i in [in1r, in1i, in2r, in2i]]), 'mismatched types'
    ZERO = llvmlite.ir.Constant(ftype, 0.0)
    out = context.make_helper(builder, sig.return_type)
    out.imag = ZERO
    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)
    in2r_abs_ge_in2i_abs = builder.fcmp_ordered('>=', in2r_abs, in2i_abs)
    with builder.if_else(in2r_abs_ge_in2i_abs) as (then, otherwise):
        with then:
            rat = builder.fdiv(in2i, in2r)
            tmp1 = builder.fmul(in1i, rat)
            tmp2 = builder.fmul(in2i, rat)
            tmp3 = builder.fadd(in1r, tmp1)
            tmp4 = builder.fadd(in2r, tmp2)
            tmp5 = builder.fdiv(tmp3, tmp4)
            out.real = np_real_floor_impl(context, builder, floor_sig, (tmp5,))
        with otherwise:
            rat = builder.fdiv(in2r, in2i)
            tmp1 = builder.fmul(in1r, rat)
            tmp2 = builder.fmul(in2r, rat)
            tmp3 = builder.fadd(in1i, tmp1)
            tmp4 = builder.fadd(in2i, tmp2)
            tmp5 = builder.fdiv(tmp3, tmp4)
            out.real = np_real_floor_impl(context, builder, floor_sig, (tmp5,))
    return out._getvalue()

def np_complex_power_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    return numbers.complex_power_impl(context, builder, sig, args)

def real_float_power_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    return numbers.real_power_impl(context, builder, sig, args)

def np_complex_float_power_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 2)
    return numbers.complex_power_impl(context, builder, sig, args)

def np_gcd_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    return mathimpl.gcd_impl(context, builder, sig, args)

def np_lcm_impl(context, builder, sig, args):
    if False:
        return 10
    (xty, yty) = sig.args
    assert xty == yty == sig.return_type
    (x, y) = args

    def lcm(a, b):
        if False:
            while True:
                i = 10
        '\n        Like gcd, heavily cribbed from Julia.\n        '
        return 0 if a == 0 else abs(a * (b // np.gcd(b, a)))
    res = context.compile_internal(builder, lcm, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

def np_complex_sign_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    op = args[0]
    ty = sig.args[0]
    float_ty = ty.underlying_float
    ZERO = context.get_constant(float_ty, 0.0)
    ONE = context.get_constant(float_ty, 1.0)
    MINUS_ONE = context.get_constant(float_ty, -1.0)
    NAN = context.get_constant(float_ty, float('nan'))
    result = context.make_complex(builder, ty)
    result.real = ZERO
    result.imag = ZERO
    cmp_sig = typing.signature(types.boolean, *[ty] * 2)
    cmp_args = [op, result._getvalue()]
    arg1_ge_arg2 = np_complex_ge_impl(context, builder, cmp_sig, cmp_args)
    arg1_eq_arg2 = np_complex_eq_impl(context, builder, cmp_sig, cmp_args)
    arg1_lt_arg2 = np_complex_lt_impl(context, builder, cmp_sig, cmp_args)
    real_when_ge = builder.select(arg1_eq_arg2, ZERO, ONE)
    real_when_nge = builder.select(arg1_lt_arg2, MINUS_ONE, NAN)
    result.real = builder.select(arg1_ge_arg2, real_when_ge, real_when_nge)
    return result._getvalue()

def np_real_rint_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.call_fp_intrinsic(builder, 'llvm.rint', args)

def np_complex_rint_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    inner_sig = typing.signature(*[float_ty] * 2)
    out.real = np_real_rint_impl(context, builder, inner_sig, [in1.real])
    out.imag = np_real_rint_impl(context, builder, inner_sig, [in1.imag])
    return out._getvalue()

def np_real_exp_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.exp_impl(context, builder, sig, args)

def np_complex_exp_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    return cmathimpl.exp_impl(context, builder, sig, args)

def np_real_exp2_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    dispatch_table = {types.float32: 'numba_exp2f', types.float64: 'numba_exp2'}
    return _dispatch_func_by_name_type(context, builder, sig, args, dispatch_table, 'exp2')

def np_complex_exp2_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    in1 = context.make_complex(builder, ty, value=args[0])
    tmp = context.make_complex(builder, ty)
    loge2 = context.get_constant(float_ty, _NPY_LOGE2)
    tmp.real = builder.fmul(loge2, in1.real)
    tmp.imag = builder.fmul(loge2, in1.imag)
    return np_complex_exp_impl(context, builder, sig, [tmp._getvalue()])

def np_real_log_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.log_impl(context, builder, sig, args)

def np_complex_log_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    return cmathimpl.log_impl(context, builder, sig, args)

def np_real_log2_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    dispatch_table = {types.float32: 'numba_log2f', types.float64: 'numba_log2'}
    return _dispatch_func_by_name_type(context, builder, sig, args, dispatch_table, 'log2')

def np_complex_log2_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    tmp = np_complex_log_impl(context, builder, sig, args)
    tmp = context.make_complex(builder, ty, value=tmp)
    log2e = context.get_constant(float_ty, _NPY_LOG2E)
    tmp.real = builder.fmul(log2e, tmp.real)
    tmp.imag = builder.fmul(log2e, tmp.imag)
    return tmp._getvalue()

def np_real_log10_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.log10_impl(context, builder, sig, args)

def np_complex_log10_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    tmp = np_complex_log_impl(context, builder, sig, args)
    tmp = context.make_complex(builder, ty, value=tmp)
    log10e = context.get_constant(float_ty, _NPY_LOG10E)
    tmp.real = builder.fmul(log10e, tmp.real)
    tmp.imag = builder.fmul(log10e, tmp.imag)
    return tmp._getvalue()

def np_real_expm1_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.expm1_impl(context, builder, sig, args)

def np_complex_expm1_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    float_unary_sig = typing.signature(*[float_ty] * 2)
    MINUS_ONE = context.get_constant(float_ty, -1.0)
    in1 = context.make_complex(builder, ty, value=args[0])
    a = np_real_exp_impl(context, builder, float_unary_sig, [in1.real])
    out = context.make_complex(builder, ty)
    cos_imag = np_real_cos_impl(context, builder, float_unary_sig, [in1.imag])
    sin_imag = np_real_sin_impl(context, builder, float_unary_sig, [in1.imag])
    tmp = builder.fmul(a, cos_imag)
    out.imag = builder.fmul(a, sin_imag)
    out.real = builder.fadd(tmp, MINUS_ONE)
    return out._getvalue()

def np_real_log1p_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.log1p_impl(context, builder, sig, args)

def np_complex_log1p_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    float_unary_sig = typing.signature(*[float_ty] * 2)
    float_binary_sig = typing.signature(*[float_ty] * 3)
    ONE = context.get_constant(float_ty, 1.0)
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    real_plus_one = builder.fadd(in1.real, ONE)
    l = np_real_hypot_impl(context, builder, float_binary_sig, [real_plus_one, in1.imag])
    out.imag = np_real_atan2_impl(context, builder, float_binary_sig, [in1.imag, real_plus_one])
    out.real = np_real_log_impl(context, builder, float_unary_sig, [l])
    return out._getvalue()

def np_real_sqrt_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.sqrt_impl(context, builder, sig, args)

def np_complex_sqrt_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    return cmathimpl.sqrt_impl(context, builder, sig, args)

def np_int_square_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    return builder.mul(args[0], args[0])

def np_real_square_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    return builder.fmul(args[0], args[0])

def np_complex_square_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    binary_sig = typing.signature(*[sig.return_type] * 3)
    return numbers.complex_mul_impl(context, builder, binary_sig, [args[0], args[0]])

def np_real_cbrt_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)

    @register_jitable(fastmath=True)
    def cbrt(x):
        if False:
            return 10
        if x < 0:
            return -np.power(-x, 1.0 / 3.0)
        else:
            return np.power(x, 1.0 / 3.0)

    def _cbrt(x):
        if False:
            for i in range(10):
                print('nop')
        if np.isnan(x):
            return np.nan
        return cbrt(x)
    return context.compile_internal(builder, _cbrt, sig, args)

def np_int_reciprocal_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.return_type
    binary_sig = typing.signature(*[ty] * 3)
    in_as_float = context.cast(builder, args[0], ty, types.float64)
    ONE = context.get_constant(types.float64, 1)
    result_as_float = builder.fdiv(ONE, in_as_float)
    return context.cast(builder, result_as_float, types.float64, ty)

def np_real_reciprocal_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    ONE = context.get_constant(sig.return_type, 1.0)
    return builder.fdiv(ONE, args[0])

def np_complex_reciprocal_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    float_ty = ty.underlying_float
    ZERO = context.get_constant(float_ty, 0.0)
    ONE = context.get_constant(float_ty, 1.0)
    in1 = context.make_complex(builder, ty, value=args[0])
    out = context.make_complex(builder, ty)
    in1r = in1.real
    in1i = in1.imag
    in1r_abs = _fabs(context, builder, in1r)
    in1i_abs = _fabs(context, builder, in1i)
    in1i_abs_le_in1r_abs = builder.fcmp_ordered('<=', in1i_abs, in1r_abs)
    with builder.if_else(in1i_abs_le_in1r_abs) as (then, otherwise):
        with then:
            r = builder.fdiv(in1i, in1r)
            tmp0 = builder.fmul(in1i, r)
            d = builder.fadd(in1r, tmp0)
            inv_d = builder.fdiv(ONE, d)
            minus_r = builder.fsub(ZERO, r)
            out.real = inv_d
            out.imag = builder.fmul(minus_r, inv_d)
        with otherwise:
            r = builder.fdiv(in1r, in1i)
            tmp0 = builder.fmul(in1r, r)
            d = builder.fadd(tmp0, in1i)
            inv_d = builder.fdiv(ONE, d)
            out.real = builder.fmul(r, inv_d)
            out.imag = builder.fsub(ZERO, inv_d)
    return out._getvalue()

def np_real_sin_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.sin_impl(context, builder, sig, args)

def np_complex_sin_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    return cmathimpl.sin_impl(context, builder, sig, args)

def np_real_cos_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.cos_impl(context, builder, sig, args)

def np_complex_cos_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    return cmathimpl.cos_impl(context, builder, sig, args)

def np_real_tan_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.tan_impl(context, builder, sig, args)

def np_real_asin_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.asin_impl(context, builder, sig, args)

def np_real_acos_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.acos_impl(context, builder, sig, args)

def np_real_atan_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.atan_impl(context, builder, sig, args)

def np_real_atan2_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    return mathimpl.atan2_float_impl(context, builder, sig, args)

def np_real_hypot_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    return mathimpl.hypot_float_impl(context, builder, sig, args)

def np_real_sinh_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.sinh_impl(context, builder, sig, args)

def np_complex_sinh_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    fty = ty.underlying_float
    fsig1 = typing.signature(*[fty] * 2)
    x = context.make_complex(builder, ty, args[0])
    out = context.make_complex(builder, ty)
    xr = x.real
    xi = x.imag
    sxi = np_real_sin_impl(context, builder, fsig1, [xi])
    shxr = np_real_sinh_impl(context, builder, fsig1, [xr])
    cxi = np_real_cos_impl(context, builder, fsig1, [xi])
    chxr = np_real_cosh_impl(context, builder, fsig1, [xr])
    out.real = builder.fmul(cxi, shxr)
    out.imag = builder.fmul(sxi, chxr)
    return out._getvalue()

def np_real_cosh_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.cosh_impl(context, builder, sig, args)

def np_complex_cosh_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    fty = ty.underlying_float
    fsig1 = typing.signature(*[fty] * 2)
    x = context.make_complex(builder, ty, args[0])
    out = context.make_complex(builder, ty)
    xr = x.real
    xi = x.imag
    cxi = np_real_cos_impl(context, builder, fsig1, [xi])
    chxr = np_real_cosh_impl(context, builder, fsig1, [xr])
    sxi = np_real_sin_impl(context, builder, fsig1, [xi])
    shxr = np_real_sinh_impl(context, builder, fsig1, [xr])
    out.real = builder.fmul(cxi, chxr)
    out.imag = builder.fmul(sxi, shxr)
    return out._getvalue()

def np_real_tanh_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.tanh_impl(context, builder, sig, args)

def np_complex_tanh_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    fty = ty.underlying_float
    fsig1 = typing.signature(*[fty] * 2)
    ONE = context.get_constant(fty, 1.0)
    x = context.make_complex(builder, ty, args[0])
    out = context.make_complex(builder, ty)
    xr = x.real
    xi = x.imag
    si = np_real_sin_impl(context, builder, fsig1, [xi])
    ci = np_real_cos_impl(context, builder, fsig1, [xi])
    shr = np_real_sinh_impl(context, builder, fsig1, [xr])
    chr_ = np_real_cosh_impl(context, builder, fsig1, [xr])
    rs = builder.fmul(ci, shr)
    is_ = builder.fmul(si, chr_)
    rc = builder.fmul(ci, chr_)
    ic = builder.fmul(si, shr)
    sqr_rc = builder.fmul(rc, rc)
    sqr_ic = builder.fmul(ic, ic)
    d = builder.fadd(sqr_rc, sqr_ic)
    inv_d = builder.fdiv(ONE, d)
    rs_rc = builder.fmul(rs, rc)
    is_ic = builder.fmul(is_, ic)
    is_rc = builder.fmul(is_, rc)
    rs_ic = builder.fmul(rs, ic)
    numr = builder.fadd(rs_rc, is_ic)
    numi = builder.fsub(is_rc, rs_ic)
    out.real = builder.fmul(numr, inv_d)
    out.imag = builder.fmul(numi, inv_d)
    return out._getvalue()

def np_real_asinh_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.asinh_impl(context, builder, sig, args)

def np_real_acosh_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.acosh_impl(context, builder, sig, args)

def np_complex_acosh_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1)
    ty = sig.args[0]
    csig2 = typing.signature(*[ty] * 3)
    ONE = context.get_constant_generic(builder, ty, 1.0 + 0j)
    x = args[0]
    x_plus_one = numbers.complex_add_impl(context, builder, csig2, [x, ONE])
    x_minus_one = numbers.complex_sub_impl(context, builder, csig2, [x, ONE])
    sqrt_x_plus_one = np_complex_sqrt_impl(context, builder, sig, [x_plus_one])
    sqrt_x_minus_one = np_complex_sqrt_impl(context, builder, sig, [x_minus_one])
    prod_sqrt = numbers.complex_mul_impl(context, builder, csig2, [sqrt_x_plus_one, sqrt_x_minus_one])
    log_arg = numbers.complex_add_impl(context, builder, csig2, [x, prod_sqrt])
    return np_complex_log_impl(context, builder, sig, [log_arg])

def np_real_atanh_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.atanh_impl(context, builder, sig, args)

def np_real_floor_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.call_fp_intrinsic(builder, 'llvm.floor', args)

def np_real_ceil_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.call_fp_intrinsic(builder, 'llvm.ceil', args)

def np_real_trunc_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.call_fp_intrinsic(builder, 'llvm.trunc', args)

def np_real_fabs_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1)
    return mathimpl.call_fp_intrinsic(builder, 'llvm.fabs', args)

def np_complex_ge_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    ty = sig.args[0]
    (in1, in2) = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag
    xr_gt_yr = builder.fcmp_ordered('>', xr, yr)
    no_nan_xi_yi = builder.fcmp_ordered('ord', xi, yi)
    xr_eq_yr = builder.fcmp_ordered('==', xr, yr)
    xi_ge_yi = builder.fcmp_ordered('>=', xi, yi)
    first_term = builder.and_(xr_gt_yr, no_nan_xi_yi)
    second_term = builder.and_(xr_eq_yr, xi_ge_yi)
    return builder.or_(first_term, second_term)

def np_complex_le_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    ty = sig.args[0]
    (in1, in2) = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag
    xr_lt_yr = builder.fcmp_ordered('<', xr, yr)
    no_nan_xi_yi = builder.fcmp_ordered('ord', xi, yi)
    xr_eq_yr = builder.fcmp_ordered('==', xr, yr)
    xi_le_yi = builder.fcmp_ordered('<=', xi, yi)
    first_term = builder.and_(xr_lt_yr, no_nan_xi_yi)
    second_term = builder.and_(xr_eq_yr, xi_le_yi)
    return builder.or_(first_term, second_term)

def np_complex_gt_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    ty = sig.args[0]
    (in1, in2) = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag
    xr_gt_yr = builder.fcmp_ordered('>', xr, yr)
    no_nan_xi_yi = builder.fcmp_ordered('ord', xi, yi)
    xr_eq_yr = builder.fcmp_ordered('==', xr, yr)
    xi_gt_yi = builder.fcmp_ordered('>', xi, yi)
    first_term = builder.and_(xr_gt_yr, no_nan_xi_yi)
    second_term = builder.and_(xr_eq_yr, xi_gt_yi)
    return builder.or_(first_term, second_term)

def np_complex_lt_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    ty = sig.args[0]
    (in1, in2) = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag
    xr_lt_yr = builder.fcmp_ordered('<', xr, yr)
    no_nan_xi_yi = builder.fcmp_ordered('ord', xi, yi)
    xr_eq_yr = builder.fcmp_ordered('==', xr, yr)
    xi_lt_yi = builder.fcmp_ordered('<', xi, yi)
    first_term = builder.and_(xr_lt_yr, no_nan_xi_yi)
    second_term = builder.and_(xr_eq_yr, xi_lt_yi)
    return builder.or_(first_term, second_term)

def np_complex_eq_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    ty = sig.args[0]
    (in1, in2) = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag
    xr_eq_yr = builder.fcmp_ordered('==', xr, yr)
    xi_eq_yi = builder.fcmp_ordered('==', xi, yi)
    return builder.and_(xr_eq_yr, xi_eq_yi)

def np_complex_ne_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    ty = sig.args[0]
    (in1, in2) = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag
    xr_ne_yr = builder.fcmp_unordered('!=', xr, yr)
    xi_ne_yi = builder.fcmp_unordered('!=', xi, yi)
    return builder.or_(xr_ne_yr, xi_ne_yi)

def _complex_is_true(context, builder, ty, val):
    if False:
        while True:
            i = 10
    complex_val = context.make_complex(builder, ty, value=val)
    re_true = cgutils.is_true(builder, complex_val.real)
    im_true = cgutils.is_true(builder, complex_val.imag)
    return builder.or_(re_true, im_true)

def np_logical_and_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = cgutils.is_true(builder, args[0])
    b = cgutils.is_true(builder, args[1])
    return builder.and_(a, b)

def np_complex_logical_and_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = _complex_is_true(context, builder, sig.args[0], args[0])
    b = _complex_is_true(context, builder, sig.args[1], args[1])
    return builder.and_(a, b)

def np_logical_or_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = cgutils.is_true(builder, args[0])
    b = cgutils.is_true(builder, args[1])
    return builder.or_(a, b)

def np_complex_logical_or_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = _complex_is_true(context, builder, sig.args[0], args[0])
    b = _complex_is_true(context, builder, sig.args[1], args[1])
    return builder.or_(a, b)

def np_logical_xor_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = cgutils.is_true(builder, args[0])
    b = cgutils.is_true(builder, args[1])
    return builder.xor(a, b)

def np_complex_logical_xor_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    a = _complex_is_true(context, builder, sig.args[0], args[0])
    b = _complex_is_true(context, builder, sig.args[1], args[1])
    return builder.xor(a, b)

def np_logical_not_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return cgutils.is_false(builder, args[0])

def np_complex_logical_not_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    a = _complex_is_true(context, builder, sig.args[0], args[0])
    return builder.not_(a)

def np_int_smax_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 2)
    (arg1, arg2) = args
    arg1_sge_arg2 = builder.icmp_signed('>=', arg1, arg2)
    return builder.select(arg1_sge_arg2, arg1, arg2)

def np_int_umax_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 2)
    (arg1, arg2) = args
    arg1_uge_arg2 = builder.icmp_unsigned('>=', arg1, arg2)
    return builder.select(arg1_uge_arg2, arg1, arg2)

def np_real_maximum_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    (arg1, arg2) = args
    arg1_nan = builder.fcmp_unordered('uno', arg1, arg1)
    any_nan = builder.fcmp_unordered('uno', arg1, arg2)
    nan_result = builder.select(arg1_nan, arg1, arg2)
    arg1_ge_arg2 = builder.fcmp_ordered('>=', arg1, arg2)
    non_nan_result = builder.select(arg1_ge_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)

def np_real_fmax_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    (arg1, arg2) = args
    arg2_nan = builder.fcmp_unordered('uno', arg2, arg2)
    any_nan = builder.fcmp_unordered('uno', arg1, arg2)
    nan_result = builder.select(arg2_nan, arg1, arg2)
    arg1_ge_arg2 = builder.fcmp_ordered('>=', arg1, arg2)
    non_nan_result = builder.select(arg1_ge_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)

def np_complex_maximum_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty] * 2)
    (arg1, arg2) = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg1_nan, arg1, arg2)
    arg1_ge_arg2 = np_complex_ge_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_ge_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)

def np_complex_fmax_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty] * 2)
    (arg1, arg2) = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg2_nan, arg1, arg2)
    arg1_ge_arg2 = np_complex_ge_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_ge_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)

def np_int_smin_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 2)
    (arg1, arg2) = args
    arg1_sle_arg2 = builder.icmp_signed('<=', arg1, arg2)
    return builder.select(arg1_sle_arg2, arg1, arg2)

def np_int_umin_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    (arg1, arg2) = args
    arg1_ule_arg2 = builder.icmp_unsigned('<=', arg1, arg2)
    return builder.select(arg1_ule_arg2, arg1, arg2)

def np_real_minimum_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    (arg1, arg2) = args
    arg1_nan = builder.fcmp_unordered('uno', arg1, arg1)
    any_nan = builder.fcmp_unordered('uno', arg1, arg2)
    nan_result = builder.select(arg1_nan, arg1, arg2)
    arg1_le_arg2 = builder.fcmp_ordered('<=', arg1, arg2)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)

def np_real_fmin_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 2)
    (arg1, arg2) = args
    arg1_nan = builder.fcmp_unordered('uno', arg1, arg1)
    any_nan = builder.fcmp_unordered('uno', arg1, arg2)
    nan_result = builder.select(arg1_nan, arg2, arg1)
    arg1_le_arg2 = builder.fcmp_ordered('<=', arg1, arg2)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)

def np_complex_minimum_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty] * 2)
    (arg1, arg2) = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg1_nan, arg1, arg2)
    arg1_le_arg2 = np_complex_le_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)

def np_complex_fmin_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty] * 2)
    (arg1, arg2) = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg2_nan, arg1, arg2)
    arg1_le_arg2 = np_complex_le_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)

def np_int_isnan_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return cgutils.false_bit

def np_real_isnan_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return mathimpl.is_nan(builder, args[0])

def np_complex_isnan_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    (x,) = args
    (ty,) = sig.args
    complex_val = context.make_complex(builder, ty, value=x)
    return cmathimpl.is_nan(builder, complex_val)

def np_int_isfinite_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return cgutils.true_bit

def np_datetime_isfinite_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return builder.icmp_unsigned('!=', args[0], npdatetime.NAT)

def np_datetime_isnat_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return builder.icmp_signed('==', args[0], npdatetime.NAT)

def np_real_isfinite_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return mathimpl.is_finite(builder, args[0])

def np_complex_isfinite_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    (x,) = args
    (ty,) = sig.args
    complex_val = context.make_complex(builder, ty, value=x)
    return cmathimpl.is_finite(builder, complex_val)

def np_int_isinf_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return cgutils.false_bit

def np_real_isinf_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    return mathimpl.is_inf(builder, args[0])

def np_complex_isinf_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    (x,) = args
    (ty,) = sig.args
    complex_val = context.make_complex(builder, ty, value=x)
    return cmathimpl.is_inf(builder, complex_val)

def np_real_signbit_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    masks = {types.float16: context.get_constant(types.uint16, 32768), types.float32: context.get_constant(types.uint32, 2147483648), types.float64: context.get_constant(types.uint64, 9223372036854775808)}
    arg_ty = sig.args[0]
    arg_int_ty = getattr(types, f'uint{arg_ty.bitwidth}')
    arg_ll_int_ty = context.get_value_type(arg_int_ty)
    int_res = builder.and_(builder.bitcast(args[0], arg_ll_int_ty), masks[arg_ty])
    bool_res = builder.icmp_unsigned('!=', int_res, int_res.type(0))
    return bool_res

def np_real_copysign_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    _check_arity_and_homogeneity(sig, args, 2)
    return mathimpl.copysign_float_impl(context, builder, sig, args)

def np_real_nextafter_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    _check_arity_and_homogeneity(sig, args, 2)
    dispatch_table = {types.float32: 'numba_nextafterf', types.float64: 'numba_nextafter'}
    return _dispatch_func_by_name_type(context, builder, sig, args, dispatch_table, 'nextafter')

def np_real_spacing_impl(context, builder, sig, args):
    if False:
        return 10
    _check_arity_and_homogeneity(sig, args, 1)
    dispatch_table = {types.float32: 'numba_nextafterf', types.float64: 'numba_nextafter'}
    [ty] = sig.args
    inner_sig = typing.signature(sig.return_type, ty, ty)
    ll_ty = args[0].type
    ll_inf = ll_ty(np.inf)
    fnty = llvmlite.ir.FunctionType(ll_ty, [ll_ty, ll_ty])
    fn = cgutils.insert_pure_function(builder.module, fnty, name='llvm.copysign')
    ll_sinf = builder.call(fn, [ll_inf, args[0]])
    inner_args = args + [ll_sinf]
    nextafter = _dispatch_func_by_name_type(context, builder, inner_sig, inner_args, dispatch_table, 'nextafter')
    return builder.fsub(nextafter, args[0])

def np_real_ldexp_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    (x1, x2) = args
    (ty1, ty2) = sig.args
    x2 = context.cast(builder, x2, ty2, types.intc)
    f_fi_sig = typing.signature(ty1, ty1, types.intc)
    return mathimpl.ldexp_impl(context, builder, f_fi_sig, (x1, x2))