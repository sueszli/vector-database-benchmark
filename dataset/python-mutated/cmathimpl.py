"""
Implement the cmath module functions.
"""
import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
registry = Registry('cmathimpl')
lower = registry.lower

def is_nan(builder, z):
    if False:
        for i in range(10):
            print('nop')
    return builder.fcmp_unordered('uno', z.real, z.imag)

def is_inf(builder, z):
    if False:
        for i in range(10):
            print('nop')
    return builder.or_(mathimpl.is_inf(builder, z.real), mathimpl.is_inf(builder, z.imag))

def is_finite(builder, z):
    if False:
        print('Hello World!')
    return builder.and_(mathimpl.is_finite(builder, z.real), mathimpl.is_finite(builder, z.imag))

@lower(cmath.isnan, types.Complex)
def isnan_float_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    [typ] = sig.args
    [value] = args
    z = context.make_complex(builder, typ, value=value)
    res = is_nan(builder, z)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower(cmath.isinf, types.Complex)
def isinf_float_impl(context, builder, sig, args):
    if False:
        return 10
    [typ] = sig.args
    [value] = args
    z = context.make_complex(builder, typ, value=value)
    res = is_inf(builder, z)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower(cmath.isfinite, types.Complex)
def isfinite_float_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    [typ] = sig.args
    [value] = args
    z = context.make_complex(builder, typ, value=value)
    res = is_finite(builder, z)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@overload(cmath.rect)
def impl_cmath_rect(r, phi):
    if False:
        print('Hello World!')
    if all([isinstance(typ, types.Float) for typ in [r, phi]]):

        def impl(r, phi):
            if False:
                i = 10
                return i + 15
            if not math.isfinite(phi):
                if not r:
                    return abs(r)
                if math.isinf(r):
                    return complex(r, phi)
            real = math.cos(phi)
            imag = math.sin(phi)
            if real == 0.0 and math.isinf(r):
                real /= r
            else:
                real *= r
            if imag == 0.0 and math.isinf(r):
                imag /= r
            else:
                imag *= r
            return complex(real, imag)
        return impl

def intrinsic_complex_unary(inner_func):
    if False:
        i = 10
        return i + 15

    def wrapper(context, builder, sig, args):
        if False:
            for i in range(10):
                print('nop')
        [typ] = sig.args
        [value] = args
        z = context.make_complex(builder, typ, value=value)
        x = z.real
        y = z.imag
        x_is_finite = mathimpl.is_finite(builder, x)
        y_is_finite = mathimpl.is_finite(builder, y)
        inner_sig = signature(sig.return_type, *(typ.underlying_float,) * 2 + (types.boolean,) * 2)
        res = context.compile_internal(builder, inner_func, inner_sig, (x, y, x_is_finite, y_is_finite))
        return impl_ret_untracked(context, builder, sig, res)
    return wrapper
NAN = float('nan')
INF = float('inf')

@lower(cmath.exp, types.Complex)
@intrinsic_complex_unary
def exp_impl(x, y, x_is_finite, y_is_finite):
    if False:
        return 10
    'cmath.exp(x + y j)'
    if x_is_finite:
        if y_is_finite:
            c = math.cos(y)
            s = math.sin(y)
            r = math.exp(x)
            return complex(r * c, r * s)
        else:
            return complex(NAN, NAN)
    elif math.isnan(x):
        if y:
            return complex(x, x)
        else:
            return complex(x, y)
    elif x > 0.0:
        if y_is_finite:
            real = math.cos(y)
            imag = math.sin(y)
            if real != 0:
                real *= x
            if imag != 0:
                imag *= x
            return complex(real, imag)
        else:
            return complex(x, NAN)
    elif y_is_finite:
        r = math.exp(x)
        c = math.cos(y)
        s = math.sin(y)
        return complex(r * c, r * s)
    else:
        r = 0
        return complex(r, r)

@lower(cmath.log, types.Complex)
@intrinsic_complex_unary
def log_impl(x, y, x_is_finite, y_is_finite):
    if False:
        for i in range(10):
            print('nop')
    'cmath.log(x + y j)'
    a = math.log(math.hypot(x, y))
    b = math.atan2(y, x)
    return complex(a, b)

@lower(cmath.log, types.Complex, types.Complex)
def log_base_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    'cmath.log(z, base)'
    [z, base] = args

    def log_base(z, base):
        if False:
            while True:
                i = 10
        return cmath.log(z) / cmath.log(base)
    res = context.compile_internal(builder, log_base, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@overload(cmath.log10)
def impl_cmath_log10(z):
    if False:
        print('Hello World!')
    if not isinstance(z, types.Complex):
        return
    LN_10 = 2.302585092994046

    def log10_impl(z):
        if False:
            print('Hello World!')
        'cmath.log10(z)'
        z = cmath.log(z)
        return complex(z.real / LN_10, z.imag / LN_10)
    return log10_impl

@overload(cmath.phase)
def phase_impl(x):
    if False:
        i = 10
        return i + 15
    'cmath.phase(x + y j)'
    if not isinstance(x, types.Complex):
        return

    def impl(x):
        if False:
            while True:
                i = 10
        return math.atan2(x.imag, x.real)
    return impl

@overload(cmath.polar)
def polar_impl(x):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(x, types.Complex):
        return

    def impl(x):
        if False:
            print('Hello World!')
        (r, i) = (x.real, x.imag)
        return (math.hypot(r, i), math.atan2(i, r))
    return impl

@lower(cmath.sqrt, types.Complex)
def sqrt_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    SQRT2 = 1.4142135623730951
    ONE_PLUS_SQRT2 = 1.0 + SQRT2
    theargflt = sig.args[0].underlying_float
    MAX = mathimpl.DBL_MAX if theargflt.bitwidth == 64 else mathimpl.FLT_MAX
    THRES = MAX / ONE_PLUS_SQRT2

    def sqrt_impl(z):
        if False:
            i = 10
            return i + 15
        'cmath.sqrt(z)'
        a = z.real
        b = z.imag
        if a == 0.0 and b == 0.0:
            return complex(abs(b), b)
        if math.isinf(b):
            return complex(abs(b), b)
        if math.isnan(a):
            return complex(a, a)
        if math.isinf(a):
            if a < 0.0:
                return complex(abs(b - b), math.copysign(a, b))
            else:
                return complex(a, math.copysign(b - b, b))
        if abs(a) >= THRES or abs(b) >= THRES:
            a *= 0.25
            b *= 0.25
            scale = True
        else:
            scale = False
        if a >= 0:
            t = math.sqrt((a + math.hypot(a, b)) * 0.5)
            real = t
            imag = b / (2 * t)
        else:
            t = math.sqrt((-a + math.hypot(a, b)) * 0.5)
            real = abs(b) / (2 * t)
            imag = math.copysign(t, b)
        if scale:
            return complex(real * 2, imag)
        else:
            return complex(real, imag)
    res = context.compile_internal(builder, sqrt_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@lower(cmath.cos, types.Complex)
def cos_impl(context, builder, sig, args):
    if False:
        print('Hello World!')

    def cos_impl(z):
        if False:
            print('Hello World!')
        'cmath.cos(z) = cmath.cosh(z j)'
        return cmath.cosh(complex(-z.imag, z.real))
    res = context.compile_internal(builder, cos_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@overload(cmath.cosh)
def impl_cmath_cosh(z):
    if False:
        i = 10
        return i + 15
    if not isinstance(z, types.Complex):
        return

    def cosh_impl(z):
        if False:
            i = 10
            return i + 15
        'cmath.cosh(z)'
        x = z.real
        y = z.imag
        if math.isinf(x):
            if math.isnan(y):
                real = abs(x)
                imag = y
            elif y == 0.0:
                real = abs(x)
                imag = y
            else:
                real = math.copysign(x, math.cos(y))
                imag = math.copysign(x, math.sin(y))
            if x < 0.0:
                imag = -imag
            return complex(real, imag)
        return complex(math.cos(y) * math.cosh(x), math.sin(y) * math.sinh(x))
    return cosh_impl

@lower(cmath.sin, types.Complex)
def sin_impl(context, builder, sig, args):
    if False:
        return 10

    def sin_impl(z):
        if False:
            print('Hello World!')
        'cmath.sin(z) = -j * cmath.sinh(z j)'
        r = cmath.sinh(complex(-z.imag, z.real))
        return complex(r.imag, -r.real)
    res = context.compile_internal(builder, sin_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@overload(cmath.sinh)
def impl_cmath_sinh(z):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(z, types.Complex):
        return

    def sinh_impl(z):
        if False:
            while True:
                i = 10
        'cmath.sinh(z)'
        x = z.real
        y = z.imag
        if math.isinf(x):
            if math.isnan(y):
                real = x
                imag = y
            else:
                real = math.cos(y)
                imag = math.sin(y)
                if real != 0.0:
                    real *= x
                if imag != 0.0:
                    imag *= abs(x)
            return complex(real, imag)
        return complex(math.cos(y) * math.sinh(x), math.sin(y) * math.cosh(x))
    return sinh_impl

@lower(cmath.tan, types.Complex)
def tan_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15

    def tan_impl(z):
        if False:
            return 10
        'cmath.tan(z) = -j * cmath.tanh(z j)'
        r = cmath.tanh(complex(-z.imag, z.real))
        return complex(r.imag, -r.real)
    res = context.compile_internal(builder, tan_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@overload(cmath.tanh)
def impl_cmath_tanh(z):
    if False:
        print('Hello World!')
    if not isinstance(z, types.Complex):
        return

    def tanh_impl(z):
        if False:
            return 10
        'cmath.tanh(z)'
        x = z.real
        y = z.imag
        if math.isinf(x):
            real = math.copysign(1.0, x)
            if math.isinf(y):
                imag = 0.0
            else:
                imag = math.copysign(0.0, math.sin(2.0 * y))
            return complex(real, imag)
        tx = math.tanh(x)
        ty = math.tan(y)
        cx = 1.0 / math.cosh(x)
        txty = tx * ty
        denom = 1.0 + txty * txty
        return complex(tx * (1.0 + ty * ty) / denom, ty / denom * cx * cx)
    return tanh_impl

@lower(cmath.acos, types.Complex)
def acos_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    LN_4 = math.log(4)
    THRES = mathimpl.FLT_MAX / 4

    def acos_impl(z):
        if False:
            return 10
        'cmath.acos(z)'
        if abs(z.real) > THRES or abs(z.imag) > THRES:
            real = math.atan2(abs(z.imag), z.real)
            imag = math.copysign(math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4, -z.imag)
            return complex(real, imag)
        else:
            s1 = cmath.sqrt(complex(1.0 - z.real, -z.imag))
            s2 = cmath.sqrt(complex(1.0 + z.real, z.imag))
            real = 2.0 * math.atan2(s1.real, s2.real)
            imag = math.asinh(s2.real * s1.imag - s2.imag * s1.real)
            return complex(real, imag)
    res = context.compile_internal(builder, acos_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@overload(cmath.acosh)
def impl_cmath_acosh(z):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(z, types.Complex):
        return
    LN_4 = math.log(4)
    THRES = mathimpl.FLT_MAX / 4

    def acosh_impl(z):
        if False:
            print('Hello World!')
        'cmath.acosh(z)'
        if abs(z.real) > THRES or abs(z.imag) > THRES:
            real = math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4
            imag = math.atan2(z.imag, z.real)
            return complex(real, imag)
        else:
            s1 = cmath.sqrt(complex(z.real - 1.0, z.imag))
            s2 = cmath.sqrt(complex(z.real + 1.0, z.imag))
            real = math.asinh(s1.real * s2.real + s1.imag * s2.imag)
            imag = 2.0 * math.atan2(s1.imag, s2.real)
            return complex(real, imag)
    return acosh_impl

@lower(cmath.asinh, types.Complex)
def asinh_impl(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    LN_4 = math.log(4)
    THRES = mathimpl.FLT_MAX / 4

    def asinh_impl(z):
        if False:
            i = 10
            return i + 15
        'cmath.asinh(z)'
        if abs(z.real) > THRES or abs(z.imag) > THRES:
            real = math.copysign(math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4, z.real)
            imag = math.atan2(z.imag, abs(z.real))
            return complex(real, imag)
        else:
            s1 = cmath.sqrt(complex(1.0 + z.imag, -z.real))
            s2 = cmath.sqrt(complex(1.0 - z.imag, z.real))
            real = math.asinh(s1.real * s2.imag - s2.real * s1.imag)
            imag = math.atan2(z.imag, s1.real * s2.real - s1.imag * s2.imag)
            return complex(real, imag)
    res = context.compile_internal(builder, asinh_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@lower(cmath.asin, types.Complex)
def asin_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10

    def asin_impl(z):
        if False:
            i = 10
            return i + 15
        'cmath.asin(z) = -j * cmath.asinh(z j)'
        r = cmath.asinh(complex(-z.imag, z.real))
        return complex(r.imag, -r.real)
    res = context.compile_internal(builder, asin_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@lower(cmath.atan, types.Complex)
def atan_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10

    def atan_impl(z):
        if False:
            while True:
                i = 10
        'cmath.atan(z) = -j * cmath.atanh(z j)'
        r = cmath.atanh(complex(-z.imag, z.real))
        if math.isinf(z.real) and math.isnan(z.imag):
            return complex(r.imag, r.real)
        else:
            return complex(r.imag, -r.real)
    res = context.compile_internal(builder, atan_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)

@lower(cmath.atanh, types.Complex)
def atanh_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    LN_4 = math.log(4)
    THRES_LARGE = math.sqrt(mathimpl.FLT_MAX / 4)
    THRES_SMALL = math.sqrt(mathimpl.FLT_MIN)
    PI_12 = math.pi / 2

    def atanh_impl(z):
        if False:
            return 10
        'cmath.atanh(z)'
        if z.real < 0.0:
            negate = True
            z = -z
        else:
            negate = False
        ay = abs(z.imag)
        if math.isnan(z.real) or z.real > THRES_LARGE or ay > THRES_LARGE:
            if math.isinf(z.imag):
                real = math.copysign(0.0, z.real)
            elif math.isinf(z.real):
                real = 0.0
            else:
                h = math.hypot(z.real * 0.5, z.imag * 0.5)
                real = z.real / 4.0 / h / h
            imag = -math.copysign(PI_12, -z.imag)
        elif z.real == 1.0 and ay < THRES_SMALL:
            if ay == 0.0:
                real = INF
                imag = z.imag
            else:
                real = -math.log(math.sqrt(ay) / math.sqrt(math.hypot(ay, 2.0)))
                imag = math.copysign(math.atan2(2.0, -ay) / 2, z.imag)
        else:
            sqay = ay * ay
            zr1 = 1 - z.real
            real = math.log1p(4.0 * z.real / (zr1 * zr1 + sqay)) * 0.25
            imag = -math.atan2(-2.0 * z.imag, zr1 * (1 + z.real) - sqay) * 0.5
        if math.isnan(z.imag):
            imag = NAN
        if negate:
            return complex(-real, -imag)
        else:
            return complex(real, imag)
    res = context.compile_internal(builder, atanh_impl, sig, args)
    return impl_ret_untracked(context, builder, sig, res)