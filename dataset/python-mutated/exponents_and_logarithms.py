import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, handle_numpy_casting, handle_numpy_dtype, from_zero_dim_arrays_to_scalar, handle_numpy_out

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _exp(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        for i in range(10):
            print('nop')
    ret = ivy.exp(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _exp2(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        return 10
    ret = ivy.pow(2, x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _expm1(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        return 10
    ret = ivy.expm1(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _frexp(x, /, out1_2=(None, None), out=(None, None), *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        return 10
    (mant, exp) = ivy.frexp(x, out=out)
    if ivy.is_array(where):
        mant = ivy.where(where, mant, ivy.default(out[0], ivy.zeros_like(mant)), out=out[0])
        exp = ivy.where(where, exp, ivy.default(out[1], ivy.zeros_like(exp)), out=out[1])
    return (mant, exp)

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _ldexp(x1, x2, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        return 10
    ret = ivy.ldexp(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _log(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        return 10
    ret = ivy.log(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _log10(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        print('Hello World!')
    ret = ivy.log10(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _log1p(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        print('Hello World!')
    ret = ivy.log1p(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _log2(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        print('Hello World!')
    ret = ivy.log2(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logaddexp(x1, x2, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        while True:
            i = 10
    ret = ivy.logaddexp(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logaddexp2(x1, x2, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        for i in range(10):
            print('nop')
    ret = ivy.logaddexp2(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@to_ivy_arrays_and_back
def i0(x):
    if False:
        while True:
            i = 10
    return ivy.i0(x)