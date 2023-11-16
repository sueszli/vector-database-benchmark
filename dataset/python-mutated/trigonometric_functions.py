import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, handle_numpy_casting, handle_numpy_dtype, from_zero_dim_arrays_to_scalar, handle_numpy_out

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arccos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        i = 10
        return i + 15
    ret = ivy.acos(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arccosh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        print('Hello World!')
    ret = ivy.acosh(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arcsin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        i = 10
        return i + 15
    ret = ivy.asin(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arctan(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        return 10
    ret = ivy.atan(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arctan2(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        i = 10
        return i + 15
    ret = ivy.atan2(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _cos(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        for i in range(10):
            print('nop')
    ret = ivy.cos(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _deg2rad(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None, extobj=None):
    if False:
        return 10
    ret = ivy.deg2rad(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _degrees(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        print('Hello World!')
    ret = ivy.rad2deg(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _rad2deg(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        return 10
    ret = ivy.rad2deg(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _sin(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        i = 10
        return i + 15
    ret = ivy.sin(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _tan(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        print('Hello World!')
    ret = ivy.tan(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret