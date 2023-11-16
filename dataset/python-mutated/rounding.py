import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, handle_numpy_casting, handle_numpy_dtype, from_zero_dim_arrays_to_scalar, handle_numpy_out

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _ceil(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        print('Hello World!')
    ret = ivy.ceil(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _floor(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        i = 10
        return i + 15
    ret = ivy.floor(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _rint(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        i = 10
        return i + 15
    ret = ivy.round(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, x), out=out)
    return ret

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _trunc(x, /, out=None, *, where=True, casting='same_kind', order='k', dtype=None, subok=True):
    if False:
        return 10
    ret = ivy.trunc(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def around(a, decimals=0, out=None):
    if False:
        for i in range(10):
            print('nop')
    return ivy.round(a, decimals=decimals, out=out)

@handle_numpy_out
@to_ivy_arrays_and_back
def fix(x, /, out=None):
    if False:
        for i in range(10):
            print('nop')
    where = ivy.greater_equal(x, 0)
    return ivy.where(where, ivy.floor(x, out=out), ivy.ceil(x, out=out), out=out)

@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def round(a, decimals=0, out=None):
    if False:
        for i in range(10):
            print('nop')
    return ivy.round(a, decimals=decimals, out=out)