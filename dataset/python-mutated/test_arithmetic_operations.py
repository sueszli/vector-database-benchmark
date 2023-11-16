from hypothesis import assume, strategies as st
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_elementwise import _float_power_helper

@handle_frontend_test(fn_tree='numpy.add', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True)]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='add'))
def test_numpy_add(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.divide', aliases=['numpy.true_divide'], dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True)]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='divide'))
def test_numpy_divide(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    assume(not np.any(np.isclose(xs[1], 0.0)))
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, atol=0.001, rtol=0.001, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.divmod', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, allow_inf=False, large_abs_safety_factor=6, safety_factor_scale='linear', shared_dtype=True)]), where=np_frontend_helpers.where(), test_with_out=st.just(False), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='divmod'))
def test_numpy_divmod(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    assume(not np.any(np.isclose(xs[1], 0)))
    if dtype:
        assume(np.dtype(dtype) >= np.dtype(input_dtypes[0]))
        assume(np.dtype(dtype) >= np.dtype(input_dtypes[1]))
        assume(not np.any(np.isclose(xs[1].astype(dtype), 0)))
    assume('uint' not in input_dtypes[0] and 'uint' not in input_dtypes[1])
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x1=xs[0], x2=xs[1])

@handle_frontend_test(fn_tree='numpy.float_power', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : _float_power_helper()]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='float_power'))
def test_numpy_float_power(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw):
    if False:
        for i in range(10):
            print('nop')
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    xs = list(xs[0])
    input_dtypes = list(input_dtypes[0])
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    assume(casting == 'same_kind')
    assume(dtype != 'bool')
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.floor_divide', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, large_abs_safety_factor=4, shared_dtype=True, safety_factor_scale='linear')]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='floor_divide'))
def test_numpy_floor_divide(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtypes, x, casting, dtype) = dtypes_values_casting
    assume(not np.any(np.isclose(x[1], 0, rtol=0.1, atol=0.1)))
    assume(not np.any(np.isclose(x[0], 0, rtol=0.1, atol=0.1)))
    if dtype:
        assume(np.dtype(dtype) >= np.dtype(input_dtypes[0]))
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x1=x[0], x2=x[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True, atol=0.01, rtol=0.01)

@handle_frontend_test(fn_tree='numpy.fmod', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True, large_abs_safety_factor=6, small_abs_safety_factor=6, safety_factor_scale='log')]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='fmod'))
def test_numpy_fmod(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    assume(not np.any(np.isclose(xs[1], 0.0)))
    assume(not np.any(np.isclose(xs[0], 0.0)))
    if dtype:
        assume(not np.any(np.isclose(xs[1].astype(dtype), 0.0)))
        assume(not np.any(np.isclose(xs[0].astype(dtype), 0.0)))
    assume('uint' not in input_dtypes[0] and 'uint' not in input_dtypes[1])
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.mod', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, min_value=0, exclude_min=True, shared_dtype=True)]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='mod'))
def test_numpy_mod(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True, rtol=1e-05, atol=1e-05)

@handle_frontend_test(fn_tree='numpy.modf', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=1, min_value=0, exclude_min=True)]), where=np_frontend_helpers.where(), test_with_out=st.just(False), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='modf'))
def test_numpy_modf(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtypes, x, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.multiply', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True)]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='multiply'))
def test_numpy_multiply(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.negative', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'))]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='negative'))
def test_numpy_negative(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtypes, x, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.positive', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'))]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='positive'))
def test_numpy_positive(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtypes, x, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.power', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('integer'), num_arrays=2, min_value=0, max_value=7, shared_dtype=True)]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='power'))
def test_numpy_power(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.reciprocal', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), small_abs_safety_factor=4, large_abs_safety_factor=4, safety_factor_scale='log')]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='reciprocal'))
def test_numpy_reciprocal(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw):
    if False:
        return 10
    (input_dtypes, x, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    assume(not np.any(np.isclose(x[0], 0)))
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, x=x[0], out=None, where=where, casting=casting, rtol=0.01, atol=0.01, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.remainder', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True)]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='remainder'))
def test_numpy_remainder(dtypes_values_casting, where, frontend, test_flags, backend_fw, fn_tree, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    assume(not np.any(np.isclose(xs[1], 0.0)))
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, frontend=frontend, test_flags=test_flags, backend_to_test=backend_fw, fn_tree=fn_tree, on_device=on_device, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.subtract', dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(arr_func=[lambda : helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True)]), where=np_frontend_helpers.where(), number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(fn_name='subtract'))
def test_numpy_subtract(dtypes_values_casting, where, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtypes, xs, casting, dtype) = dtypes_values_casting
    (where, input_dtypes, test_flags) = np_frontend_helpers.handle_where_and_array_bools(where=where, input_dtype=input_dtypes, test_flags=test_flags)
    np_frontend_helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x1=xs[0], x2=xs[1], out=None, where=where, casting=casting, order='K', dtype=dtype, subok=True)

@handle_frontend_test(fn_tree='numpy.vdot', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2), test_with_out=st.just(False))
def test_numpy_vdot(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtypes, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, test_values=False, a=xs[0], b=xs[1])