from hypothesis import strategies as st
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_searching import _broadcastable_trio
from ...test_numpy.test_sorting_searching_counting.test_searching import _broadcastable_trio as _where_helper

@st.composite
def _searchsorted(draw):
    if False:
        while True:
            i = 10
    (dtype_x, x) = draw(helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric', full=False, key='searchsorted'), shape=(draw(st.integers(min_value=1, max_value=10)),)))
    (dtype_v, v) = draw(helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric', full=False, key='searchsorted'), min_num_dims=1))
    input_dtypes = dtype_x + dtype_v
    xs = x + v
    side = draw(st.sampled_from(['left', 'right']))
    sorter = None
    xs[0] = np.sort(xs[0], axis=-1)
    return (input_dtypes, xs, side, sorter)

@st.composite
def _unique_helper(draw):
    if False:
        for i in range(10):
            print('nop')
    (arr_dtype, arr, shape) = draw(helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric', full=False, key='searchsorted'), min_num_dims=1, min_dim_size=2, ret_shape=True))
    axis = draw(st.sampled_from(list(range(len(shape))) + [None]))
    return_index = draw(st.booleans())
    return_inverse = draw(st.booleans())
    return_counts = draw(st.booleans())
    return (arr_dtype, arr, return_index, return_inverse, return_counts, axis)

@handle_frontend_test(fn_tree='jax.numpy.argmax', dtype_and_x=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), force_int_axis=True, min_num_dims=1, valid_axis=True), keepdims=st.booleans())
def test_jax_argmax(*, dtype_and_x, keepdims, on_device, fn_tree, frontend, backend_fw, test_flags):
    if False:
        while True:
            i = 10
    (input_dtype, x, axis) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0], axis=axis, out=None, keepdims=keepdims)

@handle_frontend_test(fn_tree='jax.numpy.argsort', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), min_axis=-1, max_axis=0, min_num_dims=1, force_int_axis=True), test_with_out=st.just(False))
def test_jax_argsort(*, dtype_x_axis, frontend, backend_fw, test_flags, fn_tree, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=input_dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0], axis=axis)

@handle_frontend_test(fn_tree='jax.numpy.argwhere', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid')), test_with_out=st.just(False))
def test_jax_argwhere(dtype_and_x, frontend, backend_fw, test_flags, fn_tree, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0], size=None, fill_value=None)

@handle_frontend_test(fn_tree='jax.numpy.count_nonzero', dtype_input_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('valid'), min_num_dims=1, force_int_axis=True, valid_axis=True, allow_neg_axes=True), keepdims=st.booleans(), test_with_out=st.just(False))
def test_jax_count_nonzero(dtype_input_axis, keepdims, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x, axis) = dtype_input_axis
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0], axis=axis, keepdims=keepdims)

@handle_frontend_test(fn_tree='jax.numpy.extract', broadcastables=_broadcastable_trio())
def test_jax_extract(broadcastables, frontend, backend_fw, test_flags, fn_tree, on_device):
    if False:
        while True:
            i = 10
    (cond, xs, dtype) = broadcastables
    helpers.test_frontend_function(input_dtypes=dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, condition=cond, arr=xs[0])

@handle_frontend_test(fn_tree='jax.numpy.flatnonzero', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric')), test_with_out=st.just(False))
def test_jax_flatnonzero(dtype_and_x, frontend, backend_fw, test_flags, fn_tree, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0])

@handle_frontend_test(fn_tree='jax.numpy.nanargmax', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), min_axis=-1, max_axis=0, min_num_dims=1, force_int_axis=True), keep_dims=st.booleans(), test_with_out=st.just(False))
def test_jax_nanargmax(dtype_x_axis, frontend, backend_fw, test_flags, fn_tree, on_device, keep_dims):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=input_dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0], axis=axis, keepdims=keep_dims)

@handle_frontend_test(fn_tree='jax.numpy.nanargmin', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), min_axis=-1, max_axis=0, min_num_dims=1, force_int_axis=True), keep_dims=st.booleans(), test_with_out=st.just(False))
def test_jax_nanargmin(dtype_x_axis, frontend, backend_fw, test_flags, fn_tree, on_device, keep_dims):
    if False:
        i = 10
        return i + 15
    (input_dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=input_dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0], axis=axis, keepdims=keep_dims)

@handle_frontend_test(fn_tree='jax.numpy.nonzero', dtype_and_a=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid')), test_with_out=st.just(False))
def test_jax_nonzero(dtype_and_a, frontend, backend_fw, test_flags, fn_tree, on_device):
    if False:
        return 10
    (dtype, a) = dtype_and_a
    helpers.test_frontend_function(input_dtypes=dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=a[0])

@handle_frontend_test(fn_tree='jax.numpy.searchsorted', dtype_x_v_side_sorter=_searchsorted(), test_with_out=st.just(False))
def test_jax_searchsorted(dtype_x_v_side_sorter, frontend, backend_fw, test_flags, fn_tree, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtypes, xs, side, sorter) = dtype_x_v_side_sorter
    helpers.test_frontend_function(input_dtypes=input_dtypes, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=xs[0], v=xs[1], side=side, sorter=sorter)

@handle_frontend_test(fn_tree='jax.numpy.sort', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), min_axis=-1, max_axis=0, min_num_dims=1, force_int_axis=True), test_with_out=st.just(False))
def test_jax_sort(*, dtype_x_axis, frontend, backend_fw, fn_tree, on_device, test_flags):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=input_dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0], axis=axis)

@handle_frontend_test(fn_tree='jax.numpy.sort_complex', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), min_num_dims=1, min_dim_size=1, min_axis=-1, max_axis=0), test_with_out=st.just(False))
def test_jax_sort_complex(*, dtype_x_axis, frontend, backend_fw, test_flags, fn_tree, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=input_dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0], test_values=False)

@handle_frontend_test(fn_tree='jax.numpy.unique', fn_inputs=_unique_helper(), test_with_out=st.just(False))
def test_jax_unique(fn_inputs, backend_fw, frontend, test_flags, fn_tree, on_device):
    if False:
        return 10
    (arr_dtype, arr, return_index, return_inverse, return_counts, axis) = fn_inputs
    helpers.test_frontend_function(input_dtypes=arr_dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, ar=arr[0], return_index=return_index, return_inverse=return_inverse, return_counts=return_counts, axis=axis)

@handle_frontend_test(fn_tree='jax.numpy.where', broadcastables=_where_helper(), only_cond=st.booleans(), size=st.integers(min_value=1, max_value=20), fill_value=st.one_of(st.integers(-10, 10), st.floats(-10, 10), st.booleans()))
def test_jax_where(*, broadcastables, only_cond, size, fill_value, frontend, backend_fw, fn_tree, on_device, test_flags):
    if False:
        for i in range(10):
            print('nop')
    (cond, x1, x2, dtype) = broadcastables
    if only_cond:
        (x1, x2) = (None, None)
    else:
        (size, fill_value) = (None, None)
    helpers.test_frontend_function(input_dtypes=['bool', dtype], fn_tree=fn_tree, on_device=on_device, test_flags=test_flags, frontend=frontend, backend_to_test=backend_fw, condition=cond, x=x1, y=x2, size=size, fill_value=fill_value)