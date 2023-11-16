import sys
import ivy
from hypothesis import assume, strategies as st
from ivy.functional.frontends.tensorflow.nn import _convolution_broadcast_helper
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_nn import _x_and_filters
import numpy as np
import math
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.globals as test_globals
from ivy_tests.test_ivy.helpers import handle_frontend_test, assert_all_close, BackendHandler
dtype_shared = st.shared(st.sampled_from(helpers.get_dtypes('numeric')), key='dtype')

@st.composite
def _LinSpace_helper(draw):
    if False:
        i = 10
        return i + 15
    shape = ()
    dtype = draw(st.sampled_from(['float32', 'float64']))
    start = draw(helpers.array_values(dtype=dtype, shape=shape, min_value=-5.0, max_value=5.0))
    stop = draw(helpers.array_values(dtype=dtype, shape=shape, min_value=-4.0, max_value=10.0))
    return ([dtype] * 2, start, stop)

@st.composite
def _arrays_idx_n_dtypes(draw):
    if False:
        for i in range(10):
            print('nop')
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key='num_dims'))
    num_arrays = draw(st.shared(helpers.ints(min_value=2, max_value=4), key='num_arrays'))
    common_shape = draw(helpers.list_of_size(x=helpers.ints(min_value=2, max_value=3), size=num_dims - 1))
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(helpers.list_of_size(x=helpers.ints(min_value=2, max_value=3), size=num_arrays))
    xs = []
    input_dtypes = draw(helpers.array_dtypes(available_dtypes=draw(helpers.get_dtypes('float')), shared_dtype=True))
    for (ud, dt) in zip(unique_dims, input_dtypes):
        x = draw(helpers.array_values(shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:], dtype=dt))
        xs.append(x)
    return (xs, input_dtypes, unique_idx)

@st.composite
def _dtypes(draw):
    if False:
        i = 10
        return i + 15
    return draw(st.shared(helpers.list_of_size(x=st.sampled_from(draw(helpers.get_dtypes('numeric'))), size=1), key='dtype'))

@st.composite
def _fill_value(draw):
    if False:
        print('Hello World!')
    dtype = draw(_dtypes())[0]
    with BackendHandler.update_backend(test_globals.CURRENT_BACKEND) as ivy_backend:
        if ivy_backend.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        elif ivy_backend.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

@st.composite
def _get_shared_dtype(draw):
    if False:
        i = 10
        return i + 15
    return st.shared(st.sampled_from(draw(helpers.get_dtypes('numeric'))), key='dtype')

@st.composite
def _get_splits(draw, as_list=False):
    if False:
        return 10
    'Generate valid splits, either by generating an integer that evenly divides the\n    axis or a list of splits that sum to the length of the axis being split.'
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key='value_shape'))
    axis = draw(st.shared(helpers.get_axis(shape=shape, force_int=True), key='target_axis'))

    @st.composite
    def get_int_split(draw):
        if False:
            while True:
                i = 10
        if shape[axis] == 0:
            return 0
        factors = []
        for i in range(1, shape[axis] + 1):
            if shape[axis] % i == 0:
                factors.append(i)
        return draw(st.sampled_from(factors))

    @st.composite
    def get_list_split(draw):
        if False:
            print('Hello World!')
        num_or_size_splits = []
        while sum(num_or_size_splits) < shape[axis]:
            split_value = draw(helpers.ints(min_value=1, max_value=shape[axis] - sum(num_or_size_splits)))
            num_or_size_splits.append(split_value)
        return num_or_size_splits
    if as_list:
        return draw(get_list_split())
    else:
        return draw(get_int_split())

@st.composite
def _multiple_shape_helper(draw):
    if False:
        return 10
    (input_dtype, input_array, input_shape) = draw(helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), ret_shape=True))
    input_dims = len(input_shape)
    dt_n_multiples = draw(helpers.dtype_and_values(available_dtypes=['int32', 'int64'], min_value=0, max_value=10, shape=draw(helpers.get_shape(min_num_dims=1, max_num_dims=1, min_dim_size=input_dims, max_dim_size=input_dims))))
    return (input_dtype, input_array, dt_n_multiples)

@st.composite
def _pad_helper(draw, return_constant_values=False):
    if False:
        while True:
            i = 10
    (dtype, input, shape) = draw(helpers.dtype_and_values(min_num_dims=1, ret_shape=True))
    ndim = len(shape)
    (padding_dtype, paddings) = draw(helpers.dtype_and_values(available_dtypes=['int32', 'int64'], shape=(ndim, 2), min_value=0, max_value=10))
    if return_constant_values:
        (_, constant_values) = draw(helpers.dtype_and_values(dtype=dtype, shape=(1,)))
        return (dtype, input[0], padding_dtype, paddings[0], constant_values[0][0])
    return (dtype, input[0], padding_dtype, paddings[0])

@st.composite
def _permute_dims_helper(draw):
    if False:
        i = 10
        return i + 15
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key='shape'))
    dims = [x for x in range(len(shape))]
    permutation = draw(st.permutations(dims))
    return permutation

@st.composite
def _pow_helper_shared_dtype(draw):
    if False:
        i = 10
        return i + 15
    (dtype, x) = draw(helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True))
    (dtype1, dtype2) = dtype
    (x1, x2) = x
    if 'int' in dtype2:
        x2 = ivy.nested_map(lambda x: abs(x), x2, include_derived={'list': True})
    if ivy.is_int_dtype(dtype2):
        max_val = ivy.iinfo(dtype2).max
    else:
        max_val = ivy.finfo(dtype2).max
    max_x1 = np.max(np.abs(x1))
    if max_x1 in [0, 1]:
        max_value = None
    else:
        max_value = int(math.log(max_val) / math.log(max_x1))
        if abs(max_value) > abs(max_val) / 40 or max_value < 0:
            max_value = None
    return ([dtype1, dtype2], [x1, x2])

@st.composite
def _reshape_helper(draw):
    if False:
        i = 10
        return i + 15
    shape = draw(helpers.get_shape(min_num_dims=1))
    reshape_shape = draw(helpers.reshape_shapes(shape=shape))
    dtype = draw(helpers.array_dtypes(num_arrays=1))
    x = draw(helpers.array_values(dtype=dtype[0], shape=shape))
    return (x, dtype, reshape_shape)

@st.composite
def _squeeze_helper(draw):
    if False:
        i = 10
        return i + 15
    shape = draw(st.shared(helpers.get_shape(), key='value_shape'))
    valid_axes = []
    for (index, axis) in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    valid_axes.insert(0, None)
    axis = draw(st.sampled_from(valid_axes))
    return [axis] if axis is not None else axis

@st.composite
def df(draw, data_format):
    if False:
        for i in range(10):
            print('nop')
    data_format = draw(data_format)
    return data_format

@st.composite
def reverse_helper(draw):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x, shape) = draw(helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), min_num_dims=1, max_num_dims=8, ret_shape=True))
    (axis_dtype, axis) = draw(helpers.dtype_and_values(available_dtypes=['bool'], min_num_dims=1, max_num_dims=1, num_arrays=1, shape=(len(shape),)))
    return (dtype, x, axis_dtype, axis)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.AccumulateNV2', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False), shape=helpers.get_shape(min_num_dims=1))
def test_tensorflow_AccumulateNV2(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device, shape):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, inputs=x[0], shape=shape)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Acos', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Acos(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Acosh', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Acosh(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Add', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Add(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.AddN', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), min_num_dims=1, large_abs_safety_factor=8, small_abs_safety_factor=8, safety_factor_scale='log', min_value=-10000.0, max_value=10000.0), test_with_out=st.just(False))
def test_tensorflow_AddN(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, inputs=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.AddV2', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_AddV2(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Angle', dtype_and_xs=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('complex')), Tout=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Angle(*, dtype_and_xs, Tout, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, xs) = dtype_and_xs
    if input_dtype[0] == 'complex128':
        Tout = 'float64'
    elif input_dtype[0] == 'complex64':
        Tout = 'float32' if Tout else None
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=xs[0], Tout=Tout)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.ApproximateEqual', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True, large_abs_safety_factor=20, small_abs_safety_factor=20, safety_factor_scale='log'), tol=st.floats(1e-05, 0.001), test_with_out=st.just(False))
def test_tensorflow_ApproximateEqual(*, dtype_and_x, tol, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1], tolerance=tol)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.ArgMax', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-5, max_value=5, allow_inf=False), output_type=st.sampled_from(['int16', 'int32', 'int64']), test_with_out=st.just(False))
def test_tensorflow_ArgMax(*, dtype_x_axis, output_type, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], dimension=axis, output_type=output_type)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.ArgMin', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-5, max_value=5, allow_inf=False), output_type=st.sampled_from(['int32', 'int64']), test_with_out=st.just(False))
def test_tensorflow_ArgMin(*, dtype_x_axis, output_type, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], dimension=axis, output_type=output_type)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Asin', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Asin(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Atan', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Atan(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Atan2', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Atan2(*, dtype_and_x, frontend, test_flags, fn_tree, on_device, backend_fw):
    if False:
        while True:
            i = 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, y=xs[0], x=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Atanh', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Atanh(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.BandedTriangularSolve', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2), test_with_out=st.just(False), lower=st.booleans(), adjoint=st.booleans())
def test_tensorflow_BandedTriangularSolve(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device, lower, adjoint):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, matrix=x[0], rhs=x[1], lower=lower, adjoint=adjoint)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.BatchMatMul', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2), test_with_out=st.just(False), adj_x=st.booleans(), adj_y=st.booleans())
def test_tensorflow_BatchMatMul(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device, adj_x, adj_y):
    if False:
        return 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1], adj_x=adj_x, adj_y=adj_y)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.BatchMatMulV2', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2), test_with_out=st.just(False), adj_x=st.booleans(), adj_y=st.booleans())
def test_tensorflow_BatchMatMulV2(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device, adj_x, adj_y):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1], adj_x=adj_x, adj_y=adj_y)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.BatchMatMulV3', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2), test_with_out=st.just(False), Tout=st.sampled_from(['float32', 'float64']), adj_x=st.booleans(), adj_y=st.booleans())
def test_tensorflow_BatchMatMulV3(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device, Tout, adj_x, adj_y):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1], Tout=Tout, adj_x=adj_x, adj_y=adj_y)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.BitwiseAnd', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('integer'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_BitwiseAnd(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.BitwiseOr', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('integer'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_BitwiseOr(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.BitwiseXor', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('integer'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_BitwiseXor(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.BroadcastTo', array_and_shape=helpers.array_and_broadcastable_shape(_get_shared_dtype()), test_with_out=st.just(False))
def test_tensorflow_BroadcastTo(*, array_and_shape, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (x, to_shape) = array_and_shape
    helpers.test_frontend_function(input_dtypes=[x.dtype], backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x, shape=to_shape)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Ceil', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Ceil(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Cholesky', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_value=0, max_value=10, shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x]))), test_with_out=st.just(False))
def test_tensorflow_Cholesky(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, x) = dtype_and_x
    x = x[0]
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 0.001
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x, rtol=0.0001, atol=0.0001)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Complex', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2), test_with_out=st.just(False), Tout=st.sampled_from(['complex64', 'complex128']))
def test_tensorflow_Complex(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device, Tout):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, real=x[0], imag=x[1], Tout=Tout)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Concat', xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(), test_with_out=st.just(False))
def test_tensorflow_Concat(*, xs_n_input_dtypes_n_unique_idx, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (xs, input_dtypes, unique_idx) = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, concat_dim=unique_idx, values=xs)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.ConcatV2', xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(), test_with_out=st.just(False), number_positional_args=st.just(0))
def test_tensorflow_ConcatV2(xs_n_input_dtypes_n_unique_idx, test_flags, frontend, backend_fw, fn_tree):
    if False:
        return 10
    (xs, input_dtypes, unique_idx) = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(input_dtypes=input_dtypes, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, values=xs, axis=unique_idx)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Conj', dtype_and_xs=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('complex')), test_with_out=st.just(False))
def test_tensorflow_Conj(*, dtype_and_xs, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, xs) = dtype_and_xs
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=xs[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Conv2D', x_f_d_df=_x_and_filters(dtypes=helpers.get_dtypes('float', full=False), data_format=st.sampled_from(['NHWC']), padding=st.sampled_from(['SAME', 'VALID', 'EXPLICIT']), type='2d', dilation_min=1, dilation_max=1), test_with_out=st.just(False), number_positional_args=st.just(0))
def test_tensorflow_Conv2D(*, x_f_d_df, test_flags, frontend, backend_fw, fn_tree, on_device):
    if False:
        print('Hello World!')
    (input_dtype, x, filters, dilation, data_format, stride, padding) = x_f_d_df
    channel_index = data_format.find('C')
    stride = _convolution_broadcast_helper(stride, num_spatial_dims=2, channel_index=channel_index, name='strides')
    dilation = _convolution_broadcast_helper(dilation, num_spatial_dims=2, channel_index=channel_index, name='dilations')
    explicit_padding = None
    if isinstance(padding, list):
        explicit_padding = padding
        padding = 'EXPLICIT'
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, input=x, filter=filters, strides=stride, padding=padding, explicit_paddings=explicit_padding, data_format=data_format, dilations=dilation, use_cudnn_on_gpu=True)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Conv3D', x_f_d_df=_x_and_filters(dtypes=helpers.get_dtypes('float', full=False), data_format=st.sampled_from(['NDHWC']), padding=st.sampled_from(['SAME', 'VALID']), type='3d', dilation_min=1, dilation_max=1), test_with_out=st.just(False), number_positional_args=st.just(0))
def test_tensorflow_Conv3D(*, x_f_d_df, test_flags, frontend, backend_fw, fn_tree, on_device):
    if False:
        print('Hello World!')
    (input_dtype, x, filters, dilation, data_format, stride, padding) = x_f_d_df
    stride = _convolution_broadcast_helper(stride, num_spatial_dims=3, channel_index=4, name='strides')
    dilation = _convolution_broadcast_helper(dilation, num_spatial_dims=3, channel_index=4, name='dilations')
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, input=x, filter=filters, strides=stride, padding=padding, data_format=data_format, dilations=dilation)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Cos', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Cos(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Cosh', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Cosh(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Cross', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=1, max_num_dims=5, min_dim_size=3, max_dim_size=3, safety_factor_scale='log', num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Cross(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=xs[0], b=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Cumprod', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-5, max_value=5), exclusive=st.booleans(), reverse=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Cumprod(*, dtype_x_axis, exclusive, reverse, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], axis=axis, exclusive=exclusive, reverse=reverse)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Cumsum', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-5, max_value=5), exclusive=st.booleans(), reverse=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Cumsum(*, dtype_x_axis, exclusive, reverse, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, rtol=0.01, atol=0.01, x=x[0], axis=axis, exclusive=exclusive, reverse=reverse)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.CumulativeLogsumexp', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), axis=st.just(0), test_with_out=st.just(False), exclusive=st.booleans(), reverse=st.booleans())
def test_tensorflow_CumulativeLogsumexp(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device, axis, exclusive, reverse):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, x=x[0], axis=axis, exclusive=exclusive, reverse=reverse)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.DebugGradientIdentity', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_DebugGradientIdentity(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Diag', dtype_and_x=helpers.dtype_and_values(available_dtypes=['float32', 'float64', 'int32', 'int64'], min_num_dims=1, max_num_dims=1, min_value=-1e+30, max_value=1e+30), test_with_out=st.just(False))
def test_tensorflow_Diag(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, diagonal=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Div', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Div(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Elu', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), min_value=-3, max_value=3, min_num_dims=1, max_num_dims=3, min_dim_size=1, max_dim_size=3), name=st.just(None), test_with_out=st.just(False), number_positional_args=st.just(0))
def test_tensorflow_Elu(*, dtype_and_x, name, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, features=x[0], name=name)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Equal', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Equal(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.EuclideanNorm', dtype_values_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('float'), min_num_dims=3, max_num_dims=5, min_dim_size=1, max_dim_size=4, min_axis=-3, max_axis=2, valid_axis=True, allow_neg_axes=True), keep_dims=st.booleans(), test_with_out=st.just(False), number_positional_args=st.just(0))
def test_tensorflow_EuclideanNorm(dtype_values_axis, keep_dims, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, values, axis) = dtype_values_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=values[0], axis=axis, keep_dims=keep_dims)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Exp', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Exp(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Expm1', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Expm1(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.FFT', dtype_and_x=helpers.dtype_and_values(min_num_dims=1, min_dim_size=2, large_abs_safety_factor=15, small_abs_safety_factor=15, safety_factor_scale='log', available_dtypes=helpers.get_dtypes('complex')), test_with_out=st.just(False))
def test_tensorflow_FFT(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], rtol=0.01, atol=0.01)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.FFT2D', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('complex'), min_value=-100000.0, max_value=100000.0, min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=5, large_abs_safety_factor=2.5, small_abs_safety_factor=2.5, safety_factor_scale='log'))
def test_tensorflow_FFT2D(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], rtol=0.01, atol=0.01)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.FFT3D', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('complex'), min_value=-100000.0, max_value=100000.0, min_num_dims=3, max_num_dims=5, min_dim_size=2, max_dim_size=5, large_abs_safety_factor=2.5, small_abs_safety_factor=2.5, safety_factor_scale='log'))
def test_tensorflow_FFT3D(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], rtol=0.01, atol=0.01)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Fill', shape=helpers.get_shape(allow_none=False, min_num_dims=1, min_dim_size=1), fill_value=_fill_value(), dtypes=_dtypes(), test_with_out=st.just(False))
def test_tensorflow_Fill(*, shape, fill_value, dtypes, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    helpers.test_frontend_function(input_dtypes=dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, rtol=1e-05, dims=shape, value=fill_value)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Floor', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Floor(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.FloorDiv', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_FloorDiv(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.FloorMod', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_FloorMod(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Gather', params_indices_others=helpers.array_indices_axis(array_dtypes=helpers.get_dtypes('numeric'), indices_dtypes=['int32', 'int64'], disable_random_axis=True, axis_zero=True, min_num_dims=1, max_num_dims=5, min_dim_size=1, max_dim_size=10), test_with_out=st.just(False))
def test_tensorflow_Gather(*, params_indices_others, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtypes, params, indices) = params_indices_others
    helpers.test_frontend_function(input_dtypes=dtypes, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, params=params, indices=indices, validate_indices=True)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Greater', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Greater(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.GreaterEqual', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_GreaterEqual(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Identity', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric')), test_with_out=st.just(False))
def test_tensorflow_Identity(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.IdentityN', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric')), test_with_out=st.just(False))
def test_tensorflow_IdentityN(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Igamma', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), num_arrays=2, shared_dtype=True, abs_smallest_val=1e-05, min_num_dims=2, max_num_dims=2, min_dim_size=3, max_dim_size=3, min_value=2, max_value=100, allow_nan=False), test_with_out=st.just(False))
def test_tensorflow_Igamma(*, dtype_and_x, on_device, fn_tree, backend_fw, frontend, test_flags):
    if False:
        while True:
            i = 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, frontend=frontend, backend_to_test=backend_fw, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, rtol=0.0001, a=xs[0], x=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Imag', dtype_and_xs=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid')), send_Tout=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Imag(*, dtype_and_xs, send_Tout, frontend, test_flags, fn_tree, on_device, backend_fw):
    if False:
        i = 10
        return i + 15
    (input_dtype, xs) = dtype_and_xs
    if input_dtype[0] == 'complex128':
        send_Tout = 'float64'
    elif input_dtype[0] == 'complex64':
        send_Tout = 'float32' if send_Tout else None
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=xs[0], Tout=send_Tout)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Inv', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric')), test_with_out=st.just(False))
def test_tensorflow_Inv(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.InvGrad', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=1, num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_InvGrad(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, y=x[0], dy=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Invert', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('integer')), test_with_out=st.just(False))
def test_tensorflow_Invert(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.LeakyRelu', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=1), test_with_out=st.just(False), alpha=helpers.floats(min_value=0, max_value=1))
def test_tensorflow_LeakyReLU(*, dtype_and_x, alpha, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x) = dtype_and_x
    return helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, features=x[0], alpha=alpha)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.LeftShift', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('integer'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_LeftShift(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Less', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Less(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.LessEqual', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_LessEqual(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.LinSpace', dtype_and_params=_LinSpace_helper(), num=helpers.ints(min_value=2, max_value=10))
def test_tensorflow_LinSpace(*, dtype_and_params, num, on_device, fn_tree, frontend, test_flags, backend_fw):
    if False:
        while True:
            i = 10
    (dtype, start, stop) = dtype_and_params
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, start=start, stop=stop, num=num, on_device=on_device)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Log', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Log(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Log1p', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), safety_factor_scale='log'), test_with_out=st.just(False))
def test_tensorflow_Log1p(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.LogSoftmax', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_LogSoftmax(*, dtype_and_x, on_device, fn_tree, frontend, test_flags, backend_fw):
    if False:
        return 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, logits=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.LogicalNot', dtype_and_x=helpers.dtype_and_values(dtype=['bool'], num_arrays=1, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_LogicalNot(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.LogicalOr', dtype_and_x=helpers.dtype_and_values(dtype=['bool', 'bool'], num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_LogicalOr(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.MatMul', dtype_and_x=helpers.dtype_and_values(available_dtypes=['float32', 'float64', 'int32', 'int64'], shape=(3, 3), num_arrays=2, shared_dtype=True, large_abs_safety_factor=10, small_abs_safety_factor=10, safety_factor_scale='log'), transpose_a=st.booleans(), transpose_b=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_MatMul(*, dtype_and_x, transpose_a, transpose_b, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, atol=0.01, a=x[0], b=x[1], transpose_a=transpose_a, transpose_b=transpose_b)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.MatrixDeterminant', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])), min_value=-5, max_value=5), test_with_out=st.just(False))
def test_tensorflow_MatrixDeterminant(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.MatrixInverse', dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), shape=helpers.ints(min_value=2, max_value=10).map(lambda x: tuple([x, x]))).filter(lambda x: np.linalg.cond(x[1][0].tolist()) < 1 / sys.float_info.epsilon), adjoint=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_MatrixInverse(*, dtype_x, adjoint, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], adjoint=adjoint, rtol=1e-05, atol=0.0001)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Max', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-5, max_value=5), keep_dims=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Max(*, dtype_x_axis, keep_dims, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], axis=axis, keep_dims=keep_dims)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.MaxPool3D', aliases=['tensorflow.nn.max_pool3d'], data_format=st.sampled_from(['NDHWC', 'NCDHW']), x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=5), test_with_out=st.just(False))
def test_tensorflow_MaxPool3D(*, x_k_s_p, data_format, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x, ksize, strides, padding) = x_k_s_p
    data_format = data_format
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], ksize=ksize, strides=strides, padding=padding, data_format=data_format)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Maximum', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Maximum(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Mean', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('float'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-10, max_value=3), keep_dims=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Mean(*, dtype_x_axis, keep_dims, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], axis=axis, keep_dims=keep_dims, rtol=0.01, atol=0.01)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Min', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-5, max_value=5), keep_dims=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Min(*, dtype_x_axis, keep_dims, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], axis=axis, keep_dims=keep_dims)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Minimum', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Minimum(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Mod', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Mod(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Mul', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Mul(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Neg', dtype_and_x=helpers.dtype_and_values(available_dtypes=['float32', 'float64', 'int8', 'int16', 'int32', 'int64']), test_with_out=st.just(False))
def test_tensorflow_Neg(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.NotEqual', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_NotEqual(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.NthElement', array_indices_axis=helpers.array_indices_axis(array_dtypes=helpers.get_dtypes('numeric'), indices_dtypes=['int32'], min_num_dims=1, min_dim_size=1, disable_random_axis=True), reverse=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_NthElement(*, array_indices_axis, reverse, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x, n) = array_indices_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x, n=n.flatten()[0], reverse=reverse)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.OnesLike', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric')), test_with_out=st.just(False))
def test_tensorflow_OnesLike(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Pack', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('valid'), valid_axis=True, force_int_axis=True, min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Pack(dtype_x_axis, fn_tree, frontend, test_flags, backend_fw):
    if False:
        i = 10
        return i + 15
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, values=x, axis=axis)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Pad', dtype_x_paddings=_pad_helper(), number_positional_args=st.just(0), test_with_out=st.just(False))
def test_tensorflow_Pad(dtype_x_paddings, frontend, test_flags, fn_tree, backend_fw):
    if False:
        return 10
    (dtype, x, padding_dtype, paddings) = dtype_x_paddings
    helpers.test_frontend_function(input_dtypes=dtype + padding_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, input=x, paddings=paddings)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.PadV2', dtype_x_paddings=_pad_helper(return_constant_values=True), test_with_out=st.just(False))
def test_tensorflow_PadV2(dtype_x_paddings, frontend, test_flags, fn_tree, backend_fw):
    if False:
        i = 10
        return i + 15
    (dtype, x, padding_dtype, paddings, constant_values) = dtype_x_paddings
    helpers.test_frontend_function(input_dtypes=dtype + padding_dtype + dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, input=x, paddings=paddings, constant_values=constant_values)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Pow', dtype_and_x=_pow_helper_shared_dtype(), test_with_out=st.just(False))
def test_tensorflow_Pow(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Prod', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-5, max_value=5), keep_dims=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Prod(*, dtype_x_axis, keep_dims, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], axis=axis, keep_dims=keep_dims)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Real', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False), Tout=st.sampled_from(['float32', 'float64']))
def test_tensorflow_Real(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device, Tout):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, input=x[0], Tout=Tout)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.RealDiv', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True, large_abs_safety_factor=8, small_abs_safety_factor=8, safety_factor_scale='log'), test_with_out=st.just(False))
def test_tensorflow_RealDiv(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, atol=0.001, rtol=0.001, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Reciprocal', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_value=1), test_with_out=st.just(False))
def test_tensorflow_Reciprocal(dtype_and_x, frontend, test_flags, fn_tree, backend_fw):
    if False:
        while True:
            i = 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Relu', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Relu(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, features=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Relu6', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Relu6(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, features=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Reshape', test_with_out=st.just(False), x_reshape=_reshape_helper())
def test_tensorflow_Reshape(*, x_reshape, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (x, dtype, shape) = x_reshape
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, tensor=x, shape=shape)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Reverse', dtype_x_axis=reverse_helper())
def test_tensorflow_Reverse(*, dtype_x_axis, frontend, fn_tree, test_flags, on_device, backend_fw):
    if False:
        i = 10
        return i + 15
    (dtype, x, axis_dtype, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype + axis_dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, tensor=x[0], dims=axis[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.RightShift', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('integer'), num_arrays=2, shared_dtype=True, min_value=0, max_value=8), test_with_out=st.just(False))
def test_tensorflow_RightShift(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Round', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Round(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Rsqrt', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Rsqrt(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Shape', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Shape(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.ShapeN', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), max_num_dims=4), output_dtype=st.sampled_from(['int32', 'int64']), test_with_out=st.just(False))
def test_tensorflow_ShapeN(*, dtype_and_x, output_dtype, on_device, fn_tree, frontend, test_flags, backend_fw):
    if False:
        print('Hello World!')
    (input_dtype, input) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=input, out_type=output_dtype)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Sigmoid', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Sigmoid(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Sign', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), large_abs_safety_factor=5, small_abs_safety_factor=5, safety_factor_scale='log'), test_with_out=st.just(False))
def test_tensorflow_Sign(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Sinh', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Sinh(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Size', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), max_num_dims=4), output_dtype=st.sampled_from(['int32', 'int64']), test_with_out=st.just(False))
def test_tensorflow_Size(*, dtype_and_x, frontend, test_flags, backend_fw, fn_tree, on_device, output_dtype):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], out_type=output_dtype)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Softmax', dtype_values_axis=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Softmax(dtype_values_axis, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, values) = dtype_values_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, test_flags=test_flags, frontend=frontend, fn_tree=fn_tree, on_device=on_device, atol=0.01, logits=values[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Softplus', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Softplus(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, features=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Softsign', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Softsign(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, features=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Split', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), shape=st.shared(helpers.get_shape(min_num_dims=1), key='value_shape')), axis=st.shared(helpers.get_axis(shape=st.shared(helpers.get_shape(min_num_dims=1), key='value_shape'), force_int=True), key='target_axis'), num_splits=_get_splits())
def test_tensorflow_Split(*, dtype_and_x, axis, num_splits, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, value) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, value=value[0], axis=axis, num_split=num_splits)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.SplitV', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), shape=st.shared(helpers.get_shape(min_num_dims=1), key='value_shape')), axis=st.shared(helpers.get_axis(shape=st.shared(helpers.get_shape(min_num_dims=1), key='value_shape'), force_int=True), key='target_axis'), size_splits=_get_splits(as_list=True), test_with_out=st.just(False))
def test_tensorflow_SplitV(*, dtype_and_x, axis, size_splits, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtype, value) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, value=value[0], axis=axis, size_splits=size_splits, num_split=len(size_splits))

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Sqrt', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Sqrt(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Square', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric')), test_with_out=st.just(False))
def test_tensorflow_Square(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.SquaredDifference', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_SquaredDifference(*, dtype_and_x, frontend, test_flags, fn_tree, on_device, backend_fw):
    if False:
        i = 10
        return i + 15
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], y=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Squeeze', dtype_value=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), shape=st.shared(helpers.get_shape(), key='value_shape')), axis=_squeeze_helper(), test_with_out=st.just(False))
def test_tensorflow_Squeeze(dtype_value, axis, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, xs) = dtype_value
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=xs[0], axis=axis)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Sub', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('numeric'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Sub(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Sum', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('numeric'), valid_axis=True, force_int_axis=True, min_num_dims=1, min_value=-5, max_value=5), keep_dims=st.booleans(), test_with_out=st.just(False))
def test_tensorflow_Sum(*, dtype_x_axis, keep_dims, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], axis=axis, keep_dims=keep_dims)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Svd', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), min_value=0, max_value=10, shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x]))), full_matrices=st.booleans(), compute_uv=st.just(True))
def test_tensorflow_Svd(*, dtype_and_x, full_matrices, compute_uv, frontend, test_flags, fn_tree, on_device, backend_fw):
    if False:
        print('Hello World!')
    (dtype, x) = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 0.001
    (ret, frontend_ret) = helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, test_values=False, input=x, full_matrices=full_matrices, compute_uv=compute_uv)
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]
    (u, s, vh) = ret
    (frontend_s, frontend_u, frontend_vh) = frontend_ret
    assert_all_close(ret_np=u @ np.diag(s) @ vh, ret_from_gt_np=frontend_u @ np.diag(frontend_s) @ frontend_vh.T, rtol=0.01, atol=0.01, ground_truth_backend=frontend)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Tan', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Tan(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Tanh', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_Tanh(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.TanhGrad', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_TanhGrad(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, y=xs[0], dy=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Tile', all_arguments=_multiple_shape_helper())
def test_tensorflow_Tile(*, all_arguments, test_flags, frontend, fn_tree, on_device, backend_fw):
    if False:
        while True:
            i = 10
    (input_dtype, input_matrix, dt_and_multiples) = all_arguments
    (dt_mul, multiples) = dt_and_multiples
    helpers.test_frontend_function(input_dtypes=input_dtype + dt_mul, input=input_matrix[0], multiples=multiples[0], test_flags=test_flags, backend_to_test=backend_fw, frontend=frontend, fn_tree=fn_tree, on_device=on_device)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.TruncateDiv', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('integer'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_TruncateDiv(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtype, xs) = dtype_and_x
    assume(not np.any(np.isclose(xs[1], 0)))
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Unpack', dtype_x_axis=helpers.dtype_values_axis(available_dtypes=helpers.get_dtypes('valid'), valid_axis=True, force_int_axis=True, min_num_dims=1), test_with_out=st.just(False))
def test_tensorflow_Unpack(*, dtype_x_axis, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype, x, axis) = dtype_x_axis
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, value=x[0], num=x[0].shape[axis], axis=axis)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Xdivy', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Xdivy(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Xlog1py', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Xlog1py(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Xlogy', dtype_and_x=helpers.dtype_and_values(available_dtypes=['float16', 'float32', 'float64'], num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Xlogy(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    (input_dtype, xs) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=xs[0], y=xs[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Zeta', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid'), min_num_dims=1, max_num_dims=1, num_arrays=2, shared_dtype=True), test_with_out=st.just(False))
def test_tensorflow_Zeta(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], q=x[1])

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Roll', dtype_and_values=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), shape=st.shared(helpers.get_shape(min_num_dims=1), key='shape')), shift=helpers.get_axis(shape=st.shared(helpers.get_shape(min_num_dims=1), key='shape'), force_tuple=True), axis=helpers.get_axis(shape=st.shared(helpers.get_shape(min_num_dims=1), key='shape'), force_tuple=True))
def test_tensorflow_roll(*, dtype_and_values, shift, axis, on_device, fn_tree, frontend, test_flags, backend_fw):
    if False:
        while True:
            i = 10
    (input_dtype, value) = dtype_and_values
    if isinstance(shift, int) and isinstance(axis, tuple):
        axis = axis[0]
    if isinstance(shift, tuple) and isinstance(axis, tuple):
        if len(shift) != len(axis):
            mn = min(len(shift), len(axis))
            shift = shift[:mn]
            axis = axis[:mn]
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=value[0], shift=shift, axis=axis)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.Transpose', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), shape=st.shared(helpers.get_shape(min_num_dims=1), key='shape')), perm=_permute_dims_helper(), test_with_out=st.just(False))
def test_tensorflow_transpose(*, dtype_and_x, perm, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0], perm=perm)

@handle_frontend_test(fn_tree='tensorflow.raw_ops.ZerosLike', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float')), test_with_out=st.just(False))
def test_tensorflow_zeros_like(*, dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, x=x[0])