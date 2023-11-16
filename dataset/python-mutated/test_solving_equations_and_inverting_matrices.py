import sys
import numpy as np
from hypothesis import strategies as st, assume
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

@st.composite
def _get_inv_square_matrices(draw):
    if False:
        while True:
            i = 10
    dim_size = draw(helpers.ints(min_value=1, max_value=10))
    batch_shape = draw(st.sampled_from([2, 4, 6, 8, 10]))
    generated_shape = (dim_size,) * batch_shape
    generated_ind = int(np.floor(len(generated_shape) / 2))
    (handpicked_shape, handpicked_ind) = draw(st.sampled_from([[(24, 6, 4), 1], [(8, 3, 6, 4), 2], [(6, 7, 8, 16, 21), 3]]))
    (shape, ind) = draw(st.sampled_from([(generated_shape, generated_ind), (handpicked_shape, handpicked_ind)]))
    input_dtype = draw(helpers.get_dtypes('float', index=1, full=False).filter(lambda x: x not in ['float16', 'bfloat16']))
    invertible = False
    while not invertible:
        a = draw(helpers.array_values(dtype=input_dtype[0], shape=shape, min_value=-100, max_value=100))
        try:
            np.linalg.inv(a)
            invertible = True
        except np.linalg.LinAlgError:
            pass
    return (input_dtype, a, ind)

@handle_frontend_test(fn_tree='numpy.linalg.inv', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), small_abs_safety_factor=2, safety_factor_scale='log', shape=helpers.ints(min_value=2, max_value=20).map(lambda x: tuple([x, x]))).filter(lambda x: np.linalg.cond(x[1][0].tolist()) < 1 / sys.float_info.epsilon), test_with_out=st.just(False))
def test_numpy_inv(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0])

@handle_frontend_test(fn_tree='numpy.linalg.lstsq', x=helpers.get_first_solve_matrix(adjoint=True), y=helpers.get_second_solve_matrix(), test_with_out=st.just(False))
def test_numpy_lstsq(x, y, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    (dtype1, a, _) = x
    (dtype2, b) = y
    (ret, ret_gt) = helpers.test_frontend_function(input_dtypes=[dtype1, dtype2], frontend=frontend, test_flags=test_flags, backend_to_test=backend_fw, fn_tree=fn_tree, on_device=on_device, a=a, b=b, test_values=False)
    for (ret_f, ret_gtt) in zip(ret, ret_gt):
        return

@handle_frontend_test(fn_tree='numpy.linalg.pinv', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=2, max_num_dims=2), test_with_out=st.just(False))
def test_numpy_pinv(dtype_and_x, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        return 10
    (dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, a=x[0])

@handle_frontend_test(fn_tree='numpy.linalg.solve', x=helpers.get_first_solve_batch_matrix(), y=helpers.get_second_solve_batch_matrix(), test_with_out=st.just(False))
def test_numpy_solve(x, y, frontend, test_flags, fn_tree, backend_fw, on_device):
    if False:
        print('Hello World!')
    (dtype1, x1, _) = x
    (dtype2, x2, _) = y
    helpers.test_frontend_function(input_dtypes=[dtype1, dtype2], backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, rtol=0.0001, atol=0.0001, a=x1, b=x2)

@handle_frontend_test(fn_tree='numpy.linalg.tensorinv', params=_get_inv_square_matrices(), test_with_out=st.just(False))
def test_numpy_tensorinv(*, params, test_flags, on_device, fn_tree, frontend, backend_fw):
    if False:
        print('Hello World!')
    (dtype, x, ind) = params
    if backend_fw == 'paddle':
        assume(x.ndim <= 9)
    helpers.test_frontend_function(input_dtypes=dtype, backend_to_test=backend_fw, test_flags=test_flags, rtol=0.1, atol=0.1, frontend=frontend, fn_tree=fn_tree, on_device=on_device, a=x, ind=ind)