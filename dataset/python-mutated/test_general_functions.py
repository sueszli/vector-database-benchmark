import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test, BackendHandler
from ivy.functional.frontends.jax import vmap
from hypothesis import strategies as st
import jax

def _fn1(x, y):
    if False:
        print('Hello World!')
    return ivy.matmul(x, y)

def _fn2(x, y):
    if False:
        while True:
            i = 10
    return ivy.vecdot(x, y)

def _fn3(x, y):
    if False:
        print('Hello World!')
    return ivy.add(x, y)

@handle_frontend_test(fn_tree='jax.general_functions.device_get', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid')))
def test_jax_device_get(*, dtype_and_x, test_flags, fn_tree, frontend, backend_fw, on_device):
    if False:
        for i in range(10):
            print('nop')
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        (dtype, x) = dtype_and_x
        dtype = dtype[0]
        x = x[0]
        x = ivy_backend.asarray(x)
        if test_flags.as_variable and ivy_backend.is_float_dtype(dtype):
            x = ivy_backend.functional.ivy.gradients._variable(x)
        x_on_dev = ivy_backend.functional.frontends.jax.device_get(x).ivy_array
        dev_from_new_x = ivy_backend.dev(x_on_dev)
        assert dev_from_new_x == 'cpu'

@handle_frontend_test(fn_tree='jax.general_functions.device_put', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('valid')))
def test_jax_device_put(*, dtype_and_x, test_flags, fn_tree, frontend, backend_fw, on_device):
    if False:
        i = 10
        return i + 15
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        (dtype, x) = dtype_and_x
        dtype = dtype[0]
        x = x[0]
        x = ivy_backend.asarray(x)
        if test_flags.as_variable and ivy_backend.is_float_dtype(dtype):
            x = ivy_backend.functional.ivy.gradients._variable(x)
        device = ivy_backend.dev(x)
        x_on_dev = ivy_backend.functional.frontends.jax.device_put(x, on_device).ivy_array
        dev_from_new_x = ivy_backend.dev(x_on_dev)
        assert dev_from_new_x == device

@handle_frontend_test(fn_tree='jax.general_functions.vmap', func=st.sampled_from([_fn1, _fn2, _fn3]), dtype_and_arrays_and_axes=helpers.arrays_and_axes(allow_none=False, min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=10, num=2, return_dtype=True), in_axes_as_cont=st.booleans())
def test_jax_vmap(func, dtype_and_arrays_and_axes, in_axes_as_cont, backend_fw):
    if False:
        i = 10
        return i + 15
    (dtype, generated_arrays, in_axes) = dtype_and_arrays_and_axes
    ivy.set_backend(backend_fw)
    arrays = [ivy.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        vmapped_func = vmap(func, in_axes=in_axes, out_axes=0)
    else:
        vmapped_func = vmap(func, in_axes=0, out_axes=0)
    assert callable(vmapped_func)
    try:
        fw_res = helpers.flatten_and_to_np(ret=vmapped_func(*arrays), backend=backend_fw)
        fw_res = fw_res if len(fw_res) else None
    except Exception:
        fw_res = None
    ivy.previous_backend()
    ivy.set_backend('jax')
    arrays = [ivy.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        jax_vmapped_func = jax.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        jax_vmapped_func = jax.vmap(func, in_axes=0, out_axes=0)
    assert callable(jax_vmapped_func)
    try:
        jax_res = helpers.flatten_and_to_np(ret=jax_vmapped_func(*arrays), backend='jax')
        jax_res = jax_res if len(jax_res) else None
    except Exception:
        jax_res = None
    ivy.previous_backend()
    if fw_res is not None and jax_res is not None:
        helpers.value_test(ret_np_flat=fw_res, ret_np_from_gt_flat=jax_res, rtol=0.1, atol=0.1, backend=backend_fw, ground_truth_backend='jax')
    elif fw_res is None and jax_res is None:
        pass
    else:
        assert False, "One of the results is None while other isn't"