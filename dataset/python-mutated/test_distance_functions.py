from hypothesis import strategies as st
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

@handle_frontend_test(fn_tree='torch.nn.functional.cosine_similarity', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_value=2, max_value=5, min_dim_size=2, shared_dtype=True, num_arrays=2), dim=st.integers(min_value=-1, max_value=0))
def test_torch_cosine_similarity(*, dtype_and_x, dim, on_device, fn_tree, frontend, test_flags, backend_fw):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, rtol=0.1, x1=x[0], x2=x[1], dim=dim)

@handle_frontend_test(fn_tree='torch.nn.functional.pairwise_distance', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), num_arrays=2, min_dim_size=2, max_dim_size=5, min_num_dims=2, min_value=2, max_value=5, allow_inf=False), p=st.integers(min_value=0, max_value=2), keepdim=st.booleans())
def test_torch_pairwise_distance(*, dtype_and_x, p, keepdim, on_device, fn_tree, frontend, test_flags, backend_fw):
    if False:
        while True:
            i = 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, rtol=0.1, x1=x[0], x2=x[1], p=p, keepdim=keepdim)

@handle_frontend_test(fn_tree='torch.nn.functional.pdist', dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes('float'), min_num_dims=2, max_num_dims=2, min_dim_size=10, max_dim_size=10, min_value=1.0, max_value=100000.0), p=st.integers(min_value=0, max_value=100000.0))
def test_torch_pdist(*, dtype_and_x, p, on_device, fn_tree, frontend, test_flags, backend_fw):
    if False:
        return 10
    (input_dtype, x) = dtype_and_x
    helpers.test_frontend_function(input_dtypes=input_dtype, backend_to_test=backend_fw, frontend=frontend, test_flags=test_flags, fn_tree=fn_tree, on_device=on_device, input=x[0], p=p)