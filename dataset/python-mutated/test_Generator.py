from hypothesis import strategies as st
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method
CLASS_TREE = 'ivy.functional.frontends.numpy.random.Generator'

@handle_frontend_method(class_tree=CLASS_TREE, init_tree='numpy.random.Generator', method_name='multinomial', n=helpers.ints(min_value=2, max_value=10), dtype=helpers.get_dtypes('float', full=False), size=st.tuples(st.integers(min_value=1, max_value=10), st.integers(min_value=2, max_value=2)))
def test_numpy_multinomial(n, dtype, on_device, size, init_flags, method_flags, frontend_method_data, frontend, backend_fw):
    if False:
        for i in range(10):
            print('nop')
    helpers.test_frontend_method(init_input_dtypes=dtype, init_flags=init_flags, backend_to_test=backend_fw, method_flags=method_flags, init_all_as_kwargs_np={'bit_generator': np.random.PCG64()}, method_input_dtypes=dtype, method_all_as_kwargs_np={'n': n, 'pvals': np.array([1 / n] * n, dtype=dtype[0]), 'size': size}, frontend=frontend, frontend_method_data=frontend_method_data, test_values=False, on_device=on_device)