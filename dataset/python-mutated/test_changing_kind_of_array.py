import numpy as np
import ivy.functional.frontends.numpy as np_frontend
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test, BackendHandler

@handle_frontend_test(fn_tree='numpy.asmatrix', arr=helpers.dtype_and_values(min_num_dims=2, max_num_dims=2))
def test_numpy_asmatrix(arr, backend_fw):
    if False:
        print('Hello World!')
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        (dtype, x) = arr
        ret = np_frontend.asmatrix(x[0])
        ret_gt = np.asmatrix(x[0])
        assert ret.shape == ret_gt.shape
        assert ivy_backend.all(ivy_backend.flatten(ret._data) == np.ravel(ret_gt))

@handle_frontend_test(fn_tree='numpy.asscalar', arr=helpers.array_values(dtype=helpers.get_dtypes('numeric'), shape=1))
def test_numpy_asscalar(arr: np.ndarray):
    if False:
        for i in range(10):
            print('nop')
    ret_1 = arr.item()
    ret_2 = np_frontend.asscalar(arr)
    assert ret_1 == ret_2