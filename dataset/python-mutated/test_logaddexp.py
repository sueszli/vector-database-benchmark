import unittest
import numpy as np
import paddle

def ref_logaddexp_old(x, y):
    if False:
        i = 10
        return i + 15
    y = np.broadcast_to(y, x.shape)
    out = np.log1p(np.exp(-np.absolute(x - y))) + np.maximum(x, y)
    return out

def ref_logaddexp(x, y):
    if False:
        return 10
    return np.logaddexp(x, y)

class TestLogsumexpAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.place = paddle.CUDAPlace(0) if paddle.base.core.is_compiled_with_cuda() else paddle.CPUPlace()

    def api_case(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.uniform(-1, 1, self.xshape).astype(self.dtype)
        self.y = np.random.uniform(-1, 1, self.yshape).astype(self.dtype)
        out_ref = ref_logaddexp(self.x, self.y)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.logaddexp(x, y)
        np.testing.assert_allclose(out.numpy(), out_ref, atol=1e-06)

    def test_api(self):
        if False:
            i = 10
            return i + 15
        self.xshape = [1, 2, 3, 4]
        self.yshape = [1, 2, 3, 4]
        self.dtype = np.float64
        self.api_case()

    def test_api_broadcast(self):
        if False:
            while True:
                i = 10
        self.xshape = [1, 2, 3, 4]
        self.yshape = [1, 2, 3, 1]
        self.dtype = np.float32
        self.api_case()

    def test_api_bigdata(self):
        if False:
            return 10
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.float32
        self.api_case()

    def test_api_int32(self):
        if False:
            for i in range(10):
                print('nop')
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.int32
        self.api_case()

    def test_api_int64(self):
        if False:
            print('Hello World!')
        self.xshape = [10, 200, 300]
        self.yshape = [10, 200, 300]
        self.dtype = np.int64
        self.api_case()
if __name__ == '__main__':
    unittest.main()