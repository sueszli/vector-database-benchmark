import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir_exe_and_pir_api
import paddle
SEED = 2020
np.random.seed(SEED)

class A:
    pass

def function(x: A) -> A:
    if False:
        for i in range(10):
            print('nop')
    t: A = A()
    return 2 * x

class TestTypeHint(Dy2StTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        self.x = np.zeros(shape=1, dtype=np.int32)
        self._init_dyfunc()

    def _init_dyfunc(self):
        if False:
            for i in range(10):
                print('nop')
        self.dyfunc = function

    def _run_static(self):
        if False:
            print('Hello World!')
        return self._run(to_static=True)

    def _run_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        return self._run(to_static=False)

    def _run(self, to_static):
        if False:
            while True:
                i = 10
        tensor_x = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(tensor_x)
        else:
            ret = self.dyfunc(tensor_x)
        if hasattr(ret, 'numpy'):
            return ret.numpy()
        else:
            return ret

    @test_legacy_and_pir_exe_and_pir_api
    def test_ast_to_func(self):
        if False:
            return 10
        static_numpy = self._run_static()
        dygraph_numpy = self._run_dygraph()
        print(static_numpy, dygraph_numpy)
        np.testing.assert_allclose(dygraph_numpy, static_numpy, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()