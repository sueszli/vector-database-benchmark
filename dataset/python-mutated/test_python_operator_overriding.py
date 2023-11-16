import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import framework
paddle.enable_static()

class TestPythonOperatorOverride(unittest.TestCase):

    def check_result(self, fn, place, dtype):
        if False:
            i = 10
            return i + 15
        shape = [9, 10]
        x_data = np.random.random(size=shape).astype(dtype)
        y_data = np.random.random(size=shape).astype(dtype)
        python_out = fn(x_data, y_data)
        x_var = paddle.static.create_global_var(name='x', shape=shape, value=0.0, dtype=dtype, persistable=True)
        y_var = paddle.static.create_global_var(name='y', shape=shape, value=0.0, dtype=dtype, persistable=True)
        out = fn(x_var, y_var)
        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        base_out = exe.run(base.default_main_program(), feed={'x': x_data, 'y': y_data}, fetch_list=[out])
        np.testing.assert_array_equal(python_out, base_out[0])

    def test_override(self):
        if False:
            for i in range(10):
                print('nop')
        compare_fns = [lambda _a, _b: _a == _b, lambda _a, _b: _a != _b, lambda _a, _b: _a < _b, lambda _a, _b: _a <= _b, lambda _a, _b: _a > _b, lambda _a, _b: _a >= _b]
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        dtypes = ['int32', 'float32']
        for place in places:
            for dtype in dtypes:
                for compare_fn in compare_fns:
                    with framework.program_guard(framework.Program(), framework.Program()):
                        self.check_result(compare_fn, place, dtype)
if __name__ == '__main__':
    unittest.main()