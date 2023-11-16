import unittest
import cinn
import numpy as np
import scipy
from cinn import Target, ir, lang, pe, runtime
from cinn.poly import create_stages

class TestPEElementwise(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.m = 32
        self.n = 32
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k32
        self.target.os = Target.OS.Linux
        cinn.set_target(self.target)
        self.unary_data = []

    def test_unary(self):
        if False:
            return 10
        for (fn_name, pe_fn, np_fn, dtype, low, high) in [('exp', pe.exp, np.exp, 'float32', -10, 10), ('erf', pe.erf, scipy.special.erf, 'float32', -99, 99), ('sqrt', pe.sqrt, np.sqrt, 'float32', 0.1, 10), ('log', pe.log, np.log, 'float32', 0.1, 99), ('log2', pe.log2, np.log2, 'float32', 0.1, 99), ('log10', pe.log10, np.log10, 'float32', 0.1, 99), ('floor', pe.floor, np.floor, 'float32', -99, 99), ('ceil', pe.ceil, np.ceil, 'float32', -99, 99), ('round', pe.round, np.round, 'float32', -99, 99), ('trunc', pe.trunc, np.trunc, 'float32', -99, 99), ('cos', pe.cos, np.cos, 'float32', -2.0 * np.pi, 2.0 * np.pi), ('cosh', pe.cosh, np.cosh, 'float32', -2.0 * np.pi, 2.0 * np.pi), ('tan', pe.tan, np.tan, 'float32', -2.0 * np.pi, 2.0 * np.pi), ('tanh', pe.tanh, np.tanh, 'float32', -2.0 * np.pi, 2.0 * np.pi), ('tanh', pe.tanh, np.tanh, 'float32', -2.0 * np.pi, 2.0 * np.pi), ('sin', pe.sin, np.sin, 'float32', -2.0 * np.pi, 2.0 * np.pi), ('sinh', pe.sinh, np.sinh, 'float32', -2.0 * np.pi, 2.0 * np.pi), ('isnan', pe.isnan, np.isnan, 'float32', -99, 99), ('isfinite', pe.isfinite, np.isfinite, 'float32', -99, 99), ('isinf', pe.isinf, np.isinf, 'float32', -99, 99), ('negative', pe.negative, np.negative, 'float32', -99, 99), ('bitwise_not', pe.bitwise_not, np.bitwise_not, 'int32', -99, 99), ('sigmoid', pe.sigmoid, lambda x: 1 / (1 + np.exp(-x)), 'float32', -99, 99), ('sign', pe.sign, np.sign, 'float32', -99, 99), ('abs', pe.abs, np.abs, 'float32', -99, 99), ('rsqrt', pe.rsqrt, lambda x: np.ones_like(x) / np.sqrt(x), 'float32', 0.1, 99)]:
            self.compiler = cinn.Compiler.create(self.target)
            is_round = fn_name == 'round'
            is_bool = (fn_name == 'isnan') | (fn_name == 'isfinite') | (fn_name == 'isinf') | (fn_name == 'logical_not')
            self.union_tester(fn_name, pe_fn, np_fn, dtype, low, high, is_round, is_bool)

    def union_tester(self, fn_name, cinn_fn, np_fn, dtype='float32', low=0, high=1, is_round=False, is_bool=False):
        if False:
            while True:
                i = 10
        (m, n) = (ir.Expr(_) for _ in (self.m, self.n))
        x = lang.Placeholder(dtype, 'x', [m, n])
        y = cinn_fn(x.to_tensor())
        func_name = 'test_' + fn_name
        args = [x.to_tensor()]
        for out in y:
            args.append(out)
        stages = create_stages(args)
        func = lang.lower(func_name, stages, args)
        builder = lang.Module.Builder('elementwise_module', self.target)
        builder.add_function(func)
        module = builder.build()
        self.compiler.build(module)
        fn = self.compiler.lookup(func_name)
        (x_data, x_buf, out_buf, *args) = self.create_data(dtype, low, high, is_round, is_bool)
        fn(args)
        self.assertTrue(np.allclose(out_buf.numpy(), self.create_target_data(x_data, np_fn), atol=0.0001), func_name)

    def create_target_data(self, x_data, np_target_fn):
        if False:
            return 10
        return np_target_fn(x_data)

    def create_data(self, dtype, low, high, is_round, is_bool):
        if False:
            return 10
        self.unary_data.clear()
        if not self.unary_data:
            x_data = np.around(np.random.uniform(low, high, (self.m, self.n)).astype(dtype), 2)
            if is_round:
                x_data += (np.abs(np.fmod(x_data, 1)) - 0.5 < 1e-06) * 0.0001
            x = runtime.cinn_buffer_t(x_data, runtime.cinn_x86_device)
            if is_bool:
                out = runtime.cinn_buffer_t(np.zeros([self.m, self.n]).astype(np.bool_), runtime.cinn_x86_device)
            else:
                out = runtime.cinn_buffer_t(np.zeros([self.m, self.n]).astype(dtype), runtime.cinn_x86_device)
            self.unary_data = [x_data, x, out, runtime.cinn_pod_value_t(x), runtime.cinn_pod_value_t(out)]
        return self.unary_data
if __name__ == '__main__':
    unittest.main()