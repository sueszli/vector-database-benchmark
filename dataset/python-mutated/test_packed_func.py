import unittest
from math import isclose
import cinn
from cinn import ir

class TestPackedFunc(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_lambda(self):
        if False:
            i = 10
            return i + 15
        add3 = ir.register_packed_func('test_packed_func_add3')(lambda x, y, z: x + y + z)
        self.assertEqual(add3(1, 2, 3), 6)
        self.assertEqual(ir.get_global_func('test_packed_func_add3'), add3)
        self.assertTrue(isinstance(add3, ir.PackedFunc))

    def test_normal_function(self):
        if False:
            i = 10
            return i + 15

        @ir.register_packed_func('test_packed_func_mul')
        def mul(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x * y
        self.assertTrue(isclose(mul(2.3, 3.0), 6.9, abs_tol=1e-05))
        self.assertEqual(mul(4, 5), 20)

    def test_callable_object(self):
        if False:
            while True:
                i = 10

        class Accumulator:

            def __init__(self, init):
                if False:
                    while True:
                        i = 10
                self.init = init

            def __call__(self, *args):
                if False:
                    i = 10
                    return i + 15
                r = cinn.CINNValue(self.init)
                for arg in args:
                    r = r + arg
                return r
        accumulate = ir.register_packed_func('accumulate_float')(Accumulator(1.0))
        self.assertTrue(isclose(accumulate(1.0, 2.0, 3.0, 4.0), 11.0))

    def test_cxx_register(self):
        if False:
            print('Hello World!')
        add_int = ir.Registry.get('test_add_int64')
        self.assertEqual(add_int(2, 3), 5)
        add_expr = ir.Registry.get('test_add_expr')
        x = ir.Expr(1)
        y = ir.Expr(2)
        z = x + y
        r = add_expr(x, y)
        self.assertEqual(r.node_type(), z.node_type())
        mul_float = ir.Registry.get('test_mul_float')
        self.assertTrue(isclose(mul_float(2.4, 2.5), 6.0, abs_tol=1e-05))
if __name__ == '__main__':
    unittest.main()