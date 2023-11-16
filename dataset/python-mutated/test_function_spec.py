import unittest
from test_declarative import foo_func
import paddle
from paddle.jit.dy2static.function_spec import FunctionSpec
from paddle.static import InputSpec
paddle.enable_static()

class TestFunctionSpec(unittest.TestCase):

    def test_constructor(self):
        if False:
            return 10
        foo_spec = FunctionSpec(foo_func)
        args_name = foo_spec.args_name
        self.assertListEqual(args_name, ['a', 'b', 'c', 'd'])
        self.assertTrue(foo_spec.dygraph_function == foo_func)
        self.assertIsNone(foo_spec.input_spec)

    def test_verify_input_spec(self):
        if False:
            i = 10
            return i + 15
        a_spec = InputSpec([None, 10], name='a')
        b_spec = InputSpec([10], name='b')
        with self.assertRaises(TypeError):
            foo_spec = FunctionSpec(foo_func, input_spec=a_spec)
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec, b_spec])
        self.assertTrue(len(foo_spec.flat_input_spec) == 2)

    def test_unified_args_and_kwargs(self):
        if False:
            return 10
        foo_spec = FunctionSpec(foo_func)
        (args, kwargs) = foo_spec.unified_args_and_kwargs([10, 20], {'c': 4})
        self.assertTupleEqual(args, (10, 20, 4, 2))
        self.assertTrue(len(kwargs) == 0)
        (args, kwargs) = foo_spec.unified_args_and_kwargs([], {'a': 10, 'b': 20, 'd': 4})
        self.assertTupleEqual(args, (10, 20, 1, 4))
        self.assertTrue(len(kwargs) == 0)
        (args, kwargs) = foo_spec.unified_args_and_kwargs([10], {'b': 20})
        self.assertTupleEqual(args, (10, 20, 1, 2))
        self.assertTrue(len(kwargs) == 0)
        with self.assertRaises(ValueError):
            foo_spec.unified_args_and_kwargs([10, 20, 30, 40, 50], {'c': 4})
        with self.assertRaises(ValueError):
            foo_spec.unified_args_and_kwargs([10], {'c': 4})

    def test_args_to_input_spec(self):
        if False:
            for i in range(10):
                print('nop')
        a_spec = InputSpec([None, 10], name='a', stop_gradient=True)
        b_spec = InputSpec([10], name='b', stop_gradient=True)
        a_tensor = paddle.static.data(name='a_var', shape=[4, 10])
        b_tensor = paddle.static.data(name='b_var', shape=[4, 10])
        kwargs = {'c': 1, 'd': 2}
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec, b_spec])
        (input_with_spec, _) = foo_spec.args_to_input_spec((a_tensor, b_tensor, 1, 2), {})
        self.assertTrue(len(input_with_spec) == 4)
        self.assertTrue(input_with_spec[0] == a_spec)
        ans_b_spec = InputSpec([4, 10], name='b', stop_gradient=True)
        self.assertTrue(input_with_spec[1] == ans_b_spec)
        self.assertTrue(input_with_spec[2] == 1)
        self.assertTrue(input_with_spec[3] == 2)
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec])
        (input_with_spec, _) = foo_spec.args_to_input_spec((a_tensor, b_tensor), {})
        self.assertTrue(len(input_with_spec) == 2)
        self.assertTrue(input_with_spec[0] == a_spec)
        self.assertTupleEqual(input_with_spec[1].shape, (4, 10))
        self.assertEqual(input_with_spec[1].name, 'b_var')
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec])
        with self.assertRaises(ValueError):
            input_with_spec = foo_spec.args_to_input_spec((a_tensor, b_tensor), {'c': 4})
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec, b_spec])
        with self.assertRaises(ValueError):
            input_with_spec = foo_spec.args_to_input_spec((a_tensor,), {})
if __name__ == '__main__':
    unittest.main()