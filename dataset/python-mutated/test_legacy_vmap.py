from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn.functional as F
from torch import Tensor
from torch._vmap_internals import vmap
import functools
import itertools
import warnings
from torch.testing._internal.common_device_type import instantiate_device_type_tests
import types
FALLBACK_REGEX = 'There is a performance drop'

class EnableVmapFallbackWarnings:

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        if False:
            while True:
                i = 10
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)

class TestVmapAPI(TestCase):

    def test_non_tensor_output_raises(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, "got type <class 'float'> as the return"):
            output = vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            if False:
                return 10
            return (x, 3)
        with self.assertRaisesRegex(ValueError, "got type <class 'int'> for return 1"):
            vmap(multiple_outputs)(torch.ones(3))

    def test_different_map_dim_size_raises(self):
        if False:
            while True:
                i = 10
        x = torch.randn(2)
        y = torch.randn(3)
        expected_msg = 'Expected all tensors to have the same size in the mapped dimension'
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z['x'] + z['y'], in_dims=({'x': 0, 'y': 0},))({'x': x, 'y': y})

    def test_func_with_no_inputs(self):
        if False:
            return 10
        expected_msg = 'got no inputs'

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            return torch.randn(3)

        def bar(x):
            if False:
                while True:
                    i = 10
            return torch.randn(3)
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(foo)()
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(bar)()

    def test_constant_function(self):
        if False:
            while True:
                i = 10
        output = vmap(lambda x: torch.tensor(3.14))(torch.ones(3))
        self.assertEqual(output, torch.tensor([3.14, 3.14, 3.14]))

    def test_single_input(self):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 3)

        def square(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x
        output = vmap(square)(x)
        self.assertEqual(output, x * x)

    def test_multiple_inputs(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul)(x, y)
        self.assertEqual(output, x * y)

    def test_multiple_outputs(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                while True:
                    i = 10
            return (x * x, x * x * x)
        x = torch.randn(3)
        outputs = vmap(foo)(x)
        self.assertEqual(outputs[0], x * x)
        self.assertEqual(outputs[1], x * x * x)

    def test_multiple_outputs_error_cases(self):
        if False:
            while True:
                i = 10

        def returns_tuple_of_tensors(x):
            if False:
                return 10
            return (x, x)

        def returns_list_of_two_tensors(x):
            if False:
                print('Hello World!')
            return [x, x]

        def returns_list_of_one_tensor(x):
            if False:
                return 10
            return [x]
        x = torch.randn(3)
        vmap(returns_tuple_of_tensors)(x)
        msg = "must only return Tensors, got type <class 'list'>"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_two_tensors)(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_one_tensor)(x)

    def test_nested_with_same_map_dim(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        output = vmap(vmap(torch.mul))(x, y)
        self.assertEqual(output, x * y)
        output = vmap(vmap(vmap(torch.mul)))(x, y)
        self.assertEqual(output, x * y)

    def test_nested_with_different_map_dim(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        output = vmap(lambda x: vmap(lambda y: x * y)(y))(x)
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)
        z = torch.randn(7, 3)
        output = vmap(lambda x: vmap(lambda y: vmap(lambda z: x * y * z)(z))(y))(x)
        self.assertEqual(output.shape, (2, 5, 7, 3))
        self.assertEqual(output, x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z)

    def test_noop_in_inner_vmap(self):
        if False:
            while True:
                i = 10
        x = torch.randn(3)
        y = torch.randn(5)
        output = vmap(lambda x: vmap(lambda y: x)(y))(x)
        self.assertEqual(output, x.view(3, 1).expand(3, 5))

    def test_unsupported_op_err_msg(self):
        if False:
            return 10
        tensor = torch.randn(2, 3)
        msg = "Batching rule not implemented for aten::.+; the fallback path doesn't work on out= or view ops"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(torch.ravel)(tensor)

        def out_op(x, y):
            if False:
                print('Hello World!')
            return torch.abs(x, out=y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)
        tensor = torch.randn(2)
        with self.assertRaisesRegex(RuntimeError, 'Batching rule not implemented'):
            vmap(lambda t: torch.atleast_1d([t]))(tensor)
        with self.assertRaisesRegex(RuntimeError, 'Batching rule not implemented'):
            vmap(torch.Tensor.item)(tensor)

    def test_nonzero_out_dims(self):
        if False:
            i = 10
            return i + 15
        tensor = torch.randn(2, 3)
        result = vmap(lambda x: x, out_dims=1)(tensor)
        self.assertEqual(result, tensor.permute(1, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 0, 3))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=-1)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 3, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())
        tensor = torch.randn(2, 3, 5, 7)
        other = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x, y: (x, y), out_dims=2)(tensor, other)
        self.assertEqual(result, (tensor.permute(1, 2, 0, 3), other.permute(1, 2, 0, 3)))
        ndims = 64
        shape = [2] + [1] * (ndims - 1)
        expected_shape = [1, 1, 2] + [1] * (ndims - 3)
        tensor = torch.randn(shape)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result.shape, expected_shape)

        def foo(x, y):
            if False:
                while True:
                    i = 10
            return (x, x * y, x * y * y)
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=1)(x, y)
        self.assertEqual(result, (x.permute(1, 0, 2), (x * y).permute(1, 0, 2), (x * y * y).permute(1, 0, 2)))

    def test_multiple_out_dims(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                i = 10
                return i + 15
            return (x, x)

        def bar(x, y):
            if False:
                print('Hello World!')
            return (x, x, x, x * y)
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=(0, 1))(x)
        self.assertEqual(result, (x, x.permute(1, 0, 2)))
        result = vmap(bar, out_dims=(-1, 0, 1, 2))(x, y)
        expected = (x.permute(1, 2, 0), x, x.permute(1, 0, 2), (x * y).permute(1, 2, 0))
        self.assertEqual(result, expected)

    def test_nested_out_dims(self):
        if False:
            i = 10
            return i + 15
        y = torch.randn(2, 3, 5, 7)
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y))(y)
        self.assertEqual(result.shape, (2, 5, 3, 7))
        self.assertEqual(result, y.permute(0, 2, 1, 3))
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y), out_dims=1)(y)
        self.assertEqual(result.shape, (5, 2, 3, 7))
        self.assertEqual(result, y.permute(2, 0, 1, 3))
        result = vmap(lambda y: vmap(lambda x: x, out_dims=-1)(y), out_dims=-1)(y)
        self.assertEqual(result.shape, (5, 7, 3, 2))
        self.assertEqual(result, y.permute(2, 3, 1, 0))
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        result = vmap(lambda y: vmap(lambda x: x * y, out_dims=1)(x), out_dims=-1)(y)
        self.assertEqual(result.shape, (3, 2, 5))
        self.assertEqual(result, (y.view(5, 1, 3) * x).permute(2, 1, 0))

    def test_out_dims_edge_case(self):
        if False:
            while True:
                i = 10

        def foo(x):
            if False:
                print('Hello World!')
            return x
        tensor = torch.randn(2, 3)
        expected = vmap(foo, out_dims=1)(tensor)
        result = vmap(foo, out_dims=(1,))(tensor)
        self.assertEqual(result, expected)

    def test_out_dims_must_be_int_or_tuple_of_int_err_msg(self):
        if False:
            i = 10
            return i + 15
        msg = '`out_dims` must be an int or a tuple of int'
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims='lol')(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=('lol',))(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=None)(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(None,))(tensor)

    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        if False:
            while True:
                i = 10
        msg = '`out_dims` must have one dim per output'
        x = torch.randn(2, 3, 5)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(0, 0))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0, 0, 0))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x), out_dims=(0,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0))(x)

    def test_out_dim_out_of_bounds_err_msg(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Dimension out of range'
        x = torch.randn(2, 3, 5)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=3)(x)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=-4)(x)

    def test_non_zero_in_dims(self):
        if False:
            while True:
                i = 10
        tensor = torch.randn(2, 3, 5)
        output = vmap(lambda x: x, (1,))(tensor)
        self.assertEqual(output, tensor.permute(1, 0, 2))
        self.assertEqual(output.data_ptr(), tensor.data_ptr())
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        output = vmap(torch.mul, (0, 1))(x, y)
        self.assertEqual(output, x * y.t())
        output = vmap(torch.mul, (1, 0))(x, y)
        self.assertEqual(output, x.t() * y)

    def test_none_in_dims(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul, (0, None))(x, y)
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)
        output = vmap(torch.mul, (0, None))(x, 2)
        self.assertEqual(output, x * 2)

    def test_nested_non_default_in_dims(self):
        if False:
            print('Hello World!')
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.mul), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    def test_non_default_in_dims_out_dims(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(2, 3, 5)
        result = vmap(lambda x: x, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x)
        self.assertEqual(result.data_ptr(), x.data_ptr())
        result = vmap(lambda x: x, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, x.transpose(1, 2))
        self.assertEqual(result.data_ptr(), x.data_ptr())

        def foo(x):
            if False:
                i = 10
                return i + 15
            return x * 2
        result = vmap(foo, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x * 2)
        result = vmap(foo, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, (x * 2).transpose(1, 2))
        result = vmap(vmap(foo, 1, 1), 1, 1)(x)
        self.assertEqual(result, x * 2)

    def test_accepts_nested_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        B0 = 2
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        out = vmap(lambda z: z[0] + z[1])((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1])([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z['x'] + z['y'])({'x': x, 'y': y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z['x'] + z['y'], in_dims=(0,))({'x': x, 'y': y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z['x'] + z['y'], in_dims=({'x': 0, 'y': 0},))({'x': x, 'y': y})
        self.assertEqual(out, x + y)
        out_fn = vmap(lambda z: z['x'][0] + z['x'][1][0] + z['y'][0] + z['y'][1])
        out = out_fn({'x': [x, (x,)], 'y': [y, y]})
        self.assertEqual(out, x + x + y + y)

    def test_in_dims_wrong_type_err_msg(self):
        if False:
            print('Hello World!')
        x = torch.randn(3)
        y = torch.randn(3)
        msg = 'expected `in_dims` to be int or a \\(potentially nested\\) tuple'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, [0, 0])(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, set({0}))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, 'lol')(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=[0, 0])([x, y])
        vmap(torch.mul, (0, 0))(x, y)

    def test_not_enough_in_dims_err_msg(self):
        if False:
            print('Hello World!')
        x = torch.randn(3)
        y = torch.randn(3)
        msg = 'in_dims is not compatible with the structure of `inputs`'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0,))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0, 0, 0))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0],))([x, y])
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))([x, y])
        vmap(torch.mul, (0, 0))(x, y)

    def test_integer_in_dim_but_not_tensor_input_err_msg(self):
        if False:
            i = 10
            return i + 15

        def foo(xy):
            if False:
                for i in range(10):
                    print('nop')
            return xy[0] * xy[1]

        def bar(x, yz):
            if False:
                return 10
            return x * yz[0] * yz[1]
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        msg = 'Got in_dim=0 for an input but the input is of type'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum)(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum, (0, 0))(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, 1])
        vmap(torch.sum, (0, None))(x, 0)

    def test_in_dim_not_in_tensor_err_msg(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        msg = 'Got in_dim=-?\\w for some input, but that input is a Tensor of dimensionality \\w'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-1,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def test_fallback_does_not_warn_by_default(self):
        if False:
            while True:
                i = 10
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            result = vmap(op)(x, y)
            self.assertEqual(len(wa), 1)

    def test_fallback_warns_when_warnings_are_enabled(self):
        if False:
            return 10
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                result = vmap(op)(x, y)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                result = vmap(*vmap_args)(*inputs)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def test_fallback_zero_dim(self):
        if False:
            print('Hello World!')
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        self._assert_uses_vmap_fallback((op,), (x, y))
        (B0, B1) = (0, 3)
        x = torch.randn(B0, 11)
        y = torch.randn(11)
        msg = 'The fallback path does not support vmap over dims of size 0'
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)
        x = torch.randn(B0, B1, 11)
        y = torch.randn(B1, 11)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

    def test_fallback_atan2(self):
        if False:
            i = 10
            return i + 15
        op = torch.atan2
        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)
        self._assert_uses_vmap_fallback((op,), (x, y))
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))

    def test_fallback_masked_fill(self):
        if False:
            for i in range(10):
                print('nop')

        def run_test(batch_size):
            if False:
                while True:
                    i = 10
            B0 = batch_size
            x = torch.randn(B0, 7, 11, 13)
            dim = 0
            index = torch.tensor([0, 4, 2])
            values = torch.randn(B0, 3, 11, 13)
            self._assert_uses_vmap_fallback((torch.index_add, (0, None, None, 0)), (x, dim, index, values))
            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            expected = torch.index_add(x, dim + 1, index, values.view(B0, 3, 11, 13))
            self.assertEqual(result, expected)
        run_test(batch_size=5)
        run_test(batch_size=1237)

    def test_fallback_multiple_returns(self):
        if False:
            for i in range(10):
                print('nop')
        (B0, B1, B2) = (2, 3, 1237)
        tensor = torch.randn(B0, 10)
        self._assert_uses_vmap_fallback((torch.var_mean,), (tensor,))
        result = vmap(torch.var_mean)(tensor)
        expected = torch.var_mean(tensor, dim=1)
        self.assertEqual(result, expected)
        tensor = torch.randn(B0, B1, 10)
        result = vmap(vmap(torch.var_mean))(tensor)
        expected = torch.var_mean(tensor, dim=2)
        self.assertEqual(result, expected)
        tensor = torch.randn(B0, B1, B2, 10)
        result = vmap(vmap(vmap(torch.var_mean)))(tensor)
        expected = torch.var_mean(tensor, dim=3)
        self.assertEqual(result, expected)

    def test_inplace_fallback_unary(self):
        if False:
            i = 10
            return i + 15
        op = Tensor.acos_
        (B0, B1, B2) = (2, 3, 10000)
        x = torch.randn(B0, 5)
        self._assert_uses_vmap_fallback((op,), (x,))
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op)(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op, out_dims=(1,))(x)
        self.assertTrue(result._base is x)
        self.assertEqual(result, x_orig.t().acos())
        x_orig = torch.randn(B0, B1, 5)
        x = x_orig.clone()
        result = vmap(vmap(op))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        result = vmap(vmap(vmap(op)))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

    def test_inplace_fallback_nary_same_levels(self):
        if False:
            return 10
        op = Tensor.atan2_
        outplace_op = torch.atan2
        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)
        self._assert_uses_vmap_fallback((op,), (x, y))
        B0 = 5
        x_orig = torch.randn(7, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, 7, 11)
        vmap(op, (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim(0, 2)))
        (B0, B1) = (5, 7)
        x_orig = torch.randn(B1, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, B1, 11)
        vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim([0, 1], [2, 0])))
        (B0, B1, B2) = (100, 10, 10)
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        y = torch.randn(B0, B1, B2)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))

    def test_inplace_fallback_nary_different_levels(self):
        if False:
            print('Hello World!')
        op = Tensor.atan2_
        outplace_op = torch.atan2
        (B0, B1, B2) = (2, 3, 5)
        x = torch.rand(B0, 7)
        y = torch.rand(7)
        self._assert_uses_vmap_fallback((op, (0, None)), (x, y))
        x_orig = torch.rand(B0, 7)
        x = x_orig.clone()
        y = torch.rand(7)
        vmap(op, in_dims=(0, None))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y))
        x_orig = torch.rand(B0, B1, 7)
        x = x_orig.clone()
        y = torch.rand(B0, 7)
        vmap(vmap(op, in_dims=(0, None)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, 1, 7)))
        msg = 'vmap: aten::atan2_\\(self, \\*extra_args\\) is not possible'
        x = torch.rand(7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(x, y)
        x = torch.rand(B1, 7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 0))(x, y)
        x = torch.rand(B1, 7)
        y = torch.rand(7, B0)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 1))(x, y)
        x = torch.rand(B0, 7)
        y = torch.rand(B0, B1, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(None, 0)))(x, y)

    def test_backward_unsupported_interaction(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(5)
        grad = torch.randn_like(x)
        err_msg = 'backward\\(\\) called inside torch.vmap'

        def backward_on_vmapped_tensor(x):
            if False:
                for i in range(10):
                    print('nop')
            x.sum().backward()
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_on_vmapped_tensor)(x)

        def backward_with_vmapped_grad(x, grad):
            if False:
                i = 10
                return i + 15
            x.backward(grad)
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_with_vmapped_grad)(x, grad)

        def completely_unrelated_backward(y):
            if False:
                return 10
            x.sum().backward()
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(completely_unrelated_backward)(y)

    def test_grad_unsupported_interaction(self):
        if False:
            for i in range(10):
                print('nop')
        input_tensor = torch.randn(3, requires_grad=True)
        err_msg = 'autograd.grad.* called inside torch.vmap'
        captured = torch.randn(3, requires_grad=True)

        def output_to_grad_is_vmapped(input_tensor):
            if False:
                print('Hello World!')
            output = (captured * input_tensor).sum()
            return torch.autograd.grad([output], [captured])[0]
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(output_to_grad_is_vmapped)(input_tensor)
        output = (input_tensor ** 2).sum()

        def input_to_grad_is_vmapped(input_tensor):
            if False:
                while True:
                    i = 10
            return torch.autograd.grad([output], [input_tensor])[0]
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(input_to_grad_is_vmapped)(input_tensor)

    def test_batched_gradient_basic(self):
        if False:
            return 10
        N = 3
        x = torch.randn(N, requires_grad=True)
        y = torch.randn(N)

        def vjp_mul(v):
            if False:
                while True:
                    i = 10
            return torch.autograd.grad([x * y], [x], grad_outputs=[v])[0]
        batched_v = torch.eye(N)
        jacobian = vmap(vjp_mul)(batched_v)
        self.assertEqual(jacobian, torch.diagflat(y))

    def test_functools_partial(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(3)
        y = torch.randn(2, 3)
        result = vmap(functools.partial(torch.mul, x))(y)
        self.assertEqual(result, x * y)

    def test_nn_module(self):
        if False:
            for i in range(10):
                print('nop')
        tensor = torch.randn(2, 3)
        model = torch.nn.Linear(3, 3, bias=False)
        result = vmap(model)(tensor)
        self.assertEqual(result, model(tensor))

    def test_fallback_with_undefined_grad(self):
        if False:
            return 10
        B0 = 7
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        weight = torch.randn(3, 3, 1, 1)
        v = torch.randn(B0, 2, 3, 4, 5)

        def get_vjp(v):
            if False:
                while True:
                    i = 10
            result = torch.nn.functional.conv2d(x, weight)
            (grad_x,) = torch.autograd.grad(result, x, v)
            return grad_x
        self._assert_uses_vmap_fallback([get_vjp], [v])

def slice_inputs(inputs, bdims, i):
    if False:
        return 10
    result = []
    for (inp, bdim) in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)

def reference_vmap(op, inputs, in_dims=0, out_dims=0):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(in_dims, int):
        in_dims = (in_dims,) * len(inputs)
    bdim_sizes = [inp.size(dim) for (inp, dim) in zip(inputs, in_dims) if dim is not None]
    assert all((bdim_size == bdim_sizes[0] for bdim_size in bdim_sizes))
    bdim_size = bdim_sizes[0]
    results = tuple((op(*slice_inputs(inputs, in_dims, i)) for i in range(bdim_size)))
    assert len(results) > 0
    op_has_single_return = not isinstance(results[0], tuple)
    if op_has_single_return:
        assert all((isinstance(result, torch.Tensor) for result in results))
        if isinstance(out_dims, int):
            out_dims = (out_dims,) * 1
        return torch.stack(results, dim=out_dims[0])
    assert all((isinstance(result, tuple) for result in results))
    num_returns = len(results[0])
    assert all((len(result) == num_returns for result in results))
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
    return tuple((torch.stack(result_shards, out_dim) for (result_shards, out_dim) in zip(zip(*results), out_dims)))

class TensorFactory:

    @staticmethod
    def rand(size, device='cpu', dtype=torch.float):
        if False:
            return 10
        return torch.rand(size, device=device, dtype=dtype)

    @staticmethod
    def randn(size, device='cpu', dtype=torch.float):
        if False:
            i = 10
            return i + 15
        return torch.randn(size, device=device, dtype=dtype)

    @staticmethod
    def randp1(size, device='cpu', dtype=torch.float):
        if False:
            print('Hello World!')
        return torch.rand(size, device=device, dtype=dtype) + 1

def _vmap_test(self, op, inputs, in_dims=0, out_dims=0, check_view=False, check_propagates_grad=True):
    if False:
        for i in range(10):
            print('nop')
    result = vmap(op, in_dims, out_dims)(*inputs)
    reference_result = reference_vmap(op, inputs, in_dims, out_dims)
    self.assertEqual(result, reference_result)
    op_has_single_return = not isinstance(result, tuple)
    if check_view:
        result_as_tuple = (result,) if op_has_single_return else result
        for output in result_as_tuple:
            input0_base = inputs[0] if inputs[0]._base is None else inputs[0]._base
            self.assertTrue(output._base is input0_base, msg='result was not a view of the first input!')
    if not check_propagates_grad:
        return
    inputs_clone = list(inputs)
    inputs_clone[0] = inputs[0].clone().requires_grad_()
    result = vmap(op, in_dims, out_dims)(*inputs_clone)
    result_as_tuple = (result,) if op_has_single_return else result
    self.assertTrue(result[0].requires_grad)

def should_allow_vmap_fallback_usage(fn):
    if False:
        print('Hello World!')
    return getattr(fn, '_allow_vmap_fallback_usage', False)

def allowVmapFallbackUsage(fn):
    if False:
        return 10
    fn._allow_vmap_fallback_usage = True
    return fn

class Namespace:

    class TestVmapBase(TestCase):

        def __init__(self, method_name='runTest'):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(method_name)
            test_method = getattr(self, method_name, None)
            if test_method is None:
                return
            if not should_allow_vmap_fallback_usage(test_method):
                setattr(self, method_name, self._wrap_method_with_vmap_fallback_check(test_method))

        def _wrap_method_with_vmap_fallback_check(self, method):
            if False:
                return 10
            msg = 'Expected the test to not invoke the vmap fallback path, i.e., all of the operators being tested in this test should have batching rules implemented. If you are intentionally testing something to do with the fallback path, use allowVmapFallbackUsage. Otherwise, please make sure that batching rules are implemented for the operator(s) being tested.'

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                with warnings.catch_warnings(record=True) as wa:
                    warnings.simplefilter('always')
                    with EnableVmapFallbackWarnings():
                        method(*args, **kwargs)
                    for captured_warning in wa:
                        self.assertNotRegex(str(captured_warning.message), FALLBACK_REGEX, msg)
            return types.MethodType(wrapper, self)

        @allowVmapFallbackUsage
        def test_vmap_fallback_check_ok(self):
            if False:
                print('Hello World!')
            op_using_fallback = torch.var_mean
            vmap(op_using_fallback)(torch.rand(3))

        def test_vmap_fallback_check(self):
            if False:
                return 10

            @self._wrap_method_with_vmap_fallback_check
            def no_fallback(self):
                if False:
                    while True:
                        i = 10
                pass
            op_using_fallback = torch.var_mean

            @self._wrap_method_with_vmap_fallback_check
            def uses_fallback(self):
                if False:
                    print('Hello World!')
                vmap(op_using_fallback)(torch.rand(3))
            no_fallback(self)
            with self.assertRaises(AssertionError):
                uses_fallback(self)

class TestVmapOperators(Namespace.TestVmapBase):

    def _vmap_test(self, *args, **kwargs):
        if False:
            return 10
        return _vmap_test(self, *args, **kwargs)

    def _vmap_view_test(self, *args, **kwargs):
        if False:
            return 10
        self._vmap_test(*args, **kwargs, check_view=True)

    def _test_unary(self, op, getter, device, *args, **kwargs):
        if False:
            while True:
                i = 10
        test = functools.partial(self._vmap_test, *args, **kwargs)
        (B0, B1) = (7, 11)
        test(op, [getter([B0, 3], device)])
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2)
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2, out_dims=2)
        test(vmap(op), [getter([B0, B1], device)])
        test(vmap(op), [getter([B1, 2, 5, B0, 3], device)], in_dims=2)
        test(vmap(op, in_dims=2), [getter([2, 5, B0, B1, 3], device)], in_dims=2, out_dims=2)

    def test_unary_pointwise_ops(self):
        if False:
            return 10
        cases = [(torch.abs, TensorFactory.randn), (torch.acos, TensorFactory.rand), (torch.asin, TensorFactory.rand), (torch.atan, TensorFactory.rand), (torch.ceil, TensorFactory.randn), (torch.cos, TensorFactory.rand), (torch.cosh, TensorFactory.rand), (torch.digamma, TensorFactory.rand), (torch.exp, TensorFactory.randn), (torch.expm1, TensorFactory.randn), (torch.floor, TensorFactory.randn), (torch.frac, TensorFactory.randn), (torch.lgamma, TensorFactory.rand), (torch.log, TensorFactory.randp1), (torch.log10, TensorFactory.randp1), (torch.log1p, TensorFactory.randp1), (torch.log2, TensorFactory.randp1), (torch.neg, TensorFactory.randn), (torch.reciprocal, TensorFactory.randp1), (torch.relu, TensorFactory.randn), (torch.round, TensorFactory.randn), (torch.rsqrt, TensorFactory.randp1), (torch.sigmoid, TensorFactory.randn), (torch.sign, TensorFactory.randn), (torch.sin, TensorFactory.rand), (torch.sinh, TensorFactory.rand), (torch.sqrt, TensorFactory.rand), (torch.tan, TensorFactory.rand), (torch.tanh, TensorFactory.rand), (torch.trunc, TensorFactory.randn)]
        for (op, getter) in cases:
            self._test_unary(op, getter, 'cpu')

    def test_clone(self):
        if False:
            return 10
        self._test_unary(lambda x: x.clone(), TensorFactory.randn, 'cpu')
        self._test_unary(lambda x: x.clone(memory_format=torch.preserve_format), TensorFactory.randn, 'cpu')
        self._test_unary(lambda x: x.clone(memory_format=torch.contiguous_format), TensorFactory.randn, 'cpu')

        def clone_contiguous(x):
            if False:
                return 10
            return x.clone(memory_format=torch.contiguous_format)
        (B0, B1) = (3, 5)
        x = torch.randn(2, B0, 7)
        y = vmap(clone_contiguous, in_dims=1, out_dims=1)(x)
        self.assertTrue(y.movedim(1, 0).is_contiguous())
        self.assertTrue(y[:, 0, :].is_contiguous())
        x = torch.randn(2, B0, 7, B1)
        y = vmap(vmap(clone_contiguous, in_dims=2), in_dims=1)(x)
        self.assertTrue(y.is_contiguous())
        self.assertTrue(y[0][0].is_contiguous())
        msg = 'only supported with memory_format torch.preserve_format or torch.contiguous_format'
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last))(torch.randn(B0))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last_3d))(torch.randn(B0))

    def test_binary_pointwise_ops(self):
        if False:
            print('Hello World!')

        def get_number(getter):
            if False:
                i = 10
                return i + 15
            return getter([]).item()

        def make_case(op, input_getter=TensorFactory.randn):
            if False:
                while True:
                    i = 10
            return (op, input_getter)
        cases = [make_case(torch.add), make_case(lambda x, y: x + y), make_case(torch.sub), make_case(lambda x, y: x - y), make_case(torch.mul), make_case(lambda x, y: x * y), make_case(torch.div, input_getter=TensorFactory.randp1), make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1), make_case(torch.pow, input_getter=TensorFactory.randp1), make_case(lambda x, y: x ** y, input_getter=TensorFactory.randp1)]
        test = self._vmap_test
        for (op, getter) in cases:
            device = 'cpu'
            (B0, B1) = (7, 11)
            test(op, (getter([B0, 3], device), getter([B0, 3], device)))
            test(op, (getter([B0], device), getter([B0, 2, 3], device)))
            test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
            test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1), out_dims=1)
            test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
            test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(0, None))
            test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
            test(vmap(op, in_dims=(None, 0)), (getter([B0, 2, 3], device), getter([B1, 3], device)), in_dims=(0, None))
            number = get_number(getter)
            self._test_unary(lambda t: op(t, number), getter, device)
            number = get_number(getter)
            self._test_unary(lambda t: op(number, t), getter, device)
            test(op, (getter([B0], device), getter([B0], device, dtype=torch.double)))
            test(op, (getter([B0], device, dtype=torch.double), getter([B0], device)))
            test(op, (getter([B0], device), getter([B0], device)))
            test(op, (getter([B0, 2], device), getter([B0], device, torch.double)))
            test(op, (getter([B0], device, torch.double), getter([B0, 2], device)))
            if not torch.cuda.is_available():
                continue

    def test_as_strided(self):
        if False:
            print('Hello World!')

        def _test(sizes, strides, offset, tensor, lambd):
            if False:
                print('Hello World!')
            result = vmap(lambda t: t.as_strided(sizes, strides, offset))(tensor)
            expected = vmap(lambd)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)
        B0 = 5
        tensors = [torch.randn(B0, 2, 3), torch.randn(B0, 3, 2).transpose(1, 2), torch.randn(2, B0, 2, 3)[1], torch.randn(B0, 2, 4, 3, 7)[:, :, 0, :, 0], torch.randn(B0, 2, 4, 3, 7)[:, :, 2, :, 1]]
        for x in tensors:
            (S0, S1) = x.stride()[1:]
            offset = x.storage_offset()
            _test([5, 5, 2, 3], [0, 0, S0, S1], offset, x, lambda x: x.expand(5, 5, 2, 3))
            _test([3, 2], [S1, S0], offset, x, lambda x: x.transpose(0, 1))
            _test([2], [S0], offset + S1, x, lambda x: x[:, 1])
        B1 = 7
        x = torch.randn(B1, B0, 2, 3)
        (S0, S1) = x.stride()[2:]
        result = vmap(vmap(lambda t: t.as_strided([5, 5, 2, 3], [0, 0, S0, S1])), in_dims=1)(x)
        expected = vmap(vmap(lambda t: t.expand(5, 5, 2, 3)), in_dims=1)(x)
        self.assertTrue(result._base is expected._base)
        self.assertEqual(result, expected)
        with self.assertRaisesRegex(RuntimeError, 'size and stride must have the same length'):
            x = torch.randn(B0, 2, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([1, 1, 1], [1, 1]))(x)
        msg = 'batch dims being vmapped over are at the front of the tensor'
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([2, 3], [B0 * 3, 1]))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 2, 3, B1).movedim(3, 1)
            vmap(vmap(lambda x: x.as_strided([2, 3], [B1 * 3, B1])))(x)
        msg = 'This is not supported inside of vmap'
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3)
            vmap(lambda x: x.as_strided([3], [1], 1))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3, 5)
            vmap(lambda x: x.as_strided([4, 4], [4, 1], 0))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, B1, 3, 5)
            vmap(vmap(lambda x: x.as_strided([4, 4], [4, 1], 0)))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3)[1]
            vmap(lambda x: x.as_strided([3], [1], B0 * 3 - 1))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 0, 3)
            vmap(lambda x: x.as_strided([3], [1]))(x)

    def test_bmm(self):
        if False:
            for i in range(10):
                print('nop')
        op = torch.bmm
        test = self._vmap_test
        (B0, B1) = (7, 11)
        msg = 'Shape mismatch'
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(1, None))
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op, in_dims=(None, 0)), (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)), in_dims=(None, 1))
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(vmap(op), (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)), in_dims=(1, 0))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)), in_dims=(None, 0))

    def test_cat(self):
        if False:
            return 10
        test = self._vmap_test
        (B0, B1) = (5, 7)

        def get_op(dim):
            if False:
                for i in range(10):
                    print('nop')

            def op(*tensors):
                if False:
                    i = 10
                    return i + 15
                return torch.cat(tensors, dim=dim)
            return op
        test(get_op(0), (torch.rand(B0, 2), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(2), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(3, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(17, 2), torch.rand(17, 3, B0)), in_dims=(None, 2))
        test(vmap(get_op(0), in_dims=(0, None)), (torch.rand(B1, 2), torch.rand(B0, 3)), in_dims=(None, 0))
        test(vmap(get_op(0), in_dims=(0, 0)), (torch.rand(B1, 2), torch.rand(B0, B1, 3)), in_dims=(None, 0))

    def test_conj(self):
        if False:
            i = 10
            return i + 15
        op = torch.conj

        def run_test(dtype):
            if False:
                i = 10
                return i + 15

            def get(shape):
                if False:
                    return 10
                return torch.randn(shape, dtype=dtype)
            (B0, B1) = (7, 11)
            test = self._vmap_test
            test(op, [get([B0, 3])])
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, B0, 3])], in_dims=2)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)
        run_test(torch.float)
        run_test(torch.cfloat)
        real_tensor = torch.randn(3)
        result = vmap(op)(real_tensor)
        self.assertEqual(result.data_ptr(), real_tensor.data_ptr())

    def test_contiguous(self):
        if False:
            print('Hello World!')
        op = Tensor.contiguous
        self._test_unary(op, TensorFactory.randn, 'cpu')
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)
        x = x.movedim(0, 2)
        result = vmap(Tensor.contiguous, in_dims=2, out_dims=2)(x)
        self.assertTrue(result is x)
        msg = 'NYI: querying is_contiguous inside of vmap for memory_format'
        tensor = torch.randn(B0, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last_3d))(tensor)

    def test_stride(self):
        if False:
            i = 10
            return i + 15
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            assert x.stride() == (7 * 5, 7, 1)
            return x
        vmap(foo)(x)
        x = torch.randn(2, B0, 5, 7).movedim(1, 0)

        def bar(x):
            if False:
                i = 10
                return i + 15
            assert x.stride() == (7 * 5 * B0, 7, 1)
            return x
        vmap(bar)(x)

    def test_chunk(self):
        if False:
            i = 10
            return i + 15
        test = self._vmap_view_test
        op = torch.chunk
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 1024), 15, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 9, 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), 4, 0), in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)), (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)

    def test_clamp(self):
        if False:
            i = 10
            return i + 15
        clamp_cases = ((lambda t: t.clamp(min=-0.5), TensorFactory.randn), (lambda t: t.clamp(max=0.5), TensorFactory.randn), (lambda t: t.clamp(min=-0.5, max=0.5), TensorFactory.randn), (lambda t: t.clamp_min(min=-0.5), TensorFactory.randn), (lambda t: t.clamp_max(max=0.5), TensorFactory.randn))
        for (op, getter) in clamp_cases:
            self._test_unary(op, getter, 'cpu')

    def test_comparison_ops(self):
        if False:
            while True:
                i = 10
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        getter = TensorFactory.randn
        (B0, B1) = (7, 11)
        ops = (torch.eq, lambda x, y: x == y, torch.gt, lambda x, y: x > y, torch.ge, lambda x, y: x >= y, torch.le, lambda x, y: x <= y, torch.lt, lambda x, y: x < y, torch.ne, lambda x, y: x != y)
        for op in ops:
            test(op, (getter([B0, 3]), getter([B0, 3])))
            test(op, (getter([B0]), getter([B0, 2, 3])))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1), out_dims=1)
            test(op, (getter([B0]), getter([2, 3])), in_dims=(0, None))
            test(op, (getter([2, 3]), getter([B0, 3])), in_dims=(0, None))
            test(vmap(op), (getter([B0, B1, 2, 3]), getter([B0, B1, 3])))
            test(vmap(op, in_dims=(None, 0)), (getter([B0, 2, 3]), getter([B1, 3])), in_dims=(0, None))
            number = getter([]).item()
            self._test_unary(lambda t: op(t, number), getter, 'cpu', check_propagates_grad=False)

    def test_diagonal(self):
        if False:
            return 10
        tensor = torch.randn(3, 5, 7, 11, 13)
        test = self._vmap_view_test
        op = torch.diagonal
        test(op, (tensor, 1, 0, 1), in_dims=(0, None, None, None))
        test(op, (tensor, 0, 2, -1), in_dims=(0, None, None, None))
        test(op, (tensor, 2, 1, 2), in_dims=(1, None, None, None))
        test(op, (tensor, 0, -2, -1), in_dims=(1, None, None, None), out_dims=1)
        test(vmap(lambda t: op(t, 0, 0, -1)), (tensor,), in_dims=1, out_dims=1)
        test(vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3), (tensor,), in_dims=1, out_dims=1)

    def test_dot(self):
        if False:
            i = 10
            return i + 15
        op = torch.dot
        test = self._vmap_test
        (B0, B1) = (7, 11)
        msg = 'Shape mismatch'
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2))
        test(op, (torch.rand(B0, 5), torch.rand(5)), in_dims=(0, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, B0, 5), torch.rand(5)), in_dims=(1, None))
        test(op, (torch.rand(5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(vmap(op, in_dims=(None, 0)), (torch.rand(5), torch.rand(B1, B0, 5)), in_dims=(None, 1))
        test(op, (torch.rand(B0, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, 5), torch.rand(B0, 5)), in_dims=(None, 0))

    def test_expand_as(self):
        if False:
            while True:
                i = 10
        op = torch.Tensor.expand_as
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 1, 5), torch.rand(B0, 2, 3, 5)))
        test(op, (torch.rand(B0, 1, 5), torch.rand(2, 3, 5)), in_dims=(0, None))
        test(op, (torch.rand(1, 5), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B0, B1, 2, 3, 5)))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B1, B0, 2, 3, 5)), in_dims=(0, 1))
        test(vmap(op), (torch.rand(B0, B1), torch.rand(B1, 2, 3, 5)), in_dims=(0, None))
        test(vmap(vmap(op)), (torch.rand(B0, B1, B2), torch.rand(B0, B1, B2, 2, 3, 5)))

    def test_fill_and_zero_inplace(self):
        if False:
            while True:
                i = 10
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        (B0, B1) = (7, 11)
        ops = (lambda t: t.fill_(0.1), lambda t: t.fill_(torch.tensor(0.2)), lambda t: t.zero_())
        for op in ops:
            test(op, [TensorFactory.randn([B0, 3])])
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2)
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)
            test(vmap(op), [TensorFactory.randn([B0, B1])])
            test(vmap(op), [TensorFactory.randn([B1, 2, 5, B0, 3])], in_dims=2)
            test(vmap(op, in_dims=2), [TensorFactory.randn([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)
        (B0, B1) = (3, 5)
        test(Tensor.fill_, [TensorFactory.randn([B0, B1]), TensorFactory.randn(B0)])
        with self.assertRaisesRegex(RuntimeError, "output with shape .+ doesn't match the broadcast shape"):
            vmap(Tensor.fill_, (None, 0))(TensorFactory.randn([B0, B1]), TensorFactory.randn([B0]))

    def _test_complex_views(self, op, dtypes):
        if False:
            i = 10
            return i + 15
        test = self._vmap_view_test

        def run_test(op, dtype):
            if False:
                while True:
                    i = 10

            def get(shape):
                if False:
                    i = 10
                    return i + 15
                return torch.randn(shape, dtype=dtype)
            (B0, B1) = (7, 11)
            test(op, [get([B0, 3])])
            test(op, [get([3, B0])], in_dims=1)
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, 3, B0])], in_dims=4)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)
        for dtype in dtypes:
            run_test(op, dtype)

    def test_real(self):
        if False:
            i = 10
            return i + 15
        self._test_complex_views(torch.real, dtypes=[torch.cfloat, torch.cdouble])

    def test_imag(self):
        if False:
            i = 10
            return i + 15
        self._test_complex_views(torch.imag, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_real(self):
        if False:
            while True:
                i = 10
        self._test_complex_views(torch.view_as_real, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_complex(self):
        if False:
            print('Hello World!')

        def run_test(dtype):
            if False:
                print('Hello World!')

            def get(shape):
                if False:
                    i = 10
                    return i + 15
                return torch.randn(shape, dtype=dtype)
            op = torch.view_as_complex
            test = self._vmap_view_test
            (B0, B1) = (7, 11)
            test(op, [get([B0, 3, 2])])
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2)
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2, out_dims=2)
            test(vmap(op), [get([B0, B1, 2])])
            test(vmap(op), [get([B1, 2, 5, B0, 3, 2])], in_dims=2)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3, 2])], in_dims=2, out_dims=2)
            test(op, [get([3, B0, 2])], in_dims=1)
            test(vmap(op, in_dims=1), [get([3, B1, B0, 2])], in_dims=2)
            test(op, [get([B0, 2]).transpose(0, 1)], in_dims=1)
            test(vmap(op, in_dims=1), [get([B0, B1, 2]).movedim(1, 2)])
            test(vmap(op, in_dims=2), [get([B0, 3, B1, 2]).movedim(2, 3)])
            msg = 'Tensor must have a last dimension with stride 1'
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([2, B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op, in_dims=1), in_dims=1)(get([2, B0, B1]))
            msg = 'Input tensor must have one or more dimensions'
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op)(get([B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op))(get([B0, B1]))
            msg = 'Tensor must have a last dimension of size 2'
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([3, 2]))
        for dtype in [torch.float, torch.double]:
            run_test(dtype)

    def test_is_complex(self):
        if False:
            print('Hello World!')
        ctensor = torch.randn(3, dtype=torch.cfloat)
        tensor = torch.randn(3)

        def foo(x):
            if False:
                i = 10
                return i + 15
            if x.is_complex():
                return torch.tensor(1)
            else:
                return torch.tensor(0)
        self.assertEqual(vmap(foo)(ctensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(tensor), torch.tensor([0, 0, 0]))

    def test_is_floating_point(self):
        if False:
            for i in range(10):
                print('nop')
        float_tensor = torch.tensor([1.0, 2.0, 3.0])
        long_tensor = torch.tensor([1, 2, 3])

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            if x.is_floating_point():
                return torch.tensor(1)
            else:
                return torch.tensor(0)
        self.assertEqual(vmap(foo)(float_tensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(long_tensor), torch.tensor([0, 0, 0]))

    def test_is_contiguous(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                i = 10
                return i + 15
            if x.is_contiguous():
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)
        (B0, B1) = (3, 5)
        contig = torch.randn(B0, 2, 7)
        self.assertEqual(vmap(foo)(contig), torch.ones(B0))
        noncontig = torch.randn(2, B0, 7)
        self.assertEqual(vmap(foo, in_dims=1)(noncontig), torch.zeros(B0))
        noncontig = torch.randn(2, B0, 7).movedim(1, 0)
        self.assertEqual(vmap(foo)(noncontig), torch.zeros(B0))
        noncontig = torch.randn(2, 7, B0)
        self.assertEqual(vmap(foo, in_dims=2)(noncontig), torch.zeros(B0))
        contig = torch.randn(B0, B1, 3)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))
        contig = torch.randn(B1, B0, 3)
        self.assertEqual(vmap(vmap(foo), in_dims=1)(contig), torch.ones(B0, B1))
        contig = torch.randn(B1, B0, 3).movedim(0, 1)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))
        noncontig = torch.randn(B0, 3, B1)
        self.assertEqual(vmap(vmap(foo, in_dims=1))(noncontig), torch.zeros(B0, B1))

        def bar(x):
            if False:
                print('Hello World!')
            assert x.is_contiguous()
            return x
        vmap(bar)(torch.randn(B0, 0, 3))
        vmap(bar, in_dims=1)(torch.randn(0, B0, 3))
        vmap(bar)(torch.randn(B0, 0, 3).mT)

        def baz(x, memory_format):
            if False:
                return 10
            x.is_contiguous(memory_format=memory_format)
            return x
        msg = 'NYI: querying is_contiguous inside of vmap for memory_format'
        tensor = torch.randn(B0, 2, 7, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last_3d))(tensor)

    def test_movedim(self):
        if False:
            return 10
        op = torch.movedim
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 5), 0, 1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 0, 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 2, B0, 5), 0, 1), in_dims=(2, None, None))
        test(vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)), (torch.rand(B1, 2, B0, 5, B2), 0, 1), in_dims=(2, None, None))
        test(op, (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2]), in_dims=(0, None, None))
        test(op, (torch.rand(2, 3, B0, 5), [1, 0], [0, 2]), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0]), in_dims=(2, None, None))
        test(vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)), (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0]), in_dims=(2, None, None))

    def test_mm(self):
        if False:
            i = 10
            return i + 15
        op = torch.mm
        test = self._vmap_test
        (B0, B1) = (7, 11)
        msg = 'Shape mismatch'
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))
        test(op, (torch.rand(B0, 2, 5), torch.rand(5, 2)), in_dims=(0, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, B0, 2, 5), torch.rand(5, 2)), in_dims=(1, None))
        test(op, (torch.rand(2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))
        test(vmap(op, in_dims=(None, 0)), (torch.rand(2, 5), torch.rand(B1, B0, 5, 2)), in_dims=(None, 1))
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2)))
        test(vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2)), in_dims=(1, 0))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, 2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))

    def test_mv(self):
        if False:
            return 10
        op = torch.mv
        test = self._vmap_test
        (B0, B1) = (7, 11)
        msg = 'Shape mismatch'
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2))
        test(op, (torch.rand(B0, 2, 5), torch.rand(5)), in_dims=(0, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, B0, 2, 5), torch.rand(5)), in_dims=(1, None))
        test(op, (torch.rand(2, 5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(vmap(op, in_dims=(None, 0)), (torch.rand(2, 5), torch.rand(B1, B0, 5)), in_dims=(None, 1))
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, 2, 5), torch.rand(B0, 5)), in_dims=(None, 0))

    def test_narrow(self):
        if False:
            print('Hello World!')
        op = torch.narrow
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 5), -1, 1, 3), in_dims=(0, None, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1, 3), in_dims=(1, None, None, None))
        test(vmap(op, in_dims=(0, None, None, None)), (torch.rand(B1, 2, B0, 5), 1, 0, 0), in_dims=(2, None, None, None))
        test(vmap(vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)), (torch.rand(B1, 2, B0, 5, B2), -1, 2, 3), in_dims=(2, None, None, None))

    def test_new_empty(self):
        if False:
            print('Hello World!')
        op = Tensor.new_empty
        (B0, B1) = (7, 11)
        result = vmap(lambda x: op(x, [2, 3]))(torch.randn(B0))
        self.assertEqual(result.shape, [B0, 2, 3])
        result = vmap(lambda x: op(x, []))(torch.randn(B0))
        self.assertEqual(result.shape, [B0])
        result = vmap(vmap(lambda x: op(x, [2, 3])))(torch.randn(B0, B1))
        self.assertEqual(result.shape, [B0, B1, 2, 3])

    def test_new_empty_strided(self):
        if False:
            while True:
                i = 10
        (B0, B1) = (7, 11)

        def _test_single_vmap(size, stride, B0):
            if False:
                while True:
                    i = 10
            x = torch.randn(B0)
            result = vmap(lambda x: x.new_empty_strided(size, stride))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0] + size)
            self.assertEqual(result.stride(), [S] + stride)

        def _test_double_vmap(size, stride, B0, B1):
            if False:
                return 10
            x = torch.randn(B0, B1)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)
            x = torch.randn(B1, B0)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)), in_dims=1)(x)
            S = x.new_empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)
        _test_single_vmap([2, 3, 5], [3 * 5, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [3 * 5, 5, 1], B0, B1)
        _test_single_vmap([2, 3, 5], [0, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [0, 5, 1], B0, B1)
        for shape in [[2, 3, 4], [0, 2, 0]]:
            for strides in [[12, 4, 1], [2, 4, 6], [0, 0, 0]]:
                _test_single_vmap(shape, strides, B0)
                _test_double_vmap(shape, strides, B0, B1)

    def test_new_zeros(self):
        if False:
            print('Hello World!')
        op = Tensor.new_zeros
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        (B0, B1) = (7, 11)
        test(lambda x: op(x, 2, 3), (torch.rand(B0),))
        test(lambda x: op(x, []), (torch.rand(B0),))
        test(vmap(lambda x: op(x, 3, 5)), (torch.rand(B0, B1),))

    def test_select(self):
        if False:
            i = 10
            return i + 15
        op = torch.select
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 5), 0, 0), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1), in_dims=(1, None, None))
        test(vmap(lambda t: op(t, 1, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(lambda t: op(t, 1, 1), in_dims=1)), (torch.rand(B1, 2, B0, B2, 5),), in_dims=2)

    def test_stack(self):
        if False:
            print('Hello World!')
        test = self._vmap_test
        (B0, B1) = (5, 7)

        def get_op(dim):
            if False:
                for i in range(10):
                    print('nop')

            def op(*tensors):
                if False:
                    while True:
                        i = 10
                return torch.stack(tensors, dim=dim)
            return op
        test(get_op(0), (torch.rand(B0, 3), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(3), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        test(vmap(get_op(0), in_dims=(0, None)), (torch.rand(B1, 2), torch.rand(B0, 2)), in_dims=(None, 0))
        test(vmap(get_op(0), in_dims=(0, 0)), (torch.rand(B1, 2), torch.rand(B0, B1, 2)), in_dims=(None, 0))

    def test_slice(self):
        if False:
            print('Hello World!')
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        test(lambda t: t[0:1], (torch.rand(B0, 3, 5),))
        test(lambda t: t[:, 1:3], (torch.rand(3, 5, B0),), in_dims=2)
        test(vmap(lambda t: t[:, 0:1], in_dims=2), (torch.rand(3, 5, B0, B1),), in_dims=2)
        test(vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2), (torch.rand(3, 5, B0, B1, B2),), in_dims=2)

    def test_squeeze(self):
        if False:
            while True:
                i = 10
        test = self._vmap_view_test
        op = torch.squeeze
        (B0, B1) = (1, 11)
        test(op, (torch.rand(B0),))
        test(op, (torch.rand(B0, 3, 5),))
        test(op, (torch.rand(1, B0, 5),), in_dims=1)
        test(op, (torch.rand(B0, 0, 1, 5, 1),))
        test(op, (torch.rand(B0, 1, 1, 1, 1),))
        test(vmap(op), (torch.rand(B0, B1, 1),))
        test(vmap(op), (torch.rand(B1, 1, B0),), in_dims=2)

    def test_sum_dim(self):
        if False:
            print('Hello World!')
        test = self._vmap_test
        (B0, B1) = (5, 7)
        test(lambda x: x.sum(()), [torch.randn([B0])])
        test(lambda x: x.sum(()), [torch.randn([B0, 2])])
        test(lambda x: x.sum(0), [torch.randn([B0])])
        test(lambda x: x.sum(-1), [torch.randn([B0])])
        test(lambda x: x.sum(0), [torch.randn([B0, 3])])
        test(lambda x: x.sum(-1), [torch.randn([2, 5, B0, 3])], in_dims=2)
        test(lambda x: x.sum(2), [torch.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)
        test(vmap(lambda x: x.sum(())), [torch.randn([B0, B1])])
        test(vmap(lambda x: x.sum(0)), [torch.randn([B0, B1])])
        test(vmap(lambda x: x.sum(-1)), [torch.randn([B0, B1])])
        test(vmap(lambda x: x.sum(-2)), [torch.randn([B1, 2, 5, B0, 3])], in_dims=2)
        test(vmap(lambda x: x.sum(2), in_dims=2), [torch.randn([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

    def test_reshape(self):
        if False:
            print('Hello World!')
        test = self._vmap_test
        (B0, B1, B2) = (7, 11, 13)
        op = torch.reshape
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None), check_view=True)
        test(op, (torch.rand(2, B0, 5), [1, 1, 10]), in_dims=(1, None), check_view=False)
        test(vmap(lambda t: t.reshape([-1])), (torch.rand(B0, B1, 2, 5),), check_view=True)
        test(vmap(vmap(lambda t: t.reshape([-1]), in_dims=2), in_dims=1), (torch.rand(3, B1, 2, B2, 5, B0),), in_dims=5, check_view=False)

    def test_reshape_as(self):
        if False:
            for i in range(10):
                print('nop')
        test = self._vmap_test
        (B0, B1, B2) = (7, 11, 13)
        op = torch.Tensor.reshape_as
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)), check_view=True)
        test(op, (torch.rand(2 * 5), torch.rand(B0, 2, 5)), in_dims=(None, 0), check_view=True)
        test(op, (torch.rand(B0, 2 * 5), torch.rand(2, 5)), in_dims=(0, None), check_view=True)
        test(op, (torch.rand(2, B0, 5), torch.rand(1, 1, 10)), in_dims=(1, None), check_view=False)
        test(vmap(op), (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)), check_view=True)
        test(vmap(vmap(op, in_dims=(2, None)), in_dims=(1, None)), (torch.rand(3, B1, 2, B2, 5, B0), torch.rand(B0, 3 * 2 * 5)), in_dims=(5, 0), check_view=False)

    def test_result_type(self):
        if False:
            return 10

        def scalar_tensor_with_dtype(op):
            if False:
                i = 10
                return i + 15

            def wrapped(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                dtype = op(*args, **kwargs)
                return torch.ones([], dtype=dtype)
            return wrapped
        test = self._vmap_test
        op = scalar_tensor_with_dtype(torch.result_type)
        B0 = 2
        test(op, (torch.randn(B0), torch.randn(B0, dtype=torch.float64)), check_propagates_grad=False)
        test(op, (torch.randn(B0), torch.randint(10, [B0], dtype=torch.int64)), check_propagates_grad=False)
        test(lambda x: op(x, 1), (torch.randn(B0),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0),), check_propagates_grad=False)
        test(lambda x: op(x, torch.tensor(1)), (torch.randn(B0),), check_propagates_grad=False)
        test(lambda x: op(x, torch.tensor(1.6, dtype=torch.double)), (torch.randn(B0),), check_propagates_grad=False)
        test(op, (torch.randn(B0, 2), torch.randn(B0, 2, dtype=torch.float64)), check_propagates_grad=False)
        test(op, (torch.randn(B0, 2), torch.randint(10, [B0, 2], dtype=torch.int64)), check_propagates_grad=False)
        test(lambda x: op(x, 1), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(lambda x: op(x, torch.tensor(1)), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(lambda x: op(x, torch.tensor(1.6, dtype=torch.double)), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(op, (torch.randn(B0, 2), torch.randn(B0, dtype=torch.float64)), check_propagates_grad=False)
        test(op, (torch.randn(B0, 2), torch.randint(10, [B0], dtype=torch.int64)), check_propagates_grad=False)

    def test_tensor_split(self):
        if False:
            while True:
                i = 10
        test = self._vmap_view_test
        op = torch.tensor_split
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 1024), 5, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 150, 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), 256, 0), in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)), (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)
        test(op, (torch.rand(B0, 2, 1024), [50, 100, 378, 890], -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), [50, 100, 212, 345, 0, 378, 890], 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), [50, 100, 212, 345, 0, 378, 890], 0), in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, [4, 8, 9, 34, 29], 1), in_dims=2)), (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)

    def test_split(self):
        if False:
            print('Hello World!')
        test = self._vmap_view_test
        op = torch.split
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 1024), 101, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 130, 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), 256, 0), in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)), (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)
        test(op, (torch.rand(B0, 2, 1024), [1, 1020, 3], -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), [100] * 10 + [24], 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), [256] * 3 + [255], 0), in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, [4] * 8 + [8] * 4, 1), in_dims=2)), (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)

    def test_trace(self):
        if False:
            while True:
                i = 10
        op = torch.trace
        test = self._vmap_test
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 5),))
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_transpose(self):
        if False:
            print('Hello World!')
        op = torch.transpose
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        test(lambda x: op(x, 0, 1), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, -1, -2), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, 3, 1), (torch.rand(B0, 2, 5, 4, 6),))
        test(lambda x: op(x, 1, 0), (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(lambda x: op(x, 0, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(lambda x: op(x, 0, 1), in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)
        for (dim1, dim2) in itertools.product([0, -1], [0, -1]):
            x = torch.rand(B0)
            result = vmap(lambda x: op(x, dim1, dim2))(x)
            self.assertTrue(result is x)

    def test_t(self):
        if False:
            for i in range(10):
                print('nop')
        op = torch.t
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 5),))
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_T_numpy(self):
        if False:
            for i in range(10):
                print('nop')

        def op(t):
            if False:
                for i in range(10):
                    print('nop')
            return t.T
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 3, 5),))
        test(op, (torch.rand(2, B0, 3, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(op), (torch.rand(B1, 2, B0, 3, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 3, B2, 5),), in_dims=2)

    def test_to(self):
        if False:
            i = 10
            return i + 15
        test = self._vmap_test
        (B0, B1) = (7, 11)
        test(lambda t: t.to('cpu'), (torch.rand(B0),))
        test(lambda t: t.to(torch.double), (torch.rand(B0),))
        test(lambda t, o: t.to(o), (torch.rand(B0), torch.randn(B0, dtype=torch.float64)))
        test(lambda t, o: t.to(o), (torch.rand(B0), torch.randn(B0, dtype=torch.float64)), in_dims=(0, None))
        test(vmap(lambda t: t.to(torch.double)), (torch.rand(B0, B1, 3),))
        test(lambda t: t.double(), (torch.rand(B0),))
        test(lambda t: t.float(), (torch.rand(B0),))
        test(lambda t: t.int(), (torch.rand(B0),), check_propagates_grad=False)
        test(lambda t: t.long(), (torch.rand(B0),), check_propagates_grad=False)

    def test_unfold(self):
        if False:
            i = 10
            return i + 15
        op = torch.Tensor.unfold
        test = self._vmap_view_test
        (B0, B1, B2) = (3, 2, 5)
        test(op, (torch.rand(B0, 7, 11), 0, 2, 1), in_dims=(0, None, None, None))
        test(op, (torch.rand(7, B0, 11), 1, 4, 2), in_dims=(1, None, None, None))
        test(vmap(op, in_dims=(0, None, None, None)), (torch.rand(B1, 7, B0, 11), 1, 5, 1), in_dims=(2, None, None, None))
        test(vmap(vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)), (torch.rand(B1, 7, B0, 11, B2), -1, 2, 4), in_dims=(2, None, None, None))

    def test_unbind(self):
        if False:
            i = 10
            return i + 15
        test = self._vmap_view_test
        op = torch.unbind
        (B0, B1, B2) = (7, 11, 13)
        test(op, (torch.rand(B0, 2, 1024), -1), in_dims=(0, None))
        test(op, (torch.rand(B0, 2, 0),))
        test(op, (torch.rand(2, B0, 7), 0), in_dims=(1, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, 1023, B0, 5), 1), in_dims=(2, None))
        test(vmap(vmap(lambda t: op(t, dim=1), in_dims=2)), (torch.rand(B1, 2, B0, 32, B2),), in_dims=2)

    def test_view(self):
        if False:
            while True:
                i = 10
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        op = torch.Tensor.view
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, None))(torch.rand(2, B0, 5), [10])
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None))
        test(op, (torch.rand(B0, 4, 5), [1, 2, 1, 10]), in_dims=(0, None))
        test(vmap(lambda t: t.view([-1])), (torch.rand(B0, B1, 2, 5, 3),))
        test(vmap(vmap(lambda t: t.reshape([-1])), in_dims=1), (torch.rand(B2, B0, B1, 3, 2, 5),), in_dims=1)

    def test_view_as(self):
        if False:
            return 10
        test = self._vmap_view_test
        (B0, B1, B2) = (7, 11, 13)
        op = torch.Tensor.view_as
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, 0))(torch.rand(2, B0, 5), torch.rand(B0, 10))
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)))
        test(op, (torch.rand(2 * 5), torch.rand(B0, 2, 5)), in_dims=(None, 0))
        test(op, (torch.rand(B0, 2 * 5), torch.rand(2, 5)), in_dims=(0, None))
        test(op, (torch.rand(B0, 4, 5), torch.rand(2, 1, 1, 10)), in_dims=(0, None))
        test(vmap(op), (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)))
        test(vmap(vmap(op, in_dims=(0, None)), in_dims=(0, None)), (torch.rand(B1, B2, B0, 3, 2, 5), torch.rand(B0, 3 * 2 * 5)), in_dims=(2, 0))

    def test_no_random_op_support(self):
        if False:
            return 10
        B0 = 2
        captured = torch.rand(3)
        random_ops = [(torch.bernoulli, (torch.rand(B0, 1),)), (lambda t: torch.bernoulli(t, p=0.5), (torch.rand(B0, 1),)), (lambda t: torch.multinomial(t, 2), (torch.rand(B0, 3),)), (torch.normal, (torch.randn(B0, 1), torch.randn(B0, 1))), (lambda t: torch.normal(t, 1.0), (torch.randn(B0, 1),)), (lambda t: torch.normal(0.0, t), (torch.randn(B0, 1),)), (torch.poisson, (torch.rand(B0, 1),)), (torch.rand_like, (torch.rand(B0, 1),)), (torch.randn_like, (torch.rand(B0, 1),)), (lambda t: torch.randint_like(t, 2), (torch.rand(B0, 1),)), (lambda t: torch.randint_like(t, 0, 2), (torch.rand(B0, 1),)), (lambda t: torch.bernoulli(captured), (torch.rand(B0),)), (lambda t: torch.bernoulli(captured, p=0.5), (torch.rand(B0),)), (lambda t: torch.multinomial(captured, 2), (torch.rand(B0),)), (lambda t: torch.normal(captured, captured), (torch.randn(B0),)), (lambda t: torch.normal(captured, 1.0), (torch.randn(B0),)), (lambda t: torch.normal(0.0, captured), (torch.randn(B0),)), (lambda t: torch.poisson(captured), (torch.rand(B0),)), (lambda t: torch.rand_like(captured), (torch.rand(B0),)), (lambda t: torch.randn_like(captured), (torch.rand(B0),)), (lambda t: torch.randint_like(captured, 2), (torch.rand(B0),)), (lambda t: torch.randint_like(captured, 0, 2), (torch.rand(B0),)), (lambda t: t.bernoulli_(), (torch.randn(B0, 1),)), (lambda t: t.cauchy_(), (torch.randn(B0, 1),)), (lambda t: t.exponential_(), (torch.randn(B0, 1),)), (lambda t: t.geometric_(0.5), (torch.randn(B0, 1),)), (lambda t: t.log_normal_(), (torch.randn(B0, 1),)), (lambda t: t.normal_(), (torch.randn(B0, 1),)), (lambda t: t.random_(), (torch.randn(B0, 1),)), (lambda t: t.random_(0, 2), (torch.randn(B0, 1),)), (lambda t: t.random_(2), (torch.randn(B0, 1),)), (lambda t: t.uniform_(), (torch.randn(B0, 1),)), (lambda t: captured.bernoulli_(), (torch.randn(B0),)), (lambda t: captured.cauchy_(), (torch.randn(B0),)), (lambda t: captured.exponential_(), (torch.randn(B0),)), (lambda t: captured.geometric_(0.5), (torch.randn(B0),)), (lambda t: captured.log_normal_(), (torch.randn(B0),)), (lambda t: captured.normal_(), (torch.randn(B0),)), (lambda t: captured.random_(), (torch.randn(B0),)), (lambda t: captured.random_(0, 2), (torch.randn(B0),)), (lambda t: captured.random_(2), (torch.randn(B0),)), (lambda t: captured.uniform_(), (torch.randn(B0),)), (lambda t: torch.rand(1), (torch.randn(B0),)), (lambda t: torch.randn(1), (torch.randn(B0),)), (lambda t: torch.randint(5, [1]), (torch.randn(B0),)), (lambda t: torch.randperm(5), (torch.randn(B0),))]
        for (op, args) in random_ops:
            with self.assertRaisesRegex(RuntimeError, 'vmap: We do not yet support calling random operations'):
                vmap(op)(*args)

def construct_v(output, batch_size):
    if False:
        return 10
    return torch.randn(batch_size, *output.shape, dtype=output.dtype, device=output.device)

def as_tuple(x):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)

def differentiable(args):
    if False:
        print('Hello World!')
    return tuple((arg for arg in as_tuple(args) if isinstance(arg, torch.Tensor) and arg.requires_grad))

def _get_rand_no_zeros(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    requires_grad = kwargs.get('requires_grad', False)
    kwargs_without_requires_grad = kwargs.copy()
    kwargs_without_requires_grad['requires_grad'] = False
    result = torch.rand(*args, **kwargs_without_requires_grad)
    return result.clamp_min_(0.1).requires_grad_(requires_grad)

class TestVmapBatchedGradient(Namespace.TestVmapBase):

    def _vmap_test(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return _vmap_test(self, *args, **kwargs)

    def _batched_grad_test(self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3):
        if False:
            i = 10
            return i + 15
        if kwargs is None:
            kwargs = {}
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        batched_vectors = tuple((construct_v(out, batch_size) for out in outputs))

        def vector_jacobian_product(*vectors):
            if False:
                for i in range(10):
                    print('nop')
            return torch.autograd.grad(outputs, differentiable(args), vectors, retain_graph=True)
        self._vmap_test(vector_jacobian_product, batched_vectors, check_propagates_grad=False)

    def _batched_grad_grad_test(self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3):
        if False:
            while True:
                i = 10
        if kwargs is None:
            kwargs = {}
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        ones = tuple((torch.ones_like(out) for out in outputs))
        first_grads = torch.autograd.grad(outputs, differentiable(args), ones, create_graph=True)
        first_grads = differentiable(first_grads)
        self.assertNotEqual(len(first_grads), 0, 'None of the first grads depend on the input!')
        batched_vectors = tuple((construct_v(grad, batch_size) for grad in first_grads))

        def vector_hessian_product(*vectors):
            if False:
                for i in range(10):
                    print('nop')
            outputs = torch.autograd.grad(first_grads, differentiable(args), vectors, retain_graph=True, allow_unused=True)
            outputs = tuple((out for out in outputs if out is not None))
            assert len(outputs) > 0
            return outputs
        self._vmap_test(vector_hessian_product, batched_vectors, check_propagates_grad=False)

    def _test_arithmetic(self, op, device, test_grad_grad=True):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 3, requires_grad=True, device=device)
        y = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        scalar = 3.14
        self._batched_grad_test(op, (x, y))
        self._batched_grad_test(op, (scalar, y))
        self._batched_grad_test(op, (x, scalar))
        if test_grad_grad:
            self._batched_grad_grad_test(op, (x, y))

    def test_add(self, device):
        if False:
            return 10
        self._test_arithmetic(torch.add, device, test_grad_grad=False)
        self._test_arithmetic(lambda x, y: x + y, device, test_grad_grad=False)

    def test_sub(self, device):
        if False:
            return 10
        self._test_arithmetic(torch.sub, device, test_grad_grad=False)
        self._test_arithmetic(lambda x, y: x - y, device, test_grad_grad=False)

    def test_mul(self, device):
        if False:
            return 10
        self._test_arithmetic(torch.mul, device)
        self._test_arithmetic(lambda x, y: x * y, device)

    def test_div(self, device):
        if False:
            return 10
        self._test_arithmetic(torch.div, device)
        self._test_arithmetic(lambda x, y: x / y, device)

    @allowVmapFallbackUsage
    def test_binary_cross_entropy(self, device):
        if False:
            print('Hello World!')
        x = torch.sigmoid(torch.randn(3, 2, device=device, requires_grad=True))
        target = torch.rand(3, 2, device=device)
        op = functools.partial(F.binary_cross_entropy, target=target)
        self._batched_grad_test(op, (x,), {})
        self._batched_grad_grad_test(op, (x,), {})

    def test_expand(self, device):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(2, 3, device=device, requires_grad=True)

        def op(x):
            if False:
                i = 10
                return i + 15
            return x.expand(5, 5, 2, 3)
        self._batched_grad_test(op, (x,))

    @allowVmapFallbackUsage
    def test_index(self, device):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 3, requires_grad=True, device=device)
        index = torch.tensor([[0, 0], [1, 1]], device=device)

        def op(x):
            if False:
                i = 10
                return i + 15
            y = x * x
            return y[index]
        self._batched_grad_test(op, (x,))
        self._batched_grad_grad_test(op, (x,))

    def test_lgamma(self, device):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(Tensor.lgamma, (x,))
        self._batched_grad_grad_test(Tensor.lgamma, (x,))

    def test_log(self, device):
        if False:
            print('Hello World!')
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(torch.log, (x,))
        self._batched_grad_grad_test(torch.log, (x,))

    def test_logsumexp(self, device):
        if False:
            while True:
                i = 10
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)

        def op(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.logsumexp(x, -1)
        self._batched_grad_test(op, (x,))
        self._batched_grad_grad_test(op, (x,))

    def test_log1p(self, device):
        if False:
            print('Hello World!')
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(torch.log1p, (x,))
        self._batched_grad_grad_test(torch.log1p, (x,))

    @allowVmapFallbackUsage
    def test_max(self, device):
        if False:
            return 10
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.max, (x,))

    @allowVmapFallbackUsage
    def test_median(self, device):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.median, (x,))

    @allowVmapFallbackUsage
    def test_min(self, device):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.min, (x,))

    def test_permute(self, device):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        def op(x):
            if False:
                while True:
                    i = 10
            return x.permute(2, 0, 1)
        self._batched_grad_test(op, (x,))

    def test_reshape(self, device):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        def op(x):
            if False:
                while True:
                    i = 10
            return x.reshape([2 * 3, 5])
        self._batched_grad_test(op, (x,))

    def test_sigmoid(self, device):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(Tensor.sigmoid, (x,))
        self._batched_grad_grad_test(Tensor.sigmoid, (x,))

    def test_stack(self, device):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 3, device=device, requires_grad=True)
        y = torch.randn(2, 3, device=device, requires_grad=True)

        def op(x, y):
            if False:
                return 10
            return torch.stack([x, y])
        self._batched_grad_test(op, (x, y))

    def test_select(self, device):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x[1], (x,))
        self._batched_grad_test(lambda x: x.select(1, 2), (x,))
        self._batched_grad_test(lambda x: x.select(-1, 0), (x,))

    def test_slice(self, device):
        if False:
            return 10
        x = torch.randn(2, 3, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x[0:1], (x,))
        self._batched_grad_test(lambda x: x[:, 1:3], (x,))
        self._batched_grad_test(lambda x: x[..., 1:3], (x,))

    def test_trace(self, device):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(Tensor.trace, (x,))

    def test_threshold(self, device):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: F.threshold(x, 0.5, 0.0), (x,))

    @allowVmapFallbackUsage
    def test_inplace_on_view(self, device):
        if False:
            while True:
                i = 10
        leaf = torch.randn(4, 5, requires_grad=True)

        def func(leaf):
            if False:
                return 10
            base = leaf * leaf
            view = base[0]
            view.cos_()
            return view
        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    @allowVmapFallbackUsage
    def test_inplace_manyview(self, device):
        if False:
            print('Hello World!')
        leaf = torch.randn(4, 4, 5, requires_grad=True)

        def func(leaf):
            if False:
                for i in range(10):
                    print('nop')
            base = leaf * leaf
            view = base.transpose(0, 2)
            view = view[1]
            view = view.diagonal()
            view = view[::2]
            view.cos_()
            return view
        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    def test_diagonal(self, device):
        if False:
            print('Hello World!')
        x = torch.randn(4, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x.diagonal(1, 0, 1), (x,))
        x = torch.randn(3, 4, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x.diagonal(0, -1, -2), (x,))

    @allowVmapFallbackUsage
    def test_unrelated_output(self, device):
        if False:
            i = 10
            return i + 15
        B0 = 3
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            if False:
                while True:
                    i = 10
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res
        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))

    @allowVmapFallbackUsage
    def test_unrelated_output_multiple_grad(self, device):
        if False:
            while True:
                i = 10
        B0 = 3
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            if False:
                for i in range(10):
                    print('nop')
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res
        _ = vjp(gy[0])
        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))
instantiate_device_type_tests(TestVmapBatchedGradient, globals(), None)
if __name__ == '__main__':
    run_tests()