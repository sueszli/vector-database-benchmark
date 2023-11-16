import unittest
from functools import partial
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import append_backward
from paddle.base.framework import Program, program_guard
paddle.enable_static()

class TestAPISwitchCase(unittest.TestCase):

    def test_return_single_var(self):
        if False:
            return 10

        def fn_1():
            if False:
                for i in range(10):
                    print('nop')
            return paddle.tensor.fill_constant(shape=[4, 2], dtype='int32', value=1)

        def fn_2():
            if False:
                return 10
            return paddle.tensor.fill_constant(shape=[4, 2], dtype='int32', value=2)

        def fn_3():
            if False:
                print('Hello World!')
            return paddle.tensor.fill_constant(shape=[4, 3], dtype='int32', value=3)
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            index_1 = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=1)
            index_2 = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=2)
            index_5 = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=5)
            out_0 = paddle.static.nn.switch_case(branch_index=index_1, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
            out_1 = paddle.static.nn.switch_case(branch_index=index_1, branch_fns=(fn_1, fn_2, fn_3))
            out_2 = paddle.static.nn.switch_case(branch_index=index_5, branch_fns=((1, fn_1), (2, fn_2)), default=fn_3)
            out_3 = paddle.static.nn.switch_case(branch_index=index_2, branch_fns=[(1, fn_1), (2, fn_2)])
            out_4 = paddle.static.nn.switch_case(branch_index=index_5, branch_fns=[(1, fn_1), (3, fn_2), (2, fn_3)])
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(main_program, fetch_list=[out_0, out_1, out_2, out_3, out_4])
            np.testing.assert_allclose(res[0], 1, rtol=1e-05, err_msg=f'result is {res[0]} but answer is {1}')
            np.testing.assert_allclose(res[1], 2, rtol=1e-05, err_msg=f'result is {res[1]} but answer is {2}')
            np.testing.assert_allclose(res[2], 3, rtol=1e-05, err_msg=f'result is {res[2]} but answer is {3}')
            np.testing.assert_allclose(res[3], 2, rtol=1e-05, err_msg=f'result is {res[3]} but answer is {2}')
            np.testing.assert_allclose(res[4], 2, rtol=1e-05, err_msg=f'result is {res[4]} but answer is {2}')

    def test_0d_tensor(self):
        if False:
            return 10

        def fn_1():
            if False:
                for i in range(10):
                    print('nop')
            return paddle.full(shape=[], dtype='int32', fill_value=1)

        def fn_2():
            if False:
                while True:
                    i = 10
            return paddle.full(shape=[], dtype='int32', fill_value=2)

        def fn_3():
            if False:
                for i in range(10):
                    print('nop')
            return paddle.full(shape=[], dtype='int32', fill_value=3)
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            index_1 = paddle.full(shape=[], dtype='int32', fill_value=1)
            index_2 = paddle.full(shape=[], dtype='int32', fill_value=2)
            index_5 = paddle.full(shape=[], dtype='int32', fill_value=5)
            out_0 = paddle.static.nn.switch_case(branch_index=index_1, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
            out_1 = paddle.static.nn.switch_case(branch_index=index_1, branch_fns=(fn_1, fn_2, fn_3))
            out_2 = paddle.static.nn.switch_case(branch_index=index_5, branch_fns=((1, fn_1), (2, fn_2)), default=fn_3)
            out_3 = paddle.static.nn.switch_case(branch_index=index_2, branch_fns=[(1, fn_1), (2, fn_2)])
            out_4 = paddle.static.nn.switch_case(branch_index=index_5, branch_fns=[(1, fn_1), (3, fn_2), (2, fn_3)])
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(main_program, fetch_list=[out_0, out_1, out_2, out_3, out_4])
            np.testing.assert_allclose(res[0], 1, rtol=1e-05, err_msg=f'result is {res[0]} but answer is {1}')
            self.assertEqual(res[0].shape, ())
            np.testing.assert_allclose(res[1], 2, rtol=1e-05, err_msg=f'result is {res[1]} but answer is {2}')
            self.assertEqual(res[1].shape, ())
            np.testing.assert_allclose(res[2], 3, rtol=1e-05, err_msg=f'result is {res[2]} but answer is {3}')
            self.assertEqual(res[2].shape, ())
            np.testing.assert_allclose(res[3], 2, rtol=1e-05, err_msg=f'result is {res[3]} but answer is {2}')
            self.assertEqual(res[3].shape, ())
            np.testing.assert_allclose(res[4], 2, rtol=1e-05, err_msg=f'result is {res[4]} but answer is {2}')
            self.assertEqual(res[4].shape, ())

    def test_0d_tensor_backward(self):
        if False:
            return 10
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = paddle.full(shape=[], dtype='float32', fill_value=-2.0)
            x.stop_gradient = False
            pred = paddle.full(shape=[], dtype='int32', fill_value=2)
            out = paddle.static.nn.switch_case(branch_index=pred, branch_fns=[(1, lambda : x), (2, lambda : 2 * x)], default=lambda : -x)
            append_backward(out)
        place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
        exe = base.Executor(place)
        res = exe.run(main_program, fetch_list=[out.name, x.grad_name])
        np.testing.assert_allclose(np.asarray(res[0]), np.array(-4.0), rtol=1e-05)
        self.assertEqual(res[0].shape, ())
        np.testing.assert_allclose(np.asarray(res[1]), np.array(2.0), rtol=1e-05)
        self.assertEqual(res[1].shape, ())

    def test_0d_tensor_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()

        def fn_1():
            if False:
                i = 10
                return i + 15
            return paddle.full(shape=[], dtype='int32', fill_value=1)

        def fn_2():
            if False:
                while True:
                    i = 10
            return paddle.full(shape=[], dtype='int32', fill_value=2)

        def fn_3():
            if False:
                for i in range(10):
                    print('nop')
            return paddle.full(shape=[], dtype='int32', fill_value=3)
        index_1 = paddle.full(shape=[], dtype='int32', fill_value=1)
        index_2 = paddle.full(shape=[], dtype='int32', fill_value=2)
        index_5 = paddle.full(shape=[], dtype='int32', fill_value=5)
        out_0 = paddle.static.nn.switch_case(branch_index=index_1, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
        out_1 = paddle.static.nn.switch_case(branch_index=index_1, branch_fns=(fn_1, fn_2, fn_3))
        out_2 = paddle.static.nn.switch_case(branch_index=index_5, branch_fns=((1, fn_1), (2, fn_2)), default=fn_3)
        out_3 = paddle.static.nn.switch_case(branch_index=index_2, branch_fns=[(1, fn_1), (2, fn_2)])
        out_4 = paddle.static.nn.switch_case(branch_index=index_5, branch_fns=[(1, fn_1), (3, fn_2), (2, fn_3)])
        np.testing.assert_allclose(out_0, 1, rtol=1e-05, err_msg=f'result is {out_0} but answer is {1}')
        self.assertEqual(out_0.shape, [])
        np.testing.assert_allclose(out_1, 2, rtol=1e-05, err_msg=f'result is {out_1} but answer is {2}')
        self.assertEqual(out_1.shape, [])
        np.testing.assert_allclose(out_2, 3, rtol=1e-05, err_msg=f'result is {out_2} but answer is {3}')
        self.assertEqual(out_2.shape, [])
        np.testing.assert_allclose(out_3, 2, rtol=1e-05, err_msg=f'result is {out_3} but answer is {2}')
        self.assertEqual(out_3.shape, [])
        np.testing.assert_allclose(out_4, 2, rtol=1e-05, err_msg=f'result is {out_4} but answer is {2}')
        self.assertEqual(out_4.shape, [])
        paddle.enable_static()

    def test_return_var_tuple(self):
        if False:
            print('Hello World!')

        def fn_1():
            if False:
                while True:
                    i = 10
            return (paddle.tensor.fill_constant(shape=[1, 2], dtype='int32', value=1), paddle.tensor.fill_constant(shape=[2, 3], dtype='float32', value=2))

        def fn_2():
            if False:
                return 10
            return (paddle.tensor.fill_constant(shape=[3, 4], dtype='int32', value=3), paddle.tensor.fill_constant(shape=[4, 5], dtype='float32', value=4))

        def fn_3():
            if False:
                print('Hello World!')
            return (paddle.tensor.fill_constant(shape=[5], dtype='int32', value=5), paddle.tensor.fill_constant(shape=[5, 6], dtype='float32', value=6))
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            index_1 = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=1)
            out = paddle.static.nn.switch_case(index_1, ((1, fn_1), (2, fn_2)), fn_3)
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            ret = exe.run(main_program, fetch_list=out)
            np.testing.assert_allclose(np.asarray(ret[0]), np.full((1, 2), 1, np.int32), rtol=1e-05)
            np.testing.assert_allclose(np.asarray(ret[1]), np.full((2, 3), 2, np.float32), rtol=1e-05)

class TestAPISwitchCase_Nested(unittest.TestCase):

    def test_nested_switch_case(self):
        if False:
            print('Hello World!')

        def fn_1(x=1):
            if False:
                i = 10
                return i + 15
            out = paddle.static.nn.switch_case(branch_index=paddle.tensor.fill_constant(shape=[1], dtype='int32', value=x), branch_fns={1: partial(paddle.tensor.fill_constant, shape=[1], dtype='int32', value=1), x: partial(paddle.tensor.fill_constant, shape=[2], dtype='int32', value=x)})
            return out

        def fn_2(x=2):
            if False:
                i = 10
                return i + 15
            out = paddle.static.nn.switch_case(branch_index=paddle.tensor.fill_constant(shape=[1], dtype='int32', value=2), branch_fns={1: partial(paddle.tensor.fill_constant, shape=[4, 3], dtype='int32', value=1), 2: partial(fn_1, x=x)})
            return out

        def fn_3():
            if False:
                return 10
            out = paddle.static.nn.switch_case(branch_index=paddle.tensor.fill_constant(shape=[1], dtype='int32', value=3), branch_fns={1: partial(paddle.tensor.fill_constant, shape=[4, 3], dtype='int32', value=1), 3: partial(fn_2, x=3)})
            return out
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            index_1 = paddle.static.data(name='index_1', shape=[1], dtype='uint8')
            index_2 = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=2)
            index_3 = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=3)
            out_1 = paddle.static.nn.switch_case(branch_index=index_1, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
            out_2 = paddle.static.nn.switch_case(branch_index=index_2, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
            out_3 = paddle.static.nn.switch_case(branch_index=index_3, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(main_program, feed={'index_1': np.array([1], dtype='uint8')}, fetch_list=[out_1, out_2, out_3])
            np.testing.assert_allclose(res[0], 1, rtol=1e-05, err_msg=f'result is {res[0]} but answer is {1}')
            np.testing.assert_allclose(res[1], 2, rtol=1e-05, err_msg=f'result is {res[1]} but answer is {2}')
            np.testing.assert_allclose(res[2], 3, rtol=1e-05, err_msg=f'result is {res[2]} but answer is {3}')

    def test_nested_switch_0d_tensor(self):
        if False:
            print('Hello World!')

        def fn_1(x=1):
            if False:
                return 10
            out = paddle.static.nn.switch_case(branch_index=paddle.full(shape=[], dtype='int32', fill_value=x), branch_fns={1: partial(paddle.full, shape=[], dtype='int32', fill_value=1), x: partial(paddle.full, shape=[], dtype='int32', fill_value=x)})
            return out

        def fn_2(x=2):
            if False:
                for i in range(10):
                    print('nop')
            out = paddle.static.nn.switch_case(branch_index=paddle.full(shape=[], dtype='int32', fill_value=2), branch_fns={1: partial(paddle.full, shape=[], dtype='int32', fill_value=1), 2: partial(fn_1, x=x)})
            return out

        def fn_3():
            if False:
                print('Hello World!')
            out = paddle.static.nn.switch_case(branch_index=paddle.full(shape=[], dtype='int32', fill_value=3), branch_fns={1: partial(paddle.full, shape=[], dtype='int32', fill_value=1), 3: partial(fn_2, x=3)})
            return out
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            index_1 = paddle.static.data(name='index_1', shape=[1], dtype='uint8')
            index_2 = paddle.full(shape=[], dtype='int32', fill_value=2)
            index_3 = paddle.full(shape=[], dtype='int64', fill_value=3)
            out_1 = paddle.static.nn.switch_case(branch_index=index_1, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
            out_2 = paddle.static.nn.switch_case(branch_index=index_2, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
            out_3 = paddle.static.nn.switch_case(branch_index=index_3, branch_fns={1: fn_1, 2: fn_2, 3: fn_3})
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(main_program, feed={'index_1': np.array([1], dtype='uint8')}, fetch_list=[out_1, out_2, out_3])
            np.testing.assert_allclose(res[0], 1, rtol=1e-05, err_msg=f'result is {res[0]} but answer is {1}')
            self.assertEqual(res[0].shape, ())
            np.testing.assert_allclose(res[1], 2, rtol=1e-05, err_msg=f'result is {res[1]} but answer is {2}')
            self.assertEqual(res[1].shape, ())
            np.testing.assert_allclose(res[2], 3, rtol=1e-05, err_msg=f'result is {res[2]} but answer is {3}')
            self.assertEqual(res[2].shape, ())

class TestAPISwitchCase_Error(unittest.TestCase):

    def test_error(self):
        if False:
            i = 10
            return i + 15

        def fn_1():
            if False:
                i = 10
                return i + 15
            return paddle.tensor.fill_constant(shape=[4, 2], dtype='int32', value=1)

        def fn_2():
            if False:
                return 10
            return paddle.tensor.fill_constant(shape=[4, 2], dtype='int32', value=2)

        def fn_3():
            if False:
                while True:
                    i = 10
            return paddle.tensor.fill_constant(shape=[4, 3], dtype='int32', value=3)
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            key_float32 = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.23)
            key_int32 = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=0.23)

            def type_error_branch_index():
                if False:
                    while True:
                        i = 10
                paddle.static.nn.switch_case(branch_index=1, branch_fns=[(1, fn_1)], default=fn_3)
            self.assertRaises(TypeError, type_error_branch_index)

            def dtype_error_branch_index():
                if False:
                    while True:
                        i = 10
                paddle.static.nn.switch_case(branch_index=key_float32, branch_fns=[(1, fn_1)], default=fn_3)
            self.assertRaises(TypeError, dtype_error_branch_index)

            def type_error_branch_fns():
                if False:
                    print('Hello World!')
                paddle.static.nn.switch_case(branch_index=key_int32, branch_fns=1, default=fn_3)
            self.assertRaises(TypeError, type_error_branch_fns)

            def type_error_index_fn_pair_1():
                if False:
                    i = 10
                    return i + 15
                paddle.static.nn.switch_case(branch_index=key_int32, branch_fns=[1], default=fn_3)
            self.assertRaises(TypeError, type_error_index_fn_pair_1)

            def type_error_index_fn_pair_2():
                if False:
                    print('Hello World!')
                paddle.static.nn.switch_case(branch_index=key_int32, branch_fns=[(1, 2, 3)], default=fn_3)
            self.assertRaises(TypeError, type_error_index_fn_pair_2)

            def type_error_key():
                if False:
                    i = 10
                    return i + 15
                paddle.static.nn.switch_case(branch_index=key_int32, branch_fns=[(2.3, 2)], default=fn_3)
            self.assertRaises(TypeError, type_error_key)

            def value_error_key():
                if False:
                    return 10
                paddle.static.nn.switch_case(branch_index=key_int32, branch_fns=[(2, fn_1), (2, fn_2)], default=fn_3)
            self.assertRaises(ValueError, value_error_key)

            def type_error_fn():
                if False:
                    return 10
                paddle.static.nn.switch_case(branch_index=key_int32, branch_fns=[(1, 1), (2, fn_2)], default=fn_3)
            self.assertRaises(TypeError, type_error_fn)

            def type_error_default():
                if False:
                    print('Hello World!')
                paddle.static.nn.switch_case(branch_index=key_int32, branch_fns=[(1, fn_1), (2, fn_2)], default=1)
            self.assertRaises(TypeError, type_error_default)
if __name__ == '__main__':
    unittest.main()