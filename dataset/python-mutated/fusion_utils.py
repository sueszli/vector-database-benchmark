import numpy
import cupy
from cupy import testing
scalar_types = (numpy.generic, int, float, complex)

def check_fusion(generate_inputs_name='generate_inputs', generate_inputs_args=None, check_array=None, check_array_kwargs=None, accept_error=()):
    if False:
        print('Hello World!')
    "Decorator for tests for ``cupy.fuse``.\n\n    This decorator checks the results of the original function is equals to\n    that of the fused function.\n\n    Args:\n        generate_input_args(tuple): argument tuple passed to\n            ``generate_input``. Defaults to ``()``.\n        check_array(function): testing function which compares\n            ``{numpy/cupy}.ndarray`` objects.\n            Defaults to ``testing.assert_allclose``.\n        check_array_kwargs(dict): keyword arguments passed to\n            ``check_array``. Defaults to ``{'rtol': 3e-3, 'atol': 3e-3}``.\n        accept_error(Exception or tuple of Exception):\n            Specify acceptable errors.\n    "
    if generate_inputs_args is None:
        generate_inputs_args = ()
    if check_array is None:
        check_array = testing.assert_allclose
    if check_array_kwargs is None:
        check_array_kwargs = {'rtol': 0.003, 'atol': 0.003}
    if not isinstance(accept_error, (tuple, list)):
        accept_error = (accept_error,)

    def check(xp, actual, expected):
        if False:
            for i in range(10):
                print('nop')
        if expected is None:
            assert actual is None
        elif isinstance(expected, scalar_types + (numpy.ndarray,)):
            assert isinstance(actual, scalar_types + (xp.ndarray,))
            check_array(actual, expected, **check_array_kwargs)
        elif isinstance(expected, (list, tuple)):
            assert type(actual) == type(expected)
            for (item_actual, item_expected) in zip(actual, expected):
                check(xp, item_actual, item_expected)
        elif isinstance(expected, dict):
            assert isinstance(actual, dict)
            for (key, expected_value) in expected:
                actual_value = actual.pop(key)
                check_array(xp, actual_value, expected_value)
            assert len(actual) == 0
        else:
            assert False

    def call(func, args, kwargs):
        if False:
            print('Hello World!')
        try:
            ret = func(*args, **kwargs)
            err = None
        except Exception as e:
            if not isinstance(e, accept_error):
                raise
            ret = None
            err = e
        return (ret, err)

    def check_result(xp, actual, expected):
        if False:
            while True:
                i = 10
        (ret_a, err_a) = actual
        (ret_e, err_e) = expected
        if err_e is None:
            if err_a is not None:
                raise err_a
            check(xp, ret_a, ret_e)
        elif err_a is None:
            raise err_e

    def deco(func):
        if False:
            while True:
                i = 10

        def wrapper(self, **generate_inputs_kwargs):
            if False:
                while True:
                    i = 10
            generate_inputs = getattr(self, generate_inputs_name)
            impl_np = func(self, numpy, **generate_inputs_kwargs)
            impl_cp = func(self, cupy, **generate_inputs_kwargs)

            @cupy.fuse()
            def impl_fuse_np(*args, **kwargs):
                if False:
                    print('Hello World!')
                return impl_np(*args, **kwargs)

            @cupy.fuse()
            def impl_fuse_cp(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return impl_cp(*args, **kwargs)
            (args_np, kwargs_np) = generate_inputs(numpy, *generate_inputs_args, **generate_inputs_kwargs)
            (args_cp, kwargs_cp) = generate_inputs(cupy, *generate_inputs_args, **generate_inputs_kwargs)
            (args_fuse_np, kwargs_fuse_np) = generate_inputs(numpy, *generate_inputs_args, **generate_inputs_kwargs)
            (args_fuse_cp, kwargs_fuse_cp) = generate_inputs(cupy, *generate_inputs_args, **generate_inputs_kwargs)
            result_np = call(impl_np, args_np, kwargs_np)
            result_cp = call(impl_cp, args_cp, kwargs_cp)
            result_fuse_np = call(impl_fuse_np, args_fuse_np, kwargs_fuse_np)
            result_fuse_cp = call(impl_fuse_cp, args_fuse_cp, kwargs_fuse_cp)
            check_result(cupy, result_cp, result_np)
            check_result(numpy, result_fuse_np, result_np)
            check_result(cupy, result_fuse_cp, result_np)
            (_, err) = result_np
            if err is None:
                check(cupy, args_cp, args_np)
                check(numpy, args_fuse_np, args_np)
                check(cupy, args_fuse_cp, args_np)
        return wrapper
    return deco

def can_use_grid_synchronization():
    if False:
        for i in range(10):
            print('nop')
    return not cupy.cuda.runtime.is_hip and cupy.cuda.runtime.runtimeGetVersion() >= 9000 and (int(cupy.cuda.device.get_compute_capability()) >= 70)