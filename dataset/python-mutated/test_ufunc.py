import itertools
import unittest
from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils

def _permutate_shapes(shapes_list):
    if False:
        for i in range(10):
            print('nop')
    permutated_shapes_set = set()
    for shapes in shapes_list:
        for permutated_shapes in itertools.permutations(shapes):
            permutated_shapes_set.add(permutated_shapes)
    return list(permutated_shapes_set)

@testing.parameterize(*testing.product({'shapes': _permutate_shapes([((1,), (1,)), ((3, 4), (3, 4)), ((10,), (1,)), ((3, 4), (3, 1)), ((3, 4), (1, 4)), ((3, 4), (4,)), ((3, 4), (1, 1)), ((3, 4), (1,)), ((2, 3, 4), (1, 1, 1)), ((3, 1), (1, 4)), ((2, 1, 4), (3, 1)), ((0,), (0,)), ((0,), (1,)), ((2, 0, 3), (2, 0, 3)), ((2, 0, 3), (0, 1)), ((256, 256), (256,)), ((256, 256), (256, 1))])}))
class TestFusionBroadcast(unittest.TestCase):

    def generate_inputs(self, xp):
        if False:
            for i in range(10):
                print('nop')
        (shape1, shape2) = self.shapes
        x = testing.shaped_random(shape1, xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random(shape2, xp, 'int64', scale=10, seed=1)
        return ((x, y), {})

    @fusion_utils.check_fusion()
    def test_broadcast(self, xp):
        if False:
            while True:
                i = 10
        return lambda x, y: x + y

@testing.parameterize(*testing.product({'shapes': _permutate_shapes([((2,), (3,)), ((2,), (0,)), ((3, 2), (3, 3)), ((3, 2), (2, 2)), ((3,), (1, 2))])}))
class TestFusionBroadcastInvalid(unittest.TestCase):

    def generate_inputs(self, xp):
        if False:
            i = 10
            return i + 15
        (shape1, shape2) = self.shapes
        x = testing.shaped_random(shape1, xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random(shape2, xp, 'int64', scale=10, seed=1)
        return ((x, y), {})

    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_broadcast(self, xp):
        if False:
            for i in range(10):
                print('nop')
        return lambda x, y: x + y

    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_broadcast_inplace(self, xp):
        if False:
            for i in range(10):
                print('nop')

        def impl(x, y):
            if False:
                for i in range(10):
                    print('nop')
            x += y
        return impl

class TestFusionParseInput(unittest.TestCase):

    def generate_inputs(self, xp):
        if False:
            while True:
                i = 10
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        return ((x,), {})

    @fusion_utils.check_fusion()
    def test_add(self, xp):
        if False:
            for i in range(10):
                print('nop')
        return lambda x: x + x

    @fusion_utils.check_fusion(accept_error=(ValueError, TypeError))
    def test_add_too_less_param(self, xp):
        if False:
            return 10
        return lambda x: xp.add(x)

    @fusion_utils.check_fusion(accept_error=(ValueError, TypeError))
    def test_add_too_much_param(self, xp):
        if False:
            print('Hello World!')
        return lambda x: xp.add(x, x, x, x)

    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_add_none(self, xp):
        if False:
            while True:
                i = 10
        return lambda x: x + None

    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_add_object(self, xp):
        if False:
            return 10
        return lambda x: x + object()

    @fusion_utils.check_fusion()
    def test_add_kwargs_out_none(self, xp):
        if False:
            for i in range(10):
                print('nop')

        def impl(x):
            if False:
                return 10
            xp.add(x, x, out=None)
        return impl

    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_add_out_object(self, xp):
        if False:
            return 10

        def impl(x):
            if False:
                print('Hello World!')
            xp.add(x, x, object())
            return x
        return impl

    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_add_kwargs_out_object(self, xp):
        if False:
            for i in range(10):
                print('nop')

        def impl(x):
            if False:
                while True:
                    i = 10
            xp.add(x, x, out=object())
            return x
        return impl

    @fusion_utils.check_fusion()
    def test_divmod(self, xp):
        if False:
            return 10
        return lambda x: xp.divmod(x, x)

class TestFusionOutDtype(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        if False:
            print('Hello World!')
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=1)
        z = testing.shaped_random((3, 4), xp, dtype2, scale=10, seed=2)
        return ((x, y, z), {})

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'), full=True, no_complex=True)
    @fusion_utils.check_fusion(accept_error=TypeError)
    @testing.with_requires('numpy>=1.13')
    def test_outarg(self, xp, dtype1, dtype2):
        if False:
            i = 10
            return i + 15

        def impl(x, y, z):
            if False:
                return 10
            xp.add(x, y, out=z)
            return z
        return impl

class TestFusionScalar(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        if False:
            i = 10
            return i + 15
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        return ((array,), {})

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_python_scalar_r(self, xp, dtype1, dtype2):
        if False:
            for i in range(10):
                print('nop')

        def func(array):
            if False:
                return 10
            py_scalar = dtype2(1).item()
            return array + py_scalar
        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_numpy_scalar_r(self, xp, dtype1, dtype2):
        if False:
            while True:
                i = 10

        def func(array):
            if False:
                for i in range(10):
                    print('nop')
            np_scalar = dtype2(1)
            return array + np_scalar
        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_python_scalar_l(self, xp, dtype1, dtype2):
        if False:
            return 10

        def func(array):
            if False:
                print('Hello World!')
            py_scalar = dtype2(1).item()
            return py_scalar + array
        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_numpy_scalar_l(self, xp, dtype1, dtype2):
        if False:
            i = 10
            return i + 15

        def func(array):
            if False:
                i = 10
                return i + 15
            np_scalar = dtype2(1)
            return np_scalar + array
        return func

    def python_scalar_param_r(self, xp, dtype1, dtype2):
        if False:
            for i in range(10):
                print('nop')
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        py_scalar = dtype2(1).item()
        return ((array, py_scalar), {})

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='python_scalar_param_r')
    def test_python_scalar_param_r(self, xp, dtype1, dtype2):
        if False:
            while True:
                i = 10

        def func(array, py_scalar):
            if False:
                return 10
            return array + py_scalar
        return func

    def python_scalar_param_l(self, xp, dtype1, dtype2):
        if False:
            i = 10
            return i + 15
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        py_scalar = dtype2(1).item()
        return ((py_scalar, array), {})

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='python_scalar_param_l')
    def test_python_scalar_param_l(self, xp, dtype1, dtype2):
        if False:
            print('Hello World!')

        def func(py_scalar, array):
            if False:
                for i in range(10):
                    print('nop')
            return py_scalar + array
        return func

    def numpy_scalar_param_r(self, xp, dtype1, dtype2):
        if False:
            return 10
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        np_scalar = dtype2(1)
        return ((array, np_scalar), {})

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_param_r')
    def test_numpy_scalar_param_r(self, xp, dtype1, dtype2):
        if False:
            while True:
                i = 10

        def func(array, np_scalar):
            if False:
                for i in range(10):
                    print('nop')
            return array + np_scalar
        return func

    def numpy_scalar_param_l(self, xp, dtype1, dtype2):
        if False:
            i = 10
            return i + 15
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        np_scalar = dtype2(1)
        return ((np_scalar, array), {})

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_param_l')
    def test_numpy_scalar_param_l(self, xp, dtype1, dtype2):
        if False:
            i = 10
            return i + 15

        def func(np_scalar, array):
            if False:
                for i in range(10):
                    print('nop')
            return np_scalar + array
        return func

    def numpy_scalar_params_binop(self, xp, dtype1, dtype2):
        if False:
            for i in range(10):
                print('nop')
        scalar1 = dtype1(1)
        scalar2 = dtype2(1)
        array = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        return ((scalar1, scalar2, array), {})

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_params_binop')
    def test_numpy_scalar_params_binop(self, xp, dtype1, dtype2):
        if False:
            print('Hello World!')

        def func(scalar1, scalar2, array):
            if False:
                i = 10
                return i + 15
            dtype = (scalar1 + scalar2).dtype
            return array.astype(dtype)
        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_params_binop')
    def test_scalar_inplace_update(self, xp, dtype1, dtype2):
        if False:
            while True:
                i = 10

        def func(scalar1, scalar2, array):
            if False:
                print('Hello World!')
            scalar1_copy = scalar1
            scalar1 += scalar2
            return array + scalar1 + scalar1_copy
        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_param_r')
    def test_scalar_inplace_update_with_array(self, xp, dtype1, dtype2):
        if False:
            print('Hello World!')

        def func(array, scalar):
            if False:
                while True:
                    i = 10
            scalar += array
            return scalar
        return func