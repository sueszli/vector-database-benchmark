import unittest
from unittest import mock
import cupy
from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils

class CreateMock(object):

    def __init__(self, target):
        if False:
            for i in range(10):
                print('nop')
        self.target = eval(target)
        self.retvals = []

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ret = self.target(*args, **kwargs)
        self.retvals.append(ret)
        return ret

    def check_number_of_ops(self, loops, memories, variables, lookup, mutate):
        if False:
            return 10
        pass

def check_number_of_ops(loops, memories, variables, lookup, mutate):
    if False:
        for i in range(10):
            print('nop')

    def wrapper(test_method):
        if False:
            i = 10
            return i + 15

        def new_impl(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            target = 'cupy._core._fusion_trace.TraceImpl'
            with mock.patch(target, CreateMock(target)) as m:
                result = test_method(self, *args, **kwargs)
                m.check_number_of_ops(loops, memories, variables, lookup, mutate)
            return result
        return new_impl
    return wrapper

class TestOptimizations(unittest.TestCase):

    def generate_inputs(self, xp):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=1)
        return ((x, y), {})

    def generate_input_broadcast(self, xp):
        if False:
            for i in range(10):
                print('nop')
        x = testing.shaped_random((3, 1, 4), xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random((3, 5, 4), xp, 'int64', scale=10, seed=1)
        return ((x, y), {})

    def generate_input_same_memory(self, xp):
        if False:
            for i in range(10):
                print('nop')
        x = testing.shaped_random((4, 4), xp, 'int64', scale=10, seed=0)
        return ((x, x, x.T), {})

    @check_number_of_ops(loops=1, memories=3, variables=3, lookup=[2], mutate=[1])
    @fusion_utils.check_fusion()
    def test_one_elementwise_op(self, xp):
        if False:
            return 10
        return lambda x, y: x + y

    @check_number_of_ops(loops=1, memories=3, variables=3, lookup=[2], mutate=[1])
    @fusion_utils.check_fusion()
    def test_fuse_elementwise_op_1(self, xp):
        if False:
            while True:
                i = 10

        def impl(x, y):
            if False:
                return 10
            return x + x + y + y
        return impl

    @check_number_of_ops(loops=1, memories=3, variables=3, lookup=[2], mutate=[1])
    @fusion_utils.check_fusion()
    def test_fuse_elementwise_op_2(self, xp):
        if False:
            while True:
                i = 10

        def impl(x, y):
            if False:
                print('Hello World!')
            z = x + y
            return z + z
        return impl

    @check_number_of_ops(loops=1, memories=4, variables=4, lookup=[3], mutate=[1])
    @fusion_utils.check_fusion()
    def test_fuse_elementwise_ops_4(self, xp):
        if False:
            while True:
                i = 10

        def impl(x, y):
            if False:
                i = 10
                return i + 15
            res = 0
            for i in range(10):
                res += x + y
            return res
        return impl

    @check_number_of_ops(loops=0, memories=1, variables=1, lookup=[], mutate=[])
    @fusion_utils.check_fusion()
    def test_ignore_op(self, xp):
        if False:
            print('Hello World!')

        def impl(x, y):
            if False:
                return 10
            z = x + y
            return x
        return impl

    @check_number_of_ops(loops=1, memories=4, variables=4, lookup=[2], mutate=[2])
    @fusion_utils.check_fusion()
    def test_returns_tuple(self, xp):
        if False:
            for i in range(10):
                print('nop')

        def impl(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return (x + y, y - x)
        return impl

    @check_number_of_ops(loops=2, memories=4, variables=6, lookup=[1, 3], mutate=[1, 1])
    @fusion_utils.check_fusion(generate_inputs_name='generate_input_broadcast')
    def test_different_shapes(self, xp):
        if False:
            for i in range(10):
                print('nop')

        def impl(x, y):
            if False:
                i = 10
                return i + 15
            r1 = x + x
            r2 = x + y
            r3 = y + y
            return r1 * r2 * r3
        return impl

    @check_number_of_ops(loops=1, memories=2, variables=2, lookup=[2], mutate=[2])
    @fusion_utils.check_fusion()
    def test_inplace_elementwise_1(self, xp):
        if False:
            print('Hello World!')

        def impl(x, y):
            if False:
                for i in range(10):
                    print('nop')
            x += y
            y += x
            x += y
        return impl

    @check_number_of_ops(loops=1, memories=3, variables=3, lookup=[2], mutate=[2])
    @fusion_utils.check_fusion()
    def test_inplace_elementwise_2(self, xp):
        if False:
            i = 10
            return i + 15

        def impl(x, y):
            if False:
                print('Hello World!')
            x += y
            return x + x
        return impl

    @check_number_of_ops(loops=1, memories=1, variables=1, lookup=[1], mutate=[1])
    @fusion_utils.check_fusion(generate_inputs_name='generate_input_same_memory')
    def test_inplace_same_variable(self, xp):
        if False:
            print('Hello World!')

        def impl(x, y, z):
            if False:
                print('Hello World!')
            x += y
            x += y
        return impl

    @check_number_of_ops(loops=4, memories=3, variables=4, lookup=[1, 2, 1, 2], mutate=[1, 1, 1, 1])
    @fusion_utils.check_fusion(generate_inputs_name='generate_input_same_memory')
    def test_inplace_same_memory_space(self, xp):
        if False:
            while True:
                i = 10

        def impl(x, y, z):
            if False:
                i = 10
                return i + 15
            x += z
            x += z
        return impl

    @check_number_of_ops(loops=1, memories=2, variables=2, lookup=[1], mutate=[1])
    @fusion_utils.check_fusion()
    def test_one_reduction_op(self, xp):
        if False:
            print('Hello World!')
        return lambda x, y: xp.sum(x, axis=0)

    @check_number_of_ops(loops=1, memories=2, variables=3, lookup=[1], mutate=[1])
    @fusion_utils.check_fusion()
    def test_one_reduction_op_rotate(self, xp):
        if False:
            for i in range(10):
                print('nop')
        return lambda x, y: xp.sum(x, axis=1)

    @check_number_of_ops(loops=2, memories=4, variables=4, lookup=[2, 1], mutate=[1, 1])
    @fusion_utils.check_fusion()
    def test_one_fuse_reduction_premap(self, xp):
        if False:
            while True:
                i = 10

        def impl(x, y):
            if False:
                return 10
            premap = x + y
            return xp.sum(premap, axis=0)
        return impl