"""Tests for function_cache."""
import dataclasses
import itertools
import timeit
from typing import Any, Optional
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.core.function.polymorphism import function_type
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.types import trace

class MockGenericType(trace.TraceType):

    def __init__(self, obj):
        if False:
            return 10
        self._object = obj

    def is_subtype_of(self, other):
        if False:
            i = 10
            return i + 15
        return self == other

    def most_specific_common_supertype(self, others):
        if False:
            for i in range(10):
                print('nop')
        return None

    def placeholder_value(self, placeholder_context=None):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, trace.TraceType):
            return NotImplemented
        return isinstance(other, MockGenericType) and self._object == other._object

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self._object)

class MockIntGenericType(MockGenericType):

    def most_specific_common_supertype(self, others):
        if False:
            print('Hello World!')
        if all([self._object == other._object for other in others]):
            return MockIntGenericType(self._object)
        else:
            return None

class MockSubtypeOf2(MockGenericType):

    def is_subtype_of(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, MockGenericType):
            return False
        return other._object == 2

class MockSupertypes2With3(MockGenericType):

    def most_specific_common_supertype(self, others):
        if False:
            while True:
                i = 10
        if self._object == 2 and isinstance(others[0]._object, int):
            return MockSupertypes2With3(3)
        else:
            return None

class MockShape(trace.TraceType):

    def __init__(self, *shape: Optional[int]):
        if False:
            i = 10
            return i + 15
        self.shape = shape

    def is_subtype_of(self, other: 'MockShape') -> bool:
        if False:
            while True:
                i = 10
        if len(self.shape) != len(other.shape):
            return False
        if any((o is not None and s != o for (s, o) in zip(self.shape, other.shape))):
            return False
        return True

    def most_specific_common_supertype(self, _):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.shape)

    def __repr__(self):
        if False:
            return 10
        return str(self)

    def __hash__(self) -> int:
        if False:
            return 10
        return hash(self.shape)

    def __eq__(self, other: 'MockShape') -> bool:
        if False:
            return 10
        return self.shape == other.shape

    def placeholder_value(self, placeholder_context):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

def make_single_param_type(type_constraint):
    if False:
        i = 10
        return i + 15
    return function_type.FunctionType([function_type.Parameter('x', function_type.Parameter.POSITIONAL_ONLY, False, type_constraint)])

@dataclasses.dataclass(frozen=True)
class MockFunction:
    function_type: Any
    test_string: str

def make_type(value):
    if False:
        return 10
    typing_context = trace_type.InternalTracingContext()
    value_type = trace_type.from_value(value, typing_context)
    f_type = make_single_param_type(value_type)
    return f_type

class FunctionCacheTest(test.TestCase):

    def testConcreteFunctionDictRetainsInsertedKeys(self):
        if False:
            while True:
                i = 10
        cache = function_cache.FunctionCache()
        f_type_1 = make_type(1)
        self.assertIsNone(cache.lookup(f_type_1))
        f_type_2 = make_type(2)
        f_type_3 = make_type(3)
        cache.add(MockFunction(f_type_1, 'test_1'))
        cache.add(MockFunction(f_type_2, 'test_2'))
        self.assertEqual(cache.lookup(f_type_1).test_string, 'test_1')
        self.assertEqual(cache.lookup(f_type_2).test_string, 'test_2')
        self.assertIsNone(cache.lookup(f_type_3))

    def testClearRemovesAllConcreteFunctions(self):
        if False:
            for i in range(10):
                print('nop')
        cache = function_cache.FunctionCache()
        f_type_1 = make_type(1)
        f_type_2 = make_type(2)
        f_type_3 = make_type(3)
        cache.add(MockFunction(f_type_1, 'test_1'))
        cache.add(MockFunction(f_type_2, 'test_2'))
        self.assertEqual(cache.lookup(f_type_1).test_string, 'test_1')
        self.assertEqual(cache.lookup(f_type_2).test_string, 'test_2')
        self.assertIsNone(cache.lookup(f_type_3))
        cache.clear()
        self.assertIsNone(cache.lookup(f_type_1))
        self.assertIsNone(cache.lookup(f_type_2))
        self.assertIsNone(cache.lookup(f_type_3))

    def testDeleteRemovesConcreteFunctions(self):
        if False:
            while True:
                i = 10
        cache = function_cache.FunctionCache()
        f_type_1 = make_type(1)
        cache.add(MockFunction(f_type_1, 'test_1'))
        self.assertEqual(cache.lookup(f_type_1).test_string, 'test_1')
        cache.delete(f_type_1)
        self.assertIsNone(cache.lookup(f_type_1))
        f_type_2 = make_single_param_type(MockSubtypeOf2(2))
        cache.add(MockFunction(f_type_2, 'test_2'))
        self.assertEqual(cache.lookup(f_type_2).test_string, 'test_2')
        f_type_3 = make_single_param_type(MockSubtypeOf2(3))
        self.assertEqual(cache.lookup(f_type_3).test_string, 'test_2')
        cache.delete(f_type_2)
        self.assertIsNone(cache.lookup(f_type_2))
        self.assertIsNone(cache.lookup(f_type_3))

    def testMostSpecificFunctionCacheKeyIsLookedUp(self):
        if False:
            for i in range(10):
                print('nop')
        ctx = function_cache.FunctionContext(0)
        cache = function_cache.FunctionCache()
        cache.add(MockFunction(make_single_param_type(MockShape(1, 2, None)), 'a'), ctx)
        cache.add(MockFunction(make_single_param_type(MockShape(1, 2, 3)), 'b'), ctx)
        self.assertEqual(cache.lookup(make_single_param_type(MockShape(1, 2, 3)), ctx).test_string, 'b')

    def testFirstMostSpecificFunctionCacheKeyIsLookedUp(self):
        if False:
            return 10
        ctx = function_cache.FunctionContext(0)
        cache = function_cache.FunctionCache()
        cache.add(MockFunction(make_single_param_type(MockShape(1, 2, None)), 'a'), ctx)
        cache.add(MockFunction(make_single_param_type(MockShape(1, None, 3)), 'b'), ctx)
        self.assertEqual(cache.lookup(make_single_param_type(MockShape(1, 2, 3)), ctx).test_string, 'a')

    def testMostSpecificFunctionCacheKeyIsOrderAgnostic(self):
        if False:
            while True:
                i = 10
        ctx = function_cache.FunctionContext(0)
        keys = [(MockFunction(make_single_param_type(MockShape(1, 1, 1)), 'a'), ctx), (MockFunction(make_single_param_type(MockShape(1, None, 1)), 'b'), ctx), (MockFunction(make_single_param_type(MockShape(None, None, 1)), 'c'), ctx), (MockFunction(make_single_param_type(MockShape(None, None, None)), 'd'), ctx)]
        for permutation in itertools.permutations(keys):
            cache = function_cache.FunctionCache()
            cache.add(permutation[0][0], permutation[0][1])
            cache.add(permutation[1][0], permutation[1][1])
            cache.add(permutation[2][0], permutation[2][1])
            cache.add(permutation[3][0], permutation[3][1])
            self.assertEqual(cache.lookup(make_single_param_type(MockShape(1, 1, 1)), ctx).test_string, 'a')
            self.assertEqual(cache.lookup(make_single_param_type(MockShape(1, 2, 1)), ctx).test_string, 'b')
            self.assertEqual(cache.lookup(make_single_param_type(MockShape(2, 2, 1)), ctx).test_string, 'c')
            self.assertEqual(cache.lookup(make_single_param_type(MockShape(2, 2, 2)), ctx).test_string, 'd')

class FunctionCacheBenchmark(test.Benchmark):

    def benchmarkCacheHit50thKeyMiss(self):
        if False:
            return 10
        cache = function_cache.FunctionCache()
        args_per_call = 5
        num_total_checks = 50
        keys = []
        for i in range(num_total_checks):
            args = []
            for j in range(args_per_call):
                args.append(array_ops.zeros([i, j]))
            keys.append(make_type(args))
        for key in keys[:-1]:
            cache.add(MockFunction(key, 'testing'))
        iterations = 10000
        subtyping_time = timeit.timeit(lambda : cache.lookup(keys[-1]), number=iterations)
        equality_time = timeit.timeit(lambda : cache.lookup(keys[-1]), number=iterations)
        self.report_benchmark(name='cache_hit_50th_f_type_miss', iters=iterations, wall_time=subtyping_time + equality_time, metrics=[{'name': 'cache_hit_50th_f_type_miss_subtype_avg_ms', 'value': subtyping_time / iterations * 1000}, {'name': 'cache_hit_50th_f_type_miss_equality_avg_ms', 'value': equality_time / iterations * 1000}, {'name': 'cache_hit_50th_f_type_miss_subtype_over_equality_ratio', 'value': subtyping_time / equality_time}])

    def benchmarkCacheHit50thKeyEqual(self):
        if False:
            i = 10
            return i + 15
        cache = function_cache.FunctionCache()
        args_per_call = 5
        num_total_checks = 50
        keys = []
        for i in range(num_total_checks):
            args = []
            for j in range(args_per_call):
                args.append(array_ops.zeros([i, j]))
            keys.append(make_type(args))
        for key in keys:
            cache.add(MockFunction(key, 'testing'))
        iterations = 10000
        subtyping_time = timeit.timeit(lambda : cache.lookup(keys[-1]), number=iterations)
        equality_time = timeit.timeit(lambda : cache.lookup(keys[-1]), number=iterations)
        self.report_benchmark(name='cache_hit_50th_f_type_equal', iters=iterations, wall_time=subtyping_time + equality_time, metrics=[{'name': 'cache_hit_50th_f_type_equal_subtype_avg_ms', 'value': subtyping_time / iterations * 1000}, {'name': 'cache_hit_50th_f_type_equal_equality_avg_ms', 'value': equality_time / iterations * 1000}, {'name': 'cache_hit_50th_f_type_subtype_over_equality_ratio', 'value': subtyping_time / equality_time}])

    def benchmarkCacheHit50thKeyKnownSubtype(self):
        if False:
            i = 10
            return i + 15
        cache = function_cache.FunctionCache()
        args_per_call = 5
        num_total_checks = 50
        keys = []
        for i in range(num_total_checks - 1):
            args = []
            for j in range(args_per_call):
                args.append(array_ops.zeros([i, j]))
            keys.append(make_type(args))
        for key in keys:
            cache.add(MockFunction(key, 'testing'))
        cache.add(MockFunction(make_single_param_type(MockSubtypeOf2(2)), 'testing'))
        cache.lookup(make_single_param_type(MockSubtypeOf2(3)))
        iterations = 10000
        lookup_key = make_single_param_type(MockSubtypeOf2(2))
        subtyping_time = timeit.timeit(lambda : cache.lookup(lookup_key), number=iterations)
        self.report_benchmark(name='cache_hit_50th_f_type_known_subtype', iters=iterations, wall_time=subtyping_time, metrics=[{'name': 'cache_hit_50th_f_type_known_subtype_avg_ms', 'value': subtyping_time / iterations * 1000}])

    def benchmarkCacheHit50thKeyUnknownSubtype(self):
        if False:
            for i in range(10):
                print('nop')
        cache = function_cache.FunctionCache()
        args_per_call = 5
        num_total_checks = 50
        keys = []
        for i in range(num_total_checks - 1):
            args = []
            for j in range(args_per_call):
                args.append(array_ops.zeros([i, j]))
            keys.append(make_type(args))

        def setup():
            if False:
                return 10
            cache.clear()
            for key in keys:
                cache.add(MockFunction(key, 'testing'))
            cache.add(MockFunction(make_single_param_type(MockSubtypeOf2(3)), 'testing'))
        iterations = 10000
        lookup_key = make_single_param_type(MockSubtypeOf2(2))
        subtyping_time = sum(timeit.repeat(stmt=lambda : cache.lookup(lookup_key), setup=setup, repeat=iterations, number=1))
        self.report_benchmark(name='cache_hit_50th_f_type_unknown_subtype', iters=iterations, wall_time=subtyping_time, metrics=[{'name': 'cache_hit_50th_f_type_unknown_subtype_avg_ms', 'value': subtyping_time / iterations * 1000}])
if __name__ == '__main__':
    test.main()