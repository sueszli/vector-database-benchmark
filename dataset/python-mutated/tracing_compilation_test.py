import copy
import functools
import itertools
import weakref
from absl.testing import parameterized
import numpy
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache as function_cache_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.util import compat
from tensorflow.python.util import nest
try:
    import attr
except ImportError:
    attr = None

def compiled_fn(fn=None, **tracing_options):
    if False:
        while True:
            i = 10
    'Decorator that compiles/calls wrapped function.'
    if fn is None:
        return functools.partial(compiled_fn, **tracing_options)
    signature = tracing_options.pop('input_signature', None)
    (function_type, default_values) = function_type_utils.make_function_type(fn, signature)
    tracing_options['polymorphic_type'] = function_type
    tracing_options['default_values'] = default_values

    def wrapped(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        bound_args = function_type.bind_with_defaults(args, kwargs, default_values)
        return tracing_compilation.call_function(bound_args.args, bound_args.kwargs, tracing_compilation.TracingOptions(fn, **tracing_options))

    def trace(*args, **kwargs):
        if False:
            return 10
        return tracing_compilation.trace_function(args, kwargs, tracing_compilation.TracingOptions(fn, **tracing_options))
    wrapped.get_concrete_function = trace
    return wrapped

class TracingCompilationTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testBackwardNoneGradient(self):
        if False:
            return 10
        model = variables.Variable(1.0, name='model')
        count = variables.Variable(0)

        @compiled_fn
        def forward_pass(value):
            if False:
                i = 10
                return i + 15
            count.assign_add(1)
            residuals = value - model
            loss = 0.5 * math_ops.reduce_mean(math_ops.pow(residuals, 2))
            return (loss, count)

        def reduce_fn(x):
            if False:
                while True:
                    i = 10
            if context.executing_eagerly():
                with backprop.GradientTape() as t:
                    (loss, count) = forward_pass(x)
                return (t.gradient(loss, model), count)
            (loss, count) = forward_pass(x)
            grad_only = gradients_impl.gradients(loss, model)
            return (grad_only, count)
        (g, _) = reduce_fn(constant_op.constant([7.0]))
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(nest.flatten(self.evaluate(g)), [-6.0])

    def testExternalControlDependency(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default(), self.test_session():
            v = variables.Variable(1.0)
            v.initializer.run()
            op = v.assign_add(1.0)

            @compiled_fn
            def f():
                if False:
                    print('Hello World!')
                with ops.control_dependencies([op]):
                    return 1.0
            self.evaluate(f())
            self.assertAllEqual(self.evaluate(v), 2.0)

    def testInputShapeFunctionRelaxation(self):
        if False:
            i = 10
            return i + 15
        unknown_dim = [False]
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(reduce_retracing=True, function_cache=function_cache)
        def func(a):
            if False:
                while True:
                    i = 10
            if a._shape_tuple()[0] is None:
                unknown_dim[0] = True
            return a + 1
        func(constant_op.constant([]))
        self.assertFalse(unknown_dim[0])
        self.assertLen(function_cache, 1)
        func(constant_op.constant([1.0]))
        self.assertTrue(unknown_dim[0])
        self.assertLen(function_cache, 2)
        func(constant_op.constant([1.0, 2.0]))
        self.assertTrue(unknown_dim[0])

    def testNestedInputShapeFunctionRelaxation(self):
        if False:
            for i in range(10):
                print('nop')
        unknown_dim = [False]
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(reduce_retracing=True, function_cache=function_cache)
        def func(a_, b_=None):
            if False:
                for i in range(10):
                    print('nop')
            del a_
            self.assertEqual(b_[0]._shape_tuple(), ())
            if b_[1]._shape_tuple()[0] is None:
                unknown_dim[0] = True
            return b_[0] + 1
        a = 'hi'
        b0 = constant_op.constant(1.0)
        func(a, b_=[b0, constant_op.constant([])])
        self.assertFalse(unknown_dim[0])
        self.assertLen(function_cache, 1)
        func(a, b_=[b0, constant_op.constant([1.0])])
        self.assertTrue(unknown_dim[0])
        self.assertLen(function_cache, 2)
        func(a, b_=[b0, constant_op.constant([1.0, 1.0])])
        self.assertTrue(unknown_dim[0])
        self.assertLen(function_cache, 2)
        unknown_dim[0] = False
        a = 'bye'
        func(a, b_=[b0, constant_op.constant([])])
        self.assertFalse(unknown_dim[0])
        self.assertLen(function_cache, 3)
        func(a, b_=[b0, constant_op.constant([1.0])])
        self.assertTrue(unknown_dim[0])
        self.assertLen(function_cache, 4)

    @test_util.run_v2_only
    def testGraphEagerIsolation(self):
        if False:
            return 10

        def f_py():
            if False:
                while True:
                    i = 10
            self.v = variables.Variable(1.0)
            return self.v.read_value()
        f = lambda : tracing_compilation.call_function(tracing_options=tracing_compilation.TracingOptions(f_py, 'f'))
        self.assertAllEqual(f(), 1.0)
        with ops.Graph().as_default():
            self.assertEqual(f().shape, ())

    @test_util.run_v2_only
    def testCompilationNumpyArraysConvertedToTensors(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(x, tensor_lib.Tensor)
            return x
        x = random_ops.random_uniform([2, 2]).numpy()
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(f, function_cache=function_cache)
        defined(x)
        self.assertLen(function_cache, 1)
        x = random_ops.random_uniform([2, 2]).numpy()
        defined(x)
        self.assertLen(function_cache, 1)
        np_ones = numpy.ones([], numpy.float32)
        np_zeros = numpy.zeros([], numpy.float32)
        tf_ones = array_ops.ones([])
        tf_zeros = array_ops.zeros([])
        self.assertEqual(1.0, defined(np_ones).numpy())
        self.assertLen(function_cache, 2)
        self.assertEqual(0.0, defined(np_zeros).numpy())
        self.assertEqual(1.0, defined(tf_ones).numpy())
        self.assertEqual(0.0, defined(tf_zeros).numpy())
        self.assertLen(function_cache, 2)
        mutable = numpy.ones([], numpy.float32)
        self.assertEqual(1.0, defined(mutable).numpy())
        mutable.fill(0)
        self.assertEqual(0.0, defined(mutable).numpy())

        class MyNdarray(numpy.ndarray):
            pass
        self.assertEqual(1.0, defined(np_ones.view(MyNdarray)).numpy())
        self.assertEqual(0.0, defined(np_zeros.view(MyNdarray)).numpy())
        self.assertLen(function_cache, 2)

    @test_util.run_v2_only
    def testNumpyDtypeInputSupported(self):
        if False:
            for i in range(10):
                print('nop')

        @compiled_fn
        def f(x, dtype):
            if False:
                return 10
            return constant_op.constant(dtype(x))
        self.assertEqual(f(1, numpy.float32).numpy(), numpy.float32(1))
        self.assertEqual(f(2, numpy.float32).numpy(), numpy.float32(2))
        self.assertEqual(f(1, numpy.int32).numpy(), numpy.int32(1))
        self.assertEqual(f(2, numpy.int32).numpy(), numpy.int32(2))

    @test_util.run_v2_only
    def testCompilationNumpyArraysConvertedToTensorsInKwargs(self):
        if False:
            for i in range(10):
                print('nop')

        def f(**kwargs):
            if False:
                i = 10
                return i + 15
            x = kwargs.pop('x')
            self.assertIsInstance(x, tensor_lib.Tensor)
            return x
        x = random_ops.random_uniform([2, 2]).numpy()
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(f, function_cache=function_cache)
        defined(x=x)
        self.assertLen(function_cache, 1)
        x = random_ops.random_uniform([2, 2]).numpy()
        defined(x=x)
        self.assertLen(function_cache, 1)
        self.assertEqual(1.0, defined(x=numpy.ones([])).numpy())
        self.assertEqual(0.0, defined(x=numpy.zeros([])).numpy())
        self.assertEqual(1.0, defined(x=array_ops.ones([])).numpy())
        self.assertEqual(0.0, defined(x=array_ops.zeros([])).numpy())

    @test_util.run_v2_only
    def testFuncListAttr(self):
        if False:
            i = 10
            return i + 15

        @compiled_fn
        def test_function(val):
            if False:
                return 10

            def fn1():
                if False:
                    i = 10
                    return i + 15
                return array_ops.ones([10])
            fn2 = lambda : array_ops.ones([10]) * 2

            def fn3(x=3):
                if False:
                    while True:
                        i = 10
                return array_ops.ones([10]) * x
            fn4 = functools.partial(fn3, x=4)
            fn5 = functools.partial(fn3, 5)
            return gen_functional_ops.case(val, [], [dtypes.float32], [compiled_fn(f).get_concrete_function() for f in (fn1, fn2, fn3, fn4, fn5)])
        ones = array_ops.ones([10])
        self.assertAllEqual([ones], test_function(0))
        self.assertAllEqual([ones * 2], test_function(1))
        self.assertAllEqual([ones * 3], test_function(2))
        self.assertAllEqual([ones * 4], test_function(3))
        self.assertAllEqual([ones * 5], test_function(4))
        self.assertAllEqual([ones * 5], test_function(22))

    @test_util.enable_control_flow_v2
    def testVariableInLoopInFunction(self):
        if False:
            while True:
                i = 10

        def test_function_py():
            if False:
                print('Hello World!')

            def loop_test(_):
                if False:
                    i = 10
                    return i + 15
                return False

            def loop_body(_):
                if False:
                    i = 10
                    return i + 15
                return variable_scope.get_variable('a', shape=())
            return while_loop.while_loop(loop_test, loop_body, [0.0])
        test_function = tracing_compilation.trace_function(tracing_options=tracing_compilation.TracingOptions(test_function_py, 'test_function'))
        self.assertEqual(test_function().shape, [])

    @test_util.run_in_graph_and_eager_modes
    def testCompilationForcesResourceVariables(self):
        if False:
            i = 10
            return i + 15

        def variable_creator():
            if False:
                return 10
            self.v = variables.Variable(0.0)
            return self.v.read_value()
        defined = tracing_compilation.trace_function(tracing_options=tracing_compilation.TracingOptions(variable_creator, 'variable_creator'))
        defined()
        self.assertIsInstance(self.v, resource_variable_ops.ResourceVariable)

    @test_util.run_gpu_only
    @test_util.run_in_graph_and_eager_modes
    def testFunctionWithResourcesOnDifferentDevices(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device('/cpu:0'):
            v_cpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])
        with ops.device('/gpu:0'):
            v_gpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

        def sum_gather():
            if False:
                for i in range(10):
                    print('nop')
            cpu_result = math_ops.reduce_sum(array_ops.gather(v_cpu, [1, 2]))
            gpu_result = math_ops.reduce_sum(array_ops.gather(v_gpu, [1, 2]))
            return (cpu_result, gpu_result)
        defined = compiled_fn(sum_gather)
        if not context.executing_eagerly():
            self.evaluate(variables.global_variables_initializer())
        expected = self.evaluate(sum_gather())
        self.assertAllEqual(expected, self.evaluate(defined()))

    @test_util.assert_no_new_pyobjects_executing_eagerly
    def testCallOptionsMemory(self):
        if False:
            return 10

        @compiled_fn
        def model(x):
            if False:
                return 10
            return x + constant_op.constant(1.0)
        context.context().function_call_options = None
        model(constant_op.constant(2.0))

    @test_util.run_in_graph_and_eager_modes
    def testVariablesPlacedOnOutsideDevice(self):
        if False:
            i = 10
            return i + 15

        class _Obj(object):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.v = None

            @compiled_fn
            def f(self):
                if False:
                    print('Hello World!')
                if self.v is None:
                    self.v = variables.Variable(1.0)
                return self.v + 1.0
        has_device = _Obj()
        with ops.device('cpu:0'):
            has_device.f()
        self.assertIn('CPU', has_device.v.device)

    def testCacheObjectHashCollisions(self):
        if False:
            while True:
                i = 10

        class Foo:

            def __hash__(self):
                if False:
                    return 10
                return 42

        def func(foo):
            if False:
                while True:
                    i = 10
            return constant_op.constant([id(foo)])
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(func, function_cache=function_cache)
        foo_1 = Foo()
        defined(foo_1)
        self.assertLen(function_cache, 1)
        foo_2 = Foo()
        defined(foo_2)
        self.assertLen(function_cache, 2)

    def testCacheTensorDtypeCollision(self):
        if False:
            print('Hello World!')

        def func(t):
            if False:
                i = 10
                return i + 15
            return t + t
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(func, function_cache=function_cache)
        t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
        defined(t)
        self.assertLen(function_cache, 1)
        t = constant_op.constant([[1.0]], dtype=dtypes.complex128)
        defined(t)
        self.assertLen(function_cache, 2)

    def testCacheTensorShapeCollision(self):
        if False:
            print('Hello World!')

        def func(t):
            if False:
                for i in range(10):
                    print('nop')
            return t + t
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(func, function_cache=function_cache)
        t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
        defined(t)
        self.assertLen(function_cache, 1)
        t = constant_op.constant([1.0], dtype=dtypes.complex64)
        defined(t)
        self.assertLen(function_cache, 2)

    def testCacheTensorShapeDtypeCollision(self):
        if False:
            print('Hello World!')

        def func(t):
            if False:
                print('Hello World!')
            return t + t
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(func, function_cache=function_cache)
        t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
        defined(t)
        self.assertLen(function_cache, 1)
        t = constant_op.constant([1.0], dtype=dtypes.complex128)
        defined(t)
        self.assertLen(function_cache, 2)

    def testCacheTensorUnknownShapesCollisionRelaxedShapes(self):
        if False:
            i = 10
            return i + 15

        def func(t):
            if False:
                for i in range(10):
                    print('nop')
            return t + t
        with context.graph_mode(), self.cached_session():
            function_cache = function_cache_lib.FunctionCache()
            defined = compiled_fn(func, reduce_retracing=True, function_cache=function_cache)
            p = array_ops.placeholder(dtype=dtypes.float32, shape=[])
            defined(p)
            self.assertLen(function_cache, 1)
            p = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
            defined(p)
            self.assertLen(function_cache, 2)
            p = array_ops.placeholder(dtype=dtypes.float32, shape=[2])
            defined(p)
            self.assertLen(function_cache, 2)
            t = constant_op.constant([1.0, 1.0, 1.0], dtype=dtypes.float32)
            defined(t)
            self.assertLen(function_cache, 2)

    def testPythonFunctionWithDefaultArgs(self):
        if False:
            print('Hello World!')

        def func(foo, bar=1, baz=2):
            if False:
                return 10
            del foo
            del bar
            del baz
            return
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(func, function_cache=function_cache)
        defined(0, baz=20)
        self.assertLen(function_cache, 1)
        defined(1)
        self.assertLen(function_cache, 2)
        defined(foo=1)
        self.assertLen(function_cache, 2)
        defined(1, 2, 3)
        self.assertLen(function_cache, 3)
        defined(1, bar=2, baz=3)
        self.assertLen(function_cache, 3)
        defined(1, baz=3, bar=2)
        self.assertLen(function_cache, 3)

    @test_util.run_v2_only
    def testFunctoolsPartialUnwrappedCorrectly(self):
        if False:
            for i in range(10):
                print('nop')

        def full_function(a, b, c=3):
            if False:
                print('Hello World!')
            return (a, b, c)
        partial = functools.partial(full_function, 1, c=4)
        (a, b, c) = partial(2)
        defined = compiled_fn(partial)
        (func_a, func_b, func_c) = defined(2)
        self.assertEqual(func_a.numpy(), a)
        self.assertEqual(func_b.numpy(), b)
        self.assertEqual(func_c.numpy(), c)

    def testInputSignatureWithMatchingInputs(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(a.shape, (2,))
            return a
        function_cache = function_cache_lib.FunctionCache()
        signature = [tensor_lib.TensorSpec(shape=(2,), dtype=dtypes.float32)]
        defined = compiled_fn(foo, input_signature=signature, function_cache=function_cache)
        a = array_ops.ones([2])
        self.assertAllEqual(a, defined(a))
        self.assertLen(function_cache, 1)
        self.assertAllEqual(a, defined.get_concrete_function()(a))
        self.assertAllEqual(a, defined.get_concrete_function(a)(a))
        self.assertAllEqual(a, defined.get_concrete_function(tensor_lib.TensorSpec((2,), dtype=dtypes.float32))(a))
        self.assertLen(function_cache, 1)

        def bar(a):
            if False:
                i = 10
                return i + 15
            self.assertEqual(a._shape_tuple(), (2, None))
            return a
        signature = [tensor_lib.TensorSpec((2, None), dtypes.float32)]
        defined = compiled_fn(bar, input_signature=signature)
        a = array_ops.ones([2, 1])
        out = defined(a)
        self.assertLen(function_cache, 1)
        self.assertAllEqual(out, a)
        b = array_ops.ones([2, 3])
        out = defined(b)
        self.assertLen(function_cache, 1)
        self.assertAllEqual(out, b)

    def testInputSignatureWithDictInPositionalArgs(self):
        if False:
            while True:
                i = 10
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def f(*_args, **_kwargs):
            if False:
                i = 10
                return i + 15
            return None
        f(1, x=2)
        self.assertLen(function_cache, 1)
        f(1, x=2)
        self.assertLen(function_cache, 1)
        f(1, {'x': 2})
        self.assertLen(function_cache, 2)

    def testInputSignatureWithCompatibleInputs(self):
        if False:
            print('Hello World!')
        rank2_spec = tensor_lib.TensorSpec(shape=(None, None), dtype=dtypes.float32)

        @compiled_fn(input_signature=[rank2_spec])
        def func(a):
            if False:
                return 10
            self.assertEqual([None, None], a.shape.as_list())
            return array_ops.shape(a)
        self.assertAllEqual([3, 1], func([[0], [1.0], [1]]))
        self.assertAllEqual([2, 2], func(numpy.array([[1, 1], [2, 2]])))
        with self.assertRaises(TypeError):
            func([0.0, 1.0, 2.0])
        with self.assertRaises(TypeError):
            func([['wrong dtype']])

    @test_util.run_v2_only
    def testNestedInputSignatures(self):
        if False:
            for i in range(10):
                print('nop')

        def expected_foo(a, b):
            if False:
                i = 10
                return i + 15
            return [a, b]
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(input_signature=[[tensor_lib.TensorSpec((2, None), dtypes.float32)] * 2, tensor_lib.TensorSpec((1,), dtypes.float32)], function_cache=function_cache)
        def foo(a, b):
            if False:
                print('Hello World!')
            self.assertEqual(a[0]._shape_tuple(), (2, None))
            self.assertEqual(a[1]._shape_tuple(), (2, None))
            self.assertEqual(b._shape_tuple(), (1,))
            return [a, b]
        a = array_ops.ones([2, 1])
        b = array_ops.ones([1])
        expected = expected_foo([a, a], b)
        out = foo([a, a], b)
        self.assertLen(function_cache, 1)
        nest.assert_same_structure(out, expected)
        self.assertAllEqual(out[0][0], a)
        self.assertAllEqual(out[0][1], a)
        self.assertAllEqual(out[1], b)
        a = array_ops.ones([2, 3])
        b = array_ops.ones([2, 5])
        c = array_ops.ones([1])
        expected = expected_foo([a, b], c)
        out = foo([a, b], c)
        self.assertLen(function_cache, 1)
        nest.assert_same_structure(out, expected)
        self.assertAllEqual(out[0][0], a)
        self.assertAllEqual(out[0][1], b)
        self.assertAllEqual(out[1], c)
        a = a.numpy().tolist()
        b = b.numpy().tolist()
        c = c.numpy().tolist()
        out = foo([a, b], c)
        self.assertLen(function_cache, 1)
        nest.assert_same_structure(out, expected)
        self.assertAllEqual(out[0][0], a)
        self.assertAllEqual(out[0][1], b)
        self.assertAllEqual(out[1], c)

    @test_util.run_v2_only
    def testNestedInputSignaturesWithDict(self):
        if False:
            for i in range(10):
                print('nop')

        def expected_bar(a):
            if False:
                return 10
            return a

        @compiled_fn(input_signature=[{'a': tensor_lib.TensorSpec((2, None), dtypes.float32), 'b': tensor_lib.TensorSpec((2, None), dtypes.float32), 'c': tensor_lib.TensorSpec((1,), dtypes.float32)}])
        def bar(a):
            if False:
                print('Hello World!')
            self.assertEqual(a['a']._shape_tuple(), (2, None))
            self.assertEqual(a['b']._shape_tuple(), (2, None))
            self.assertEqual(a['c']._shape_tuple(), (1,))
            return a
        a = array_ops.ones([2, 3])
        b = array_ops.ones([1])
        inputs = {'a': a, 'b': a, 'c': b}
        expected = expected_bar(inputs)
        out = bar(inputs)
        nest.assert_same_structure(out, expected)
        self.assertAllEqual(out['a'], expected['a'])
        self.assertAllEqual(out['b'], expected['b'])
        self.assertAllEqual(out['c'], expected['c'])
        a = a.numpy().tolist()
        b = b.numpy().tolist()
        inputs = {'a': a, 'b': a, 'c': b}
        out = bar(inputs)
        nest.assert_same_structure(out, expected)
        self.assertAllEqual(out['a'], expected['a'])
        self.assertAllEqual(out['b'], expected['b'])
        self.assertAllEqual(out['c'], expected['c'])

    def testInputSignatureMustBeSequenceOfTensorSpecs(self):
        if False:
            print('Hello World!')

        def foo(a, b):
            if False:
                i = 10
                return i + 15
            del a
            del b
        signature = {'t1': tensor_lib.TensorSpec([], dtypes.float32)}
        with self.assertRaisesRegex(TypeError, 'input_signature must be either a tuple or a list.*'):
            compiled_fn(foo, input_signature=signature)

    def testInputsIncompatibleWithNestedSignatureRaisesError(self):
        if False:
            return 10

        def foo(a, b):
            if False:
                print('Hello World!')
            return [a, b]
        signature = [[tensor_lib.TensorSpec((1,), dtypes.float32)] * 2, [tensor_lib.TensorSpec((1,), dtypes.float32)] * 2]
        defined = compiled_fn(foo, input_signature=signature)
        a = array_ops.ones([1])
        with self.assertRaises(TypeError):
            defined([a, a, a], [a])
        with self.assertRaises(TypeError):
            defined([a], [a, a, a])
        defined([a, a], [a, a])

    @test_util.run_v2_only
    def testUnderspecifiedInputSignature(self):
        if False:
            i = 10
            return i + 15

        @compiled_fn(input_signature=[tensor_lib.TensorSpec([], dtypes.float32)])
        def foo(a, training=True):
            if False:
                for i in range(10):
                    print('nop')
            if training:
                return a
            else:
                return -1.0 * a
        x = constant_op.constant(1.0)
        with self.assertRaises(ValueError):
            foo(x, training=False)
        self.assertAllEqual(x.numpy(), foo(x).numpy())

    @test_util.run_v2_only
    def testInputSignatureWithPartialFunction(self):
        if False:
            return 10

        def full_function(a, b, c=3.0):
            if False:
                print('Hello World!')
            return (a, b, c)
        partial = functools.partial(full_function, 1, c=4)
        (a, b, c) = partial(2.0)
        signature = [tensor_lib.TensorSpec([], dtypes.float32)]
        defined = compiled_fn(partial, input_signature=signature)
        x = constant_op.constant(2.0)
        (func_a, func_b, func_c) = defined(x)
        self.assertEqual(func_a.numpy(), a)
        self.assertEqual(func_b.numpy(), b)
        self.assertEqual(func_c.numpy(), c)

    @test_util.run_v2_only
    def testInputSignatureWithKeywordPositionalArgs(self):
        if False:
            print('Hello World!')
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(input_signature=[tensor_lib.TensorSpec([], dtypes.float32), tensor_lib.TensorSpec([], dtypes.int64)], function_cache=function_cache)
        def foo(flt, integer):
            if False:
                print('Hello World!')
            return (flt, integer)
        flt = constant_op.constant(1.0)
        integer = constant_op.constant(2, dtypes.int64)
        (out1, out2) = foo(flt, integer)
        self.assertLen(function_cache, 1)
        self.assertEqual(out1.numpy(), 1.0)
        self.assertEqual(out2.numpy(), 2)
        (out1, out2) = foo(flt=flt, integer=integer)
        self.assertLen(function_cache, 1)
        self.assertEqual(out1.numpy(), 1.0)
        self.assertEqual(out2.numpy(), 2)
        (out1, out2) = foo(integer=integer, flt=flt)
        self.assertLen(function_cache, 1)
        self.assertEqual(out1.numpy(), 1.0)
        self.assertEqual(out2.numpy(), 2)
        (out1, out2) = foo(flt, integer=integer)
        self.assertLen(function_cache, 1)
        self.assertEqual(out1.numpy(), 1.0)
        self.assertEqual(out2.numpy(), 2)

    @test_util.run_v2_only
    def testInputSignatureWithKeywordArgs(self):
        if False:
            while True:
                i = 10

        def foo(a, b, **kwargs):
            if False:
                print('Hello World!')
            del kwargs
            return (a, b)
        x = compiled_fn(foo, input_signature=[tensor_lib.TensorSpec([], dtypes.float32), tensor_lib.TensorSpec([], dtypes.int32)]).get_concrete_function()
        result = x(constant_op.constant(5.0), constant_op.constant(5))
        self.assertAllEqual(result, [5.0, 5])

    def testInputSignatureWithCompositeTensors(self):
        if False:
            return 10

        def f(rt):
            if False:
                while True:
                    i = 10
            self.assertEqual(rt.values.shape.as_list(), [None])
            self.assertEqual(rt.row_splits.shape.as_list(), [4])
            return rt
        signature = [ragged_tensor.RaggedTensorSpec(shape=[3, None], dtype=dtypes.int32)]
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(f, input_signature=signature, function_cache=function_cache)
        rt1 = ragged_factory_ops.constant([[1], [], [2, 3, 4]])
        out1 = defined(rt1)
        self.assertLen(function_cache, 1)
        self.assertAllEqual(out1.values, rt1.values)
        self.assertAllEqual(out1.row_splits, rt1.row_splits)
        rt2 = ragged_factory_ops.constant([[1, 2], [3, 4], [5]])
        out2 = defined(rt2)
        self.assertLen(function_cache, 1)
        self.assertAllEqual(out2.values, rt2.values)
        self.assertAllEqual(out2.row_splits, rt2.row_splits)
        rt3 = ragged_factory_ops.constant([[1, 2], [3, 4], [5], [6]])
        with self.assertRaises(TypeError):
            defined(rt3)
        rt4 = ragged_factory_ops.constant([[1.0, 2.0], [], [3.0]])
        with self.assertRaises(TypeError):
            defined(rt4)
        rt5 = ragged_factory_ops.constant([[[1]], [[2]], [[3]]])
        with self.assertRaises(ValueError):
            defined(rt5)

    @test_util.run_v2_only
    def testInputSignatureWithKeywordOnlyArgs(self):
        if False:
            return 10

        def f(a, b, c=3, *, d=4):
            if False:
                return 10
            self.assertIsInstance(a, tensor_lib.Tensor)
            self.assertIsInstance(b, tensor_lib.Tensor)
            self.assertIsInstance(c, int)
            self.assertIsInstance(d, (int, tensor_lib.Tensor))
            return a + b + c + d
        signature = [tensor_lib.TensorSpec(shape=[], dtype=dtypes.int32), tensor_lib.TensorSpec(shape=[], dtype=dtypes.int32)]
        defined = compiled_fn(f, input_signature=signature)
        self.assertEqual(defined(1, 2).numpy(), 10)
        defined = compiled_fn(functools.partial(f, c=4), input_signature=signature)
        self.assertEqual(defined(1, 2).numpy(), 11)
        defined = compiled_fn(functools.partial(f, d=5), input_signature=signature)
        self.assertEqual(defined(1, 2).numpy(), 11)
        defined = compiled_fn(functools.partial(f, d=array_ops.constant(5)), input_signature=signature)
        self.assertEqual(defined(1, 2).numpy(), 11)
        mod = module.Module()
        save(mod, '/tmp/kwonlyf', defined.get_concrete_function(*signature))
        loaded = load('/tmp/kwonlyf')
        result = loaded.signatures['serving_default'](a=array_ops.constant(1), b=array_ops.constant(2), d=array_ops.constant(5))
        self.assertEqual(result['output_0'].numpy(), 11)

    def testInputSignatureWithKeywordOnlyArgsNoDefaults(self):
        if False:
            print('Hello World!')
        signature = [tensor_lib.TensorSpec(shape=[], dtype=dtypes.int32), tensor_lib.TensorSpec(shape=[], dtype=dtypes.int32)]

        def test_func(a, *, b):
            if False:
                while True:
                    i = 10
            return a + b
        with self.assertRaisesRegex(TypeError, 'Since input_signature is defined, keyword-only parameter `b` must have a default value'):
            compiled_fn(test_func, input_signature=signature)
        test_func_lambda = lambda a, *, b: a + b
        with self.assertRaisesRegex(TypeError, 'Since input_signature is defined, keyword-only parameter `b` must have a default value'):
            compiled_fn(test_func_lambda, input_signature=signature)

    def testTensorKeywordArguments(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(a, b):
            if False:
                for i in range(10):
                    print('nop')
            del a
            return b
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(foo, function_cache=function_cache)
        a = constant_op.constant(2.0)
        b = constant_op.constant([1.0, 2.0])
        one = defined(a, b)
        self.assertLen(function_cache, 1)
        two = defined(a=a, b=b)
        self.assertLen(function_cache, 1)
        three = defined(b=b, a=a)
        self.assertLen(function_cache, 1)
        four = defined(a, b=b)
        self.assertLen(function_cache, 1)
        five = defined(b, a)
        self.assertLen(function_cache, 2)
        six = defined(a=b, b=a)
        self.assertLen(function_cache, 2)
        seven = defined(b=a, a=b)
        self.assertLen(function_cache, 2)
        self.assertAllEqual(one, [1.0, 2.0])
        self.assertAllEqual(two, [1.0, 2.0])
        self.assertAllEqual(three, [1.0, 2.0])
        self.assertAllEqual(four, [1.0, 2.0])
        self.assertAllEqual(five, 2.0)
        self.assertAllEqual(six, 2.0)
        self.assertAllEqual(seven, 2.0)

    def testFunctionWithInvalidAttribute(self):
        if False:
            i = 10
            return i + 15

        def add(x, y):
            if False:
                return 10
            return math_ops.add(x, y)
        with self.assertRaisesRegex(ValueError, 'Tracing compilation does not support `experimental_1` as an attribute.'):
            tracing_compilation.trace_function((1, 2), tracing_options=tracing_compilation.TracingOptions(add, 'add', attributes={'experimental_1': 'value1'}))

    def testRegisterFunction(self):
        if False:
            i = 10
            return i + 15

        @compiled_fn(name='add', function_cache=function_cache_lib.FunctionCache())
        def add(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.add(x, y)

        def matmul(x, y):
            if False:
                while True:
                    i = 10
            return math_ops.matmul(x, y)
        defun_matmul = compiled_fn(matmul, name='matmul', function_cache=function_cache_lib.FunctionCache())
        with context.graph_mode(), self.cached_session():
            with ops.get_default_graph().as_default():
                t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
                concrete_func_matmul = defun_matmul.get_concrete_function(t, t)
                concrete_func_matmul.add_to_graph()
                concrete_func_matmul.add_gradient_functions_to_graph()
                concrete_func_add = add.get_concrete_function(t, t)
                concrete_func_add.add_to_graph()
                concrete_func_add.add_gradient_functions_to_graph()
                graph = ops.get_default_graph()
                self.assertLen(graph._functions, 6)
                functions = list(graph._functions.values())
                captured_function_names = [f.cached_definition.signature.name for f in functions]
                expected_func_name_regex = ['.*inference.*matmul.*', '.*forward.*matmul.*', '.*inference.*backward.*matmul.*', '.*inference.*add.*', '.*forward.*add.*', '.*inference.*backward.*add.*']
                for i in range(len(functions)):
                    self.assertRegex(captured_function_names[i], expected_func_name_regex[i])
                self.assertEqual(functions[1].cached_definition.attr['backward_function_name'].s, functions[2].name)
                self.assertEqual(functions[2].cached_definition.attr['forward_function_name'].s, functions[1].name)
                self.assertEqual(functions[4].cached_definition.attr['backward_function_name'].s, functions[5].name)
                self.assertEqual(functions[5].cached_definition.attr['forward_function_name'].s, functions[4].name)
                sq = defun_matmul(t, t)
                double = add(t, t)
                self.assertAllEqual(sq.eval().reshape(-1), [7, 10, 15, 22])
                self.assertAllEqual(double.eval().reshape(-1), [2, 4, 6, 8])
                self.assertLen(graph._functions, 6)
                functions = list(graph._functions.values())
                for i in range(len(functions)):
                    self.assertEqual(captured_function_names[i], functions[i].cached_definition.signature.name)

    @test_util.run_v2_only
    def testRegisterConcreteFunction(self):
        if False:
            print('Hello World!')

        @compiled_fn(name='py_add', function_cache=function_cache_lib.FunctionCache())
        def py_add(x, y):
            if False:
                print('Hello World!')
            return math_ops.add(x, y)
        py_add(array_ops.ones([]), array_ops.ones([]))
        add = py_add.get_concrete_function(tensor_lib.TensorSpec(None, dtypes.float32), tensor_lib.TensorSpec(None, dtypes.float32))

        @compiled_fn(name='py_composite', function_cache=function_cache_lib.FunctionCache())
        def py_composite(x, y):
            if False:
                print('Hello World!')
            return (x, add(x, y))
        py_composite(array_ops.ones([]), array_ops.ones([]))
        composite = py_composite.get_concrete_function(tensor_lib.TensorSpec(None, dtypes.float32), tensor_lib.TensorSpec(None, dtypes.float32))
        with context.graph_mode(), self.cached_session():
            with ops.get_default_graph().as_default():
                t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
                composite.add_to_graph()
                composite.add_gradient_functions_to_graph()
                graph = ops.get_default_graph()
                self.assertLen(graph._functions, 6)
                functions = list(graph._functions.values())
                captured_function_names = [f.cached_definition.signature.name for f in functions]
                expected_func_name_regex = ['.*inference.*py_composite.*', '.*inference.*py_add.*', '.*forward.*py_composite.*', '.*forward.*py_add.*', '.*inference.*backward.*py_composite.*', '.*inference.*backward.*py_add.*']
                for (expected, found) in zip(expected_func_name_regex, captured_function_names):
                    self.assertRegex(found, expected)
                (composite_t, composite_double) = composite(t, t)
                double = add(t, t)
                self.assertAllEqual([[2, 4], [6, 8]], self.evaluate(double))
                self.assertAllEqual([[2, 4], [6, 8]], self.evaluate(composite_double))
                self.assertAllEqual([[1, 2], [3, 4]], self.evaluate(composite_t))
                self.assertLen(graph._functions, 6)

    def testEagerCaptures(self):
        if False:
            return 10
        with context.eager_mode():
            large_tensor = array_ops.ones(shape=(256,))
            self.assertGreater(256, capture_container._EAGER_CONST_THRESHOLD)
            small_tensor = array_ops.ones(shape=(4,))
            self.assertLessEqual(4, capture_container._EAGER_CONST_THRESHOLD)
            v = resource_variable_ops.ResourceVariable(0.0)
        for (captured, op_type) in [(large_tensor, 'Placeholder'), (small_tensor, 'Const'), (v, 'Placeholder')]:

            @compiled_fn
            def test_fn():
                if False:
                    for i in range(10):
                        print('nop')
                return captured + 1
            g = test_fn.get_concrete_function().graph
            internal_captures = g.internal_captures
            self.assertLen(internal_captures, 1)
            self.assertEqual(internal_captures[0].op.type, op_type)

    def testRegisterFunctionWithInputSignature(self):
        if False:
            print('Hello World!')

        def matmul(x, y):
            if False:
                i = 10
                return i + 15
            return math_ops.matmul(x, y)
        defun_matmul = compiled_fn(matmul, input_signature=[tensor_lib.TensorSpec(shape=(2, 2), dtype=dtypes.float32), tensor_lib.TensorSpec(shape=(2, 2), dtype=dtypes.float32)], function_cache=function_cache_lib.FunctionCache())
        with context.graph_mode(), self.cached_session():
            with ops.get_default_graph().as_default():
                t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
                concrete_func = defun_matmul.get_concrete_function(t, t)
                concrete_func.add_to_graph()
                concrete_func.add_gradient_functions_to_graph()
                graph = ops.get_default_graph()
                self.assertLen(graph._functions, 3)
                concrete_func = defun_matmul.get_concrete_function()
                concrete_func.add_to_graph()
                concrete_func.add_gradient_functions_to_graph()
                graph = ops.get_default_graph()
                self.assertLen(graph._functions, 3)

    def testRegisterFunctionWithCache(self):
        if False:
            i = 10
            return i + 15

        def matmul(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.matmul(x, y)
        defun_matmul = compiled_fn(matmul, function_cache=function_cache_lib.FunctionCache())
        with context.graph_mode(), self.cached_session():
            with ops.get_default_graph().as_default():
                t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
                t2 = constant_op.constant([[2.0, 3.0], [4.0, 5.0]])
                concrete_func_t = defun_matmul.get_concrete_function(t, t)
                concrete_func_t.add_to_graph()
                concrete_func_t.add_gradient_functions_to_graph()
                concrete_func_t2 = defun_matmul.get_concrete_function(t2, t2)
                concrete_func_t2.add_to_graph()
                concrete_func_t2.add_gradient_functions_to_graph()
                graph = ops.get_default_graph()
                self.assertLen(graph._functions, 3)

    @test_util.run_v2_only
    def testCallingFunctionWithDifferentVariables(self):
        if False:
            i = 10
            return i + 15

        @compiled_fn
        def foo(v):
            if False:
                return 10
            v.assign_add(1.0)
            return v.read_value()
        v = resource_variable_ops.ResourceVariable(0.0)
        graph_function = foo.get_concrete_function(v)
        self.assertLen(graph_function.inputs, 1)
        self.assertEmpty(graph_function.captured_inputs)
        self.assertEqual(float(graph_function(v)), 1.0)
        self.assertEqual(float(graph_function(v)), 2.0)
        w = resource_variable_ops.ResourceVariable(0.0)

        @compiled_fn
        def bar(v):
            if False:
                print('Hello World!')
            del v
            return constant_op.constant(1.0)
        graph_function = bar.get_concrete_function(v)
        self.assertEqual(float(graph_function(v)), 1.0)
        self.assertEqual(float(graph_function(w)), 1.0)

    def testCallingFunctionWithNonTensorsFails(self):
        if False:
            for i in range(10):
                print('nop')

        @compiled_fn
        def foo(x):
            if False:
                print('Hello World!')
            return x
        graph_function = foo.get_concrete_function(constant_op.constant(1.0))
        with self.assertRaises((TypeError, ValueError)):
            graph_function('Not a Tensor.')

    @parameterized.parameters([(compiled_fn(attributes={'api_implements': 'random_boost', 'api_preferred_device': 'CPU'}), compiled_fn(attributes={'api_implements': 'random_boost', 'api_preferred_device': 'GPU'})), (compiled_fn(attributes={'api_implements': 'random_boost', 'api_preferred_device': 'CPU'}), compiled_fn(attributes={'api_implements': 'random_boost', 'api_preferred_device': 'GPU'}))])
    @test_util.run_v2_only
    def testSwapImplementationWithGrapplerPlugin(self, cpu_decorator, gpu_decorator):
        if False:
            for i in range(10):
                print('nop')
        rewrites = rewriter_config_pb2.RewriterConfig()
        rewrites.implementation_selector = rewriter_config_pb2.RewriterConfig.ON
        rewrites.min_graph_nodes = -1
        graph_options = config_pb2.GraphOptions(rewrite_options=rewrites, build_cost_model=1)
        config_proto = config_pb2.ConfigProto(graph_options=graph_options)
        with context.graph_mode(), self.cached_session(config=config_proto, graph=ops.Graph(), use_gpu=True):

            @cpu_decorator
            def cpu_boost(x):
                if False:
                    return 10
                return math_ops.add(x, 2.0)

            @gpu_decorator
            def gpu_boost(x):
                if False:
                    print('Hello World!')
                return math_ops.add(x, 4.0)
            x = constant_op.constant(1.0)
            concrete_func = cpu_boost.get_concrete_function(x)
            concrete_func.add_to_graph()
            concrete_func.add_gradient_functions_to_graph()
            y = gpu_boost(x)
            y_value = self.evaluate(y)
            if test.is_gpu_available():
                self.assertEqual(y_value, 5.0)
            else:
                self.assertEqual(y_value, 3.0)

    @test_util.disable_tfrt("b/174712583: TFRT doesn't support behavior equivalent to implementation_selector for function")
    def testSwapImplementationInEager(self):
        if False:
            i = 10
            return i + 15
        if not context.executing_eagerly():
            self.skipTest('eager only')
        context.context().set_optimizer_experimental_options({'min_graph_nodes': -1, 'implementation_selector': True, 'disable_meta_optimizer': False})

        @compiled_fn(attributes={'api_implements': 'foo', 'api_preferred_device': 'CPU'})
        def on_cpu(x):
            if False:
                i = 10
                return i + 15
            return x + 2

        @compiled_fn(attributes={'api_implements': 'foo', 'api_preferred_device': 'GPU'})
        def on_gpu(x):
            if False:
                print('Hello World!')
            return x + 4

        @compiled_fn
        def run_on_cpu(t):
            if False:
                while True:
                    i = 10
            concrete_func = on_cpu.get_concrete_function(t)
            concrete_func.add_to_graph()
            concrete_func.add_gradient_functions_to_graph()
            with ops.device('CPU:0'):
                return on_gpu(t)
        self.assertEqual(run_on_cpu(constant_op.constant(1)).numpy(), 3)

    def testCompilationFunctionSeparateGraphs(self):
        if False:
            print('Hello World!')
        with context.graph_mode():
            add_cache = function_cache_lib.FunctionCache()

            @compiled_fn(function_cache=add_cache)
            def add(x):
                if False:
                    while True:
                        i = 10
                return x + 5
            maybe_add_cache = function_cache_lib.FunctionCache()

            @compiled_fn(function_cache=maybe_add_cache)
            def maybe_add(x, should_add):
                if False:
                    for i in range(10):
                        print('nop')
                if should_add:
                    return add(x)
                else:
                    return x
            with ops.Graph().as_default():
                x = constant_op.constant(11)
                maybe_add(x, True)
                self.assertLen(maybe_add_cache, 1)
                self.assertLen(add_cache, 1)
                maybe_add(x, False)
                self.assertLen(maybe_add_cache, 2)
                self.assertLen(add_cache, 1)
            with ops.Graph().as_default():
                x = constant_op.constant(11)
                maybe_add(x, True)
                self.assertLen(maybe_add_cache, 3)
                self.assertLen(add_cache, 2)

    def testCacheKeyOverlappingShapes(self):
        if False:
            while True:
                i = 10
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def defined(t):
            if False:
                print('Hello World!')
            return t
        defined(array_ops.zeros([12, 1]))
        self.assertLen(function_cache, 1)
        defined(array_ops.zeros([1, 21]))
        self.assertLen(function_cache, 2)
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def defined_again(t):
            if False:
                while True:
                    i = 10
            return defined(t)
        defined_again.get_concrete_function(array_ops.zeros([12, 1]))
        self.assertLen(function_cache, 1)
        defined_again.get_concrete_function(array_ops.zeros([1, 21]))
        self.assertLen(function_cache, 2)

    def testCacheTensorSpecIdenticalToTensor(self):
        if False:
            return 10

        @compiled_fn(function_cache=function_cache_lib.FunctionCache())
        def defined(t):
            if False:
                print('Hello World!')
            return t
        z = array_ops.zeros([2, 2])
        z_spec = tensor_lib.TensorSpec.from_tensor(z)
        self.assertIs(defined.get_concrete_function(z_spec), defined.get_concrete_function(z))

    def testCacheKeyNestedLists(self):
        if False:
            return 10
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def defined(l):
            if False:
                i = 10
                return i + 15
            return l
        a = constant_op.constant(1.0)
        b = constant_op.constant(2.0)
        c = constant_op.constant(3.0)
        defined([[a], b, c])
        self.assertLen(function_cache, 1)
        defined([[a, b], c])
        self.assertLen(function_cache, 2)

    def testCacheKeyAttrsClass(self):
        if False:
            for i in range(10):
                print('nop')
        if attr is None:
            self.skipTest('attr module is unavailable.')

        @attr.s
        class TestClass:
            a = attr.ib()
            b = attr.ib()
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def defined(l):
            if False:
                i = 10
                return i + 15
            return l
        defined(TestClass(constant_op.constant(1.0), [constant_op.constant(2.0), constant_op.constant(3.0)]))
        self.assertLen(function_cache, 1)
        defined(TestClass(constant_op.constant(1.0), [constant_op.constant(2.0), constant_op.constant(3.0)]))
        self.assertLen(function_cache, 1)
        defined(TestClass([constant_op.constant(1.0), constant_op.constant(2.0)], constant_op.constant(3.0)))
        self.assertLen(function_cache, 2)

    def testDistinctVariablesNoRetracing(self):
        if False:
            print('Hello World!')
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def defined(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return a + b + c
        x = resource_variable_ops.ResourceVariable(0.0)
        y = resource_variable_ops.ResourceVariable(0.0)
        z = resource_variable_ops.ResourceVariable(0.0)
        defined(x, y, z)
        self.assertLen(function_cache, 1)
        defined(z, y, x)
        self.assertLen(function_cache, 1)

    def testRetracingOnDifferentVaribleCombinationPatterns(self):
        if False:
            for i in range(10):
                print('nop')
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def defined(a, b, c):
            if False:
                while True:
                    i = 10
            return a + b + c
        x = resource_variable_ops.ResourceVariable(0.0)
        y = resource_variable_ops.ResourceVariable(0.0)
        z = resource_variable_ops.ResourceVariable(0.0)
        defined(x, y, z)
        self.assertLen(function_cache, 1)
        defined(x, x, z)
        self.assertLen(function_cache, 2)
        defined(y, y, z)
        self.assertLen(function_cache, 2)
        defined(z, y, y)
        self.assertLen(function_cache, 3)
        defined(z, y, y)
        self.assertLen(function_cache, 3)

    @test_util.run_v2_only
    def testDeepcopyVariableNoRetracing(self):
        if False:
            i = 10
            return i + 15
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def defined(a, b, c):
            if False:
                return 10
            return a + b + c
        x = resource_variable_ops.ResourceVariable(0.0)
        y = resource_variable_ops.ResourceVariable(0.0)
        z = resource_variable_ops.ResourceVariable(0.0)
        defined(x, y, z)
        self.assertLen(function_cache, 1)
        x_copy = copy.deepcopy(x)
        defined(x_copy, y, z)
        self.assertLen(function_cache, 1)

    @test_util.disable_tfrt('b/173429686')
    @test_util.run_v2_only
    def testExecutorType(self):
        if False:
            for i in range(10):
                print('nop')

        @compiled_fn
        def add_five(x):
            if False:
                while True:
                    i = 10
            return x + 5
        self.assertEqual(5, add_five(constant_op.constant(0, dtype=dtypes.int32)).numpy())
        with self.assertRaisesRegex(errors.NotFoundError, 'NON_EXISTENT_EXECUTOR'):
            with context.function_executor_type('NON_EXISTENT_EXECUTOR'):
                add_five(constant_op.constant(0, dtype=dtypes.int32))
        for executor_type in ('', 'DEFAULT', None):
            with context.function_executor_type(executor_type):
                self.assertAllEqual(5, add_five(constant_op.constant(0, dtype=dtypes.int32)).numpy())

    @test_util.assert_no_garbage_created
    def testReferenceCycles(self):
        if False:
            for i in range(10):
                print('nop')
        fn = compiled_fn(lambda x: 2.0 * x)
        fn(constant_op.constant(4.0))
        weak_fn = weakref.ref(fn)
        del fn
        self.assertIs(None, weak_fn())

    @test_util.run_in_graph_and_eager_modes
    def testShapeCaching(self):
        if False:
            for i in range(10):
                print('nop')

        @compiled_fn
        def func(x):
            if False:
                while True:
                    i = 10
            return array_ops.shape(x)

        @compiled_fn(input_signature=[tensor_lib.TensorSpec([None, None], dtypes.float32)])
        def calls_func(x):
            if False:
                print('Hello World!')
            return func(x)
        self.assertAllEqual([1, 1], self.evaluate(func(array_ops.zeros([1, 1]))))
        self.assertAllEqual([2, 2], self.evaluate(func(array_ops.zeros([2, 2]))))
        self.assertAllEqual([3, 3], self.evaluate(calls_func(array_ops.zeros([3, 3]))))

    def testLimitedRetracing(self):
        if False:
            print('Hello World!')
        trace_count = [0]
        function_cache = function_cache_lib.FunctionCache()

        @compiled_fn(function_cache=function_cache)
        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            trace_count[0] += 1
            return x
        for _ in range(50):
            func(constant_op.constant(3.0))
            func(constant_op.constant(4.0))
            func(constant_op.constant([[1.0, 2.0]]))
            func(constant_op.constant([[]]))
            func(constant_op.constant([[3.0, 4.0], [5.0, 6.0]]))
            func(constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
        self.assertLess(trace_count[0], 13)

class CompilationCollectionTest(test.TestCase):

    def testCollectionValueAccess(self):
        if False:
            return 10
        'Read values from graph collections inside of defun.'
        with ops.Graph().as_default() as g:
            with self.session(graph=g):
                x = 2
                y = 5
                ops.add_to_collection('x', x)
                ops.add_to_collection('y', y)

                @compiled_fn
                def fn():
                    if False:
                        return 10
                    x_const = constant_op.constant(ops.get_collection('x')[0])
                    y_const = constant_op.constant(ops.get_collection('y')[0])
                    z = math_ops.add(x_const, y_const)
                    ops.add_to_collection('z', 7)
                    return z
                self.assertEqual(7, int(self.evaluate(fn())))
                self.assertEqual(ops.get_collection('x'), [2])
                self.assertEqual(ops.get_collection('y'), [5])
                self.assertEqual(ops.get_collection('z'), [])

    def testCollectionVariableValueAccess(self):
        if False:
            for i in range(10):
                print('nop')
        'Read variable value from graph collections inside of defun.'
        with ops.Graph().as_default() as g:
            with self.session(graph=g):
                v = resource_variable_ops.ResourceVariable(1.0)

                @compiled_fn
                def f():
                    if False:
                        print('Hello World!')
                    return v.read_value()
                self.evaluate(variables.global_variables_initializer())
                self.assertEqual(1.0, float(self.evaluate(f())))
                self.assertLen(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), 1)

class MultiDeviceCompilationTest(test.TestCase, parameterized.TestCase):

    @test_util.run_gpu_only
    def testMultiDeviceOutput(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that functions can produce outputs on multiple devices.'

        @compiled_fn
        def func(a, b, transpose_a):
            if False:
                for i in range(10):
                    print('nop')
            with ops.device('/device:CPU:0'):
                m1 = math_ops.matmul(a, b, transpose_a=transpose_a)
            with ops.device('/device:GPU:0'):
                m2 = math_ops.matmul(a, b, transpose_a=transpose_a)
            return (m1, m2)
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        (m1, m2) = func(t, t, transpose_a=True)
        self.assertAllEqual(m1.numpy(), [[10, 14], [14, 20]])
        self.assertRegex(m1.backing_device, 'CPU')
        self.assertAllEqual(m2.numpy(), [[10, 14], [14, 20]])
        self.assertRegex(m2.backing_device, 'GPU')

    @test_util.run_gpu_only
    def testEmptyBody(self):
        if False:
            i = 10
            return i + 15

        @compiled_fn
        def func(a, b):
            if False:
                while True:
                    i = 10
            return (b, a)
        with ops.device('/device:CPU:0'):
            a = array_ops.identity(3.0)
        with ops.device('/device:GPU:0'):
            b = array_ops.identity(5.0)
        (m1, m2) = func(a, b)
        self.assertAllEqual(m1.numpy(), 5.0)
        self.assertRegex(m1.backing_device, 'GPU')
        self.assertAllEqual(m2.numpy(), 3.0)
        self.assertRegex(m2.backing_device, 'CPU')

    @test_util.run_gpu_only
    def testMultiDeviceInt32(self):
        if False:
            i = 10
            return i + 15
        "Tests that multi-device functions can take and output INT32s.\n\n    When an INT32 device tensor is fed into a function, it is copied to CPU\n    by the eager runtime. The function sees all INT32 inputs on CPU.\n\n    We set allocator attribute 'on_host' for INT32 outputs. They can be\n    partitioned into the GPU component function, but will be allocated on\n    CPU nevertheless.\n\n    There is experimental support for `ints_on_device` in\n    FunctionLibraryRuntime now. We can try that.\n    "
        with ops.device('/device:CPU:0'):
            int_cpu = constant_op.constant(3, dtype=dtypes.int32)
            resource = resource_variable_ops.ResourceVariable(5, dtype=dtypes.int32)
        with ops.device('/device:GPU:0'):
            int_gpu = constant_op.constant(7, dtype=dtypes.int32)

        @compiled_fn
        def func(int_cpu, resource, int_gpu):
            if False:
                print('Hello World!')
            with ops.device('/device:CPU:0'):
                m1 = int_cpu * resource + int_gpu
            with ops.device('/device:GPU:0'):
                m2 = int_gpu * resource + int_cpu + 1
            return (m1, m2)
        (m1, m2) = func(int_cpu, resource, int_gpu)
        self.assertAllEqual(m1.numpy(), 22)
        self.assertRegex(m1.backing_device, 'CPU')
        self.assertAllEqual(m2.numpy(), 39)
        self.assertRegex(m2.backing_device, 'CPU')
        (m1, m2) = func(int_gpu, resource, int_cpu)
        self.assertAllEqual(m1.numpy(), 38)
        self.assertRegex(m1.backing_device, 'CPU')
        self.assertAllEqual(m2.numpy(), 23)
        self.assertRegex(m2.backing_device, 'CPU')

    @test_util.run_gpu_only
    def testMultiDeviceColocateWith(self):
        if False:
            return 10
        "Tests that function's outputs respect colocation constraints."

        @compiled_fn
        def func(a, b):
            if False:
                while True:
                    i = 10
            with ops.colocate_with(a):
                ra = 2 * a
            with ops.colocate_with(b):
                rb = 3 * b
            return (ra, rb)
        devices = ['/device:CPU:0', '/device:GPU:0']
        for (dev1, dev2) in itertools.product(devices, devices):
            with ops.device(dev1):
                a = array_ops.identity(1.0)
            with ops.device(dev2):
                b = array_ops.identity(10.0)
            (ra, rb) = func(a, b)
            self.assertEqual(ra.numpy(), 2.0)
            self.assertRegex(ra.backing_device, dev1)
            self.assertEqual(rb.numpy(), 30.0)
            self.assertRegex(rb.backing_device, dev2)

    @test_util.run_gpu_only
    def testMultiDeviceResources(self):
        if False:
            print('Hello World!')
        with ops.device('/device:CPU:0'):
            c1 = resource_variable_ops.ResourceVariable(2.0)
            c2 = resource_variable_ops.ResourceVariable(7.0)
        with ops.device('/device:GPU:0'):
            g1 = resource_variable_ops.ResourceVariable(3.0)
            g2 = resource_variable_ops.ResourceVariable(5.0)

        @compiled_fn
        def func(resource1, resource2):
            if False:
                return 10
            with ops.device('/device:CPU:0'):
                result1 = resource1 * g2
            with ops.device('/device:GPU:0'):
                result2 = resource2 * c2
            return (result1, result2)
        (r1, r2) = func(c1, g1)
        self.assertEqual(r1.numpy(), 10.0)
        self.assertRegex(r1.backing_device, 'CPU')
        self.assertEqual(r2.numpy(), 21.0)
        self.assertRegex(r2.backing_device, 'GPU')
        (r1, r2) = func(g1, c1)
        self.assertEqual(r1.numpy(), 15.0)
        self.assertRegex(r1.backing_device, 'CPU')
        self.assertEqual(r2.numpy(), 14.0)
        self.assertRegex(r2.backing_device, 'GPU')

    @test_util.run_gpu_only
    def testOutputResources(self):
        if False:
            print('Hello World!')
        with ops.device('/device:CPU:0'):
            c1 = resource_variable_ops.ResourceVariable(2.0)
        with ops.device('/device:GPU:0'):
            g1 = resource_variable_ops.ResourceVariable(3.0)

        @compiled_fn
        def func(resource1, resource2):
            if False:
                while True:
                    i = 10
            with ops.device('/device:CPU:0'):
                result1 = resource1 * 5
            with ops.device('/device:GPU:0'):
                result2 = resource2 * 7
            return (result1, resource1.handle, result2, resource2.handle)
        (r1, res1, r2, res2) = func(c1, g1)
        self.assertEqual(r1.numpy(), 10.0)
        self.assertRegex(r1.backing_device, 'CPU')
        self.assertEqual(r2.numpy(), 21.0)
        self.assertRegex(r2.backing_device, 'GPU')

        def check_handle(handle, expected_value):
            if False:
                while True:
                    i = 10
            self.assertRegex(handle.backing_device, 'CPU')
            tensor = gen_resource_variable_ops.read_variable_op(handle, dtypes.float32)
            self.assertEqual(tensor.numpy(), expected_value)
        check_handle(res1, 2.0)
        check_handle(res2, 3.0)
        (r1, res1, r2, res2) = func(g1, c1)
        self.assertEqual(r1.numpy(), 15.0)
        self.assertRegex(r1.backing_device, 'CPU')
        self.assertEqual(r2.numpy(), 14.0)
        self.assertRegex(r2.backing_device, 'GPU')
        check_handle(res1, 3.0)
        check_handle(res2, 2.0)

    @test_util.run_gpu_only
    def testPassResourceThroughNestedFunctionCall(self):
        if False:
            while True:
                i = 10
        'Test passing GPU resource to noinline function call placed on CPU.\n\n    PartitionedCallOp must not enforce any particular device assignment for the\n    resource output. Inner function marked as `_nospecialize`, so Grappler would\n    not prune unused function output.\n    '
        with ops.device('/device:GPU:0'):
            g1 = resource_variable_ops.ResourceVariable(3.0)

        @compiled_fn(attributes={'_noinline': True, '_nospecialize': True})
        def inner(resource1):
            if False:
                while True:
                    i = 10
            return (resource1 * 2, resource1.handle)

        @compiled_fn
        def outer(resource1):
            if False:
                return 10
            with ops.device('/device:CPU:0'):
                (r1, _) = inner(resource1)
            return r1
        r1 = outer(g1)
        self.assertEqual(r1.numpy(), 6.0)
        self.assertRegex(r1.backing_device, 'CPU')

    @test_util.run_gpu_only
    def testReturnResourceFromNestedFunctionCall(self):
        if False:
            return 10
        'Test returning GPU resource from noinline function call placed on CPU.\n\n    When inferring output devices for the return value, do not set a device for\n    returns of DT_RESOURCE data type based on the device assignment of the node\n    that produced that resource. As an example function call placed on CPU can\n    return resources on GPU.\n    '
        with ops.device('/device:GPU:0'):
            g1 = resource_variable_ops.ResourceVariable(3.0)

        @compiled_fn(attributes={'_noinline': True})
        def inner(resource1):
            if False:
                i = 10
                return i + 15
            resource1.assign_add(2.0)
            return (resource1 * 2, resource1.handle)

        @compiled_fn
        def outer(resource1):
            if False:
                print('Hello World!')
            with ops.device('/device:CPU:0'):
                (r1, res1) = inner(resource1)
            return (r1, res1)
        (r1, res1) = outer(g1)
        self.assertEqual(r1.numpy(), 10.0)
        self.assertRegex(r1.backing_device, 'CPU')

        def check_handle(handle, expected_value):
            if False:
                i = 10
                return i + 15
            self.assertRegex(handle.backing_device, 'CPU')
            tensor = gen_resource_variable_ops.read_variable_op(handle, dtypes.float32)
            self.assertEqual(tensor.numpy(), expected_value)
        check_handle(res1, 5.0)

    @test_util.run_gpu_only
    def testComplexInputOutputDevicePattern(self):
        if False:
            return 10
        'Tests input/output mapping logic in partitioning.'
        with ops.device('/device:CPU:0'):
            rc0 = resource_variable_ops.ResourceVariable(2.0)
            rc1 = resource_variable_ops.ResourceVariable(3.0)
            cc0 = array_ops.identity(5.0)
            cc1 = array_ops.identity(7.0)
        with ops.device('/device:GPU:0'):
            rg0 = resource_variable_ops.ResourceVariable(11.0)
            rg1 = resource_variable_ops.ResourceVariable(13.0)
            cg0 = array_ops.identity(17.0)
            cg1 = array_ops.identity(19.0)
        for tensor in [cc0, cc1]:
            self.assertRegex(tensor.backing_device, 'CPU:0')
        for tensor in [cg0, cg1]:
            self.assertRegex(tensor.backing_device, 'GPU:0')

        @compiled_fn
        def func(rc0, cc0, cg0, rc1, cg1, rg0, rg1, cc1):
            if False:
                return 10
            with ops.device('/device:CPU:0'):
                m1 = rc0 * cg0
            with ops.device('/device:GPU:0'):
                m2 = rg0 * cc0
            with ops.device('/device:CPU:0'):
                r1 = 1000.0 * m2 + rc1 * cg1
            with ops.device('/device:GPU:0'):
                r2 = 1000.0 * m1 + rg1 * cc1
            return (r1, r2, m2, m1)
        (r1, r2, m2, m1) = func(rc0, cc0, cg0, rc1, cg1, rg0, rg1, cc1)
        self.assertRegex(m1.backing_device, 'CPU')
        self.assertRegex(r1.backing_device, 'CPU')
        self.assertRegex(m2.backing_device, 'GPU')
        self.assertRegex(r2.backing_device, 'GPU')
        self.assertEqual(m1.numpy(), 34.0)
        self.assertEqual(r1.numpy(), 55000.0 + 3.0 * 19.0)
        self.assertEqual(m2.numpy(), 55.0)
        self.assertEqual(r2.numpy(), 34000.0 + 13.0 * 7.0)

    @test_util.run_gpu_only
    def testArgumentPruning(self):
        if False:
            while True:
                i = 10
        'Tests functions taking unnecessary arguments.'
        with ops.device('/device:CPU:0'):
            c1 = constant_op.constant(5.0)
            c2 = constant_op.constant(7.0)
        with ops.device('/device:GPU:0'):
            g1 = constant_op.constant(11.0)
            g2 = constant_op.constant(13.0)
            g3 = constant_op.constant(17.0)

        @compiled_fn
        def func(g1, g2, c1, g3, c2):
            if False:
                print('Hello World!')
            return c1 * g3 * c2
        result = func(g1, g2, c1, g3, c2)
        self.assertEqual(result.numpy(), 5.0 * 7.0 * 17.0)

class CompilationArgumentNamingTest(test.TestCase, parameterized.TestCase):
    """Tests for recognizable export signatures from concrete functions."""

    @test_util.run_v2_only
    def testBasic(self):
        if False:
            i = 10
            return i + 15

        @compiled_fn
        def fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (a + b, a * b)
        fn(array_ops.ones([]), array_ops.ones([]))
        fn_op = fn.get_concrete_function(tensor_lib.TensorSpec(shape=(None,), dtype=dtypes.float32), tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['a', 'b'], [inp.op.name for inp in fn_op.inputs])
        self.assertEqual([b'a', b'b'], [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
        self.assertLen(fn_op.graph.structured_outputs, 2)
        self.assertAllClose([3.0, 2.0], fn_op(constant_op.constant(1.0), constant_op.constant(2.0)))
        self.assertAllClose([3.0, 2.0], fn_op(a=constant_op.constant(1.0), b=constant_op.constant(2.0)))

    def testVariable(self):
        if False:
            i = 10
            return i + 15

        @compiled_fn
        def fn(a, b):
            if False:
                while True:
                    i = 10
            return (a + b, a * b)
        fn(array_ops.ones([]), array_ops.ones([]))
        fn_op = fn.get_concrete_function(tensor_lib.TensorSpec(shape=(None,), dtype=dtypes.float32), variables.Variable(1.0))
        self.assertEqual(['a', 'b'], [inp.op.name for inp in fn_op.inputs])
        self.assertEqual([b'a', b'b'], [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
        self.assertLen(fn_op.graph.structured_outputs, 2)

    def testDictReturned(self):
        if False:
            return 10

        @compiled_fn
        def fn(x, z=(1.0, 2.0), y=3.0):
            if False:
                print('Hello World!')
            (z1, z2) = z
            return {'alpha': x + y + z1, 'beta': x * y + z2}
        fn(array_ops.ones([]))
        fn_op = fn.get_concrete_function(x=tensor_lib.TensorSpec(shape=(None,), dtype=dtypes.float32), y=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['x', 'y'], [inp.op.name for inp in fn_op.inputs])
        self.assertEqual([b'x', b'y'], [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
        self.assertEqual({'alpha', 'beta'}, set(fn_op.graph.structured_outputs.keys()))
        fn_op2 = fn.get_concrete_function(z=(tensor_lib.TensorSpec(shape=(None,), dtype=dtypes.float32, name='z_first'), tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='z_second')), y=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'), x=4.0)
        self.assertEqual(['z_first', 'z_second', 'custom'], [inp.op.name for inp in fn_op2.inputs])
        self.assertEqual([b'z_first', b'z_second', b'custom'], [inp.op.get_attr('_user_specified_name') for inp in fn_op2.inputs])
        fn_op3 = fn.get_concrete_function(tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'), z=(tensor_lib.TensorSpec(shape=(None,), dtype=dtypes.float32, name='z1'), tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='z2')), y=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['custom', 'z1', 'z2', 'y'], [inp.op.name for inp in fn_op3.inputs])
        self.assertEqual([b'custom', b'z1', b'z2', b'y'], [inp.op.get_attr('_user_specified_name') for inp in fn_op3.inputs])

    def testMethod(self):
        if False:
            while True:
                i = 10

        class HasMethod(object):

            def method(self, x):
                if False:
                    i = 10
                    return i + 15
                return x
        has_method = HasMethod()
        compiled_method = compiled_fn(has_method.method)
        class_op = compiled_method.get_concrete_function(tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['x'], [inp.op.name for inp in class_op.inputs])
        self.assertEqual([b'x'], [inp.op.get_attr('_user_specified_name') for inp in class_op.inputs])
        method_op = compiled_method.get_concrete_function(tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['x'], [inp.op.name for inp in method_op.inputs])
        self.assertEqual([b'x'], [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
        self.skipTest('Not working')
        method_op = has_method.method.get_concrete_function(tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='y'))
        self.assertEqual(['y'], [inp.op.name for inp in method_op.inputs])
        self.assertEqual([b'y'], [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])

    def testMethodSignature(self):
        if False:
            print('Hello World!')

        class HasMethod(object):

            def method(self, x):
                if False:
                    return 10
                hash(self)
                return x
        has_method = HasMethod()
        compiled_method = compiled_fn(has_method.method, input_signature=(tensor_lib.TensorSpec(shape=None, dtype=dtypes.float64, name='y'),))
        method_op = compiled_method.get_concrete_function()
        self.assertEqual(['y'], [inp.op.name for inp in method_op.inputs])
        self.assertEqual([b'y'], [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
        method_op2 = compiled_method.get_concrete_function()
        self.assertEqual(['y'], [inp.op.name for inp in method_op2.inputs])
        self.assertEqual([b'y'], [inp.op.get_attr('_user_specified_name') for inp in method_op2.inputs])

    def testVariadic(self):
        if False:
            while True:
                i = 10

        @compiled_fn
        def variadic_fn(x, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return x + math_ops.add_n(list(args) + list(kwargs.values()))
        variadic_fn(array_ops.ones([]), array_ops.ones([]))
        variadic_op = variadic_fn.get_concrete_function(tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32), tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32, name='y'), tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32), tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='second_variadic'), z=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32), zz=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='cust'))
        self.assertEqual(['x', 'y', 'args_1', 'second_variadic', 'z', 'cust'], [inp.op.name for inp in variadic_op.inputs])
        self.assertEqual([b'x', b'y', b'args_1', b'second_variadic', b'z', b'cust'], [inp.op.get_attr('_user_specified_name') for inp in variadic_op.inputs])

    def testVariadicInputSignature(self):
        if False:
            print('Hello World!')

        @compiled_fn(input_signature=(tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32), tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32, name='y'), tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32), tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='z')), name='variadic_fn')
        def variadic_fn(x, *args):
            if False:
                return 10
            return x + math_ops.add_n(list(args))
        variadic_fn(array_ops.ones([]), array_ops.ones([]), array_ops.ones([]), array_ops.ones([]))
        variadic_op = variadic_fn.get_concrete_function()
        self.assertIn(b'variadic_fn', variadic_op.name)
        self.assertEqual(['x', 'y', 'args_1', 'z'], [inp.op.name for inp in variadic_op.inputs])
        self.assertEqual([b'x', b'y', b'args_1', b'z'], [inp.op.get_attr('_user_specified_name') for inp in variadic_op.inputs])

class DevicePlacementTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testMultipleDeviceCheck(self):
        if False:
            return 10

        def f():
            if False:
                i = 10
                return i + 15
            with ops.device('cpu'):
                return test_ops.device_placement_op()
        func = compiled_fn(f)
        with ops.device('cpu:0'):
            output = self.evaluate(func())
            self.assertIn(compat.as_bytes('CPU:0'), output)

    @test_util.run_in_graph_and_eager_modes
    def testDeviceAnnotationsRespected(self):
        if False:
            print('Hello World!')

        def multi_device_fn():
            if False:
                while True:
                    i = 10
            with ops.device('/cpu:0'):
                s0 = test_ops.device_placement_op()
            with ops.device('/cpu:1'):
                s1 = test_ops.device_placement_op()
            with ops.device('/cpu:2'):
                s2 = test_ops.device_placement_op()
            s3 = test_ops.device_placement_op()
            return (s0, s1, s2, s3)
        function_cache = function_cache_lib.FunctionCache()
        defined = compiled_fn(multi_device_fn, function_cache=function_cache)
        outputs = self.evaluate(defined())
        self.assertLen(function_cache, 1)
        self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
        self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
        self.assertIn(compat.as_bytes('CPU:2'), outputs[2])
        with ops.device('/cpu:3'):
            outputs = self.evaluate(defined())
        self.assertLen(function_cache, 1)
        self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
        self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
        self.assertIn(compat.as_bytes('CPU:2'), outputs[2])
        self.assertIn(compat.as_bytes('CPU:3'), outputs[3])
        with ops.device('/cpu:0'):
            outputs = self.evaluate(defined())
        self.assertLen(function_cache, 1)
        self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
        self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
        self.assertIn(compat.as_bytes('CPU:2'), outputs[2])
        self.assertIn(compat.as_bytes('CPU:0'), outputs[3])

def setUpModule():
    if False:
        while True:
            i = 10
    ops.enable_eager_execution()
    cpus = config.list_physical_devices('CPU')
    config.set_logical_device_configuration(cpus[0], [context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration()])
if __name__ == '__main__':
    test.main()