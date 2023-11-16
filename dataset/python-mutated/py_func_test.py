"""Tests for py_func op."""
import gc
import queue
import re
import numpy as np
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import batch_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test

def np_func(x, y):
    if False:
        print('Hello World!')
    return np.sinh(x) + np.cosh(y)

def matmul(x, y):
    if False:
        for i in range(10):
            print('nop')
    return math_ops.matmul(x, y)

class PyFuncTestBase(test.TestCase):

    def verifyExceptionHandling(self, py_exp, tf_exp, eager=False):
        if False:
            return 10

        def inner_exception():
            if False:
                while True:
                    i = 10
            raise py_exp('blah')

        def raise_exception():
            if False:
                return 10
            inner_exception()
        expected_regexp = ': blah.*'
        expected_regexp += 'in raise_exception.*'
        expected_regexp += 'in inner_exception.*'
        expected_regexp += ': blah'

        def expected_error_check(exception):
            if False:
                i = 10
                return i + 15
            return re.search(expected_regexp, str(exception), re.DOTALL)
        if eager:
            if context.executing_eagerly():
                with self.assertRaisesWithPredicateMatch(tf_exp, expected_error_check):
                    f = script_ops.eager_py_func(raise_exception, [], [])
                return
            else:
                f = script_ops.eager_py_func(raise_exception, [], [])
        else:
            f = script_ops.py_func(raise_exception, [], [])
        with self.assertRaisesWithPredicateMatch(tf_exp, expected_error_check):
            self.evaluate(f)

class PyFuncTest(PyFuncTestBase):
    """Encapsulates tests for py_func only."""

    def testRealDataTypes(self):
        if False:
            print('Hello World!')

        def sum_func(x, y):
            if False:
                i = 10
                return i + 15
            return x + y
        for dtype in [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.uint8, dtypes.int8, dtypes.uint16, dtypes.int16, dtypes.int32, dtypes.int64]:
            with self.cached_session():
                x = constant_op.constant(1, dtype=dtype)
                y = constant_op.constant(2, dtype=dtype)
                z = self.evaluate(script_ops.py_func(sum_func, [x, y], dtype))
                self.assertEqual(z, 3)

    def testComplexDataTypes(self):
        if False:
            i = 10
            return i + 15

        def sub_func(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x - y
        for dtype in [dtypes.complex64, dtypes.complex128]:
            with self.cached_session():
                x = constant_op.constant(1 + 1j, dtype=dtype)
                y = constant_op.constant(2 - 2j, dtype=dtype)
                z = self.evaluate(script_ops.py_func(sub_func, [x, y], dtype))
                self.assertEqual(z, -1 + 3j)

    def testBoolDataTypes(self):
        if False:
            for i in range(10):
                print('nop')

        def and_func(x, y):
            if False:
                print('Hello World!')
            return x and y
        dtype = dtypes.bool
        with self.cached_session():
            x = constant_op.constant(True, dtype=dtype)
            y = constant_op.constant(False, dtype=dtype)
            z = self.evaluate(script_ops.py_func(and_func, [x, y], dtype))
            self.assertEqual(z, False)

    def testSingleType(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            x = constant_op.constant(1.0, dtypes.float32)
            y = constant_op.constant(2.0, dtypes.float32)
            z = self.evaluate(script_ops.py_func(np_func, [x, y], dtypes.float32))
            self.assertEqual(z, np_func(1.0, 2.0).astype(np.float32))

    def testScalar(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            x = constant_op.constant(1.0, dtypes.float32)
            y = constant_op.constant(2.0, dtypes.float32)
            z = self.evaluate(script_ops.eager_py_func(np_func, [x, y], [dtypes.float32]))
            self.assertEqual(z[0], np_func(1.0, 2.0).astype(np.float32))

    @test_util.run_v1_only('b/120545219')
    def testArray(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = constant_op.constant([1.0, 2.0], dtypes.float64)
            y = constant_op.constant([2.0, 3.0], dtypes.float64)
            z = self.evaluate(script_ops.py_func(np_func, [x, y], [dtypes.float64]))
            self.assertAllEqual(z[0], np_func([1.0, 2.0], [2.0, 3.0]).astype(np.float64))

    def testComplexType(self):
        if False:
            return 10
        with self.cached_session():
            x = constant_op.constant(1 + 2j, dtypes.complex64)
            y = constant_op.constant(3 + 4j, dtypes.complex64)
            z = self.evaluate(script_ops.py_func(np_func, [x, y], dtypes.complex64))
            self.assertAllClose(z, np_func(1 + 2j, 3 + 4j))

    def testRFFT(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = constant_op.constant([1.0, 2.0, 3.0, 4.0], dtypes.float32)

            def rfft(x):
                if False:
                    while True:
                        i = 10
                return np.fft.rfft(x).astype(np.complex64)
            y = self.evaluate(script_ops.py_func(rfft, [x], dtypes.complex64))
            self.assertAllClose(y, np.fft.rfft([1.0, 2.0, 3.0, 4.0]))

    def testPythonLiteral(self):
        if False:
            while True:
                i = 10
        with self.cached_session():

            def literal(x):
                if False:
                    i = 10
                    return i + 15
                return 1.0 if float(x) == 0.0 else 0.0
            x = constant_op.constant(0.0, dtypes.float64)
            y = self.evaluate(script_ops.py_func(literal, [x], dtypes.float64))
            self.assertAllClose(y, 1.0)

    def testList(self):
        if False:
            while True:
                i = 10
        with self.cached_session():

            def list_func(x):
                if False:
                    i = 10
                    return i + 15
                return [x, x + 1]
            x = constant_op.constant(0.0, dtypes.float64)
            y = self.evaluate(script_ops.py_func(list_func, [x], [dtypes.float64] * 2))
            self.assertAllClose(y, [0.0, 1.0])

    def testTuple(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():

            def tuple_func(x):
                if False:
                    while True:
                        i = 10
                return (x, x + 1)
            x = constant_op.constant(0.0, dtypes.float64)
            y = self.evaluate(script_ops.py_func(tuple_func, [x], [dtypes.float64] * 2))
            self.assertAllClose(y, [0.0, 1.0])
        with self.cached_session():
            x = constant_op.constant(0.0, dtypes.float64)
            y = self.evaluate(script_ops.py_func(tuple_func, (x,), (dtypes.float64, dtypes.float64)))
            self.assertAllClose(y, [0.0, 1.0])

    @test_util.run_v1_only('b/120545219')
    def testStrings(self):
        if False:
            i = 10
            return i + 15

        def read_fixed_length_numpy_strings():
            if False:
                i = 10
                return i + 15
            return np.array([b' there'])

        def read_and_return_strings(x, y):
            if False:
                i = 10
                return i + 15
            return x + y
        with self.cached_session():
            x = constant_op.constant([b'hello', b'hi'], dtypes.string)
            y = self.evaluate(script_ops.py_func(read_fixed_length_numpy_strings, [], dtypes.string))
            z = self.evaluate(script_ops.py_func(read_and_return_strings, [x, y], dtypes.string))
            self.assertAllEqual(z, [b'hello there', b'hi there'])

    @test_util.run_v1_only('b/120545219')
    def testStringsAreConvertedToBytes(self):
        if False:
            for i in range(10):
                print('nop')

        def read_fixed_length_numpy_strings():
            if False:
                print('Hello World!')
            return np.array([' there'])

        def read_and_return_strings(x, y):
            if False:
                print('Hello World!')
            return x + y
        with self.cached_session():
            x = constant_op.constant(['hello', 'hi'], dtypes.string)
            y = self.evaluate(script_ops.py_func(read_fixed_length_numpy_strings, [], dtypes.string))
            z = self.evaluate(script_ops.py_func(read_and_return_strings, [x, y], dtypes.string))
            self.assertAllEqual(z, [b'hello there', b'hi there'])

    @test_util.run_v1_only('b/120545219')
    def testObjectArraysAreConvertedToBytes(self):
        if False:
            print('Hello World!')

        def read_object_array():
            if False:
                i = 10
                return i + 15
            return np.array([b' there', u' ya'], dtype=np.object_)

        def read_and_return_strings(x, y):
            if False:
                while True:
                    i = 10
            return x + y
        with self.cached_session():
            x = constant_op.constant(['hello', 'hi'], dtypes.string)
            (y,) = script_ops.py_func(read_object_array, [], [dtypes.string])
            (z,) = script_ops.py_func(read_and_return_strings, [x, y], [dtypes.string])
            self.assertListEqual(list(self.evaluate(z)), [b'hello there', b'hi ya'])

    @test_util.run_v1_only('b/120545219')
    def testStringPadding(self):
        if False:
            while True:
                i = 10
        correct = [b'this', b'is', b'a', b'test']
        with self.cached_session():
            (s,) = script_ops.py_func(lambda : [correct], [], [dtypes.string])
            self.assertAllEqual(s, correct)

    @test_util.run_v1_only('b/120545219')
    def testStringPaddingAreConvertedToBytes(self):
        if False:
            print('Hello World!')
        inp = ['this', 'is', 'a', 'test']
        correct = [b'this', b'is', b'a', b'test']
        with self.cached_session():
            (s,) = script_ops.py_func(lambda : [inp], [], [dtypes.string])
            self.assertAllEqual(s, correct)

    @test_util.run_v1_only('b/120545219')
    def testNulTerminatedStrings(self):
        if False:
            while True:
                i = 10
        inp = np.array(['this\x00', 'is\x00\x00', 'a\x00', 'test\x00\x00'], dtype=np.str_)
        correct = [b'this', b'is', b'a', b'test']
        with self.cached_session():
            (s,) = script_ops.py_func(lambda : [inp], [], [dtypes.string])
            self.assertAllEqual(s, correct)

    @test_util.run_v1_only('b/120545219')
    def testLarge(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            x = array_ops.zeros([1000000], dtype=np.float32)
            y = script_ops.py_func(lambda x: x + 1, [x], [dtypes.float32])
            z = script_ops.py_func(lambda x: x * 2, [x], [dtypes.float32])
            for _ in range(100):
                sess.run([y[0].op, z[0].op])

    def testNoInput(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = self.evaluate(script_ops.py_func(lambda : 42.0, [], dtypes.float64))
            self.assertAllClose(x, 42.0)

    @test_util.run_v1_only('b/120545219')
    def testAlias(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            np_array = np.array([1.0, 2.0], dtype=np.float32)
            tf_array = script_ops.py_func(lambda : np_array, [], [dtypes.float32])
            value = tf_array + constant_op.constant([2.0, 3.0], dtype=dtypes.float32)
            value.op.run()
            self.assertAllEqual(np_array, [1.0, 2.0])

    @test_util.run_v1_only('b/120545219')
    def testReturnUnicodeString(self):
        if False:
            return 10
        with self.cached_session():
            correct = u'你好 世界'

            def unicode_string():
                if False:
                    return 10
                return correct
            (z,) = script_ops.py_func(unicode_string, [], [dtypes.string])
            self.assertEqual(self.evaluate(z), correct.encode('utf8'))

    @test_util.run_v1_only('b/120545219')
    def testBadNumpyReturnType(self):
        if False:
            return 10
        with self.cached_session():

            def bad():
                if False:
                    print('Hello World!')
                return np.array([], dtype=[('foo', np.float32)])
            (y,) = script_ops.py_func(bad, [], [dtypes.float32])
            with self.assertRaisesRegex(errors.InternalError, 'Unsupported numpy data type'):
                self.evaluate(y)

    @test_util.run_v1_only('b/120545219')
    def testBadReturnType(self):
        if False:
            print('Hello World!')
        with self.cached_session():

            def bad():
                if False:
                    print('Hello World!')
                return {'foo': dtypes.float32}
            (z,) = script_ops.py_func(bad, [], [dtypes.int64])
            with self.assertRaisesRegex(errors.InternalError, 'Unsupported object type'):
                self.evaluate(z)

    @test_util.run_v1_only('b/120545219')
    def testReturnInput(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():

            def ident(x):
                if False:
                    print('Hello World!')
                return x[0]
            p = array_ops.placeholder(dtypes.float32)
            (z,) = script_ops.py_func(ident, [p], [dtypes.float32])
            z += 0.0
            self.assertEqual(0.0, z.eval(feed_dict={p: [0.0]}))

    def testStateful(self):
        if False:
            for i in range(10):
                print('nop')
        with session_lib.Session():
            producer = iter(range(3))
            (x,) = script_ops.py_func(lambda : next(producer), [], [dtypes.int64])
            self.assertEqual(self.evaluate(x), 0)
            self.assertEqual(self.evaluate(x), 1)
            self.assertEqual(self.evaluate(x), 2)

    @test_util.enable_tf_xla_constant_folding('b/134376434')
    def testStateless(self):
        if False:
            print('Hello World!')
        with session_lib.Session():
            producer = iter(range(3))
            (x,) = script_ops.py_func(lambda : next(producer), [], [dtypes.int64], stateful=False)
            self.assertEqual(self.evaluate(x), 0)
            self.assertEqual(self.evaluate(x), 0)
            self.assertEqual(self.evaluate(x), 0)

    @test_util.run_v1_only('b/120545219')
    def testGradientFunction(self):
        if False:
            return 10
        a = constant_op.constant(0)
        (x,) = script_ops.py_func(lambda a: 0, [a], [dtypes.int64])
        (y,) = script_ops.py_func(lambda a: 0, [a], [dtypes.int64], stateful=False)
        self.assertEqual(None, ops.get_gradient_function(x.op))
        self.assertEqual(None, ops.get_gradient_function(y.op))

    @test_util.run_v1_only('b/120545219')
    def testCOrder(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            val = [[1, 2], [3, 4]]
            (x,) = script_ops.py_func(lambda : np.array(val, order='F'), [], [dtypes.int64])
            self.assertAllEqual(val, self.evaluate(x))

    @test_util.run_v1_only('b/120545219')
    def testParallel(self):
        if False:
            print('Hello World!')
        with self.cached_session() as session:
            q = queue.Queue(1)

            def blocking_put():
                if False:
                    for i in range(10):
                        print('nop')
                q.put(42)
                q.join()
                return 42

            def blocking_get():
                if False:
                    for i in range(10):
                        print('nop')
                v = q.get(block=True)
                q.task_done()
                return v
            (x,) = script_ops.py_func(blocking_put, [], [dtypes.int64])
            (y,) = script_ops.py_func(blocking_get, [], [dtypes.int64])
            session.run([x, y])

    def testNoReturnValueStateful(self):
        if False:
            i = 10
            return i + 15

        class State:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self._value = np.array([1], np.int64)

            def _increment(self, diff):
                if False:
                    for i in range(10):
                        print('nop')
                self._value += diff

            def increment(self, diff):
                if False:
                    i = 10
                    return i + 15
                return script_ops.py_func(self._increment, [diff], [], stateful=True)

            @property
            def value(self):
                if False:
                    while True:
                        i = 10
                return self._value
        with self.cached_session():
            s = State()
            op = s.increment(constant_op.constant(2, dtypes.int64))
            ret = self.evaluate(op)
            self.assertIsNone(ret)
            self.assertAllEqual([3], s.value)

    @test_util.run_v1_only('b/120545219')
    def testNoReturnValueStateless(self):
        if False:
            i = 10
            return i + 15

        def do_nothing(unused_x):
            if False:
                return 10
            pass
        f = script_ops.py_func(do_nothing, [constant_op.constant(3, dtypes.int64)], [], stateful=False)
        with self.cached_session():
            self.assertEqual(self.evaluate(f), [])

    @test_util.run_v1_only('b/120545219')
    def testExceptionHandling(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            self.verifyExceptionHandling(ValueError, errors.InvalidArgumentError)
            self.verifyExceptionHandling(TypeError, errors.InvalidArgumentError)
            self.verifyExceptionHandling(StopIteration, errors.OutOfRangeError)
            self.verifyExceptionHandling(MemoryError, errors.ResourceExhaustedError)
            self.verifyExceptionHandling(NotImplementedError, errors.UnimplementedError)

            class WeirdError(Exception):
                pass
            self.verifyExceptionHandling(WeirdError, errors.UnknownError)

    def testFunctionReferencesAreKept(self):
        if False:
            i = 10
            return i + 15
        g = ops.Graph()
        with g.as_default():
            c = constant_op.constant([1.0], dtypes.float32)

            @batch_ops.batch_function(1, 10, 100000)
            def fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return script_ops.py_func(lambda x: x, [x], [dtypes.float32])
            result = fn(c)
            gc.collect()
            self.evaluate(result)

class PyFuncAndEagerPyFuncTest(PyFuncTestBase):
    """Encapsulates tests shared between py_func and eager_py_func."""

    def verifyPyFuncsNoIncrease(self, make_graph):
        if False:
            while True:
                i = 10
        ops.reset_default_graph()
        gc.collect()
        initial_size = script_ops._py_funcs.size()
        for _ in range(1000):
            make_graph()
        ops.reset_default_graph()
        gc.collect()
        self.assertEqual(initial_size, script_ops._py_funcs.size())

    def testCleanup(self):
        if False:
            return 10

        def make_graph():
            if False:
                print('Hello World!')
            g = ops.Graph()
            with g.as_default():
                c = constant_op.constant([1.0], dtypes.float32)
                _ = script_ops.py_func(lambda x: x + 1, [c], [dtypes.float32])
                _ = script_ops.eager_py_func(lambda x: x + 1, [c], [dtypes.float32])
                script_ops.py_func(lambda x: x + c.shape[0], [c], [dtypes.float32])
                script_ops.eager_py_func(lambda x: x + c.shape[0], [c], [dtypes.float32])
        self.verifyPyFuncsNoIncrease(make_graph)

    def testCleanupInTfFunction(self):
        if False:
            i = 10
            return i + 15
        self.skipTest('b/144098211')

        def make_graph():
            if False:
                print('Hello World!')
            g = ops.Graph()
            with g.as_default():

                @def_function.function
                def fn():
                    if False:
                        for i in range(10):
                            print('nop')
                    c = constant_op.constant([1.0], dtypes.float32)
                    _ = script_ops.py_func(lambda x: x + 1, [c], [dtypes.float32])
                    _ = script_ops.eager_py_func(lambda x: x + 1, [c], [dtypes.float32])
                    script_ops.py_func(lambda x: x + c.shape[0], [c], [dtypes.float32])
                    script_ops.eager_py_func(lambda x: x + c.shape[0], [c], [dtypes.float32])
                fn()
        self.verifyPyFuncsNoIncrease(make_graph)

class EagerPyFuncTest(PyFuncTestBase):
    """Encapsulates tests for eager_py_func only."""

    @test_util.run_in_graph_and_eager_modes
    def testEagerSingleOutputInt32(self):
        if False:
            for i in range(10):
                print('nop')
        a = array_ops.ones((3, 3), dtype=dtypes.int32)
        x = array_ops.ones((3, 1), dtype=dtypes.int32)
        output = script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.int32)
        ret = self.evaluate(output)
        self.assertAllEqual(ret, [[3], [3], [3]])

    @test_util.run_in_graph_and_eager_modes
    def testRenamedDeviceInTestClusterCorrectlyIdentifiedAsLocalhost(self):
        if False:
            return 10
        if context.executing_eagerly():
            self.skipTest("b/126565353: We don't test eager's remote execution.")
        (workers, _) = test_util.create_local_cluster(num_workers=1, num_ps=0)
        worker = workers[0]
        session = session_lib.Session(worker.target)
        with ops.device('/job:worker/task:0/cpu:0'):
            a = array_ops.ones((3, 3), dtype=dtypes.float32)
            x = array_ops.ones((3, 1), dtype=dtypes.float32)
            output = script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.float32)
        ret = session.run(output)
        self.assertAllClose(ret, [[3.0], [3.0], [3.0]])

    @test_util.run_in_graph_and_eager_modes
    def testEagerSingleOutputFloat32(self):
        if False:
            return 10
        with test_util.device(use_gpu=True):
            a = array_ops.ones((3, 3), dtype=dtypes.float32)
            x = array_ops.ones((3, 1), dtype=dtypes.float32)
            output = script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.float32)
            ret = self.evaluate(output)
            self.assertAllClose(ret, [[3.0], [3.0], [3.0]])

    @test_util.run_in_graph_and_eager_modes
    def testEagerArrayOutput(self):
        if False:
            return 10
        with test_util.device(use_gpu=True):
            a = array_ops.ones((3, 3), dtype=dtypes.float32)
            x = array_ops.ones((3, 1), dtype=dtypes.float32)
            output = script_ops.eager_py_func(lambda a, x: [matmul(a, x)], inp=[a, x], Tout=[dtypes.float32])
            ret = self.evaluate(output)
            self.assertAllEqual(ret, [[[3.0], [3.0], [3.0]]])

    @test_util.run_in_graph_and_eager_modes
    def testEagerReturnNone(self):
        if False:
            print('Hello World!')
        with test_util.device(use_gpu=True):

            def no_return_value():
                if False:
                    while True:
                        i = 10
                return
            output = script_ops.eager_py_func(no_return_value, inp=[], Tout=[])
            ret = self.evaluate(output)
            if context.executing_eagerly():
                self.assertEqual(len(ret), 0)
            else:
                self.assertIsNone(ret)

    @test_util.run_in_graph_and_eager_modes
    @test_util.disable_tfrt('b/180469928')
    def testEagerPyFuncInDefun(self):
        if False:
            print('Hello World!')
        with test_util.device(use_gpu=True):

            def wrapper():
                if False:
                    print('Hello World!')
                a = array_ops.ones((3, 3), dtype=dtypes.float32)
                x = array_ops.ones((3, 1), dtype=dtypes.float32)
                return script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.float32)
            wrapped = def_function.function(wrapper)
            ret = self.evaluate(wrapped())
            self.assertAllEqual(ret, [[3.0], [3.0], [3.0]])

    @test_util.run_in_graph_and_eager_modes
    @test_util.run_v1_only('b/120545219')
    def testEagerExceptionHandling(self):
        if False:
            for i in range(10):
                print('nop')
        with test_util.device(use_gpu=True):
            self.verifyExceptionHandling(ValueError, errors.InvalidArgumentError, eager=True)
            self.verifyExceptionHandling(TypeError, errors.InvalidArgumentError, eager=True)
            self.verifyExceptionHandling(StopIteration, errors.OutOfRangeError, eager=True)
            self.verifyExceptionHandling(MemoryError, errors.ResourceExhaustedError, eager=True)
            self.verifyExceptionHandling(NotImplementedError, errors.UnimplementedError, eager=True)

            class WeirdError(Exception):
                pass
            self.verifyExceptionHandling(WeirdError, errors.UnknownError, eager=True)

    @test_util.run_in_graph_and_eager_modes
    @test_util.run_v1_only('b/120545219')
    def testEagerReturningVariableRaisesError(self):
        if False:
            print('Hello World!')

        def return_variable():
            if False:
                print('Hello World!')
            return resource_variable_ops.ResourceVariable(0.0)
        with self.assertRaisesRegex(errors.UnknownError, 'Attempting to return a variable'):
            output = script_ops.eager_py_func(return_variable, inp=[], Tout=dtypes.float32)
            self.evaluate(output)

    @test_util.run_in_graph_and_eager_modes
    def testTapeCache(self):
        if False:
            return 10
        old_cache_size = len(script_ops.tape_cache)

        def f(x):
            if False:
                i = 10
                return i + 15
            return x ** 2
        x = constant_op.constant(3.0)
        y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.float32)
        self.assertEqual(len(script_ops.tape_cache), old_cache_size)
        with backprop.GradientTape() as tape:
            tape.watch(x)
            y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.float32)
            if context.executing_eagerly():
                self.assertEqual(len(script_ops.tape_cache), old_cache_size + 1)
            else:
                self.assertEqual(len(script_ops.tape_cache), old_cache_size)
        dy_dx = tape.gradient(y, x)
        self.evaluate(dy_dx)
        self.assertEqual(len(script_ops.tape_cache), old_cache_size)

    @test_util.run_in_graph_and_eager_modes
    def testEagerGradientTape(self):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            return x ** 2
        x = constant_op.constant(3.0)
        with backprop.GradientTape() as tape:
            tape.watch(x)
            y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.float32)
        dy_dx = tape.gradient(y, x)
        self.assertAllClose(self.evaluate(dy_dx), 6.0)
        x = constant_op.constant(3.0 + 3j)
        with backprop.GradientTape() as tape:
            tape.watch(x)
            y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.complex128)
        dy_dx = tape.gradient(y, x)
        self.assertAllClose(self.evaluate(dy_dx), 6.0 - 6j)

    @test_util.run_v1_only('b/120545219')
    def testEagerGradientGraph(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x ** 2
        x = constant_op.constant(3.0)
        y = script_ops.eager_py_func(f, inp=[x], Tout=dtypes.float32)
        dy_dx = gradients_impl.gradients(y, x)[0]
        self.assertEqual(self.evaluate(dy_dx), 6.0)

    @test_util.run_v1_only('b/120545219')
    def testEagerGradientGraphTwoOutputs(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                while True:
                    i = 10
            return (x * y, x / y)
        x = constant_op.constant(3.0)
        y = constant_op.constant(2.0)
        (fa, fb) = script_ops.eager_py_func(f, inp=[x, y], Tout=[dtypes.float32, dtypes.float32])
        dy_dx = gradients_impl.gradients(fa + fb, x)[0]
        self.assertEqual(self.evaluate(dy_dx), 2.5)

    @test_util.run_in_graph_and_eager_modes
    def testEagerGradientTapeMultipleArgs(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                print('Hello World!')
            return x ** 2 + y ** 2
        x = constant_op.constant(3.0)
        y = constant_op.constant(4.0)
        with backprop.GradientTape() as tape:
            tape.watch(x)
            tape.watch(y)
            z = script_ops.eager_py_func(f, inp=[x, y], Tout=dtypes.float32)
        (dz_dx, dz_dy) = tape.gradient(z, [x, y])
        self.assertEqual(self.evaluate(dz_dx), 6.0)
        self.assertEqual(self.evaluate(dz_dy), 8.0)

    @test_util.run_v1_only('b/120545219')
    def testEagerGradientGraphMultipleArgs(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x ** 2 + y ** 2
        x = constant_op.constant(3.0)
        y = constant_op.constant(4.0)
        z = script_ops.eager_py_func(f, inp=[x, y], Tout=dtypes.float32)
        (dz_dx, dz_dy) = gradients_impl.gradients(z, [x, y])
        self.assertEqual(self.evaluate(dz_dx), 6.0)
        self.assertEqual(self.evaluate(dz_dy), 8.0)

    @test_util.run_v1_only('b/120545219')
    def testEagerGradientGraphLogHuber(self):
        if False:
            print('Hello World!')

        def log_huber(x, m):
            if False:
                while True:
                    i = 10
            if math_ops.abs(x) <= m:
                return x ** 2
            else:
                return m ** 2 * (1 - 2 * math_ops.log(m) + math_ops.log(x ** 2))
        x = array_ops.placeholder(dtypes.float32)
        m = array_ops.placeholder(dtypes.float32)
        y = script_ops.eager_py_func(func=log_huber, inp=[x, m], Tout=dtypes.float32)
        dy_dx = gradients_impl.gradients(y, x)[0]
        with self.cached_session() as sess:
            (y, dy_dx) = sess.run([y, dy_dx], feed_dict={x: 1.0, m: 2.0})
            self.assertEqual(y, 1.0)
            self.assertEqual(dy_dx, 2.0)

    @test_util.run_v1_only('b/120545219')
    def testEagerRespectsDevicePlacementOfOp(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            return math_ops.square(x)

        def g(x):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.add(x, x)
        with ops.device('/CPU:0'):
            x = array_ops.placeholder(dtypes.float32)
            y = script_ops.eager_py_func(func=f, inp=[x], Tout=dtypes.float32)
            z = script_ops.eager_py_func(func=g, inp=[y], Tout=dtypes.float32)
        with self.session() as sess:
            output = sess.run(z, feed_dict={x: 3.0})
            self.assertEqual(output, 18.0)

    @test_util.run_in_graph_and_eager_modes
    def testEagerPyFuncOnGPUWithStrings(self):
        if False:
            print('Hello World!')

        def fn(a):
            if False:
                while True:
                    i = 10
            return str(a.dtype)
        x = constant_op.constant('x', dtype=dtypes.string)
        output = script_ops.eager_py_func(fn, inp=[x], Tout=dtypes.string)
        self.assertEqual(self.evaluate(output), "<dtype: 'string'>".encode('utf8'))

    @test_util.run_in_graph_and_eager_modes
    def testEagerPyFuncNotACallable(self):
        if False:
            return 10
        x = constant_op.constant('x', dtype=dtypes.string)
        with self.assertRaisesRegex(ValueError, 'callable'):
            _ = script_ops.eager_py_func(x, inp=[x], Tout=dtypes.string)

    def testUnsupportedToutType(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'Cannot convert .* to a TensorFlow DType.'):
            script_ops.eager_py_func(lambda x: x, [1], [{}])

    def testRaggedTensorArg(self):
        if False:
            while True:
                i = 10
        x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
        (y,) = script_ops.eager_py_func(math_ops.reduce_sum, [x], [dtypes.int32])
        self.assertAllEqual(y, 21)

    def testRaggedTensorReturn(self):
        if False:
            i = 10
            return i + 15

        def fn(v, l):
            if False:
                print('Hello World!')
            return ragged_tensor.RaggedTensor.from_row_lengths(v, l)
        values = [1, 2, 3, 4, 5, 6]
        lengths = constant_op.constant([3, 1, 2], dtypes.int64)
        out_signature = [ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)]
        (y,) = script_ops.eager_py_func(fn, [values, lengths], out_signature)
        self.assertIsInstance(y, ragged_tensor.RaggedTensor)
        self.assertAllEqual(y, [[1, 2, 3], [4], [5, 6]])

    def testRaggedTensorBroadcast(self):
        if False:
            return 10

        def fn(x):
            if False:
                print('Hello World!')
            return 2 * x

        def foo(x):
            if False:
                print('Hello World!')
            spec = ragged_tensor.RaggedTensorSpec.from_value(x)
            res = script_ops.eager_py_func(fn, [x], spec)
            return x + res
        x = ragged_factory_ops.constant([[1.0, 2.0], [3.0]])
        expected_result = [[3.0, 6.0], [9.0]]
        result1 = foo(x)
        self.assertAllEqual(result1, expected_result)
        result2 = def_function.function(foo)(x)
        self.assertAllEqual(result2, expected_result)

    def testRaggedExpectedListGotList(self):
        if False:
            for i in range(10):
                print('nop')
        x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
        x_spec = type_spec.type_spec_from_value(x)
        (y,) = script_ops.eager_py_func(lambda v: [v], [x], [x_spec])
        self.assertAllEqual(y, x)

    def testRaggedExpectedListGotTuple(self):
        if False:
            i = 10
            return i + 15
        x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
        x_spec = type_spec.type_spec_from_value(x)
        (y,) = script_ops.eager_py_func(lambda v: (v,), [x], [x_spec])
        self.assertAllEqual(y, x)

    def testRaggedExpectedListGotSingleValue(self):
        if False:
            print('Hello World!')
        x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
        x_spec = type_spec.type_spec_from_value(x)
        (y,) = script_ops.eager_py_func(lambda v: v, [x], [x_spec])
        self.assertAllEqual(y, x)

    def testRaggedNoReturnValue(self):
        if False:
            return 10
        x = ragged_factory_ops.constant([[1, 2, 3], [4], [5, 6]])
        result = self.evaluate(script_ops.eager_py_func(lambda v: None, [x], []))
        if context.executing_eagerly():
            self.assertEqual(result, [])
        else:
            self.assertIsNone(result)

    def testRaggedBadReturnTypeExpectedTensorReturnedRagged(self):
        if False:
            print('Hello World!')
        rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'py_function: func=.* returned .* which did not match Tout=.*'):
            result = script_ops.eager_py_func(lambda x: x + 3, [rt], [dtypes.int32])
            self.evaluate(result)

    def testRaggedBadReturnTypeExpectedRaggedReturnedTensor(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'py_function: func=.* returned .* which did not match Tout=.*'):
            result = script_ops.eager_py_func(func=lambda x: x, inp=[constant_op.constant([[1, 2, 3]])], Tout=[ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)])
            self.evaluate(result)
if __name__ == '__main__':
    test.main()