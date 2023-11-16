"""critical section tests."""
import itertools
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

@test_util.with_control_flow_v2
class CriticalSectionTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testCreateCriticalSection(self):
        if False:
            return 10
        cs = critical_section_ops.CriticalSection(shared_name='cs')
        v = resource_variable_ops.ResourceVariable(0.0, name='v')

        def fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            c = v.value()
            with ops.control_dependencies([c]):
                nv = v.assign_add(a * b)
                with ops.control_dependencies([nv]):
                    return array_ops.identity(c)
        num_concurrent = 100
        r = [cs.execute(lambda : fn(1.0, 2.0)) for _ in range(num_concurrent)]
        self.evaluate(v.initializer)
        r_value = self.evaluate(r)
        self.assertAllClose([2.0 * i for i in range(num_concurrent)], sorted(r_value))

    @parameterized.named_parameters((('Inner%sOuter%s' % (inner, outer), inner, outer) for (inner, outer) in itertools.product(*[(False, True)] * 2)))
    @test_util.run_in_graph_and_eager_modes
    @test_util.xla_allow_fallback('b/128495870')
    def testCriticalSectionWithControlFlow(self, outer_cond, inner_cond):
        if False:
            print('Hello World!')
        if not context.executing_eagerly() and control_flow_v2_toggles.control_flow_v2_enabled():
            self.skipTest('b/135070612')
        cs = critical_section_ops.CriticalSection(shared_name='cs')
        v = resource_variable_ops.ResourceVariable(0.0, name='v')
        num_concurrent = 100

        def fn(a, b):
            if False:
                print('Hello World!')
            c = v.read_value()

            def true_fn():
                if False:
                    while True:
                        i = 10
                with ops.control_dependencies([c]):
                    nv = v.assign_add(a * b)
                    with ops.control_dependencies([nv]):
                        return array_ops.identity(c)
            return cond.cond(array_ops.identity(inner_cond), true_fn, lambda : c)

        def execute():
            if False:
                while True:
                    i = 10
            return cs.execute(lambda : fn(1.0, 2.0))
        r = [cond.cond(array_ops.identity(outer_cond), execute, v.read_value) for _ in range(num_concurrent)]
        self.evaluate(v.initializer)
        r_value = self.evaluate(r)
        if inner_cond and outer_cond:
            self.assertAllClose([2.0 * i for i in range(num_concurrent)], sorted(r_value))
        else:
            self.assertAllClose([0] * num_concurrent, r_value)

    @test_util.run_v1_only('b/123990562 Sees CancelledError on some calls')
    def testCriticalSectionInParallelDoesntDeadlockOnError(self):
        if False:
            i = 10
            return i + 15
        cs = critical_section_ops.CriticalSection(shared_name='cs')
        v = resource_variable_ops.ResourceVariable(0.0, name='v')

        def fn(i):
            if False:
                for i in range(10):
                    print('nop')
            error = control_flow_assert.Assert(i % 2 == 1, ['Error'])
            with ops.control_dependencies([error]):
                return v.read_value()
        num_concurrent = 2

        @def_function.function(autograph=False)
        def run_concurrently():
            if False:
                i = 10
                return i + 15
            return [cs.execute(lambda : fn(i)) for i in range(num_concurrent)]
        if not context.executing_eagerly():
            run_concurrently = run_concurrently()
        self.evaluate(v.initializer)
        for _ in range(100):
            with self.assertRaisesOpError('Error'):
                if context.executing_eagerly():
                    run_concurrently()
                else:
                    self.evaluate(run_concurrently)

    @test_util.run_in_graph_and_eager_modes
    def testCreateCriticalSectionFnReturnsOp(self):
        if False:
            while True:
                i = 10
        cs = critical_section_ops.CriticalSection(shared_name='cs')
        v = resource_variable_ops.ResourceVariable(0.0, name='v')

        def fn_return_op(a, b):
            if False:
                return 10
            c = v.read_value()
            with ops.control_dependencies([c]):
                nv = v.assign_add(a * b)
                with ops.control_dependencies([nv]):
                    return control_flow_ops.no_op()
        num_concurrent = 100
        r = [cs.execute(lambda : fn_return_op(1.0, 2.0)) for _ in range(num_concurrent)]
        self.evaluate(v.initializer)
        self.evaluate(r)
        final_v = self.evaluate(v)
        self.assertAllClose(2.0 * num_concurrent, final_v)

    @test_util.run_v1_only("Collections don't exist in TF2")
    def testCollection(self):
        if False:
            print('Hello World!')
        cs = critical_section_ops.CriticalSection(shared_name='cs')
        self.assertIn(cs, ops.get_collection(critical_section_ops.CRITICAL_SECTIONS))
        add = lambda x: x + 1
        execute = cs.execute(lambda : add(1.0), name='my_execute')
        execute_op = [x for x in execute.graph.get_operations() if 'my_execute' in x.name and 'MutexLock' in x.type][0]
        self.assertIn(execute_op, [signature.op for signature in ops.get_collection(critical_section_ops.CRITICAL_SECTION_EXECUTIONS)])

    def testRecursiveCriticalSectionAccessIsIllegal(self):
        if False:
            while True:
                i = 10
        cs = critical_section_ops.CriticalSection()
        add = lambda y: y + 1

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return cs.execute(lambda : add(x))
        with self.assertRaisesRegex(ValueError, 'Attempting to lock a CriticalSection .* in which we are'):
            cs.execute(lambda : fn(1.0))

    def testRecursiveCriticalSectionAccessViaCapturedTensorIsProtected(self):
        if False:
            print('Hello World!')
        cs = critical_section_ops.CriticalSection(shared_name='cs')
        fn = array_ops.identity
        to_capture = cs.execute(lambda : fn(1.0))
        fn_captures = lambda x: x + to_capture
        to_capture_too = array_ops.identity(to_capture)
        ex_0 = cs.execute(lambda : fn_captures(1.0))
        with ops.control_dependencies([to_capture]):
            ex_1 = cs.execute(lambda : fn_captures(1.0))
        dependency = array_ops.identity(to_capture)
        fn_captures_dependency = lambda x: x + dependency
        ex_2 = cs.execute(lambda : fn_captures_dependency(1.0))
        with ops.control_dependencies([to_capture_too]):
            ex_3 = cs.execute(lambda : fn_captures_dependency(1.0))
        self.assertEqual(2.0, self.evaluate(ex_0))
        self.assertEqual(2.0, self.evaluate(ex_1))
        self.assertEqual(2.0, self.evaluate(ex_2))
        self.assertEqual(2.0, self.evaluate(ex_3))

    def testRecursiveCriticalSectionAccessWithinLoopIsProtected(self):
        if False:
            i = 10
            return i + 15
        cs = critical_section_ops.CriticalSection(shared_name='cs')

        def body_implicit_capture(i, j):
            if False:
                print('Hello World!')
            fn = lambda : j + 1
            return (i + 1, cs.execute(fn))
        (i_n, j_n) = while_loop.while_loop(lambda i, _: i < 1000, body_implicit_capture, [0, 0], parallel_iterations=25)
        i_n = array_ops.identity(i_n)
        logging.warn("\n==============\nRunning 'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock body_implicit_capture'\n==============\n")
        self.assertEqual((1000, 1000), self.evaluate((i_n, j_n)))
        logging.warn("\n==============\nSuccessfully finished running 'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock body_implicit_capture'\n==============\n")

        def body_implicit_capture_protected(i, j):
            if False:
                i = 10
                return i + 15
            fn = lambda : j + 1
            with ops.control_dependencies([j]):
                return (i + 1, cs.execute(fn))
        (i_n, j_n) = while_loop.while_loop(lambda i, _: i < 1000, body_implicit_capture_protected, [0, 0], parallel_iterations=25)
        i_n = array_ops.identity(i_n)
        logging.warn("\n==============\nRunning 'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock body_implicit_capture_protected'\n==============\n")
        self.assertEqual((1000, 1000), self.evaluate((i_n, j_n)))
        logging.warn("\n==============\nSuccessfully finished running 'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock body_implicit_capture_protected'\n==============\n")

        def body_args_capture(i, j):
            if False:
                print('Hello World!')
            fn = lambda x: x + 1
            return (i + 1, cs.execute(lambda : fn(j)))
        (i_n, j_n) = while_loop.while_loop(lambda i, _: i < 1000, body_args_capture, [0, 0], parallel_iterations=25)
        i_n = array_ops.identity(i_n)
        logging.warn("\n==============\nRunning 'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock body_args_capture'\n==============\n")
        self.assertEqual((1000, 1000), self.evaluate((i_n, j_n)))
        logging.warn("\n==============\nSuccessfully finished running 'testRecursiveCriticalSectionAccessWithinLoopDoesNotDeadlock body_args_capture'\n==============\n")

    def testRecursiveCriticalSectionAccessIsIllegalSameSharedName(self):
        if False:
            print('Hello World!')
        cs = critical_section_ops.CriticalSection(shared_name='cs')
        cs_same = critical_section_ops.CriticalSection(shared_name='cs')
        add = lambda x: x + 1

        def fn(x):
            if False:
                return 10
            return cs_same.execute(lambda : add(x))
        with self.assertRaisesRegex(ValueError, 'Attempting to lock a CriticalSection .* in which we are'):
            cs.execute(lambda : fn(1.0))

    @test_util.run_v1_only("b/123955885 Can't identify consumed resources in eager mode")
    def testMultipleCSExecutionsRequestSameResource(self):
        if False:
            return 10
        cs0 = critical_section_ops.CriticalSection()
        cs1 = critical_section_ops.CriticalSection()
        v = resource_variable_ops.ResourceVariable(0.0, name='v')
        cs0.execute(lambda : v + 1)
        cs0.execute(lambda : v - 1)
        with self.assertRaisesRegex(ValueError, 'requested exclusive resource access'):
            cs1.execute(lambda : v + 1)
        with self.assertRaisesRegex(ValueError, 'requested exclusive resource access'):
            cs1.execute(lambda : v + 1, exclusive_resource_access=False)
        v2 = resource_variable_ops.ResourceVariable(0.0, name='v2')
        cs0.execute(lambda : v2 + 1, exclusive_resource_access=False)
        cs1.execute(lambda : v2 + 1, exclusive_resource_access=False)
        with self.assertRaisesRegex(ValueError, 'requested exclusive resource access'):
            cs1.execute(lambda : v2 + 1)

    def testControlDependencyFromOutsideWhileLoopMixedWithInsideLoop(self):
        if False:
            print('Hello World!')
        cs = critical_section_ops.CriticalSection()
        v = resource_variable_ops.ResourceVariable(0, name='v')

        def body(i):
            if False:
                i = 10
                return i + 15
            add_j = lambda j: v + j + 1
            return cs.execute(lambda : add_j(i))
        out = while_loop.while_loop(lambda i: i < 10, body, [0])
        self.evaluate(v.initializer)
        self.assertEqual(10, self.evaluate(out))

    @test_util.run_in_graph_and_eager_modes
    def testInsideFunction(self):
        if False:
            while True:
                i = 10
        if test_util.is_gpu_available():
            self.skipTest('b/123899495: Colocation errors for critical sections in map on GPU')
        cs = critical_section_ops.CriticalSection()
        with ops.device('/gpu:0' if test_util.is_gpu_available() else '/cpu:0'):
            v = resource_variable_ops.ResourceVariable(1)

        def fn():
            if False:
                print('Hello World!')
            return v.read_value()
        ds = dataset_ops.Dataset.range(1)
        if test_util.is_gpu_available():
            ds = ds.apply(prefetching_ops.copy_to_device('/gpu:0')).apply(prefetching_ops.map_on_gpu(lambda _: cs.execute(fn)))
        else:
            ds = ds.map(lambda _: cs.execute(fn))

        def get_first():
            if False:
                for i in range(10):
                    print('nop')
            if context.executing_eagerly():
                return self.evaluate(dataset_ops.make_one_shot_iterator(ds).get_next())
            itr = dataset_ops.make_initializable_iterator(ds)
            self.evaluate([v.initializer, itr.initializer])
            return self.evaluate(itr.get_next())
        self.assertEqual(1, get_first())
if __name__ == '__main__':
    test.main()