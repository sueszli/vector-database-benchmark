"""Tests for variable store."""
import gc
import threading
import numpy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.util import compat
from tensorflow.python.util import tf_inspect

def run_inside_wrap_function_in_eager_mode(graph_function):
    if False:
        for i in range(10):
            print('nop')
    'Decorator to execute the same graph code in eager and graph modes.\n\n  In graph mode, we just execute the graph_function passed as argument. In eager\n  mode, we wrap the function using wrap_function and then execute the wrapped\n  result.\n\n  Args:\n    graph_function: python function containing graph code to be wrapped\n\n  Returns:\n    decorated function\n  '

    def wrap_and_execute(self):
        if False:
            for i in range(10):
                print('nop')
        if context.executing_eagerly():
            wrapped = wrap_function.wrap_function(graph_function, [self])
            wrapped()
        else:
            graph_function(self)
    return wrap_and_execute

class VariableScopeTest(test.TestCase):

    def tearDown(self):
        if False:
            print('Hello World!')
        gc.collect()
        self.assertEqual(0, len(gc.garbage))

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testGetVar(self):
        if False:
            while True:
                i = 10
        vs = variable_scope._get_default_variable_store()
        v = vs.get_variable('v', [1])
        v1 = vs.get_variable('v', [1])
        self.assertIs(v, v1)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testResource(self):
        if False:
            for i in range(10):
                print('nop')
        vs = variable_scope._get_default_variable_store()
        v1 = vs.get_variable('v', [1], use_resource=True)
        self.assertTrue(isinstance(v1, resource_variable_ops.ResourceVariable))

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testNameExists(self):
        if False:
            return 10
        vs = variable_scope._get_default_variable_store()
        v = vs.get_variable('v', [1])
        v1 = vs.get_variable('v', [1])
        self.assertIs(v, v1)
        vs.get_variable('w', [1], reuse=False)
        with self.assertRaises(ValueError):
            vs.get_variable('v', [1], reuse=False)
        vs.get_variable('v', [1], reuse=True)
        with self.assertRaises(ValueError):
            vs.get_variable('u', [1], reuse=True)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testNamelessStore(self):
        if False:
            return 10
        vs = variable_scope._get_default_variable_store()
        vs.get_variable('v1', [2])
        vs.get_variable('v2', [2])
        expected_names = ['%s:0' % name for name in ['v1', 'v2']]
        self.assertEqual(set(expected_names), set((v.name for v in vs._vars.values())))

    @test_util.run_in_graph_and_eager_modes
    def testVarScopeInitializer(self):
        if False:
            i = 10
            return i + 15
        init = init_ops.constant_initializer(0.3)
        with variable_scope.variable_scope('tower0') as tower:
            with variable_scope.variable_scope('foo', initializer=init):
                v = variable_scope.get_variable('v', [])
                self.evaluate(variables_lib.variables_initializer([v]))
                self.assertAllClose(self.evaluate(v.value()), 0.3)
            with variable_scope.variable_scope(tower, initializer=init):
                w = variable_scope.get_variable('w', [])
                self.evaluate(variables_lib.variables_initializer([w]))
                self.assertAllClose(self.evaluate(w.value()), 0.3)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeConstraint(self):
        if False:
            for i in range(10):
                print('nop')
        constraint = lambda x: 0.0 * x
        with variable_scope.variable_scope('tower1') as tower:
            with variable_scope.variable_scope('foo', constraint=constraint):
                v = variable_scope.get_variable('v', [])
                self.assertEqual(v.constraint, constraint)
            with variable_scope.variable_scope(tower, constraint=constraint):
                w = variable_scope.get_variable('w', [])
                self.assertEqual(w.constraint, constraint)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeNestingError(self):
        if False:
            i = 10
            return i + 15
        with variable_scope.variable_scope('aa'):
            scope = variable_scope.variable_scope('bb')
            scope.__enter__()
            with variable_scope.variable_scope('cc'):
                with self.assertRaises(RuntimeError):
                    scope.__exit__(None, None, None)
            scope.__exit__(None, None, None)

    @test_util.run_deprecated_v1
    def testStringDefaultInitializer(self):
        if False:
            return 10
        with self.cached_session():
            v = variable_scope.get_variable('string', shape=[], dtype=dtypes.string)
            variables_lib.global_variables_initializer().run()
            self.assertAllEqual(compat.as_bytes(self.evaluate(v)), b'')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeDType(self):
        if False:
            return 10
        with variable_scope.variable_scope('tower2') as tower:
            with variable_scope.variable_scope('foo', dtype=dtypes.float16):
                v = variable_scope.get_variable('v', [])
                self.assertEqual(v.dtype.base_dtype, dtypes.float16)
            with variable_scope.variable_scope(tower, dtype=dtypes.float16):
                w = variable_scope.get_variable('w', [])
                self.assertEqual(w.dtype.base_dtype, dtypes.float16)

    def testGetVariableInGraphNestedUnderEagerContext(self):
        if False:
            return 10
        with context.eager_mode():

            @def_function.function
            def f(v):
                if False:
                    print('Hello World!')
                self.assertEqual(type(v), resource_variable_ops.ResourceVariable)
            var = variable_scope.get_variable('should_be_resource', [])
            f(var)

    def testEagerVariableStore(self):
        if False:
            for i in range(10):
                print('nop')
        with context.eager_mode():
            store = variable_scope.EagerVariableStore()
            with store.as_default():
                v = variable_scope.get_variable('v', shape=(), trainable=True)
                w = variable_scope.get_variable('w', shape=(), trainable=False)
            self.assertTrue(v in store.variables())
            self.assertTrue(w in store.variables())
            self.assertTrue(v in store.trainable_variables())
            self.assertFalse(w in store.trainable_variables())
            self.assertFalse(v in store.non_trainable_variables())
            self.assertTrue(w in store.non_trainable_variables())
            new_store = store.copy()
            with new_store.as_default():
                new_v = variable_scope.get_variable('v')
                new_w = variable_scope.get_variable('w')
            self.assertEqual(new_v.numpy(), v.numpy())
            self.assertEqual(new_w.numpy(), w.numpy())
            self.assertTrue(new_v in new_store.variables())
            self.assertTrue(new_w in new_store.variables())
            self.assertTrue(new_v in new_store.trainable_variables())
            self.assertFalse(new_w in new_store.trainable_variables())
            self.assertFalse(new_v in new_store.non_trainable_variables())
            self.assertTrue(new_w in new_store.non_trainable_variables())
            for v in store.variables():
                v.assign(-1)
            for v in new_store.variables():
                v.assign(1)
            for v in store.variables():
                self.assertEqual(v.numpy(), -1)
            for v in new_store.variables():
                self.assertEqual(v.numpy(), 1)

    @test_util.run_in_graph_and_eager_modes
    def testEagerVariablesStoreAddsToCollections(self):
        if False:
            i = 10
            return i + 15
        store = variable_scope.EagerVariableStore()
        with store.as_default():
            trainable = variable_scope.get_variable('v1', [], trainable=True)
            not_trainable = variable_scope.get_variable('v2', [], trainable=False)
            concat = variable_scope.get_variable('v3', [], collections=[ops.GraphKeys.CONCATENATED_VARIABLES])
            self.assertEqual(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), [trainable, not_trainable])
            self.assertEqual(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES), [trainable, concat])
            self.assertEqual(ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES), [concat])

    def testEagerVariablesOutsideStoreNotAddedToCollections(self):
        if False:
            while True:
                i = 10
        with context.eager_mode():
            variable_scope.get_variable('v1', [], trainable=True)
            variable_scope.get_variable('v2', [], trainable=False)
            self.assertFalse(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
            self.assertFalse(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

    def testEagerVariableStoreWithFunctionalLayer(self):
        if False:
            print('Hello World!')
        with context.eager_mode():
            container = variable_scope.EagerVariableStore()
            x = constant_op.constant([[2.0]])
            with container.as_default():
                y = core_layers.dense(x, 1, name='my_dense', kernel_initializer=init_ops.ones_initializer())
            self.assertAllEqual(y, [[2.0]])
            self.assertEqual(len(container.variables()), 2)
            with container.as_default():
                core_layers.dense(x, 1, name='my_dense', kernel_initializer=init_ops.ones_initializer())
            self.assertEqual(len(container.variables()), 2)

    @test_util.run_in_graph_and_eager_modes
    def testInitFromNonTensorValue(self):
        if False:
            while True:
                i = 10
        v = variable_scope.get_variable('v4', initializer=4, dtype=dtypes.int32)
        self.evaluate(variables_lib.variables_initializer([v]))
        self.assertAllClose(self.evaluate(v.value()), 4)
        w = variable_scope.get_variable('w4', initializer=numpy.array([1, 2, 3]), dtype=dtypes.int64)
        self.evaluate(variables_lib.variables_initializer([w]))
        self.assertAllClose(self.evaluate(w.value()), [1, 2, 3])
        error = ValueError if context.executing_eagerly() else TypeError
        with self.assertRaises(error):
            variable_scope.get_variable('x4', initializer={})

    @test_util.run_in_graph_and_eager_modes
    def testInitFromNonInitializer(self):
        if False:
            return 10
        types = [dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.int64, dtypes.bool]
        for (i, dtype) in enumerate(types):
            x = variable_scope.get_variable(name='xx%d' % i, shape=(3, 4), dtype=dtype)
            y = variable_scope.get_variable(name='yy%d' % i, shape=(3, 4), dtype=dtype, initializer=init_ops.zeros_initializer(dtype=dtype))
            self.evaluate(variables_lib.global_variables_initializer())
            self.assertAllEqual(self.evaluate(x.value()), self.evaluate(y.value()))

    @test_util.run_deprecated_v1
    def testVarScopeCachingDevice(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            caching_device = '/job:moo'
            with variable_scope.variable_scope('tower'):
                with variable_scope.variable_scope('caching', caching_device=caching_device):
                    v = variable_scope.get_variable('v', [])
                    self.assertTrue(v.value().device.startswith(caching_device))
                    with variable_scope.variable_scope('child'):
                        v2 = variable_scope.get_variable('v', [])
                        self.assertTrue(v2.value().device.startswith(caching_device))
                    with variable_scope.variable_scope('not_cached', caching_device=''):
                        v2_not_cached = variable_scope.get_variable('v', [])
                        self.assertFalse(v2_not_cached.value().device.startswith(caching_device))
                    with variable_scope.variable_scope('not_cached_identity_device', caching_device=lambda op: op.device):
                        v2_identity_device = variable_scope.get_variable('v', [])
                        self.assertFalse(v2_identity_device.value().device.startswith(caching_device))
                    with variable_scope.variable_scope('we_will_do_it_live') as vs_live:
                        vs_live.set_caching_device('/job:live')
                        v_live = variable_scope.get_variable('v', [])
                        self.assertTrue(v_live.value().device.startswith('/job:live'))
                v_tower = variable_scope.get_variable('v', [])
                self.assertFalse(v_tower.value().device.startswith(caching_device))

    @test_util.run_in_graph_and_eager_modes
    def testVarScopeRegularizer(self):
        if False:
            return 10
        init = init_ops.constant_initializer(0.3)

        def regularizer1(v):
            if False:
                print('Hello World!')
            return math_ops.reduce_mean(v) + 0.1

        def regularizer2(v):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.reduce_mean(v) + 0.2
        with variable_scope.variable_scope('tower3', regularizer=regularizer1) as tower:
            with variable_scope.variable_scope('foo', initializer=init):
                v = variable_scope.get_variable('v', [])
                self.evaluate(variables_lib.variables_initializer([v]))
                losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
                self.assertEqual(1, len(losses))
                self.assertAllClose(self.evaluate(losses[0]), 0.4)
            with variable_scope.variable_scope(tower, initializer=init) as vs:
                u = variable_scope.get_variable('u', [])
                vs.set_regularizer(regularizer2)
                w = variable_scope.get_variable('w', [])
                x = variable_scope.get_variable('x', [], regularizer=variable_scope.no_regularizer)
                with variable_scope.variable_scope('baz', regularizer=variable_scope.no_regularizer):
                    y = variable_scope.get_variable('y', [])
                vs.set_regularizer(variable_scope.no_regularizer)
                z = variable_scope.get_variable('z', [])
                losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
                self.assertEqual(3, len(losses))
                self.evaluate(variables_lib.variables_initializer([u, w, x, y, z]))
                self.assertAllClose(self.evaluate(losses[0]), 0.4)
                self.assertAllClose(self.evaluate(losses[1]), 0.4)
                self.assertAllClose(self.evaluate(losses[2]), 0.5)
            with variable_scope.variable_scope('foo', reuse=True):
                if not context.executing_eagerly():
                    v = variable_scope.get_variable('v', [])
                    losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
                    self.assertEqual(3, len(losses))

    @test_util.run_in_graph_and_eager_modes
    def testInitializeFromValue(self):
        if False:
            print('Hello World!')
        init = constant_op.constant(0.1)
        w = variable_scope.get_variable('v', initializer=init)
        self.evaluate(variables_lib.variables_initializer([w]))
        self.assertAllClose(self.evaluate(w.value()), 0.1)
        with self.assertRaisesRegex(ValueError, 'shape'):
            variable_scope.get_variable('u', [1], initializer=init)
        with variable_scope.variable_scope('foo', initializer=init):
            v = variable_scope.get_variable('v')
            self.evaluate(variables_lib.variables_initializer([v]))
            self.assertAllClose(self.evaluate(v.value()), 0.1)
        init = constant_op.constant(1, dtype=dtypes.int32)
        t = variable_scope.get_variable('t', initializer=init)
        self.assertEqual(t.dtype.base_dtype, dtypes.int32)
        with self.assertRaisesRegex(ValueError, "don't match"):
            variable_scope.get_variable('s', initializer=init, dtype=dtypes.float64)

    @test_util.run_deprecated_v1
    def testControlDeps(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            v0 = variable_scope.get_variable('v0', [1], initializer=init_ops.constant_initializer(0))
            with ops.control_dependencies([v0.value()]):
                v1 = variable_scope.get_variable('v1', [1], initializer=init_ops.constant_initializer(1))
                add = v1 + v0
            with self.assertRaisesRegex(errors.OpError, 'uninitialized'):
                self.evaluate(v0)
            self.evaluate(v1.initializer)
            self.assertEqual(1, self.evaluate(v1))
            with self.assertRaisesRegex(errors.OpError, 'uninitialized'):
                self.evaluate(v0)
            with self.assertRaisesRegex(errors.OpError, 'uninitialized'):
                self.evaluate(add)
            self.evaluate(v0.initializer)
            self.evaluate(add)

    @test_util.run_deprecated_v1
    def testEnableResourceVariables(self):
        if False:
            return 10
        old = resource_variables_toggle._DEFAULT_USE_RESOURCE
        try:
            resource_variables_toggle.enable_resource_variables()
            self.assertIsInstance(variable_v1.VariableV1(1.0), resource_variable_ops.ResourceVariable)
            resource_variables_toggle.disable_resource_variables()
            self.assertNotIsInstance(variable_v1.VariableV1(1.0), resource_variable_ops.ResourceVariable)
        finally:
            resource_variables_toggle._DEFAULT_USE_RESOURCE = old

    @test_util.run_deprecated_v1
    def testControlFlow(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            v0 = variable_scope.get_variable('v0', [], initializer=init_ops.constant_initializer(0))
            var_dict = {}

            def var_in_then_clause():
                if False:
                    print('Hello World!')
                v1 = variable_scope.get_variable('v1', [1], initializer=init_ops.constant_initializer(1))
                var_dict['v1'] = v1
                return v1 + v0

            def var_in_else_clause():
                if False:
                    for i in range(10):
                        print('nop')
                v2 = variable_scope.get_variable('v2', [1], initializer=init_ops.constant_initializer(2))
                var_dict['v2'] = v2
                return v2 + v0
            add = cond.cond(math_ops.less(v0, 10), var_in_then_clause, var_in_else_clause)
            v1 = var_dict['v1']
            v2 = var_dict['v2']
            self.evaluate(v1.initializer)
            self.assertEqual([1], self.evaluate(v1))
            self.evaluate(v2.initializer)
            self.assertEqual([2], self.evaluate(v2))
            with self.assertRaisesRegex(errors.OpError, 'uninitialized'):
                self.evaluate(v0)
            with self.assertRaisesRegex(errors.OpError, 'uninitialized'):
                self.evaluate(add)
            self.evaluate(v0.initializer)
            self.evaluate(add)

    @test_util.run_in_graph_and_eager_modes
    def testGetVariableScope(self):
        if False:
            print('Hello World!')
        init = init_ops.constant_initializer(0.3)
        with variable_scope.variable_scope('bar'):
            new_init1 = variable_scope.get_variable_scope().initializer
            self.assertEqual(new_init1, None)
            variable_scope.get_variable_scope().set_initializer(init)
            v = variable_scope.get_variable('v', [])
            self.evaluate(variables_lib.variables_initializer([v]))
            self.assertAllClose(self.evaluate(v.value()), 0.3)
            if not context.executing_eagerly():
                variable_scope.get_variable_scope().reuse_variables()
                with self.assertRaises(ValueError):
                    variable_scope.get_variable('w', [1])
        new_init = variable_scope.get_variable_scope().initializer
        self.assertEqual(new_init, None)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScope(self):
        if False:
            i = 10
            return i + 15
        with variable_scope.variable_scope('tower4') as tower:
            self.assertEqual(tower.name, 'tower4')
            with ops.name_scope('scope') as sc:
                self.assertEqual(sc, 'tower4/scope/')
        with variable_scope.variable_scope('tower5'):
            with variable_scope.variable_scope('bar') as bar:
                self.assertEqual(bar.name, 'tower5/bar')
                with ops.name_scope('scope') as sc:
                    self.assertEqual(sc, 'tower5/bar/scope/')
        with variable_scope.variable_scope('tower6'):
            with variable_scope.variable_scope(tower, reuse=True) as tower_shared:
                self.assertEqual(tower_shared.name, 'tower4')
                with ops.name_scope('scope') as sc:
                    self.assertEqual(sc, 'tower6/tower4/scope/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeNameScope(self):
        if False:
            return 10
        with ops.name_scope('testVarScopeNameScope1'):
            with variable_scope.variable_scope('tower') as tower:
                with ops.name_scope('scope2') as sc2:
                    self.assertEqual(sc2, 'testVarScopeNameScope1/tower/scope2/')
            if not context.executing_eagerly():
                with variable_scope.variable_scope(tower):
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'testVarScopeNameScope1/tower_1/scope2/')
                with variable_scope.variable_scope('tower'):
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'testVarScopeNameScope1/tower_2/scope2/')
        with ops.name_scope('testVarScopeNameScope2'):
            with variable_scope.variable_scope('tower'):
                with ops.name_scope('scope2') as sc2:
                    self.assertEqual(sc2, 'testVarScopeNameScope2/tower/scope2/')
            if not context.executing_eagerly():
                with variable_scope.variable_scope(tower):
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'testVarScopeNameScope2/tower_1/scope2/')
        root_var_scope = variable_scope.get_variable_scope()
        with ops.name_scope('testVarScopeNameScope3'):
            with variable_scope.variable_scope(root_var_scope):
                with ops.name_scope('scope2') as sc2:
                    self.assertEqual(sc2, 'testVarScopeNameScope3/scope2/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeOriginalNameScope(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            with ops.name_scope('scope1'):
                with variable_scope.variable_scope('tower') as tower:
                    self.assertEqual(tower.original_name_scope, 'scope1/tower/')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'scope1/tower/scope2/')
            with ops.name_scope('scope2'):
                with variable_scope.variable_scope(tower) as tower1:
                    self.assertEqual(tower1.original_name_scope, 'scope1/tower/')
                    with ops.name_scope('foo') as sc2:
                        self.assertEqual(sc2, 'scope2/tower/foo/')
                with ops.name_scope(tower.original_name_scope):
                    with ops.name_scope('bar') as sc3:
                        self.assertEqual(sc3, 'scope1/tower/bar/')
            with ops.name_scope('scope2'):
                with variable_scope.variable_scope(tower):
                    with ops.name_scope(tower.original_name_scope):
                        with ops.name_scope('bar') as sc3:
                            self.assertEqual(sc3, 'scope1/tower/bar_1/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeObjectReuse(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            vs = None
            with variable_scope.variable_scope('jump', reuse=True) as scope:
                vs = scope
            with variable_scope.variable_scope(vs) as jump:
                self.assertTrue(jump.reuse)
            with variable_scope.variable_scope(vs, reuse=True) as jump_reuse:
                self.assertTrue(jump_reuse.reuse)
            with variable_scope.variable_scope(vs, reuse=False) as jump_no_reuse:
                self.assertTrue(jump_no_reuse.reuse)
            with variable_scope.variable_scope('jump', reuse=False) as scope:
                vs = scope
            with variable_scope.variable_scope(vs) as jump:
                self.assertFalse(jump.reuse)
            with variable_scope.variable_scope(vs, reuse=True) as jump_reuse:
                self.assertTrue(jump_reuse.reuse)
            with variable_scope.variable_scope(vs, reuse=False) as jump_no_reuse:
                self.assertFalse(jump_no_reuse.reuse)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeGetOrCreateReuse(self):
        if False:
            while True:
                i = 10
        with self.cached_session():

            def test_value(value):
                if False:
                    i = 10
                    return i + 15
                x = constant_op.constant(value)
                with variable_scope.variable_scope('testVarScopeGetOrCreateReuse_bar', reuse=variable_scope.AUTO_REUSE):
                    _ = state_ops.assign(variable_scope.get_variable('var', []), x)
                with variable_scope.variable_scope('testVarScopeGetOrCreateReuse_bar', reuse=variable_scope.AUTO_REUSE):
                    _ = variable_scope.get_variable('var', [])
                self.assertEqual(value, self.evaluate(x))
            test_value(42.0)
            test_value(13.0)
            test_value(17.0)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarOpScope(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            with ops.name_scope('testVarOpScope1'):
                with variable_scope.variable_scope('tower', 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'tower/w:0')
                    with ops.name_scope('testVarOpScope2') as sc2:
                        self.assertEqual(sc2, 'testVarOpScope1/tower/testVarOpScope2/')
                with variable_scope.variable_scope('tower', 'default', []):
                    with self.assertRaises(ValueError):
                        variable_scope.get_variable('w', [])
                    with ops.name_scope('testVarOpScope2') as sc2:
                        self.assertEqual(sc2, 'testVarOpScope1/tower_1/testVarOpScope2/')
            with ops.name_scope('testVarOpScope2'):
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'default/w:0')
                    with ops.name_scope('testVarOpScope2') as sc2:
                        self.assertEqual(sc2, 'testVarOpScope2/default/testVarOpScope2/')
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'default_1/w:0')
                    with ops.name_scope('testVarOpScope2') as sc2:
                        self.assertEqual(sc2, 'testVarOpScope2/default_1/testVarOpScope2/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarOpScopeUniqueNamesInterleavedSubstringScopes(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            with variable_scope.variable_scope(None, 'defaultScope1'):
                with variable_scope.variable_scope(None, 'layer'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'defaultScope1/layer/w:0')
            with variable_scope.variable_scope(None, 'defaultScope1'):
                with variable_scope.variable_scope(None, 'layer'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'defaultScope1_1/layer/w:0')
            with variable_scope.variable_scope(None, 'defaultScope'):
                with variable_scope.variable_scope(None, 'layer'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'defaultScope/layer/w:0')
            with variable_scope.variable_scope(None, 'defaultScope1'):
                with variable_scope.variable_scope(None, 'layer'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'defaultScope1_2/layer/w:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarOpScopeUniqueNamesWithJump(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            with variable_scope.variable_scope('default') as default:
                with variable_scope.variable_scope(None, 'layer'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'default/layer/w:0')
                with variable_scope.variable_scope(None, 'layer'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'default/layer_1/w:0')
                with variable_scope.variable_scope(default):
                    pass
                with variable_scope.variable_scope(None, 'layer'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'default/layer_2/w:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarOpScopeReuse(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            with variable_scope.variable_scope('outer') as outer:
                with variable_scope.variable_scope('tower', 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/tower/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/tower/scope2/')
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/default/scope2/')
            with variable_scope.variable_scope(outer, reuse=True) as outer:
                with variable_scope.variable_scope('tower', 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/tower/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_1/tower/scope2/')
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_1/default/scope2/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeGetVar(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            with variable_scope.variable_scope('root'):
                with variable_scope.variable_scope('towerA') as tower_a:
                    va = variable_scope.get_variable('v', [1])
                    self.assertEqual(va.name, 'root/towerA/v:0')
                with variable_scope.variable_scope(tower_a, reuse=True):
                    va2 = variable_scope.get_variable('v', [1])
                    self.assertIs(va2, va)
                with variable_scope.variable_scope('towerB'):
                    vb = variable_scope.get_variable('v', [1])
                    self.assertEqual(vb.name, 'root/towerB/v:0')
                with self.assertRaises(ValueError):
                    with variable_scope.variable_scope('towerA'):
                        va2 = variable_scope.get_variable('v', [1])
                with variable_scope.variable_scope('towerA', reuse=True):
                    va2 = variable_scope.get_variable('v', [1])
                    self.assertIs(va2, va)
                with variable_scope.variable_scope('foo'):
                    with variable_scope.variable_scope('bar'):
                        v = variable_scope.get_variable('v', [1])
                        self.assertEqual(v.name, 'root/foo/bar/v:0')
                        with variable_scope.variable_scope(tower_a, reuse=True):
                            va3 = variable_scope.get_variable('v', [1])
                            self.assertIs(va, va3)
                with self.assertRaises(ValueError):
                    with variable_scope.variable_scope(tower_a, reuse=True):
                        with variable_scope.variable_scope('baz'):
                            variable_scope.get_variable('v', [1])
                with self.assertRaises(ValueError) as exc:
                    with variable_scope.variable_scope(tower_a, reuse=True):
                        variable_scope.get_variable('v', [2])
                self.assertEqual('shape' in str(exc.exception), True)
                with self.assertRaises(ValueError) as exc:
                    with variable_scope.variable_scope(tower_a, reuse=True):
                        variable_scope.get_variable('v', [1], dtype=dtypes.int32)
                self.assertEqual('dtype' in str(exc.exception), True)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeOuterScope(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            with variable_scope.variable_scope('outer') as outer:
                pass
            with variable_scope.variable_scope(outer):
                self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/w:0')
                with ops.name_scope('scope2') as sc2:
                    self.assertEqual(sc2, 'outer_1/scope2/')
                with variable_scope.variable_scope('default'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_1/default/scope2/')
            with variable_scope.variable_scope(outer, reuse=True):
                self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/w:0')
                with ops.name_scope('scope2') as sc2:
                    self.assertEqual(sc2, 'outer_2/scope2/')
                with variable_scope.variable_scope('default', reuse=True):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_2/default/scope2/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarScopeNestedOuterScope(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            with variable_scope.variable_scope('outer') as outer:
                with variable_scope.variable_scope(outer):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/outer/scope2/')
                with variable_scope.variable_scope('default'):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/default/scope2/')
                with variable_scope.variable_scope(outer, reuse=True):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/outer_1/scope2/')
                with variable_scope.variable_scope('default', reuse=True):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/default_1/scope2/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarOpScopeReuseParam(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            with variable_scope.variable_scope('outer') as outer:
                with variable_scope.variable_scope('tower', 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/tower/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/tower/scope2/')
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/default/scope2/')
            with variable_scope.variable_scope(outer) as outer:
                with variable_scope.variable_scope('tower', 'default', reuse=True):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/tower/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_1/tower/scope2/')
                outer.reuse_variables()
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_1/default/scope2/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarOpScopeReuseError(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            with self.assertRaises(ValueError):
                with variable_scope.variable_scope(None, 'default', reuse=True):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/tower/w:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarOpScopeOuterScope(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            with variable_scope.variable_scope('outer') as outer:
                pass
            with variable_scope.variable_scope(outer, 'default', []):
                self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/w:0')
                with ops.name_scope('scope2') as sc2:
                    self.assertEqual(sc2, 'outer_1/scope2/')
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_1/default/scope2/')
            with variable_scope.variable_scope(outer, 'default', reuse=True):
                self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/w:0')
                with ops.name_scope('scope2') as sc2:
                    self.assertEqual(sc2, 'outer_2/scope2/')
                outer.reuse_variables()
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_2/default/scope2/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVarOpScopeNestedOuterScope(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            with variable_scope.variable_scope('outer') as outer:
                with variable_scope.variable_scope(outer, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/outer/scope2/')
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer/default/scope2/')
            with variable_scope.variable_scope(outer, 'default', reuse=True):
                self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/w:0')
                with ops.name_scope('scope2') as sc2:
                    self.assertEqual(sc2, 'outer_1/scope2/')
                with variable_scope.variable_scope(None, 'default', []):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    with ops.name_scope('scope2') as sc2:
                        self.assertEqual(sc2, 'outer_1/default/scope2/')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testBasicWhenAuxiliaryNameScopeIsFalse(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            with variable_scope.variable_scope('scope', auxiliary_name_scope=False) as scope:
                self.assertEqual(scope.original_name_scope, '')
                self.assertEqual(variable_scope.get_variable('w', []).name, 'scope/w:0')
                self.assertEqual(constant_op.constant([], name='c').name, 'c:0')
            with variable_scope.variable_scope(scope, auxiliary_name_scope=False):
                self.assertEqual(scope.original_name_scope, '')
                self.assertEqual(variable_scope.get_variable('w1', []).name, 'scope/w1:0')
                self.assertEqual(constant_op.constant([], name='c1').name, 'c1:0')
            with ops.name_scope('scope'):
                self.assertEqual(constant_op.constant([], name='c').name, 'scope/c:0')
            with variable_scope.variable_scope('outer'):
                with variable_scope.variable_scope('inner', auxiliary_name_scope=False) as inner:
                    self.assertEqual(inner.original_name_scope, 'outer/')
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/inner/w:0')
                    self.assertEqual(constant_op.constant([], name='c').name, 'outer/c:0')
                with variable_scope.variable_scope(inner, auxiliary_name_scope=False) as inner1:
                    self.assertEqual(inner1.original_name_scope, 'outer/')
                    self.assertEqual(variable_scope.get_variable('w1', []).name, 'outer/inner/w1:0')
                    self.assertEqual(constant_op.constant([], name='c1').name, 'outer/c1:0')
                with ops.name_scope('inner'):
                    self.assertEqual(constant_op.constant([], name='c').name, 'outer/inner/c:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testCreatedByDefaultNameWhenAuxiliaryNameScopeIsFalse(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            with variable_scope.variable_scope(None, default_name='default', auxiliary_name_scope=False) as scope:
                self.assertEqual(scope.original_name_scope, '')
                self.assertEqual(variable_scope.get_variable('w', []).name, 'default/w:0')
                self.assertEqual(constant_op.constant([], name='c').name, 'c:0')
            with ops.name_scope('default'):
                self.assertEqual(constant_op.constant([], name='c').name, 'default/c:0')
            with variable_scope.variable_scope('outer'):
                with variable_scope.variable_scope(None, default_name='default', auxiliary_name_scope=False) as inner:
                    self.assertEqual(inner.original_name_scope, 'outer/')
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/default/w:0')
                    self.assertEqual(constant_op.constant([], name='c').name, 'outer/c:0')
                with ops.name_scope('default'):
                    self.assertEqual(constant_op.constant([], name='c').name, 'outer/default/c:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testReenterRootScopeWhenAuxiliaryNameScopeIsFalse(self):
        if False:
            return 10
        with self.cached_session():
            root_scope = variable_scope.get_variable_scope()
            with variable_scope.variable_scope(root_scope, auxiliary_name_scope=False) as scope:
                self.assertEqual(scope.original_name_scope, '')
                self.assertEqual(variable_scope.get_variable('w', []).name, 'w:0')
                self.assertEqual(constant_op.constant([], name='c').name, 'c:0')
            with variable_scope.variable_scope('outer'):
                with variable_scope.variable_scope(root_scope, auxiliary_name_scope=False) as inner:
                    self.assertEqual(inner.original_name_scope, '')
                    self.assertEqual(variable_scope.get_variable('w1', []).name, 'w1:0')
                    self.assertEqual(constant_op.constant([], name='c1').name, 'outer/c1:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testAuxiliaryNameScopeIsInvalid(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            with self.assertRaisesRegex(TypeError, 'auxiliary_name_scope'):
                with variable_scope.variable_scope(None, default_name='scope', auxiliary_name_scope='invalid'):
                    pass
            with self.assertRaisesRegex(TypeError, 'auxiliary_name_scope'):
                with variable_scope.variable_scope('scope', auxiliary_name_scope='invalid'):
                    pass
            with variable_scope.variable_scope('scope') as scope:
                pass
            with self.assertRaisesRegex(TypeError, 'auxiliary_name_scope'):
                with variable_scope.variable_scope(scope, auxiliary_name_scope='invalid'):
                    pass

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testReuseScopeWithoutNameScopeCollision(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            with variable_scope.variable_scope('outer'):
                with variable_scope.variable_scope('inner') as inner:
                    pass
            with variable_scope.variable_scope(inner, auxiliary_name_scope=False) as scope:
                with ops.name_scope(scope.original_name_scope):
                    self.assertEqual(variable_scope.get_variable('w', []).name, 'outer/inner/w:0')
                    self.assertEqual(constant_op.constant([], name='c').name, 'outer/inner/c:0')
                with ops.name_scope('inner'):
                    self.assertEqual(constant_op.constant([], name='c').name, 'inner/c:0')
            with variable_scope.variable_scope('another'):
                with variable_scope.variable_scope(inner, auxiliary_name_scope=False) as scope1:
                    with ops.name_scope(scope1.original_name_scope):
                        self.assertEqual(variable_scope.get_variable('w1', []).name, 'outer/inner/w1:0')
                        self.assertEqual(constant_op.constant([], name='c1').name, 'outer/inner/c1:0')
                    with ops.name_scope('inner'):
                        self.assertEqual(constant_op.constant([], name='c').name, 'another/inner/c:0')

    @test_util.run_in_graph_and_eager_modes
    def testGetLocalVar(self):
        if False:
            return 10
        with variable_scope.variable_scope('outer') as outer:
            with variable_scope.variable_scope(outer, 'default', []):
                local_var = variable_scope.get_local_variable('w', [], collections=['foo'])
                self.assertEqual(local_var.name, 'outer/w:0')
        if not context.executing_eagerly():
            self.assertIn(local_var, ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
            self.assertIn(local_var, ops.get_collection('foo'))
            self.assertNotIn(local_var, ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))
            with variable_scope.variable_scope(outer, 'default', reuse=True):
                self.assertEqual(variable_scope.get_local_variable('w', []).name, 'outer/w:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testSignatureGetVarVsGetLocalVar(self):
        if False:
            i = 10
            return i + 15
        'get_{local,}variable() must take the same list of args.'
        arg_names = tf_inspect.getargspec(variable_scope.get_variable)[0]
        local_arg_names = tf_inspect.getargspec(variable_scope.get_local_variable)[0]
        self.assertEqual(arg_names, local_arg_names)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testGetVarWithDevice(self):
        if False:
            i = 10
            return i + 15
        g = ops.Graph()
        varname_type = []

        def device_func(op):
            if False:
                i = 10
                return i + 15
            if op.type in ['Variable', 'VariableV2', 'VarHandleOp']:
                varname_type.append((op.name, op.get_attr('dtype')))
            return '/device:GPU:0'
        with g.as_default():
            with ops.device(device_func):
                _ = variable_scope.get_variable('x', (100, 200))
                _ = variable_scope.get_variable('y', dtype=dtypes.int64, initializer=numpy.arange(73))
        self.assertEqual(varname_type[0], ('x', dtypes.float32))
        self.assertEqual(varname_type[1], ('y', dtypes.int64))

    @test_util.run_deprecated_v1
    def testGetCollection(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            _ = variable_scope.get_variable('testGetCollection_a', [])
            _ = variable_scope.get_variable('testGetCollection_b', [], trainable=False)
            with variable_scope.variable_scope('testGetCollection_foo_') as scope1:
                _ = variable_scope.get_variable('testGetCollection_a', [])
                _ = variable_scope.get_variable('testGetCollection_b', [], trainable=False)
                self.assertEqual([v.name for v in scope1.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)], ['testGetCollection_foo_/testGetCollection_a:0'])
                self.assertEqual([v.name for v in scope1.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)], ['testGetCollection_foo_/testGetCollection_a:0', 'testGetCollection_foo_/testGetCollection_b:0'])
            with variable_scope.variable_scope('testGetCollection_foo') as scope2:
                _ = variable_scope.get_variable('testGetCollection_a', [])
                _ = variable_scope.get_variable('testGetCollection_b', [], trainable=False)
                self.assertEqual([v.name for v in scope2.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)], ['testGetCollection_foo/testGetCollection_a:0'])
                self.assertEqual([v.name for v in scope2.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)], ['testGetCollection_foo/testGetCollection_a:0', 'testGetCollection_foo/testGetCollection_b:0'])
            scope = variable_scope.get_variable_scope()
            self.assertEqual([v.name for v in scope.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)], ['testGetCollection_a:0', 'testGetCollection_b:0', 'testGetCollection_foo_/testGetCollection_a:0', 'testGetCollection_foo_/testGetCollection_b:0', 'testGetCollection_foo/testGetCollection_a:0', 'testGetCollection_foo/testGetCollection_b:0'])
            self.assertEqual([v.name for v in scope.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)], ['testGetCollection_a:0', 'testGetCollection_foo_/testGetCollection_a:0', 'testGetCollection_foo/testGetCollection_a:0'])

    @test_util.run_deprecated_v1
    def testGetTrainableVariablesWithGetVariable(self):
        if False:
            return 10
        with self.cached_session():
            _ = variable_scope.get_variable('testGetTrainableVariables_a', [])
            with variable_scope.variable_scope('testGetTrainableVariables_foo') as scope:
                _ = variable_scope.get_variable('testGetTrainableVariables_b', [])
                _ = variable_scope.get_variable('testGetTrainableVariables_c', [], trainable=False)
                _ = variable_scope.get_variable('testGetTrainableVariables_d', [], synchronization=variable_scope.VariableSynchronization.ON_READ)
                self.assertEqual([v.name for v in scope.trainable_variables()], ['testGetTrainableVariables_foo/testGetTrainableVariables_b:0'])
                _ = variable_scope.get_variable('testGetTrainableVariables_e', [], synchronization=variable_scope.VariableSynchronization.ON_READ, trainable=True)
                self.assertEqual([v.name for v in scope.trainable_variables()], ['testGetTrainableVariables_foo/testGetTrainableVariables_b:0', 'testGetTrainableVariables_foo/testGetTrainableVariables_e:0'])
                _ = variable_scope.get_variable('testGetTrainableVariables_f', [], synchronization=variable_scope.VariableSynchronization.ON_WRITE)
                self.assertEqual([v.name for v in scope.trainable_variables()], ['testGetTrainableVariables_foo/testGetTrainableVariables_b:0', 'testGetTrainableVariables_foo/testGetTrainableVariables_e:0', 'testGetTrainableVariables_foo/testGetTrainableVariables_f:0'])

    @test_util.run_deprecated_v1
    def testGetTrainableVariablesWithVariable(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            _ = variable_v1.VariableV1(1.0, name='testGetTrainableVariables_a')
            with variable_scope.variable_scope('testGetTrainableVariables_foo') as scope:
                _ = variable_v1.VariableV1(1.0, name='testGetTrainableVariables_b')
                _ = variable_v1.VariableV1(1.0, name='testGetTrainableVariables_c', trainable=False)
                _ = variable_v1.VariableV1(1.0, name='testGetTrainableVariables_d', synchronization=variable_scope.VariableSynchronization.ON_READ)
                self.assertEqual([v.name for v in scope.trainable_variables()], ['testGetTrainableVariables_foo/testGetTrainableVariables_b:0'])
                _ = variable_v1.VariableV1(1.0, name='testGetTrainableVariables_e', synchronization=variable_scope.VariableSynchronization.ON_READ, trainable=True)
                self.assertEqual([v.name for v in scope.trainable_variables()], ['testGetTrainableVariables_foo/testGetTrainableVariables_b:0', 'testGetTrainableVariables_foo/testGetTrainableVariables_e:0'])
                _ = variable_v1.VariableV1(1.0, name='testGetTrainableVariables_f', synchronization=variable_scope.VariableSynchronization.ON_WRITE)
                self.assertEqual([v.name for v in scope.trainable_variables()], ['testGetTrainableVariables_foo/testGetTrainableVariables_b:0', 'testGetTrainableVariables_foo/testGetTrainableVariables_e:0', 'testGetTrainableVariables_foo/testGetTrainableVariables_f:0'])

    @test_util.run_deprecated_v1
    def testGetGlobalVariables(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            _ = variable_scope.get_variable('testGetGlobalVariables_a', [])
            with variable_scope.variable_scope('testGetGlobalVariables_foo') as scope:
                _ = variable_scope.get_variable('testGetGlobalVariables_b', [])
                self.assertEqual([v.name for v in scope.global_variables()], ['testGetGlobalVariables_foo/testGetGlobalVariables_b:0'])

    @test_util.run_deprecated_v1
    def testGetLocalVariables(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            _ = variable_scope.get_variable('a', [], collections=[ops.GraphKeys.LOCAL_VARIABLES])
            with variable_scope.variable_scope('foo') as scope:
                _ = variable_scope.get_variable('b', [], collections=[ops.GraphKeys.LOCAL_VARIABLES])
                _ = variable_scope.get_variable('c', [])
                self.assertEqual([v.name for v in scope.local_variables()], ['foo/b:0'])

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testGetVariableWithRefDtype(self):
        if False:
            print('Hello World!')
        v = variable_scope.get_variable('v', shape=[3, 4], dtype=dtypes.float32)
        _ = variable_scope.get_variable('w', shape=[5, 6], dtype=v.dtype)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testGetVariableWithInitializerWhichTakesNoArgs(self):
        if False:
            i = 10
            return i + 15
        v = variable_scope.get_variable('foo', initializer=lambda : [2])
        self.assertEqual(v.name, 'foo:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testGetVariableWithInitializerWhichTakesOptionalArgs(self):
        if False:
            i = 10
            return i + 15
        v = variable_scope.get_variable('foo', initializer=lambda x=True: [2])
        self.assertEqual(v.name, 'foo:0')

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testGetVariableWithInitializerWhichTakesUnprovidedArgsAndNoShape(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, "The initializer passed is not valid. It should be a callable with no arguments and the shape should not be provided or an instance of `tf.keras.initializers.*' and `shape` should be fully defined."):
            variable_scope.get_variable('foo', initializer=lambda x: [2])

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testTwoGraphs(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                i = 10
                return i + 15
            g1 = ops.Graph()
            g2 = ops.Graph()
            with g1.as_default():
                with g2.as_default():
                    with variable_scope.variable_scope('_'):
                        pass
        self.assertRaisesRegex(ValueError, "'_' is not a valid (?:root )?scope name", f)

def axis0_into1_partitioner(shape=None, **unused_kwargs):
    if False:
        for i in range(10):
            print('nop')
    part = [1] * len(shape)
    return part

def axis0_into2_partitioner(shape=None, **unused_kwargs):
    if False:
        for i in range(10):
            print('nop')
    part = [1] * len(shape)
    part[0] = 2
    return part

def axis0_into3_partitioner(shape=None, **unused_kwargs):
    if False:
        while True:
            i = 10
    part = [1] * len(shape)
    part[0] = 3
    return part

class VariableScopeWithPartitioningTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testResultNameMatchesRequested(self):
        if False:
            print('Hello World!')
        with variable_scope.variable_scope('scope0', partitioner=axis0_into2_partitioner):
            v = variable_scope.get_variable('name0', shape=(3, 1, 1))
            self.assertEqual(v.name, 'scope0/name0')
            v_concat = v.as_tensor()
            self.assertEqual(v_concat.name, 'scope0/name0:0')
            variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertIn('scope0/name0/part_0:0', [x.name for x in variables])
            self.assertIn('scope0/name0/part_1:0', [x.name for x in variables])
            self.assertNotIn('scope0/name0/part_2:0', [x.name for x in variables])

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testBreaksIfPartitioningChanges(self):
        if False:
            print('Hello World!')
        with variable_scope.variable_scope('scope0', partitioner=axis0_into2_partitioner):
            variable_scope.get_variable('name0', shape=(3, 1, 1))
        with variable_scope.variable_scope('scope0', partitioner=axis0_into3_partitioner, reuse=True):
            with self.assertRaisesRegex(ValueError, 'Trying to reuse partitioned variable .* but specified partitions .* and found partitions .*'):
                variable_scope.get_variable('name0', shape=(3, 1, 1))
        with variable_scope.variable_scope('scope0', partitioner=axis0_into1_partitioner, reuse=True):
            with self.assertRaisesRegex(ValueError, 'Trying to reuse partitioned variable .* but specified partitions .* and found partitions .*'):
                variable_scope.get_variable('name0', shape=(3, 1, 1))

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testReturnsExistingConcatenatedValueIfReuse(self):
        if False:
            for i in range(10):
                print('nop')
        with variable_scope.variable_scope('scope0', partitioner=axis0_into2_partitioner):
            v_concat = variable_scope.get_variable('name0', shape=(3, 1, 1))
            variable_scope.get_variable_scope().reuse_variables()
            v_concat_2 = variable_scope.get_variable('name0', shape=(3, 1, 1))
            self.assertEqual(v_concat, v_concat_2)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testAllowsReuseWithoutPartitioner(self):
        if False:
            while True:
                i = 10
        with variable_scope.variable_scope('scope0', partitioner=axis0_into2_partitioner):
            v = variable_scope.get_variable('name0', shape=(3, 1, 1))
        with variable_scope.variable_scope('scope0', reuse=True):
            v_reused = variable_scope.get_variable('name0')
        self.assertIs(v, v_reused)

    def testNoReuseInEagerByDefault(self):
        if False:
            for i in range(10):
                print('nop')
        with context.eager_mode():
            with variable_scope.variable_scope('scope0', partitioner=axis0_into2_partitioner):
                v1 = variable_scope.get_variable('name0', shape=(3, 1, 1))
                v2 = variable_scope.get_variable('name0', shape=(3, 1, 1))
                self.assertIsNot(v1, v2)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testPropagatePartitionerOnReopening(self):
        if False:
            while True:
                i = 10
        with variable_scope.variable_scope('scope0', partitioner=axis0_into2_partitioner) as vs:
            self.assertEqual(axis0_into2_partitioner, vs.partitioner)
            with variable_scope.variable_scope(vs) as vs1:
                self.assertEqual(axis0_into2_partitioner, vs1.partitioner)

    @test_util.run_deprecated_v1
    def testScalarIgnoresPartitioner(self):
        if False:
            while True:
                i = 10
        with variable_scope.variable_scope('scope0', partitioner=axis0_into2_partitioner):
            v = variable_scope.get_variable('name0', shape=())
            self.assertEqual(v.name, 'scope0/name0:0')
            variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertIn('scope0/name0:0', [x.name for x in variables])

    def _testPartitionConcatenatesAlongCorrectAxis(self, use_resource):
        if False:
            while True:
                i = 10

        def _part_axis_0(**unused_kwargs):
            if False:
                i = 10
                return i + 15
            return (2, 1, 1)

        def _part_axis_1(**unused_kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return (1, 2, 1)
        with variable_scope.variable_scope('root', use_resource=use_resource):
            v0 = variable_scope.get_variable('n0', shape=(2, 2, 2), partitioner=_part_axis_0)
            v1 = variable_scope.get_variable('n1', shape=(2, 2, 2), partitioner=_part_axis_1)
        self.assertEqual(v0.get_shape(), (2, 2, 2))
        self.assertEqual(v1.get_shape(), (2, 2, 2))
        n0_0 = list(v0)[0]
        n0_1 = list(v0)[1]
        self.assertEqual(n0_0.get_shape(), (1, 2, 2))
        self.assertEqual(n0_1.get_shape(), (1, 2, 2))
        n1_0 = list(v1)[0]
        n1_1 = list(v1)[1]
        self.assertEqual(n1_0.get_shape(), (2, 1, 2))
        self.assertEqual(n1_1.get_shape(), (2, 1, 2))

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testPartitionConcatenatesAlongCorrectAxis(self):
        if False:
            for i in range(10):
                print('nop')
        self._testPartitionConcatenatesAlongCorrectAxis(use_resource=False)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testPartitionConcatenatesAlongCorrectAxisResource(self):
        if False:
            for i in range(10):
                print('nop')
        self._testPartitionConcatenatesAlongCorrectAxis(use_resource=True)

    def testPartitionConcatenatesAlongCorrectAxisResourceInEager(self):
        if False:
            return 10
        with context.eager_mode():
            self._testPartitionConcatenatesAlongCorrectAxis(use_resource=True)

class VariableScopeWithCustomGetterTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testNonCallableGetterFails(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'custom_getter .* not callable:'):
            with variable_scope.variable_scope('scope0', custom_getter=3):
                variable_scope.get_variable('name0')
        with self.assertRaisesRegex(ValueError, 'custom_getter .* not callable:'):
            variable_scope.get_variable('name0', custom_getter=3)

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testNoSideEffectsWithIdentityCustomGetter(self):
        if False:
            print('Hello World!')
        called = [0]

        def custom_getter(getter, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            called[0] += 1
            return getter(*args, **kwargs)
        with variable_scope.variable_scope('scope', custom_getter=custom_getter) as scope:
            v = variable_scope.get_variable('v', [1])
        with variable_scope.variable_scope(scope, reuse=True):
            v2 = variable_scope.get_variable('v', [1])
        with variable_scope.variable_scope('new_scope') as new_scope:
            v3 = variable_scope.get_variable('v3', [1])
        with variable_scope.variable_scope(new_scope, reuse=True, custom_getter=custom_getter):
            v4 = variable_scope.get_variable('v3', [1])
        self.assertIs(v, v2)
        self.assertIs(v3, v4)
        self.assertEqual(3, called[0])

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testSynchronizationAndAggregationWithCustomGetter(self):
        if False:
            while True:
                i = 10
        called = [0]
        synchronization = variable_scope.VariableSynchronization.AUTO
        aggregation = variable_scope.VariableAggregation.NONE

        def custom_getter(getter, *args, **kwargs):
            if False:
                return 10
            called[0] += 1
            self.assertEqual(kwargs['synchronization'], synchronization)
            self.assertEqual(kwargs['aggregation'], aggregation)
            return getter(*args, **kwargs)
        with variable_scope.variable_scope('scope', custom_getter=custom_getter):
            variable_scope.get_variable('v', [1])
        self.assertEqual(1, called[0])
        with variable_scope.variable_scope('scope', custom_getter=custom_getter):
            synchronization = variable_scope.VariableSynchronization.ON_READ
            aggregation = variable_scope.VariableAggregation.MEAN
            variable_scope.get_variable('v1', [1], synchronization=synchronization, aggregation=aggregation)
        self.assertEqual(2, called[0])

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testCustomGetterWithReuse(self):
        if False:
            for i in range(10):
                print('nop')

        def custom_getter(getter, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            var = getter(*args, **kwargs)
            if kwargs['reuse']:
                return array_ops.identity(var, name='reused')
            else:
                return array_ops.identity(var, name='not_reused')
        with variable_scope.variable_scope('scope', custom_getter=custom_getter) as scope:
            v = variable_scope.get_variable('v', [1])
        with variable_scope.variable_scope(scope, reuse=True):
            v2 = variable_scope.get_variable('v', [1])
        self.assertEqual(v.name, 'not_reused:0')
        self.assertEqual(v2.name, 'reused:0')

    @test_util.run_deprecated_v1
    def testGetterThatCreatesTwoVariablesAndSumsThem(self):
        if False:
            while True:
                i = 10

        def custom_getter(getter, name, *args, **kwargs):
            if False:
                print('Hello World!')
            g_0 = getter('%s/0' % name, *args, **kwargs)
            g_1 = getter('%s/1' % name, *args, **kwargs)
            with ops.name_scope('custom_getter'):
                return g_0 + g_1
        with variable_scope.variable_scope('scope', custom_getter=custom_getter):
            v = variable_scope.get_variable('v', [1, 2, 3])
        self.assertEqual([1, 2, 3], v.get_shape())
        true_vars = variables_lib.trainable_variables()
        self.assertEqual(2, len(true_vars))
        self.assertEqual('scope/v/0:0', true_vars[0].name)
        self.assertEqual('scope/v/1:0', true_vars[1].name)
        self.assertEqual('custom_getter/add:0', v.name)
        with self.cached_session() as sess:
            variables_lib.global_variables_initializer().run()
            (np_vars, np_v) = self.evaluate([true_vars, v])
            self.assertAllClose(np_v, sum(np_vars))

    @test_util.run_deprecated_v1
    def testNestedCustomGetters(self):
        if False:
            for i in range(10):
                print('nop')

        def sum_getter(getter, name, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            g_0 = getter('%s/sum_0' % name, *args, **kwargs)
            g_1 = getter('%s/sum_1' % name, *args, **kwargs)
            with ops.name_scope('sum_getter'):
                return g_0 + g_1

        def prod_getter(getter, name, *args, **kwargs):
            if False:
                while True:
                    i = 10
            g_0 = getter('%s/prod_0' % name, *args, **kwargs)
            g_1 = getter('%s/prod_1' % name, *args, **kwargs)
            with ops.name_scope('prod_getter'):
                return g_0 * g_1
        with variable_scope.variable_scope('prod_scope', custom_getter=prod_getter):
            with variable_scope.variable_scope('sum_scope', custom_getter=sum_getter):
                with variable_scope.variable_scope('inner_sum_scope', custom_getter=sum_getter):
                    v = variable_scope.get_variable('v', [1, 2, 3])
        self.assertEqual([1, 2, 3], v.get_shape())
        true_vars = variables_lib.trainable_variables()
        self.assertEqual(8, len(true_vars))
        template = 'prod_scope/sum_scope/inner_sum_scope/v/sum_%d/sum_%d/prod_%d:0'
        self.assertEqual(template % (0, 0, 0), true_vars[0].name)
        self.assertEqual(template % (0, 0, 1), true_vars[1].name)
        self.assertEqual(template % (0, 1, 0), true_vars[2].name)
        self.assertEqual(template % (0, 1, 1), true_vars[3].name)
        self.assertEqual(template % (1, 0, 0), true_vars[4].name)
        self.assertEqual(template % (1, 0, 1), true_vars[5].name)
        self.assertEqual(template % (1, 1, 0), true_vars[6].name)
        self.assertEqual(template % (1, 1, 1), true_vars[7].name)
        with self.cached_session() as sess:
            variables_lib.global_variables_initializer().run()
            (np_vars, np_v) = self.evaluate([true_vars, v])
            self.assertAllClose(np_v, np_vars[0] * np_vars[1] + np_vars[2] * np_vars[3] + (np_vars[4] * np_vars[5] + np_vars[6] * np_vars[7]))

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVariableCreator(self):
        if False:
            print('Hello World!')
        variable_names = []

        def creator_a(next_creator, **kwargs):
            if False:
                print('Hello World!')
            variable_names.append(kwargs.get('name', ''))
            return next_creator(**kwargs)

        def creator_b(next_creator, **kwargs):
            if False:
                return 10
            kwargs['name'] = 'forced_name'
            return next_creator(**kwargs)
        with variable_scope.variable_creator_scope(creator_a):
            with variable_scope.variable_creator_scope(creator_b):
                variable_v1.VariableV1(1.0, name='one_name')
        self.assertEqual(variable_names[0], 'forced_name')
        called = [False]

        def creater_c(next_creator, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            called[0] = True
            self.assertEqual(kwargs['synchronization'], variable_scope.VariableSynchronization.ON_WRITE)
            self.assertEqual(kwargs['aggregation'], variable_scope.VariableAggregation.MEAN)
            return next_creator(**kwargs)
        with variable_scope.variable_creator_scope(creater_c):
            variable_scope.get_variable('v', [], synchronization=variable_scope.VariableSynchronization.ON_WRITE, aggregation=variable_scope.VariableAggregation.MEAN)
        self.assertTrue(called[0])

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testVariableCreatorNestingError(self):
        if False:
            print('Hello World!')

        def creator(next_creator, **kwargs):
            if False:
                print('Hello World!')
            return next_creator(**kwargs)
        graph = ops.get_default_graph()
        old_creator_stack = graph._variable_creator_stack
        try:
            scope = variable_scope.variable_creator_scope(creator)
            scope.__enter__()
            with variable_scope.variable_creator_scope(creator):
                with self.assertRaises(RuntimeError):
                    scope.__exit__(None, None, None)
        finally:
            graph._variable_creator_stack = old_creator_stack

class PartitionInfoTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testConstructorChecks(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            variable_scope._PartitionInfo(full_shape=None, var_offset=[0, 1])
        with self.assertRaises(TypeError):
            variable_scope._PartitionInfo(full_shape=[0, 1], var_offset=None)
        with self.assertRaises(TypeError):
            variable_scope._PartitionInfo(full_shape='foo', var_offset=[0, 1])
        with self.assertRaises(TypeError):
            variable_scope._PartitionInfo(full_shape=[0, 1], var_offset='foo')
        with self.assertRaises(ValueError):
            variable_scope._PartitionInfo(full_shape=[0, 1], var_offset=[0])
        with self.assertRaises(ValueError):
            variable_scope._PartitionInfo(full_shape=[1, 1], var_offset=[0, 1])

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testSingleOffset(self):
        if False:
            return 10
        partition_info = variable_scope._PartitionInfo(full_shape=[9, 3], var_offset=[4, 0])
        self.assertEqual(4, partition_info.single_offset([1, 3]))
        partition_info = variable_scope._PartitionInfo(full_shape=[9, 3], var_offset=[0, 0])
        self.assertEqual(0, partition_info.single_offset([9, 3]))

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testSingleSliceDim(self):
        if False:
            while True:
                i = 10
        partition_info = variable_scope._PartitionInfo(full_shape=[9, 3], var_offset=[4, 0])
        with self.assertRaises(TypeError):
            partition_info.single_slice_dim(None)
        with self.assertRaises(ValueError):
            partition_info.single_slice_dim([1, 2, 3])
        with self.assertRaises(ValueError):
            partition_info.single_slice_dim([6, 3])
        with self.assertRaises(ValueError):
            partition_info.single_slice_dim([1, 1])
        partition_info = variable_scope._PartitionInfo(full_shape=[9, 3], var_offset=[0, 0])
        self.assertEqual(1, partition_info.single_slice_dim([9, 2]))
        partition_info = variable_scope._PartitionInfo(full_shape=[9, 3], var_offset=[4, 0])
        self.assertEqual(0, partition_info.single_slice_dim([2, 3]))

class VariableScopeMultithreadedTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testTwoThreadsDisjointScopeEntry(self):
        if False:
            print('Hello World!')

        def thread_fn(i, graph):
            if False:
                i = 10
                return i + 15
            with graph.as_default():
                with variable_scope.variable_scope('foo'):
                    if i == 0:
                        v = variable_scope.get_variable('v', [])
                        self.assertEqual('foo/v:0', v.name)
                    else:
                        with self.assertRaises(ValueError):
                            variable_scope.get_variable('v', [])
        graph = ops.get_default_graph()
        threads = [threading.Thread(target=thread_fn, args=(i, graph)) for i in range(2)]
        threads[0].start()
        threads[0].join()
        threads[1].start()
        threads[1].join()

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testTwoThreadsNestedScopeEntry(self):
        if False:
            i = 10
            return i + 15

        def thread_fn(i, graph, run_event, pause_event):
            if False:
                i = 10
                return i + 15
            with graph.as_default():
                with variable_scope.variable_scope('foo'):
                    if i == 0:
                        v = variable_scope.get_variable('v', [])
                        self.assertEqual('foo/v:0', v.name)
                    else:
                        with self.assertRaises(ValueError):
                            variable_scope.get_variable('v', [])
                    pause_event.set()
                    run_event.wait()
        graph = ops.get_default_graph()
        run_events = [threading.Event() for _ in range(2)]
        pause_events = [threading.Event() for _ in range(2)]
        threads = [threading.Thread(target=thread_fn, args=(i, graph, run_events[i], pause_events[i])) for i in range(2)]
        threads[0].start()
        pause_events[0].wait()
        threads[1].start()
        pause_events[1].wait()
        run_events[0].set()
        run_events[1].set()
        threads[0].join()
        threads[1].join()

    @test_util.run_in_graph_and_eager_modes
    @run_inside_wrap_function_in_eager_mode
    def testReenterMainScope(self):
        if False:
            print('Hello World!')

        def thread_fn(graph, main_thread_scope):
            if False:
                while True:
                    i = 10
            with graph.as_default():
                with variable_scope.variable_scope(main_thread_scope):
                    with variable_scope.variable_scope('foo'):
                        v = variable_scope.get_variable('v', [])
                        self.assertEqual('main/foo/v:0', v.name)
                with variable_scope.variable_scope('bar'):
                    v = variable_scope.get_variable('v', [])
                    self.assertEqual('bar/v:0', v.name)
        graph = ops.get_default_graph()
        with variable_scope.variable_scope('main') as main_thread_scope:
            thread = threading.Thread(target=thread_fn, args=(graph, main_thread_scope))
            thread.start()
            thread.join()
if __name__ == '__main__':
    test.main()