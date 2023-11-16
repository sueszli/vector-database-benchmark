"""Functional test for moving_averages.py."""
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import moving_averages
from tensorflow.python.training import saver as saver_lib

class MovingAveragesTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testAssignMovingAverageWithoutZeroDebias(self):
        if False:
            print('Hello World!')
        var = variables.Variable([10.0, 11.0])
        val = constant_op.constant([1.0, 2.0], dtypes.float32)
        decay = 0.25
        if context.executing_eagerly():
            self.assertAllClose([10.0, 11.0], self.evaluate(var))
            assign = moving_averages.assign_moving_average(var, val, decay, zero_debias=False)
            self.assertAllClose([10.0 * 0.25 + 1.0 * (1.0 - 0.25), 11.0 * 0.25 + 2.0 * (1.0 - 0.25)], self.evaluate(var))
        else:
            assign = moving_averages.assign_moving_average(var, val, decay, zero_debias=False)
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose([10.0, 11.0], self.evaluate(var))
            assign.op.run()
            self.assertAllClose([10.0 * 0.25 + 1.0 * (1.0 - 0.25), 11.0 * 0.25 + 2.0 * (1.0 - 0.25)], self.evaluate(var))

    @test_util.run_in_graph_and_eager_modes
    def testAssignMovingAverage(self):
        if False:
            i = 10
            return i + 15
        var = variables.Variable([0.0, 0.0])
        val = constant_op.constant([1.0, 2.0], dtypes.float32)
        decay = 0.25
        if context.executing_eagerly():
            self.assertAllClose([0.0, 0.0], self.evaluate(var))
            assign = moving_averages.assign_moving_average(var, val, decay)
            self.assertAllClose([1.0 * (1.0 - 0.25) / (1 - 0.25), 2.0 * (1.0 - 0.25) / (1 - 0.25)], self.evaluate(var))
        else:
            assign = moving_averages.assign_moving_average(var, val, decay)
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose([0.0, 0.0], self.evaluate(var))
            assign.op.run()
            self.assertAllClose([1.0 * (1.0 - 0.25) / (1 - 0.25), 2.0 * (1.0 - 0.25) / (1 - 0.25)], self.evaluate(var))

    @test_util.deprecated_graph_mode_only
    def testAssignMovingAverageNewNamingMultipleCalls(self):
        if False:
            for i in range(10):
                print('nop')
        with variable_scope.variable_scope('scope1') as vs1:
            with variable_scope.variable_scope('scope2'):
                var = variables.Variable(1.0, name='Var')
                moving_averages.assign_moving_average(var, 0.0, 0.99)
                moving_averages.assign_moving_average(var, 0.0, 0.99)
        expected_names = ['scope1/scope2/Var:0', 'scope1/scope2/scope1/scope2/Var/biased:0', 'scope1/scope2/scope1/scope2/Var/local_step:0', 'scope1/scope2/scope1/scope2/Var/biased_1:0', 'scope1/scope2/scope1/scope2/Var/local_step_1:0']
        actual_names = [v.name for v in vs1.global_variables()]
        self.assertSetEqual(set(expected_names), set(actual_names))

    @test_util.deprecated_graph_mode_only
    def testAssignMovingAverageNewNamingMultipleCallsWithReuse(self):
        if False:
            return 10
        with variable_scope.variable_scope('scope1') as vs1:
            var = variable_scope.get_variable('Var', shape=[])
            moving_averages.assign_moving_average(var, 0.0, 0.99)
            moving_averages.assign_moving_average(var, 0.0, 0.99)
        with variable_scope.variable_scope(vs1, reuse=True):
            var = variable_scope.get_variable('Var', shape=[])
            moving_averages.assign_moving_average(var, 0.0, 0.99)
            moving_averages.assign_moving_average(var, 0.0, 0.99)

    @test_util.deprecated_graph_mode_only
    def testWeightedMovingAverage(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            decay = 0.5
            weight = array_ops.placeholder(dtypes.float32, [])
            val = array_ops.placeholder(dtypes.float32, [])
            wma = moving_averages.weighted_moving_average(val, decay, weight)
            self.evaluate(variables.global_variables_initializer())
            val_1 = 3.0
            weight_1 = 4.0
            wma_array = sess.run(wma, feed_dict={val: val_1, weight: weight_1})
            numerator_1 = val_1 * weight_1 * (1.0 - decay)
            denominator_1 = weight_1 * (1.0 - decay)
            self.assertAllClose(numerator_1 / denominator_1, wma_array)
            val_2 = 11.0
            weight_2 = 22.0
            wma_array = sess.run(wma, feed_dict={val: val_2, weight: weight_2})
            numerator_2 = numerator_1 * decay + val_2 * weight_2 * (1.0 - decay)
            denominator_2 = denominator_1 * decay + weight_2 * (1.0 - decay)
            self.assertAllClose(numerator_2 / denominator_2, wma_array)

    @test_util.deprecated_graph_mode_only
    def testWeightedMovingAverageBfloat16(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            decay = 0.5
            weight = array_ops.placeholder(dtypes.bfloat16, [])
            val = array_ops.placeholder(dtypes.bfloat16, [])
            wma = moving_averages.weighted_moving_average(val, decay, weight)
            self.evaluate(variables.global_variables_initializer())
            val_1 = 3.0
            weight_1 = 4.0
            wma_array = sess.run(wma, feed_dict={val: val_1, weight: weight_1})
            numerator_1 = val_1 * weight_1 * (1.0 - decay)
            denominator_1 = weight_1 * (1.0 - decay)
            self.assertAllClose(numerator_1 / denominator_1, wma_array)
            val_2 = 11.0
            weight_2 = 22.0
            wma_array = sess.run(wma, feed_dict={val: val_2, weight: weight_2})
            numerator_2 = numerator_1 * decay + val_2 * weight_2 * (1.0 - decay)
            denominator_2 = denominator_1 * decay + weight_2 * (1.0 - decay)
            self.assertAllClose(dtypes._np_bfloat16(numerator_2 / denominator_2), wma_array)

def _Repeat(value, dim):
    if False:
        i = 10
        return i + 15
    if dim == 1:
        return value
    return [value] * dim

class ExponentialMovingAverageTest(test.TestCase):

    def _CheckDecay(self, ema, actual_decay, dim, dynamic_decay_value=None):
        if False:
            return 10

        def _Scale(dk, steps):
            if False:
                for i in range(10):
                    print('nop')
            if ema._zero_debias:
                return 1 - dk ** steps
            else:
                return 1
        tens = _Repeat(10.0, dim)
        thirties = _Repeat(30.0, dim)
        var0 = variables.Variable(tens, name='v0')
        var1 = variables.Variable(thirties, name='v1')
        self.evaluate(variables.global_variables_initializer())
        tensor2 = var0 + var1
        if dynamic_decay_value is not None:
            self.evaluate(ema._decay.assign(dynamic_decay_value))
        update = ema.apply([var0, var1, tensor2])
        avg0 = ema.average(var0)
        avg1 = ema.average(var1)
        avg2 = ema.average(tensor2)
        self.assertItemsEqual([var0, var1], variables.moving_average_variables())
        self.assertNotIn(avg0, variables.trainable_variables())
        self.assertNotIn(avg1, variables.trainable_variables())
        self.assertNotIn(avg2, variables.trainable_variables())
        self.evaluate(variables.global_variables_initializer())
        if dynamic_decay_value is not None:
            self.evaluate(ema._decay.assign(dynamic_decay_value))
        self.assertEqual('v0/ExponentialMovingAverage:0', avg0.name)
        self.assertEqual('v1/ExponentialMovingAverage:0', avg1.name)
        self.assertEqual('add/ExponentialMovingAverage:0', avg2.name)
        self.assertAllClose(tens, self.evaluate(var0))
        self.assertAllClose(thirties, self.evaluate(var1))
        self.assertAllClose(_Repeat(10.0 + 30.0, dim), self.evaluate(tensor2))
        self.assertAllClose(tens, self.evaluate(avg0))
        self.assertAllClose(thirties, self.evaluate(avg1))
        self.assertAllClose(_Repeat(0.0, dim), self.evaluate(avg2))
        self.evaluate(update)
        dk = actual_decay
        expected = _Repeat(10.0 * dk + 10.0 * (1 - dk), dim)
        self.assertAllClose(expected, self.evaluate(avg0))
        expected = _Repeat(30.0 * dk + 30.0 * (1 - dk), dim)
        self.assertAllClose(expected, self.evaluate(avg1))
        expected = _Repeat(0.0 * dk + (10.0 + 30.0) * (1 - dk) / _Scale(dk, 1), dim)
        self.assertAllClose(expected, self.evaluate(avg2))
        self.evaluate(update)
        expected = _Repeat((10.0 * dk + 10.0 * (1 - dk)) * dk + 10.0 * (1 - dk), dim)
        self.assertAllClose(expected, self.evaluate(avg0))
        expected = _Repeat((30.0 * dk + 30.0 * (1 - dk)) * dk + 30.0 * (1 - dk), dim)
        self.assertAllClose(expected, self.evaluate(avg1))
        expected = _Repeat(((0.0 * dk + (10.0 + 30.0) * (1 - dk)) * dk + (10.0 + 30.0) * (1 - dk)) / _Scale(dk, 2), dim)
        self.assertAllClose(expected, self.evaluate(avg2))

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNoNumUpdates_Scalar(self):
        if False:
            while True:
                i = 10
        ema = moving_averages.ExponentialMovingAverage(0.25)
        self._CheckDecay(ema, actual_decay=0.25, dim=1)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNoNumUpdates_Scalar_DynamicDecay(self):
        if False:
            for i in range(10):
                print('nop')
        decay_var = variables.Variable(0.75)
        ema = moving_averages.ExponentialMovingAverage(decay_var)
        self._CheckDecay(ema, actual_decay=0.25, dim=1, dynamic_decay_value=0.25)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNoNumUpdates_Scalar_Debias(self):
        if False:
            for i in range(10):
                print('nop')
        ema = moving_averages.ExponentialMovingAverage(0.25, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.25, dim=1)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNoNumUpdates_Scalar_Debias_DynamicDecay(self):
        if False:
            while True:
                i = 10
        decay_var = variables.Variable(0.75)
        ema = moving_averages.ExponentialMovingAverage(decay_var, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.25, dim=1, dynamic_decay_value=0.25)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNoNumUpdates_Vector(self):
        if False:
            for i in range(10):
                print('nop')
        ema = moving_averages.ExponentialMovingAverage(0.25)
        self._CheckDecay(ema, actual_decay=0.25, dim=5)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNoNumUpdates_Vector_DynamicDecay(self):
        if False:
            print('Hello World!')
        decay_var = variables.Variable(0.75)
        ema = moving_averages.ExponentialMovingAverage(decay_var)
        self._CheckDecay(ema, actual_decay=0.25, dim=5, dynamic_decay_value=0.25)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNoNumUpdates_Vector_Debias(self):
        if False:
            for i in range(10):
                print('nop')
        ema = moving_averages.ExponentialMovingAverage(0.25, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.25, dim=5)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNoNumUpdates_Vector_Debias_DynamicDecay(self):
        if False:
            i = 10
            return i + 15
        decay_var = variables.Variable(0.75)
        ema = moving_averages.ExponentialMovingAverage(decay_var, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.25, dim=5, dynamic_decay_value=0.25)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNumUpdates_Scalar(self):
        if False:
            return 10
        ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1)
        self._CheckDecay(ema, actual_decay=0.181818, dim=1)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNumUpdates_Scalar_DynamicDecay(self):
        if False:
            return 10
        decay_var = variables.Variable(0.75)
        ema = moving_averages.ExponentialMovingAverage(decay_var, num_updates=1)
        self._CheckDecay(ema, actual_decay=0.181818, dim=1, dynamic_decay_value=0.25)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNumUpdates_Scalar_Debias(self):
        if False:
            print('Hello World!')
        ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.181818, dim=1)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNumUpdates_Scalar_Debias_DynamicDecay(self):
        if False:
            print('Hello World!')
        decay_var = variables.Variable(0.75)
        ema = moving_averages.ExponentialMovingAverage(decay_var, num_updates=1, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.181818, dim=1, dynamic_decay_value=0.25)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNumUpdates_Vector(self):
        if False:
            for i in range(10):
                print('nop')
        ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1)
        self._CheckDecay(ema, actual_decay=0.181818, dim=5)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNumUpdates_Vector_DynamicDecay(self):
        if False:
            return 10
        decay_var = variables.Variable(0.75)
        ema = moving_averages.ExponentialMovingAverage(decay_var, num_updates=1)
        self._CheckDecay(ema, actual_decay=0.181818, dim=5, dynamic_decay_value=0.25)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNumUpdates_Vector_Debias(self):
        if False:
            for i in range(10):
                print('nop')
        ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.181818, dim=5)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNumUpdates_Vector_Debias_DynamicDecay(self):
        if False:
            for i in range(10):
                print('nop')
        decay_var = variables.Variable(0.75)
        ema = moving_averages.ExponentialMovingAverage(decay_var, num_updates=1, zero_debias=True)
        self._CheckDecay(ema, actual_decay=0.181818, dim=5, dynamic_decay_value=0.25)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesWithControlDeps(self):
        if False:
            i = 10
            return i + 15
        v0 = variables.Variable(0, name='v0')
        add_to_v0 = v0.assign_add(1)
        v1 = variables.Variable([10.0], name='v1')
        assign_to_v1 = v1.assign([20.0])
        ema = moving_averages.ExponentialMovingAverage(0.25)
        with ops.control_dependencies([add_to_v0]):
            ema_op = ema.apply([v1])
        v1_avg = ema.average(v1)
        self.assertEqual([], v1_avg.initializer.control_inputs)
        self.assertEqual([], v1_avg.value().op.control_inputs)
        self.assertEqual([], v1_avg.value().op.control_inputs)
        self.evaluate(v1_avg.initializer)
        self.evaluate(v0.initializer)
        self.assertEqual([10.0], self.evaluate(v1_avg))
        self.evaluate(assign_to_v1)
        self.evaluate(ema_op)
        self.assertEqual(1, self.evaluate(v0))
        self.assertEqual([17.5], self.evaluate(v1_avg))

    def testBasicEager(self):
        if False:
            return 10
        v0 = variables.Variable(1.0, name='v0')
        v1 = variables.Variable(2.0, name='v1')
        ema = moving_averages.ExponentialMovingAverage(0.25, name='foo')
        op = ema.apply([v0, v1])
        if not context.executing_eagerly():
            self.evaluate(variables.global_variables_initializer())
            self.evaluate(op)
        self.evaluate(v0.assign(2.0))
        self.evaluate(v1.assign(4.0))
        self.evaluate(ema.apply([v0, v1]))
        self.assertEqual('foo', ema.name)
        self.assertEqual('v0/foo', ema.average_name(v0))
        self.assertEqual('v1/foo', ema.average_name(v1))
        self.assertAllEqual(self.evaluate(ema.average(v0)), 1.75)
        self.assertAllEqual(self.evaluate(ema.average(v1)), 3.5)

    def averageVariablesNamesHelper(self, zero_debias):
        if False:
            return 10
        v0 = variables.Variable(10.0, name='v0')
        v1 = variables.Variable(30.0, name='v1')
        v2 = variables.Variable(20.0, name='v2', trainable=False)
        tensor2 = v0 + v1
        ema = moving_averages.ExponentialMovingAverage(0.25, zero_debias=zero_debias, name='foo')
        self.assertEqual('foo', ema.name)
        self.assertEqual('v0/foo', ema.average_name(v0))
        self.assertEqual('v1/foo', ema.average_name(v1))
        self.assertEqual('add/foo', ema.average_name(tensor2))
        ema.apply([v0, v1, tensor2])
        vars_to_restore = ema.variables_to_restore()
        expected_names = [ema.average_name(v0), ema.average_name(v1), ema.average_name(tensor2), v2.op.name]
        if zero_debias:
            expected_names += [ema.average_name(tensor2) + '/biased', ema.average_name(tensor2) + '/local_step']
        self.assertEqual(sorted(expected_names), sorted(vars_to_restore.keys()))
        self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
        self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
        self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNames(self):
        if False:
            for i in range(10):
                print('nop')
        self.averageVariablesNamesHelper(zero_debias=True)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNamesNoDebias(self):
        if False:
            i = 10
            return i + 15
        self.averageVariablesNamesHelper(zero_debias=False)

    @test_util.deprecated_graph_mode_only
    def averageVariablesNamesRespectScopeHelper(self, zero_debias):
        if False:
            for i in range(10):
                print('nop')
        with variable_scope.variable_scope('scope1'):
            v0 = variables.Variable(10.0, name='v0')
            v1 = variables.Variable(30.0, name='v1')
            v2 = variables.Variable(20.0, name='v2', trainable=False)
            tensor2 = v0 + v1
        with variable_scope.variable_scope('scope2'):
            ema = moving_averages.ExponentialMovingAverage(0.25, zero_debias=zero_debias, name='foo')
            self.assertEqual('scope2/scope1/v0/foo', ema.average_name(v0))
            self.assertEqual('scope2/scope1/v1/foo', ema.average_name(v1))
            self.assertEqual('scope2/scope1/add/foo', ema.average_name(tensor2))
            ema.apply([v0, v1, tensor2])
            vars_to_restore = ema.variables_to_restore()
            expected_names = [ema.average_name(v0), ema.average_name(v1), ema.average_name(tensor2), v2.op.name]
            if zero_debias:
                sc = 'scope2/'
                expected_names += [sc + ema.average_name(tensor2) + '/biased', sc + ema.average_name(tensor2) + '/local_step']
            self.assertEqual(sorted(expected_names), sorted(vars_to_restore.keys()))
            self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
            self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
            self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNamesRespectScope(self):
        if False:
            i = 10
            return i + 15
        self.averageVariablesNamesRespectScopeHelper(zero_debias=True)

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesNamesRespectScopeNoDebias(self):
        if False:
            while True:
                i = 10
        self.averageVariablesNamesRespectScopeHelper(zero_debias=False)

    @test_util.deprecated_graph_mode_only
    def testSubsetAverageVariablesNames(self):
        if False:
            while True:
                i = 10
        v0 = variables.Variable(10.0, name='v0')
        v1 = variables.Variable(30.0, name='v1')
        v2 = variables.Variable(20.0, name='v2', trainable=False)
        tensor2 = v0 + v1
        ema = moving_averages.ExponentialMovingAverage(0.25, name='foo_avg')
        self.assertEqual('v0/foo_avg', ema.average_name(v0))
        self.assertEqual('v1/foo_avg', ema.average_name(v1))
        self.assertEqual('add/foo_avg', ema.average_name(tensor2))
        vars_to_restore = ema.variables_to_restore([v0, tensor2])
        self.assertEqual(sorted(vars_to_restore.keys()), sorted([ema.average_name(v0), ema.average_name(tensor2), v1.op.name, v2.op.name]))
        ema.apply([v0, v1, tensor2])
        self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
        self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
        self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

    def testSubsetAverageVariablesNamesEager(self):
        if False:
            i = 10
            return i + 15
        v0 = variables.Variable(10.0, name='v0')
        v1 = variables.Variable(30.0, name='v1')
        v2 = variables.Variable(20.0, name='v2', trainable=False)
        ema = moving_averages.ExponentialMovingAverage(0.25, name='foo_avg')
        self.assertEqual('v0/foo_avg', ema.average_name(v0))
        self.assertEqual('v1/foo_avg', ema.average_name(v1))
        vars_to_restore = ema.variables_to_restore([v0, v1, v2])
        self.assertAllEqual(sorted(vars_to_restore.keys()), sorted([ema.average_name(v0), ema.average_name(v1), ema.average_name(v2)]))
        ema.apply([v0, v1])
        self.assertEqual(ema.average(v0).name[:-len(':0')], ema.average_name(v0))
        self.assertEqual(ema.average(v1).name[:-len(':0')], ema.average_name(v1))

    @test_util.deprecated_graph_mode_only
    def testAverageVariablesDeviceAssignment(self):
        if False:
            i = 10
            return i + 15
        with ops.device('/job:dev_v0'):
            v0 = variables.Variable(10.0, name='v0')
        with ops.device('/job:dev_v1'):
            v1 = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='v1', container='', shared_name='')
            v1.set_shape([1])
        tensor2 = v0 + v1
        ema = moving_averages.ExponentialMovingAverage(0.25, name='foo_avg')
        with ops.device('/job:default'):
            ema.apply([v0, v1, tensor2])
        self.assertDeviceEqual('/job:dev_v0', ema.average(v0).device)
        self.assertDeviceEqual('/job:dev_v1', ema.average(v1).device)
        self.assertEqual([b'loc:@v1'], ema.average(v1).op.colocation_groups())
        self.assertDeviceEqual('/job:default', ema.average(tensor2).device)

    def _ExportAndImportGraph(self, graph):
        if False:
            return 10
        'Export and import graph into a new graph.'
        meta_graph = saver_lib.export_meta_graph(graph=graph, collection_list=graph.get_all_collection_keys())
        graph_copy = ops.Graph()
        with graph_copy.as_default():
            _ = saver_lib.import_meta_graph(meta_graph)
        return graph_copy

    @test_util.deprecated_graph_mode_only
    def testImportedGraphVariablesToRestore(self):
        if False:
            while True:
                i = 10
        g = ops.Graph()
        with g.as_default():
            variables.Variable(10.0, name='v')
        g_copy = self._ExportAndImportGraph(g)
        with g_copy.as_default():
            ema = moving_averages.ExponentialMovingAverage(0.25, name='foo_avg')
            vars_to_restore = ema.variables_to_restore()
            self.assertEqual(len(vars_to_restore), 1)
            self.assertIn('v/foo_avg', vars_to_restore)

    @test_util.deprecated_graph_mode_only
    def testCopyXlaSharding(self):
        if False:
            for i in range(10):
                print('nop')
        ema = moving_averages.ExponentialMovingAverage(0.25, name='foo_avg')
        v = variables.Variable(_Repeat(10.0, 2), name='v')
        self.assertIsNone(xla_sharding.get_tensor_sharding(v))
        v = xla_sharding.mesh_split(v, np.array([0, 1]), [0], use_sharding_op=False)
        self.assertIsNotNone(xla_sharding.get_tensor_sharding(v))
        self.evaluate(variables.global_variables_initializer())
        ema.apply([v])
        avg = ema.average(v)
        self.assertEqual(xla_sharding.get_tensor_sharding(v), xla_sharding.get_tensor_sharding(avg))
if __name__ == '__main__':
    test.main()