"""Tests for tensorflow.ops.control_flow_ops."""
import collections
import math
import re
import sys
import time
from absl.testing import parameterized
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as eager_def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop as while_loop_tf
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
import tensorflow.python.ops.tensor_array_grad
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import nest
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import variable_v1

def check_consumers(graph):
    if False:
        for i in range(10):
            print('nop')
    'Sanity check on the consumer list of the tensors.'
    consumer_count = {}
    for op in graph.get_operations():
        for v in op.inputs:
            cnt = consumer_count.get(v, 0)
            consumer_count[v] = cnt + 1
    for (k, v) in consumer_count.items():
        if len(k.consumers()) != v:
            return False
    return True

def all_fetchables():
    if False:
        while True:
            i = 10
    tensor_names = []
    graph = ops.get_default_graph()
    for op in graph.get_operations():
        for t in op.outputs:
            if graph.is_fetchable(t):
                tensor_names.append(t.name)
    return tensor_names

def all_feedables():
    if False:
        for i in range(10):
            print('nop')
    feedable_tensors = []
    graph = ops.get_default_graph()
    for op in graph.get_operations():
        for t in op.inputs:
            if graph.is_feedable(t):
                feedable_tensors.append(t)
    return feedable_tensors

def opt_cfg(do_constant_folding=True):
    if False:
        print('Hello World!')
    return config_pb2.ConfigProto(allow_soft_placement=True, graph_options=config_pb2.GraphOptions(optimizer_options=config_pb2.OptimizerOptions(opt_level=config_pb2.OptimizerOptions.L1, do_function_inlining=True, do_constant_folding=do_constant_folding)))

def isum(s, maximum_iterations=None):
    if False:
        print('Hello World!')
    i = constant_op.constant(0, name='i')
    c = lambda i, s: math_ops.less(i, 10)
    b = lambda i, s: [math_ops.add(i, 1), math_ops.add(i, s)]
    (_, r_s) = while_loop_tf.while_loop(c, b, [i, s], maximum_iterations=maximum_iterations)
    return r_s

def enqueue_print_op(s):
    if False:
        return 10
    'Enqueues an op that prints a message to be captured in the test.'
    return logging_ops.print_v2('ControlFlowOpsTest: ' + s)

def filter_test_messages(s):
    if False:
        print('Hello World!')
    'Returns a list of messages printed by enqueue_print_op.'
    prefix = 'ControlFlowOpsTest: '
    return [l[len(prefix):] for l in s.split('\n') if l.startswith(prefix)]

def tf_function_in_tf2(f):
    if False:
        return 10
    if tf2.enabled():
        return eager_def_function.function(f)
    return f

@test_util.with_eager_op_as_function
@test_util.with_control_flow_v2
class ControlFlowTest(test.TestCase, parameterized.TestCase):

    @test_util.run_v1_only('b/120545219')
    def testRefIdentity(self):
        if False:
            return 10
        with self.cached_session():
            v = variable_v1.VariableV1(7)
            v = control_flow_ops._Identity(v)
            op = state_ops.assign(v, 9)
            v2 = control_flow_ops.with_dependencies([op], v)
            self.assertTrue(isinstance(v2, tensor_lib.Tensor))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(9, self.evaluate(v2))

    @test_util.run_v1_only('b/120545219')
    def testRefEnter(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            v = variable_v1.VariableV1(7)
            enter_v = control_flow_ops._Enter(v, 'foo_1', is_constant=True)
            nine = constant_op.constant(9)
            enter_nine = gen_control_flow_ops.enter(nine, 'foo_1')
            op = state_ops.assign(enter_v, enter_nine)
            v2 = control_flow_ops.with_dependencies([op], enter_v)
            v3 = control_flow_ops.exit(v2)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(9, self.evaluate(v3))

    @test_util.run_v1_only('b/120545219')
    def testRefSwitch(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            v = variable_v1.VariableV1(7)
            p = constant_op.constant(True)
            v1 = control_flow_ops._SwitchRefOrTensor(v._ref(), p)
            v2 = state_ops.assign(v1[1], 9)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(9, self.evaluate(v2))

    def testEnterMulExit(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            data = constant_op.constant([1, 2, 3, 4, 5, 6], name='data')
            enter_data = gen_control_flow_ops.enter(data, 'foo_1', False)
            five = constant_op.constant(5)
            enter_five = gen_control_flow_ops.enter(five, 'foo_1', False)
            mul_op = math_ops.multiply(enter_data, enter_five)
            exit_op = control_flow_ops.exit(mul_op)
            result = self.evaluate(exit_op)
        self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

    @test_util.run_deprecated_v1
    def testEnterShapePropagation(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            v = variables.Variable([0.0, 0.0], dtype=dtypes.float32)
            enter_v_constant = gen_control_flow_ops.enter(v, 'frame1', is_constant=True)
            self.assertEqual(enter_v_constant.shape, [2])
            enter_v_non_constant = gen_control_flow_ops.enter(v, 'frame2', is_constant=False)
            self.assertEqual(enter_v_non_constant.shape, None)

    @test_util.run_v1_only('b/120545219')
    def testSwitchMergeIndexedSlices(self):
        if False:
            return 10
        with self.cached_session():
            values = constant_op.constant([1, 2, 3, 4, 5, 6])
            indices = constant_op.constant([0, 2, 4, 6, 8, 10])
            data = indexed_slices.IndexedSlices(values, indices)
            pred = ops.convert_to_tensor(True)
            switch_op = control_flow_ops.switch(data, pred)
            merge_op = control_flow_ops.merge(switch_op)[0]
            val = merge_op.values
            ind = merge_op.indices
        self.assertAllEqual(np.arange(1, 7), val)
        self.assertAllEqual(np.arange(0, 12, 2), ind)

    @test_util.run_v1_only('b/120545219')
    def testSwitchDeadBranch(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            data = constant_op.constant([1, 2, 3, 4, 5, 6], name='data')
            ports = ops.convert_to_tensor(True, name='ports')
            switch_op = control_flow_ops.switch(data, ports)
            dead_branch = array_ops.identity(switch_op[0])
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'Retval[0] does not have value' in str(e)):
                self.evaluate(dead_branch)

    @test_util.run_v1_only('b/120545219')
    def testSwitchMergeLess(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            data = constant_op.constant([1, 2, 3, 4, 5, 6], name='data')
            zero = ops.convert_to_tensor(0)
            one = ops.convert_to_tensor(1)
            less_op = math_ops.less(zero, one)
            switch_op = control_flow_ops.switch(data, less_op)
            merge_op = control_flow_ops.merge(switch_op)[0]
            result = self.evaluate(merge_op)
        self.assertAllEqual(np.arange(1, 7), result)

    @test_util.run_v1_only('b/120545219')
    def testSwitchMergeAddIdentity(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            data = constant_op.constant([1, 2, 3, 4, 5, 6], name='data')
            ports = ops.convert_to_tensor(False, name='ports')
            switch_op = control_flow_ops.switch(data, ports)
            one = constant_op.constant(1)
            add_op = math_ops.add(switch_op[0], one)
            id_op = array_ops.identity(switch_op[1])
            merge_op = control_flow_ops.merge([add_op, id_op])[0]
            result = self.evaluate(merge_op)
        self.assertAllEqual(np.array([x + 1 for x in [1, 2, 3, 4, 5, 6]]), result)

    @test_util.run_v1_only('b/120545219')
    def testSwitchMergeAddMul(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            data = constant_op.constant([1, 2, 3, 4, 5, 6], name='data')
            ports = ops.convert_to_tensor(True, name='ports')
            switch_op = control_flow_ops.switch(data, ports)
            one = constant_op.constant(1)
            add_op = math_ops.add(switch_op[0], one)
            five = constant_op.constant(5)
            mul_op = math_ops.multiply(switch_op[1], five)
            merge_op = control_flow_ops.merge([add_op, mul_op])[0]
            result = self.evaluate(merge_op)
        self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

    @test_util.run_v1_only('b/120545219')
    def testLoop_false(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            false = ops.convert_to_tensor(False)
            n = constant_op.constant(10)
            enter_false = gen_control_flow_ops.enter(false, 'foo_1', False)
            enter_n = gen_control_flow_ops.enter(n, 'foo_1', False)
            merge_n = control_flow_ops.merge([enter_n, enter_n], name='merge_n')[0]
            switch_n = control_flow_ops.switch(merge_n, enter_false)
            exit_n = control_flow_ops.exit(switch_n[0])
            next_n = control_flow_ops.next_iteration(switch_n[0])
            merge_n.op._update_input(1, next_n)
            result = self.evaluate(exit_n)
        self.assertAllEqual(10, result)

    @test_util.run_deprecated_v1
    def testLoop_1(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            zero = constant_op.constant(0)
            one = constant_op.constant(1)
            n = constant_op.constant(10)
            enter_i = gen_control_flow_ops.enter(zero, 'foo', False)
            enter_one = gen_control_flow_ops.enter(one, 'foo', True)
            enter_n = gen_control_flow_ops.enter(n, 'foo', True)
            with ops.device(test.gpu_device_name()):
                merge_i = control_flow_ops.merge([enter_i, enter_i])[0]
            less_op = math_ops.less(merge_i, enter_n)
            cond_op = control_flow_ops.loop_cond(less_op)
            switch_i = control_flow_ops.switch(merge_i, cond_op)
            add_i = math_ops.add(switch_i[1], enter_one)
            next_i = control_flow_ops.next_iteration(add_i)
            merge_i.op._update_input(1, next_i)
            exit_i = control_flow_ops.exit(switch_i[0])
            result = self.evaluate(exit_i)
        self.assertAllEqual(10, result)

    @test_util.run_v1_only('b/120545219')
    def testLoop_2(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            zero = constant_op.constant(0)
            one = constant_op.constant(1)
            n = constant_op.constant(10)
            enter_i = gen_control_flow_ops.enter(zero, 'foo', False)
            enter_one = gen_control_flow_ops.enter(one, 'foo', True)
            enter_n = gen_control_flow_ops.enter(n, 'foo', True)
            merge_i = control_flow_ops.merge([enter_i, enter_i])[0]
            less_op = math_ops.less(merge_i, enter_n)
            cond_op = control_flow_ops.loop_cond(less_op)
            switch_i = control_flow_ops.switch(merge_i, cond_op)
            add_i = math_ops.add(switch_i[1], enter_one)
            with ops.device(test.gpu_device_name()):
                next_i = control_flow_ops.next_iteration(add_i)
            merge_i.op._update_input(1, next_i)
            exit_i = control_flow_ops.exit(switch_i[0])
            result = self.evaluate(exit_i)
        self.assertAllEqual(10, result)

    @test_util.run_v1_only('b/120545219')
    def testDifferentFrame(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            data = array_ops.placeholder(dtypes.float32, shape=[])
            enter_1 = gen_control_flow_ops.enter(data, 'foo_1', False)
            enter_2 = gen_control_flow_ops.enter(data, 'foo_2', False)
            res = math_ops.add(enter_1, enter_2)
            with self.assertRaisesOpError('has inputs from different frames'):
                res.eval(feed_dict={data: 1.0})

    @test_util.run_deprecated_v1
    def testCondBool(self):
        if False:
            i = 10
            return i + 15
        values = constant_op.constant(10)
        fn1 = lambda : math_ops.add(values, 1)
        fn2 = lambda : math_ops.subtract(values, 1)
        with self.assertRaisesRegex(TypeError, 'must not be a Python bool'):
            _ = tf_cond.cond(False, fn1, fn2)

    @test_util.run_deprecated_v1
    def testCondInt(self):
        if False:
            while True:
                i = 10
        p = array_ops.placeholder(dtypes.bool, shape=[])
        v = constant_op.constant(10)
        fn1 = lambda : math_ops.add(v, 1)
        fn2 = lambda : math_ops.subtract(v, 1)
        y = tf_cond.cond(p, fn1, fn2)
        grad = gradients_impl.gradients(y, [v])
        self.assertAllEqual([None], grad)

    def testCondOutputShape(self):
        if False:
            return 10
        x = constant_op.constant(1.0)
        b = tf_cond.cond(constant_op.constant(True), lambda : math_ops.square(x), lambda : math_ops.subtract(x, 1.0))
        self.assertEqual(b.shape, tensor_shape.TensorShape([]))

    @test_util.run_v1_only('b/120545219')
    def testFetchable(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            x = array_ops.placeholder(dtypes.float32)
            tf_cond.cond(constant_op.constant(True), lambda : x + 2, lambda : x + 0)
            graph = ops.get_default_graph()
            for op in graph.get_operations():
                for t in op.inputs:
                    if graph.is_fetchable(t.op):
                        sess.run(t, feed_dict={x: 3})
                    else:
                        with self.assertRaisesRegex(ValueError, 'has been marked as not fetchable'):
                            sess.run(t, feed_dict={x: 3})

    @test_util.disable_control_flow_v2('Not relevant')
    @test_util.run_v1_only('b/120545219')
    def testFeedable(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            c = constant_op.constant(2)
            i0 = constant_op.constant(0)
            r = while_loop_tf.while_loop(lambda i: i < 1000, lambda i: math_ops.square(c) + i, [i0])
            self.assertEqual(1000, r.eval(feed_dict={i0: 0}))
            feedable_tensors = all_feedables()
            for t in feedable_tensors:
                sess.run(r, feed_dict={t: 3})
            graph = ops.get_default_graph()
            for op in graph.get_operations():
                for t in op.inputs:
                    if t not in feedable_tensors and t.dtype is dtypes.int32:
                        with self.assertRaisesRegex(ValueError, 'may not be fed'):
                            sess.run(r, feed_dict={t: 3})

    @test_util.run_v1_only('b/120545219')
    def testCondIndexedSlices(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            values = constant_op.constant([10])
            indices = constant_op.constant([0])
            x = indexed_slices.IndexedSlices(values, indices)
            pred = math_ops.less(1, 2)
            fn1 = lambda : indexed_slices.IndexedSlices(math_ops.add(x.values, 1), indices)
            fn2 = lambda : indexed_slices.IndexedSlices(math_ops.subtract(x.values, 1), indices)
            r = tf_cond.cond(pred, fn1, fn2)
            val = r.values
            ind = r.indices
        self.assertAllEqual([11], val)
        self.assertAllEqual([0], ind)

    def testCondMismatchedIndexedSlices(self):
        if False:
            return 10

        @eager_def_function.function
        def foo():
            if False:
                while True:
                    i = 10
            values = constant_op.constant([10])
            indices = constant_op.constant([0])
            x = indexed_slices.IndexedSlices(values, indices)
            with self.assertRaisesRegex(TypeError, 'Cannot reconcile tf.cond 0-th outputs'):
                tf_cond.cond(constant_op.constant(True), lambda : indexed_slices.IndexedSlices(math_ops.add(x.values, 1), indices), lambda : math_ops.add(x.values, 1), indices)
        foo()

    def testCondSparseTensor(self):
        if False:
            return 10
        values = constant_op.constant([2.0, 4.0], name='values')
        indices = constant_op.constant([[0], [3]], dtype=dtypes.int64, name='indices')
        shape = constant_op.constant([10], dtype=dtypes.int64, name='dense_shape')
        x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)
        pred = math_ops.less(1, 2)
        fn1 = lambda : sparse_tensor.SparseTensor(indices + 1, x.values + 1, dense_shape=shape)
        fn2 = lambda : sparse_tensor.SparseTensor(indices, x.values - 1, dense_shape=shape)
        r = tf_cond.cond(pred, fn1, fn2)
        self.assertAllEqual([3.0, 5.0], r.values)
        self.assertAllEqual([[1], [4]], r.indices)
        self.assertAllEqual(r.values.get_shape(), (2,))

    def testCondRaggedTensor(self):
        if False:
            print('Hello World!')
        rt = ragged_factory_ops.constant([[1, 2], [3], [4, 5, 6]])
        pred = math_ops.less(1, 2)
        fn1 = lambda : array_ops.concat([rt + 2, [[100]]], axis=0)
        fn2 = lambda : rt[:2] - 2
        result = tf_cond.cond(pred, fn1, fn2)
        self.assertAllEqual([3, 4, 5, 6, 7, 8, 100], result.values)
        self.assertAllEqual([0, 2, 3, 6, 7], result.row_splits)

    @test_util.run_v1_only('b/120545219')
    def testCondResource(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            rv = resource_variable_ops.ResourceVariable(True)
            self.evaluate(variables.global_variables_initializer())
            t = ops.convert_to_tensor(1.0)

            def case():
                if False:
                    i = 10
                    return i + 15
                assign = resource_variable_ops.assign_variable_op(rv.handle, False)
                with ops.control_dependencies([assign]):
                    return array_ops.identity(t)
            self.assertEqual(1.0, self.evaluate(tf_cond.cond(rv, case, lambda : t)))

    @test_util.run_deprecated_v1
    def testCondResourceGradShape(self):
        if False:
            print('Hello World!')
        rv1 = resource_variable_ops.ResourceVariable([1.0, 2.0])
        rv2 = resource_variable_ops.ResourceVariable([3.0, 4.0])
        pred = constant_op.constant(True)
        result = tf_cond.cond(pred, lambda : rv1, lambda : rv2)
        grads = gradients_impl.gradients(result, [rv1, rv2])
        self.assertAllEqual(grads[0].shape.as_list(), [2])
        self.assertAllEqual(grads[1].shape.as_list(), [2])

    @test_util.run_v1_only('b/120545219')
    def testCondWithTensorArrayGrad(self):
        if False:
            return 10
        with self.cached_session() as sess:
            with ops.device(test.gpu_device_name()):
                pred = array_ops.placeholder(dtypes.bool, [])
                x = constant_op.constant([1.0, 2.0, 3.0])
                y = tf_cond.cond(pred, lambda : map_fn.map_fn(lambda z: z * 2.0, x), lambda : constant_op.constant([1.0, 1.0, 1.0]))
                g = gradients_impl.gradients(y, x)[0]
            self.assertAllEqual(sess.run(g, {pred: True}), [2.0, 2.0, 2.0])
            self.assertAllEqual(sess.run(g, {pred: False}), [0.0, 0.0, 0.0])

    @test_util.run_v1_only('b/120545219')
    def testCondIndexedSlicesDifferentTypes(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            values = constant_op.constant([10])
            i_32 = ops.convert_to_tensor([0], name='one', dtype=dtypes.int32)
            i_64 = ops.convert_to_tensor([0], name='one', dtype=dtypes.int64)
            x = indexed_slices.IndexedSlices(values, i_32)
            pred = math_ops.less(1, 2)
            fn1 = lambda : indexed_slices.IndexedSlices(math_ops.add(x.values, 1), i_32)
            fn2 = lambda : indexed_slices.IndexedSlices(math_ops.subtract(x.values, 1), i_64)
            r = tf_cond.cond(pred, fn1, fn2)
            val = r.values
            ind = r.indices
        self.assertAllEqual([11], val)
        self.assertAllEqual([0], ind)
        self.assertTrue(ind.dtype == np.int64)

    @test_util.run_v1_only('b/120545219')
    def testCondColocation(self):
        if False:
            return 10
        with self.session():
            with ops.device('/cpu:0'):
                v = variables.Variable(7.0)
            x = constant_op.constant(10.0)
            pred = math_ops.less(1.0, 2.0)
            fn1 = lambda : math_ops.add(v, 1.0)
            fn2 = lambda : math_ops.subtract(x, 1.0)
            r = tf_cond.cond(pred, fn1, fn2)
            for op in x.graph.get_operations():
                if op.name == 'cond/Add/Switch':
                    self.assertDeviceEqual(op.device, '/cpu:0')

    def _testCond_1(self, use_gpu):
        if False:
            i = 10
            return i + 15
        with self.cached_session(use_gpu=use_gpu):
            x = constant_op.constant(10)
            pred = math_ops.less(1, 2)
            fn1 = lambda : math_ops.add(x, 1)
            fn2 = lambda : math_ops.subtract(x, 1)
            r = tf_cond.cond(pred, fn1, fn2)
            result = self.evaluate(r)
        self.assertAllEqual(11, result)

    def testCond_1(self):
        if False:
            return 10
        self._testCond_1(use_gpu=False)

    def testCond_2(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = constant_op.constant(10)
            r = tf_cond.cond(math_ops.less(1, 0), lambda : math_ops.add(x, 1), lambda : math_ops.subtract(x, 1))
            result = self.evaluate(r)
        self.assertAllEqual(9, result)

    def testCond_3(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            x = constant_op.constant(10)
            pred = math_ops.less(1, 2)
            fn1 = lambda : math_ops.add(x, 1)
            fn2 = lambda : math_ops.subtract(x, 1)
            fn3 = lambda : math_ops.add(tf_cond.cond(pred, fn1, fn2), 1)
            r = tf_cond.cond(pred, fn3, fn2)
            result = self.evaluate(r)
        self.assertAllEqual(12, result)

    @test_util.run_in_graph_and_eager_modes
    def testCondPruning(self):
        if False:
            print('Hello World!')
        v1 = variables.Variable(7)
        v2 = variables.Variable(7)
        v3 = variables.Variable(7)

        def f():
            if False:
                while True:
                    i = 10
            age = constant_op.constant(3)
            max_age = constant_op.constant(2)
            pred = math_ops.greater(age, max_age)
            fn1 = lambda : [state_ops.assign(v1, 1).op, state_ops.assign(v2, 2).op]
            fn2 = lambda : [state_ops.assign(v3, 3).op, constant_op.constant(10).op]
            r = tf_cond.cond(pred, fn1, fn2)
            self.assertEqual(len(r), 2)
            return r[1]
        f_defun = eager_def_function.function(f)
        if not context.executing_eagerly():
            with self.cached_session():
                self.evaluate(variables.global_variables_initializer())
                result = self.evaluate(f())
                self.assertEqual(True, result)
                self.assertEqual(7, self.evaluate(v1))
                self.assertEqual(2, self.evaluate(v2))
                self.assertEqual(7, self.evaluate(v3))
        result = f_defun()
        self.assertEqual(True, self.evaluate(result))
        self.assertEqual(1, self.evaluate(v1))
        self.assertEqual(2, self.evaluate(v2))
        self.assertEqual(7, self.evaluate(v3))

    def testCond_5(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            alive = constant_op.constant(True, name='alive')
            count = constant_op.constant(0, name='count')

            def body(i):
                if False:
                    for i in range(10):
                        print('nop')
                return tf_cond.cond(alive, lambda : [math_ops.less(i, 3), math_ops.add(count, 1)], lambda : [alive, count])
            for i in range(10):
                (alive, count) = body(i)
            self.assertAllEqual(4, self.evaluate(count))

    @test_util.run_v1_only('b/120545219')
    def testCond_6(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            v1 = variables.Variable([7])
            age = constant_op.constant(3)
            pred = math_ops.greater(age, 4)
            fn1 = lambda : age
            fn2 = lambda : v1
            r = tf_cond.cond(pred, fn1, fn2)
            self.evaluate(variables.global_variables_initializer())
            result = self.evaluate(r)
            self.assertAllEqual(np.array([7]), result)

    def testCond_7(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            x = constant_op.constant(10)
            y = constant_op.constant(200)
            pred = math_ops.less(1, 2)
            fn1 = lambda : [math_ops.add(x, 1), math_ops.add(x, 2)]
            fn2 = lambda : [y, y]
            r = tf_cond.cond(pred, fn1, fn2)
            self.assertAllEqual([11, 12], self.evaluate(r))

    @parameterized.parameters(dtypes.float32, dtypes.float64)
    @test_util.run_v1_only('Uses tf.gradients')
    def testCondResourceGrad(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        init = constant_op.constant([7.0], dtype=dtype)
        v1 = variables.Variable(init)
        age = constant_op.constant(3.0, dtype=dtype)
        pred = math_ops.greater(age, 4.0)
        fn1 = lambda : age
        fn2 = lambda : v1
        r = tf_cond.cond(pred, fn1, fn2)
        grad = gradients_impl.gradients(r, v1)[0]
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(grad, [1.0])

    @test_util.run_gpu_only
    @test_util.run_deprecated_v1
    def testCond_Device(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant(-10.0)

        def true_fn():
            if False:
                print('Hello World!')
            return math_ops.exp(x)
        with ops.device('CPU:0'):
            r = tf_cond.cond(constant_op.constant(True), true_fn, lambda : 0.0)
            self.assertIn('cpu', r.device.lower())
        with session.Session() as sess:
            options = config_pb2.RunOptions(output_partition_graphs=True)
            run_metadata = config_pb2.RunMetadata()
            sess.run(r, options=options, run_metadata=run_metadata)
            self.assertEqual(len(run_metadata.partition_graphs), 1)

    def _count_matching_switch_nodes_on_device(self, run_metadata, device_str, dtype):
        if False:
            print('Hello World!')
        device_graphs = [g for g in run_metadata.partition_graphs if device_str in g.node[0].device]
        self.assertLen(device_graphs, 1)
        switch_nodes = [n for n in device_graphs[0].node if n.op == 'Switch' and n.attr['T'].type == dtype.as_datatype_enum]
        return len(switch_nodes)

    @test_util.run_gpu_only
    @test_util.run_deprecated_v1
    def testCondSwitchColocatedWithInputWhenInputExplicitlyPlacedOnCPU(self):
        if False:
            i = 10
            return i + 15
        x = array_ops.placeholder(dtypes.float32)
        with ops.device('CPU:0'):
            arg = x + 10.0

        def true_fn():
            if False:
                print('Hello World!')
            with ops.device('CPU:0'):
                return arg + 1
        r = tf_cond.cond(constant_op.constant(True), true_fn, lambda : 0.0)
        config = config_pb2.ConfigProto()
        config.graph_options.rewrite_options.loop_optimization = rewriter_config_pb2.RewriterConfig.OFF
        with self.session(config=config) as sess:
            run_metadata = config_pb2.RunMetadata()
            options = config_pb2.RunOptions(output_partition_graphs=True)
            sess.run(r, feed_dict={x: -10.0}, options=options, run_metadata=run_metadata)
            self.assertLen(run_metadata.partition_graphs, 2)
            self.assertEqual(self._count_matching_switch_nodes_on_device(run_metadata, 'CPU', dtypes.float32), 1)
            self.assertEqual(self._count_matching_switch_nodes_on_device(run_metadata, 'GPU', dtypes.float32), 0)

    @test_util.run_gpu_only
    @test_util.run_deprecated_v1
    def testCondSwitchColocatedWithInputWhenInputPlacedOnCPU(self):
        if False:
            print('Hello World!')
        x = array_ops.placeholder(dtypes.float32)
        arg = dataset_ops.Dataset.range(8)

        def true_fn():
            if False:
                print('Hello World!')
            return cardinality.cardinality(arg)
        r = tf_cond.cond(constant_op.constant(True), true_fn, lambda : constant_op.constant(0, dtypes.int64))
        config = config_pb2.ConfigProto()
        config.graph_options.rewrite_options.loop_optimization = rewriter_config_pb2.RewriterConfig.OFF
        with session.Session(config=config) as sess:
            run_metadata = config_pb2.RunMetadata()
            options = config_pb2.RunOptions(output_partition_graphs=True)
            sess.run(r, feed_dict={x: -10.0}, options=options, run_metadata=run_metadata)
            self.assertLen(run_metadata.partition_graphs, 2)
            self.assertEqual(self._count_matching_switch_nodes_on_device(run_metadata, 'CPU', dtypes.variant), 1)
            self.assertEqual(self._count_matching_switch_nodes_on_device(run_metadata, 'GPU', dtypes.variant), 0)

    @test_util.run_gpu_only
    @test_util.run_deprecated_v1
    def testCondSwitchColocatedWithInputWhenInputOnGPU(self):
        if False:
            while True:
                i = 10
        x = array_ops.placeholder(dtypes.float32)
        arg = x + 10.0

        def true_fn():
            if False:
                i = 10
                return i + 15
            with ops.device('CPU:0'):
                return arg + 1
        r = tf_cond.cond(constant_op.constant(True), true_fn, lambda : 0.0)
        config = config_pb2.ConfigProto()
        config.graph_options.rewrite_options.loop_optimization = rewriter_config_pb2.RewriterConfig.OFF
        with session.Session(config=config) as sess:
            run_metadata = config_pb2.RunMetadata()
            options = config_pb2.RunOptions(output_partition_graphs=True)
            sess.run(r, feed_dict={x: -10.0}, options=options, run_metadata=run_metadata)
            self.assertEqual(len(run_metadata.partition_graphs), 2)
            self.assertEqual(self._count_matching_switch_nodes_on_device(run_metadata, 'CPU', dtypes.float32), 0)
            self.assertEqual(self._count_matching_switch_nodes_on_device(run_metadata, 'GPU', dtypes.float32), 1)

    def testCondAccessTrueBranchTensorInFalseBranchRaises(self):
        if False:
            i = 10
            return i + 15

        @eager_def_function.function
        def f():
            if False:
                while True:
                    i = 10
            c = constant_op.constant(1.0)
            inputs = {'c': c}

            def true_fn(inputs):
                if False:
                    return 10
                inputs['c'] = array_ops.identity(inputs['c'], name='true_branch')
                return inputs['c']

            def false_fn(inputs):
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.identity(inputs['c'])
            pred = constant_op.constant(True)
            return tf_cond.cond(pred, lambda : true_fn(inputs), lambda : false_fn(inputs))
        prefix = 'cond/' if context.executing_eagerly() else ''
        with self.assertRaisesRegex(ValueError, 'Tensor %strue_branch:0 in true_fn is accessed from false_fn.' % prefix):
            f()

    def testSwitchCaseAccessBranch1TensorInBranch4Raises(self):
        if False:
            return 10

        @eager_def_function.function
        def f():
            if False:
                while True:
                    i = 10
            c = constant_op.constant(1.0)
            inputs = {'c': c}

            def br1_fn(inputs):
                if False:
                    i = 10
                    return i + 15
                inputs['c'] = array_ops.identity(inputs['c'], name='br1_identity')
                return inputs['c']

            def br4_fn(inputs):
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.identity(inputs['c'])

            def other_fn():
                if False:
                    return 10
                return array_ops.identity(c)
            return control_flow_switch_case.switch_case(constant_op.constant(2), [other_fn, lambda : br1_fn(inputs), other_fn, other_fn, lambda : br4_fn(inputs)])
        prefix = 'switch_case/indexed_case/' if context.executing_eagerly() else ''
        with self.assertRaisesRegex(ValueError, 'Tensor %sbr1_identity:0 in branch 1 is accessed from branch 4.' % prefix):
            f()

    def testCondListOutput(self):
        if False:
            return 10
        with self.cached_session() as sess:
            x = constant_op.constant(10)
            y = constant_op.constant(200)
            pred = math_ops.less(1, 2)
            fn1 = lambda : [math_ops.add(x, y), math_ops.add(x, y)]
            fn2 = lambda : [y, y]
            r = tf_cond.cond(pred, fn1, fn2)
            test_result = self.evaluate(r)
            self.assertListEqual([210, 210], test_result)

    def testTupleOutput(self):
        if False:
            return 10
        with self.cached_session() as sess:
            x = constant_op.constant(10)
            y = constant_op.constant(200)
            pred = math_ops.less(1, 2)
            fn1 = lambda : (math_ops.add(x, y), math_ops.add(x, y))
            fn2 = lambda : (y, y)
            r = tf_cond.cond(pred, fn1, fn2)
            test_result = self.evaluate(r)
            self.assertTupleEqual((210, 210), test_result)

    def testDictOutput(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            x = constant_op.constant(10)
            y = constant_op.constant(200)
            pred = math_ops.less(1, 2)
            fn1 = lambda : {'a': math_ops.add(x, y), 'b': math_ops.add(x, y)}
            fn2 = lambda : {'a': y, 'b': y}
            r = tf_cond.cond(pred, fn1, fn2)
            test_result = self.evaluate(r)
            self.assertDictEqual({'a': 210, 'b': 210}, test_result)

    def testEmbeddedListOutput(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant(10)
        y = constant_op.constant(200)
        pred = math_ops.less(1, 2)
        fn1 = lambda : [[math_ops.add(x, y), math_ops.add(x, y)]]
        fn2 = lambda : [[y, y]]
        r = tf_cond.cond(pred, fn1, fn2, strict=True)
        test_result = self.evaluate(r)
        self.assertListEqual([[210, 210]], test_result)

    def testEmbeddedTupleOutput(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            x = constant_op.constant(10)
            y = constant_op.constant(200)
            pred = math_ops.less(1, 2)
            fn1 = lambda : (math_ops.add(x, y), math_ops.add(x, y))
            fn2 = lambda : (y, y)
            r = tf_cond.cond(pred, fn1, fn2)
            test_result = self.evaluate(r)
            self.assertTupleEqual((210, 210), test_result)

    def testEmbeddedDictOutput(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            x = constant_op.constant(10)
            y = constant_op.constant(200)
            pred = math_ops.less(1, 2)
            fn1 = lambda : {'a': {'c': math_ops.add(x, y)}, 'b': {'d': math_ops.add(x, y)}}
            fn2 = lambda : {'a': {'c': y}, 'b': {'d': y}}
            r = tf_cond.cond(pred, fn1, fn2)
            test_result = self.evaluate(r)
            self.assertDictEqual({'a': {'c': 210}, 'b': {'d': 210}}, test_result)

    @test_util.run_v1_only('b/120545219')
    def testCheckNestedOutputStruct(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            x = constant_op.constant(10)
            y = constant_op.constant(200)
            pred = math_ops.less(1, 2)
            fn1 = lambda : {'a': math_ops.add(x, y), 'b': math_ops.add(x, y)}
            fn2 = lambda : {'c': y, 'd': y}
            v1_msg = "The two structures don't have the same nested structure"
            v2_msg = 'true_fn and false_fn arguments to tf.cond must have the same number, type, and overall structure of return values.'
            with self.assertRaisesRegex(TypeError if control_flow_util.ENABLE_CONTROL_FLOW_V2 else ValueError, v2_msg if control_flow_util.ENABLE_CONTROL_FLOW_V2 else v1_msg):
                tf_cond.cond(pred, fn1, fn2)

    @test_util.run_v1_only('b/120545219')
    def testCondWithControl(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            control_holder = array_ops.placeholder(dtypes.float32, shape=())
            a = constant_op.constant(3)

            def true_branch():
                if False:
                    for i in range(10):
                        print('nop')
                with ops.control_dependencies([control_holder]):
                    _ = a + 1
                return a + 2
            r = tf_cond.cond(constant_op.constant(True), true_branch, lambda : constant_op.constant(1))
            result = sess.run(r, feed_dict={control_holder: 5.0})
            self.assertEqual(5, result)

    @test_util.run_v1_only('b/120545219')
    def testUninitializedRefIdentity(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            v = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='v', container='', shared_name='')
            inited = state_ops.is_variable_initialized(v)
            (v_f, v_t) = control_flow_ops.ref_switch(v, inited)
            v_f_op = gen_array_ops.ref_identity(v_f)
            v_t_op = gen_array_ops.ref_identity(v_t)
            with ops.control_dependencies([v_f_op]):
                assign_v = state_ops.assign(v, [1.0])
            with ops.control_dependencies([v_t_op]):
                orig_v = array_ops.identity(v)
            merged_op = control_flow_ops.merge([assign_v, orig_v])
            self.assertAllEqual([1.0], self.evaluate(merged_op.output))

    def testCondSwitchIdentity(self):
        if False:
            return 10
        with session.Session(config=opt_cfg()) as sess:
            pred = constant_op.constant(True)

            def fn1():
                if False:
                    i = 10
                    return i + 15
                return control_flow_ops.no_op()

            def fn2():
                if False:
                    i = 10
                    return i + 15
                return control_flow_assert.Assert(False, ['Wrong branch!!!'])
            r = tf_cond.cond(pred, fn1, fn2)
            self.evaluate(r)

    def testCondRecvIdentity(self):
        if False:
            i = 10
            return i + 15
        with session.Session(config=opt_cfg()) as sess:
            with ops.device(test.gpu_device_name()):
                pred = constant_op.constant(True)

            def fn1():
                if False:
                    for i in range(10):
                        print('nop')
                return control_flow_ops.no_op()

            def fn2():
                if False:
                    i = 10
                    return i + 15
                with ops.device('/cpu:0'):
                    return control_flow_assert.Assert(False, ['Wrong branch!!!'])
            r = tf_cond.cond(pred, fn1, fn2)
            self.evaluate(r)

    @test_util.run_deprecated_v1
    @test_util.enable_control_flow_v2
    def testDisableLoweringSwitchMerge(self):
        if False:
            i = 10
            return i + 15
        if test_util.is_gpu_available():
            self.skipTest("Single threaded executor doesn't support partitioned graphs.  Skipping GPU test.")
        run_opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata_no_lowering = config_pb2.RunMetadata()
        run_metadata_with_lowering = config_pb2.RunMetadata()
        config = opt_cfg(do_constant_folding=False)
        pred = array_ops.placeholder_with_default(constant_op.constant(True), shape=())
        r = tf_cond.cond(pred, lambda : True, lambda : False)
        with session.Session(config=config) as sess:
            r_value = sess.run(r, options=run_opts, run_metadata=run_metadata_with_lowering)
            self.assertEqual(r_value, True)
        config.experimental.executor_type = 'SINGLE_THREADED_EXECUTOR'
        with session.Session(config=config) as sess:
            r_value = sess.run(r, options=run_opts, run_metadata=run_metadata_no_lowering)
            self.assertEqual(r_value, True)
        self.assertTrue(any(('switch' in ns.node_name for dev_stat in run_metadata_with_lowering.step_stats.dev_stats for ns in dev_stat.node_stats)))
        self.assertTrue(all(('switch' not in ns.node_name for dev_stat in run_metadata_no_lowering.step_stats.dev_stats for ns in dev_stat.node_stats)))

    @test_util.run_v1_only('b/120545219')
    def testCondGrad_1(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = constant_op.constant(10.0, name='x')
            pred = math_ops.less(1, 2)
            fn1 = lambda : array_ops.identity(x)
            fn2 = lambda : array_ops.identity(x)
            r = tf_cond.cond(pred, fn1, fn2)
            grad = gradients_impl.gradients(r, [x])[0]
            self.assertAllEqual(1.0, self.evaluate(grad))

    @test_util.run_deprecated_v1
    @test_util.enable_control_flow_v2
    def testCondComputeGradAfterSessRunFails(self):
        if False:
            return 10
        with self.cached_session():
            x = constant_op.constant(10.0, name='x')
            pred = math_ops.less(1, 2)

            def true_fn():
                if False:
                    return 10
                a = x * x
                return a * a

            def false_fn():
                if False:
                    while True:
                        i = 10
                return x * x
            r = tf_cond.cond(pred, true_fn, false_fn)
            self.assertAllEqual(r, 10000.0)
            grad = gradients_impl.gradients(r, [x])[0]
            with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'Connecting to invalid output 1 of source node cond which has 1 outputs. Try using tf.compat.v1.experimental.output_all_intermediates\\(True\\).'):
                self.evaluate(grad)

    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testCondComputeGradAfterSessRun(self):
        if False:
            return 10
        with self.cached_session():
            x = constant_op.constant(10.0, name='x')
            pred = math_ops.less(1, 2)

            def true_fn():
                if False:
                    return 10
                a = x * x
                return a * a

            def false_fn():
                if False:
                    while True:
                        i = 10
                return x * x
            r = tf_cond.cond(pred, true_fn, false_fn)
            self.assertAllEqual(r, 10000.0)
            grad = gradients_impl.gradients(r, [x])[0]
            self.assertAllEqual(grad, 4000.0)

    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testNestedCondComputeGradAfterSessRun(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            x = constant_op.constant(10.0, name='x')
            pred = math_ops.less(1, 2)

            def true_fn():
                if False:
                    return 10

                def inner_true_fn():
                    if False:
                        i = 10
                        return i + 15
                    a = x * x
                    return a * a

                def inner_false_fn():
                    if False:
                        return 10
                    return x * x
                return tf_cond.cond(constant_op.constant(True), inner_true_fn, inner_false_fn)

            def false_fn():
                if False:
                    for i in range(10):
                        print('nop')
                return x * x
            r = tf_cond.cond(pred, true_fn, false_fn)
            self.assertAllEqual(r, 10000.0)
            grad = gradients_impl.gradients(r, [x])[0]
            self.assertAllEqual(grad, 4000.0)

    @test_util.run_deprecated_v1
    def testCondGrad_2(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            c = array_ops.placeholder(dtypes.int32, shape=[])
            x = constant_op.constant(10.0)
            pred = math_ops.less(c, 2)
            fn1 = lambda : math_ops.multiply(x, 42.0)
            fn2 = lambda : math_ops.multiply(x, 3.0)
            r = tf_cond.cond(pred, fn1, fn2)
            grad = gradients_impl.gradients(r, [x])[0]
            self.assertAllEqual(42.0, grad.eval(feed_dict={c: 1}))
            self.assertAllEqual(3.0, grad.eval(feed_dict={c: 3}))

    @test_util.disable_control_flow_v2('b/110550782 (gradient w.r.t external variable)')
    @test_util.run_deprecated_v1
    def testCondGrad_3(self):
        if False:
            return 10
        with self.cached_session():
            c = array_ops.placeholder(dtypes.int32, shape=[])
            ox = constant_op.constant(10.0)
            pred = math_ops.less(c, 2)

            def fn1(x):
                if False:
                    while True:
                        i = 10
                m = x * x
                return gradients_impl.gradients(m, [ox])[0]
            fn2 = lambda : math_ops.multiply(ox, 3.0)
            y = math_ops.multiply(7.0, ox)
            r = tf_cond.cond(pred, lambda : fn1(y), fn2)
            self.assertAllEqual(980.0, r.eval(feed_dict={c: 1}))
            self.assertAllEqual(30.0, r.eval(feed_dict={c: 3}))

    @test_util.run_deprecated_v1
    def testCondGradMultiDevice(self):
        if False:
            while True:
                i = 10
        config = config_pb2.ConfigProto(device_count={'CPU': 2}, allow_soft_placement=True)
        with self.cached_session(config=config) as sess:
            pred = array_ops.placeholder(dtypes.bool, [])
            x = array_ops.placeholder(dtypes.float32)
            y = array_ops.placeholder(dtypes.float32)
            with ops.device('/cpu:0'):
                z = tf_cond.cond(pred, lambda : x * y * 2.0, lambda : 2.0)
            with ops.device('/cpu:1'):
                grad = gradients_impl.gradients(z, x)[0]
            with ops.device('/cpu:0'):
                grad_grad = gradients_impl.gradients(grad, x)[0]
            self.assertEqual(sess.run(grad, {pred: True, x: 1.0, y: 2.0}), 4.0)
            self.assertEqual(sess.run(grad, {pred: False, x: 1.0, y: 2.0}), 0.0)
            if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
                self.assertIsNone(grad_grad)
                return
            self.assertEqual(sess.run(grad_grad, {pred: True, x: 1.0, y: 2.0}), 0.0)
            self.assertEqual(sess.run(grad_grad, {pred: False, x: 1.0, y: 2.0}), 0.0)

    @test_util.run_v1_only('b/120545219')
    def testNestedCond_Simple(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = constant_op.constant(0.0, name='X')
            y = tf_cond.cond(constant_op.constant(True), lambda : x, lambda : tf_cond.cond(x < 1.0, lambda : x, lambda : x))
            result = gradients_impl.gradients(y, x)[0]
            self.assertEqual(1.0, self.evaluate(result))
            z = tf_cond.cond(constant_op.constant(False), lambda : x, lambda : tf_cond.cond(x < 1.0, lambda : x, lambda : x))
            result = gradients_impl.gradients(z, x)[0]
            self.assertEqual(1.0, self.evaluate(result))

    @test_util.run_v1_only('b/120545219')
    def testCondGrad_Gather(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            v1 = variables.Variable([1.0, 42.0])
            c = array_ops.placeholder(dtypes.int32, shape=[])
            pred = math_ops.less(c, 2)
            fn1 = lambda : array_ops.identity(v1)
            fn2 = lambda : array_ops.gather(v1, [1, 1])
            r = tf_cond.cond(pred, fn1, fn2)
            grad = gradients_impl.gradients(r, [v1])[0]
            self.evaluate(variables.global_variables_initializer())
            if control_flow_util.ENABLE_CONTROL_FLOW_V2:
                self.assertIsInstance(grad, indexed_slices.IndexedSlices)
            grad_value = sess.run(grad, feed_dict={c: 1})
            self.assertAllEqual(gradient_checker_v2._to_numpy(grad_value), [1.0, 1.0])
            grad_value = sess.run(grad, feed_dict={c: 3})
            self.assertAllEqual(gradient_checker_v2._to_numpy(grad_value), [0.0, 2.0])

    @test_util.run_deprecated_v1
    def testCondGrad_ResourceVarSparseRead(self):
        if False:
            return 10
        var = resource_variable_ops.ResourceVariable(np.ones((4, 2), dtype=np.float32))
        x = constant_op.constant(1.0)
        r = tf_cond.cond(constant_op.constant(True), lambda : x * math_ops.reduce_sum(var.sparse_read([1, 2])), lambda : constant_op.constant(np.zeros((2, 3)), dtype=dtypes.float32))
        grad = gradients_impl.gradients(r, var)[0]
        self.evaluate(variables.global_variables_initializer())
        grad_val = self.evaluate(grad)
        self.assertIsInstance(grad_val, indexed_slices.IndexedSlicesValue)
        self.assertAllEqual(gradient_checker_v2._to_numpy(grad_val), [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

    def testCondGrad_MultiGather(self):
        if False:
            while True:
                i = 10
        var = resource_variable_ops.ResourceVariable(np.ones((4, 2), dtype=np.float32))
        x1 = constant_op.constant(np.ones((3, 3), dtype=np.float32))
        x2 = constant_op.constant(2.0)

        def true_fn():
            if False:
                for i in range(10):
                    print('nop')
            y1 = var.sparse_read([1, 2])
            y2 = array_ops.gather(x1, [2]) * x2
            y3 = x2 * [1.0, 1.0, 1.0]
            return (y1, y2, y3)

        def false_fn():
            if False:
                for i in range(10):
                    print('nop')
            y1 = np.zeros((2, 2), dtype=np.float32)
            y2 = array_ops.gather(x1, [2]) * x2
            y3 = array_ops.gather(x1, [2])
            return (y1, y2, y3)

        @eager_def_function.function
        def foo():
            if False:
                print('Hello World!')
            r = tf_cond.cond(constant_op.constant(True), true_fn, false_fn)
            return gradients_impl.gradients(r, [var, x1, x2])
        grad = foo()
        self.evaluate(variables.global_variables_initializer())
        (var_grad, x1_grad, x2_grad) = self.evaluate(grad)
        self.assertIsInstance(var_grad, indexed_slices.IndexedSlicesValue)
        self.assertAllEqual(gradient_checker_v2._to_numpy(var_grad), [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0]])
        self.assertIsInstance(x1_grad, indexed_slices.IndexedSlicesValue)
        self.assertAllEqual(gradient_checker_v2._to_numpy(x1_grad), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        self.assertIsInstance(x1_grad, indexed_slices.IndexedSlicesValue)
        self.assertEqual(gradient_checker_v2._to_numpy(x2_grad), 6.0)

    @test_util.run_v1_only('b/120545219')
    def testCondPredicateTensor(self):
        if False:
            while True:
                i = 10
        'Regression test for lowering predicate from non-first output of an op.'

        @eager_def_function.function
        def foo():
            if False:
                return 10
            return (constant_op.constant('foo'), constant_op.constant(True))
        r = tf_cond.cond(foo()[1], lambda : 1.0, lambda : 2.0)
        self.assertEqual(self.evaluate(r), 1.0)

    @test_util.run_v1_only('Tests Session.run() pruning logic.')
    def testCondFeedConstantPredicate(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            value = constant_op.constant(37.0)
            predicate = constant_op.constant(True)
            cond_output = tf_cond.cond(predicate, lambda : constant_op.constant(0.0), lambda : value)
            result = array_ops.identity(cond_output)
            self.assertEqual(37.0, sess.run(result, feed_dict={predicate: False}))
            self.assertEqual(0.0, sess.run(result, feed_dict={predicate: True}))
            self.assertEqual(0.0, sess.run(result))

    @test_util.run_v1_only('Tests Session.run() pruning logic.')
    def testCondFeedPlaceholderWithDefaultPredicate(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            value = constant_op.constant(37.0)
            predicate = array_ops.placeholder_with_default(constant_op.constant(True), [])
            cond_output = tf_cond.cond(predicate, lambda : constant_op.constant(0.0), lambda : value)
            result = array_ops.identity(cond_output)
            self.assertAllEqual(37.0, sess.run(result, feed_dict={predicate: False}))
            self.assertAllEqual(0.0, sess.run(result, feed_dict={predicate: True}))
            self.assertAllEqual(0.0, sess.run(result))

    def testCondTensorDeps(self):
        if False:
            i = 10
            return i + 15
        t = array_ops.identity(1.0)

        @eager_def_function.function
        def f():
            if False:
                for i in range(10):
                    print('nop')
            with ops.control_dependencies([t]):
                return array_ops.identity(2.0)
        f.get_concrete_function()

    @test_util.run_in_graph_and_eager_modes
    def testCondAutoControlDeps(self):
        if False:
            print('Hello World!')
        if test_util.is_gpu_available():
            self.skipTest('b/128676188 causes OOM on opensource gpu tests')
        print_prefix = 'testCondAutoControlDeps: '

        def branch_fn():
            if False:
                i = 10
                return i + 15
            enqueue_print_op('A')
            enqueue_print_op('B')
            with ops.control_dependencies([enqueue_print_op('C')]):
                return constant_op.constant(10)

        def build_cond():
            if False:
                while True:
                    i = 10
            return tf_cond.cond(constant_op.constant(True), branch_fn, lambda : 0)

        def build_nested_cond():
            if False:
                while True:
                    i = 10
            return tf_cond.cond(constant_op.constant(True), build_cond, lambda : 0)
        if not context.executing_eagerly():
            with self.cached_session():
                with self.captureWritesToStream(sys.stderr) as printed:
                    self.assertEqual(self.evaluate(build_cond()), 10)
                self.assertEqual(['C'], filter_test_messages(printed.contents()))
                with self.captureWritesToStream(sys.stderr) as printed:
                    self.assertEqual(self.evaluate(build_nested_cond()), 10)
                self.assertEqual(['C'], filter_test_messages(printed.contents()))
        if control_flow_util.ENABLE_CONTROL_FLOW_V2:

            @eager_def_function.function
            def cond():
                if False:
                    for i in range(10):
                        print('nop')
                return build_cond()
            with self.captureWritesToStream(sys.stderr) as printed:
                self.assertEqual(self.evaluate(cond()), 10)
            self.assertEqual(['A', 'B', 'C'], filter_test_messages(printed.contents()))

            @eager_def_function.function
            def nested_cond():
                if False:
                    i = 10
                    return i + 15
                return build_nested_cond()
            with self.captureWritesToStream(sys.stderr) as printed:
                self.assertEqual(self.evaluate(nested_cond()), 10)
            self.assertEqual(['A', 'B', 'C'], filter_test_messages(printed.contents()))

        def pruned_cond():
            if False:
                for i in range(10):
                    print('nop')
            return build_cond()
        pruned_cond = wrap_function.wrap_function(pruned_cond, [])
        with self.captureWritesToStream(sys.stderr) as printed:
            self.assertEqual(self.evaluate(pruned_cond()), 10)
        self.assertEqual(['C'], filter_test_messages(printed.contents()))

        def pruned_nested_cond():
            if False:
                while True:
                    i = 10
            return build_nested_cond()
        pruned_nested_cond = wrap_function.wrap_function(pruned_nested_cond, [])
        with self.captureWritesToStream(sys.stderr) as printed:
            self.assertEqual(self.evaluate(pruned_nested_cond()), 10)
        self.assertEqual(['C'], filter_test_messages(printed.contents()))

    @test_util.run_in_graph_and_eager_modes
    @test_util.disable_tfrt('b/179459136')
    def testWhileAutoControlDeps(self):
        if False:
            i = 10
            return i + 15
        if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
            return

        def cond(i, unused_x):
            if False:
                while True:
                    i = 10
            enqueue_print_op('A')
            return i < 2

        def body(i, x):
            if False:
                i = 10
                return i + 15
            enqueue_print_op('B')
            with ops.control_dependencies([enqueue_print_op('C')]):
                x = array_ops.identity(x)
            with ops.control_dependencies([enqueue_print_op('D')]):
                return (i + 1, x)

        def build_while():
            if False:
                return 10
            return while_loop_tf.while_loop(cond, body, [constant_op.constant(0), constant_op.constant(0)])

        def build_nested_while():
            if False:
                while True:
                    i = 10
            return tf_cond.cond(constant_op.constant(True), build_while, lambda : [0, 0])
        if not context.executing_eagerly():
            with self.cached_session():
                with self.captureWritesToStream(sys.stderr) as printed:
                    self.assertEqual(self.evaluate(build_while()[0]), 2)
                self.assertEqual(['D', 'D'], filter_test_messages(printed.contents()))
                with self.captureWritesToStream(sys.stderr) as printed:
                    self.assertEqual(self.evaluate(build_nested_while()[0]), 2)
                self.assertEqual(['D', 'D'], filter_test_messages(printed.contents()))

        @eager_def_function.function
        def while_loop():
            if False:
                while True:
                    i = 10
            return build_while()[0]
        with self.captureWritesToStream(sys.stderr) as printed:
            self.assertEqual(self.evaluate(while_loop()), 2)
        self.assertEqual(['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A'], filter_test_messages(printed.contents()))

        @eager_def_function.function
        def nested_while_loop():
            if False:
                for i in range(10):
                    print('nop')
            return build_nested_while()[0]
        with self.captureWritesToStream(sys.stderr) as printed:
            self.assertEqual(self.evaluate(nested_while_loop()), 2)
        self.assertEqual(['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A'], filter_test_messages(printed.contents()))

        def pruned_while():
            if False:
                return 10
            return build_while()[0]
        pruned_while = wrap_function.wrap_function(pruned_while, [])
        with self.captureWritesToStream(sys.stderr) as printed:
            self.assertEqual(self.evaluate(pruned_while()), 2)
        self.assertEqual(['D', 'D'], filter_test_messages(printed.contents()))

        def pruned_nested_while():
            if False:
                i = 10
                return i + 15
            return build_nested_while()[0]
        pruned_nested_while = wrap_function.wrap_function(pruned_nested_while, [])
        with self.captureWritesToStream(sys.stderr) as printed:
            self.assertEqual(self.evaluate(pruned_nested_while()), 2)
        self.assertEqual(['D', 'D'], filter_test_messages(printed.contents()))

    def testWhile_1(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            n = constant_op.constant(0)
            c = lambda x: math_ops.less(x, 10000)
            b = lambda x: math_ops.add(x, 1)
            r = while_loop_tf.while_loop(c, b, [n], parallel_iterations=20)
            self.assertEqual(10000, self.evaluate(r))

    @test_util.run_v1_only('b/120545219')
    def testWhileExternalControlDependencies(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            v = variables.Variable(0.0)
            self.evaluate(v.initializer)
            increment = v.assign_add(1.0).read_value()

            def body_fn(i):
                if False:
                    return 10
                with ops.control_dependencies([increment]):
                    return i + 1
            result = while_loop_tf.while_loop(cond=lambda i: i < 2, body=body_fn, loop_vars=[1])
            self.assertAllEqual(result, 2)
            self.assertAllEqual(v.read_value(), 1.0)

    @test_util.run_v1_only('b/120545219')
    def testWhileExternalControlDependenciesNoInput(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            v = variables.Variable(0.0)
            self.evaluate(v.initializer)
            increment = v.assign_add(1.0).read_value()

            def body_fn(unused_i):
                if False:
                    print('Hello World!')
                with ops.control_dependencies([increment]):
                    return constant_op.constant(5, name='five')
            result = while_loop_tf.while_loop(cond=lambda i: i < 5, body=body_fn, loop_vars=[0])
            self.evaluate(result)
            self.assertAllEqual(self.evaluate(v), 1.0)

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileWithRefs_1(self):
        if False:
            return 10
        with self.cached_session() as sess:
            x = variable_v1.VariableV1(0)._ref()
            i = constant_op.constant(0)
            c = lambda i, x: math_ops.less(i, 100)
            self.assertEqual(x.dtype, dtypes.int32_ref)

            def b(i, x):
                if False:
                    print('Hello World!')
                self.assertEqual(x.dtype, dtypes.int32_ref)
                return (i + 1, gen_array_ops.ref_identity(x))
            r = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=5)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(r[0].dtype, dtypes.int32)
            self.assertEqual(r[1].dtype, dtypes.int32_ref)
            (value_i, value_x) = self.evaluate(r)
        self.assertEqual(100, value_i)
        self.assertEqual(0, value_x)

    def testWhile_2(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            s = constant_op.constant(0)
            r = isum(s)
            self.assertAllEqual(45, self.evaluate(r))

    def testWhileWithMaximumIterations(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            s = constant_op.constant([1, 2, 3, 4, 5])
            r = isum(s, maximum_iterations=3)
            self.assertAllEqual([1 + 3, 2 + 3, 3 + 3, 4 + 3, 5 + 3], self.evaluate(r))

    @test_util.run_v1_only('b/120545219')
    def testWhileWithMaximumIterationsAndSingleArgument(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            r = while_loop_tf.while_loop(lambda i: i < 3, lambda i: i + 1, [0], maximum_iterations=1)
            self.assertEqual(1, self.evaluate(r))

    @test_util.run_v1_only('b/120545219')
    def testXLAGradInLoop(self):
        if False:
            return 10
        input1 = array_ops.placeholder(dtype=dtypes.float32, shape=[None, None])
        input2 = array_ops.placeholder(dtype=dtypes.float32, shape=[None, None])

        def cond(i1, i2):
            if False:
                return 10
            return False

        def body(i1, i2):
            if False:
                i = 10
                return i + 15
            return (math_ops.add(i1, i2), math_ops.add(i1, i2))
        xla_context = control_flow_ops.XLAControlFlowContext()
        xla_context.Enter()
        (out1, _) = while_loop_tf.while_loop(cond, body, (input1, input2), maximum_iterations=2)
        g = gradients_impl.gradients(out1, [input1])
        for op in out1.graph.get_operations():
            if op.type == 'BroadcastGradientArgs':
                self.assertEqual(op.inputs[0].op.type, 'Shape')
                self.assertEqual(op.inputs[1].op.type, 'Shape')
        xla_context.Exit()

    @test_util.disable_control_flow_v2('b/115776323 (max_iters)')
    @test_util.run_v1_only('b/120545219')
    def testSingleNestedMaximumIterationsWhileLoopGradientInXLAContext(self):
        if False:
            print('Hello World!')
        v = constant_op.constant(1.0)

        def training_loop_with_gradient(i):
            if False:
                while True:
                    i = 10
            out = while_loop_tf.while_loop(lambda i_, _: i_ < 3, lambda i_, j: [i_ + 1, j * v], [0, 1.0], maximum_iterations=i)
            g = gradients_impl.gradients(out, v)
            with ops.control_dependencies(g):
                return i + 1
        xla_context = control_flow_ops.XLAControlFlowContext()
        xla_context.Enter()
        loop = while_loop_tf.while_loop(lambda i: i < 3, training_loop_with_gradient, [0])
        xla_context.Exit()
        loop_execute = array_ops.identity(loop)
        self.assertEqual(3, self.evaluate(loop_execute))

    @test_util.run_v1_only('b/120545219')
    def testInvalidMaximumIterationsWhileLoopGradientInXLAContext(self):
        if False:
            for i in range(10):
                print('nop')
        if control_flow_util.ENABLE_CONTROL_FLOW_V2:
            self.skipTest('WhileV2 does lazy evaluation of maximum_iterations')
        v = constant_op.constant(1.0)

        def inner_body(i, x):
            if False:
                i = 10
                return i + 15
            out = while_loop_tf.while_loop(lambda i, _: i < 3, lambda i, j: [i + 1, j * v], [0, x], maximum_iterations=i)
            return out

        def create_while_loop(maximum_iterations=None):
            if False:
                return 10
            return while_loop_tf.while_loop(lambda i, _: i < 3, inner_body, [0, 1.0], maximum_iterations=maximum_iterations)
        loop_no_xla = create_while_loop(maximum_iterations=5)
        gs = gradients_impl.gradients(loop_no_xla, v)
        self.evaluate(gs)
        xla_context = control_flow_ops.XLAControlFlowContext()
        xla_context.Enter()
        loop_no_maxiter = create_while_loop()
        loop_with_maxiter = create_while_loop(maximum_iterations=2)
        xla_context.Exit()
        with self.assertRaisesRegex(ValueError, "Cannot create a gradient accumulator for tensor '.+' inside XLA while_loop because maximum_iterations was not passed to the tf.while_loop call \\('.+'\\)."):
            _ = gradients_impl.gradients(loop_no_maxiter, v)
        with self.assertRaisesRegex(ValueError, "Cannot create a gradient accumulator for tensor '.+' inside XLA while_loop. maximum_iterations tensor '.+' for while_loop context '.+' must be statically known \\(e.g. a constant value or known shape dimension\\), or be defined at or outside the while loop context '.*' \\(currently defined in '.*'\\)"):
            _ = gradients_impl.gradients(loop_with_maxiter, v)

    @test_util.run_v1_only('b/120545219')
    def testInvalidMaximumIterationsFromSiblingContextWhileLoopInXLAContext(self):
        if False:
            print('Hello World!')
        v = constant_op.constant(1.0)

        def create_while_loop():
            if False:
                for i in range(10):
                    print('nop')
            max_iter_holder = []

            def create_mi():
                if False:
                    return 10
                max_iter_holder.append(array_ops.placeholder(dtypes.int32, shape=()))
                return 1.0
            _ = tf_cond.cond(constant_op.constant(True), create_mi, create_mi)
            return while_loop_tf.while_loop(lambda i, _: i < 3, lambda i, x: (i + 1, v * x), (0, 1.0), maximum_iterations=max_iter_holder[0])
        if control_flow_util.ENABLE_CONTROL_FLOW_V2:
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()
            with self.assertRaisesRegex(ValueError, 'must be from the same graph.*'):
                loop = create_while_loop()
            xla_context.Exit()
        else:
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()
            loop = create_while_loop()
            xla_context.Exit()
            with self.assertRaisesRegex(ValueError, "Cannot create a gradient accumulator for tensor '.+' inside XLA while_loop. maximum_iterations tensor '.*Placeholder:0' for while_loop context '.+' must be statically known \\(e.g. a constant value or known shape dimension\\), or be defined at or outside the while loop context '' \\(currently defined in 'cond/.+'\\)"):
                _ = gradients_impl.gradients(loop, v)

    @test_util.run_v1_only('b/120545219')
    def testNestedWhileLoopWithMaxItersFromOuterContextInXLAContext(self):
        if False:
            print('Hello World!')
        if test_util.is_gpu_available():
            self.skipTest('b/128646372, b/128645947 fails in opensource build')
        v = constant_op.constant(1.0)
        p = array_ops.placeholder(dtype=dtypes.int32)

        def mid_body_builder(iterations):
            if False:
                i = 10
                return i + 15

            def mid_body(i, x):
                if False:
                    for i in range(10):
                        print('nop')
                r = while_loop_tf.while_loop(lambda *_: True, lambda i, x: (i + 1, v * x), (0, x), maximum_iterations=iterations, name='inner')
                return (i + 1, gradients_impl.gradients(x + r[1], v)[0])
            return mid_body

        def outer_body(i, x):
            if False:
                print('Hello World!')
            iterations = array_ops.size(p, name='iterations')
            return (i + 1, x + while_loop_tf.while_loop(lambda *_: True, mid_body_builder(iterations), (0, x), maximum_iterations=iterations, name='mid')[1])

        def create_while_loop():
            if False:
                while True:
                    i = 10
            with ops.device('/cpu:0'):
                r = while_loop_tf.while_loop(lambda *_: True, outer_body, (0, 1.0), maximum_iterations=5, name='outer')
                return array_ops.identity(r[1])
        xla_context = control_flow_ops.XLAControlFlowContext()
        xla_context.Enter()
        final_with_xla_context = create_while_loop()
        xla_context.Exit()
        final_without_xla_context = create_while_loop()
        with self.session(use_gpu=False) as sess:
            opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata_without_xla_context = config_pb2.RunMetadata()
            run_metadata = config_pb2.RunMetadata()
            final_value_without_xla_context = sess.run(final_without_xla_context, feed_dict={p: [0, 0, 0]}, options=opts, run_metadata=run_metadata_without_xla_context)
            final_value_with_xla_context = sess.run(final_with_xla_context, feed_dict={p: [0, 0, 0]}, options=opts, run_metadata=run_metadata)
            if control_flow_util.ENABLE_CONTROL_FLOW_V2:
                for dev in run_metadata_without_xla_context.step_stats.dev_stats:
                    if '/device:CPU' in dev.device:
                        node_stats = dev.node_stats
                stack_push_count = len([x for x in node_stats if re.match('.*TensorListPushBack_?\\d*', x.node_name)])
            else:
                for dev in run_metadata.step_stats.dev_stats:
                    if '/device:CPU' in dev.device:
                        node_stats = dev.node_stats
                stack_push_op = 'StackPushV2'
                stack_push_count = len([x for x in node_stats if x.node_name.endswith('StackPushV2')])
            self.assertEqual(stack_push_count, 5 * 3 * 3, str(node_stats))
            self.assertAllClose(final_value_with_xla_context, final_value_without_xla_context)

    @test_util.run_deprecated_v1
    def testWhile_3(self):
        if False:
            return 10
        with self.cached_session():

            def compute(i, m, c, o):
                if False:
                    i = 10
                    return i + 15
                (m, c) = [math_ops.add(m, 1), math_ops.add(c, 1)]
                o = math_ops.add(o, m)
                o = math_ops.add(o, c)
                i = math_ops.add(i, 1)
                return [i, m, c, o]
            i = ops.convert_to_tensor(0)
            m = ops.convert_to_tensor(0)
            c = ops.convert_to_tensor(0)
            o = ops.convert_to_tensor(0)
            d = ops.convert_to_tensor(100)
            r = while_loop_tf.while_loop(lambda i, m, c, o: math_ops.less(i, d), compute, [i, m, c, o])
            result = r[3]
        self.assertAllEqual(10100, result)

    @test_util.run_deprecated_v1
    def testWhile_4(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():

            def compute(i, m, c, o):
                if False:
                    return 10
                (m, c) = [array_ops.gather(x, i), array_ops.gather(x, i)]
                o = math_ops.add(o, m)
                o = math_ops.add(o, c)
                i = math_ops.add(i, 1)
                return [i, m, c, o]
            i = ops.convert_to_tensor(0)
            m = ops.convert_to_tensor(0)
            c = ops.convert_to_tensor(0)
            o = ops.convert_to_tensor(0)
            x = ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
            s = array_ops.size(x)
            r = while_loop_tf.while_loop(lambda i, m, c, o: math_ops.less(i, s), compute, [i, m, c, o])
            result = r[3]
        self.assertAllEqual(42, result)

    @test_util.run_v1_only('b/120545219')
    def testWhile_5(self):
        if False:
            return 10
        with self.cached_session():

            def compute(i, c, o):
                if False:
                    i = 10
                    return i + 15
                c = array_ops.strided_slice(x, array_ops.expand_dims(i, 0), [1] + array_ops.expand_dims(i, 0))
                o = array_ops.concat([o, c], 0)
                i = math_ops.add(i, 1)
                return [i, c, o]
            i = ops.convert_to_tensor(0)
            c = ops.convert_to_tensor([0])
            o = ops.convert_to_tensor([0])
            x = ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
            s = array_ops.size(x)
            r = while_loop_tf.while_loop(lambda i, c, o: math_ops.less(i, s), compute, [i, c, o], [i.get_shape(), tensor_shape.unknown_shape(), tensor_shape.unknown_shape()])
            result = r[2]
        self.assertAllEqual(np.array([0, 1, 2, 3, 4, 5, 6]), result)

    @test_util.run_gpu_only
    @test_util.run_deprecated_v1
    def testWhile_Device(self):
        if False:
            return 10

        def body(x):
            if False:
                while True:
                    i = 10
            return math_ops.exp(x)
        with ops.device('CPU:0'):
            r = while_loop_tf.while_loop(lambda x: x < 10, body, [constant_op.constant(-10.0)])
            self.assertIn('cpu', r.device.lower())
        with session.Session() as sess:
            options = config_pb2.RunOptions(output_partition_graphs=True)
            run_metadata = config_pb2.RunMetadata()
            sess.run(r, options=options, run_metadata=run_metadata)
            self.assertEqual(len(run_metadata.partition_graphs), 1)

    @test_util.disable_control_flow_v2('b/116338794 (buffer_reuse)')
    @test_util.run_v1_only('b/120545219')
    def testBufferForwarding(self):
        if False:
            for i in range(10):
                print('nop')
        run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()
        with self.cached_session() as sess:
            with ops.device('/cpu:0'):
                c = constant_op.constant(2)
                i0 = constant_op.constant(0)
                r = while_loop_tf.while_loop(lambda i: i < 1000, lambda i: math_ops.square(c) + i, [i0])
            r_val = sess.run(r, options=run_options, run_metadata=run_metadata)
            self.assertEqual(1000, r_val)
            self.assertTrue(run_metadata.HasField('step_stats'))
            unique_allocs = set()
            for node_stat in run_metadata.step_stats.dev_stats[0].node_stats:
                for output in node_stat.output:
                    unique_allocs.add(output.tensor_description.allocation_description.ptr)
            self.assertLess(len(unique_allocs), 756)

    def _testWhile_Gpu_1(self, use_gpu):
        if False:
            print('Hello World!')
        with self.cached_session(use_gpu=use_gpu):
            n = constant_op.constant(1.0)
            c = lambda x: math_ops.less(x, 10.0)
            b = lambda x: math_ops.add(x, 1.0)
            r = while_loop_tf.while_loop(c, b, [n])
            self.assertAllClose(10.0, self.evaluate(r))

    def testWhile_Gpu_1(self):
        if False:
            print('Hello World!')
        self._testWhile_Gpu_1(use_gpu=False)
        self._testWhile_Gpu_1(use_gpu=True)

    def _testWhile_Gpu_2(self, use_gpu):
        if False:
            return 10
        with self.cached_session(use_gpu=use_gpu):
            n = constant_op.constant(1.0)
            c = lambda x: math_ops.less(x, 10.0)

            def b(x):
                if False:
                    i = 10
                    return i + 15
                with ops.device('/cpu:0'):
                    return math_ops.add(x, 1.0)
            r = while_loop_tf.while_loop(c, b, [n])
            self.assertAllClose(10.0, self.evaluate(r))

    def testWhile_Gpu_2(self):
        if False:
            return 10
        self._testWhile_Gpu_2(use_gpu=False)
        self._testWhile_Gpu_2(use_gpu=True)

    def testWhileShape(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            i = constant_op.constant(0)
            m = array_ops.ones([2, 2])
            c = lambda i, j: math_ops.less(i, 2)

            def _b(i, j):
                if False:
                    print('Hello World!')
                new_i = math_ops.add(i, 1)
                new_j = array_ops.tile(j, [2, 2])
                return [new_i, new_j]
            r = while_loop_tf.while_loop(c, _b, [i, m], [i.get_shape(), tensor_shape.unknown_shape()])
            r = r[1] * array_ops.ones([8, 8])
            self.assertAllEqual(np.ones((8, 8)), self.evaluate(r))

    @test_util.disable_control_flow_v2('b/131265085')
    @test_util.run_v1_only('b/131265085')
    def testWhileBadShape(self):
        if False:
            return 10
        x = constant_op.constant([2.0, 4.0], name='values')
        i = constant_op.constant(0)
        c = lambda i, _: math_ops.less(i, 10)
        b = lambda i, x: [i + 1, x + 1]
        with self.assertRaisesRegex(ValueError, 'is not compatible with'):
            while_loop_tf.while_loop(c, b, [i, x], [i.shape, tensor_shape.TensorShape([5])])

    @test_util.run_in_graph_and_eager_modes
    def testWhileBadBodyReturn(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant([2.0, 4.0], name='values')
        i = constant_op.constant(0)
        c = lambda i, *x: math_ops.less(i, 10)
        b = lambda i, *x: (i, i) + x
        with self.assertRaisesRegex(ValueError, "The two structures don't have the same nested structure."):
            while_loop_tf.while_loop(c, b, [i, x])

    @test_util.run_deprecated_v1
    def testWhileWithNonTensorInput_Scalar(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            n = 0
            c = lambda x: x < 10000
            b = lambda x: x + 1
            r = while_loop_tf.while_loop(c, b, [n], parallel_iterations=20)
            self.assertEqual(10000, self.evaluate(r))

    def testWhileWithNonTensorInput_Vector(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            n = np.array([0])
            c = lambda x: x[0] < 10000
            b = lambda x: array_ops_stack.stack([x[0] + 1])
            r = while_loop_tf.while_loop(c, b, [n], parallel_iterations=20)
            self.assertEqual([10000], self.evaluate(r))

    def testWhileShapeInference(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            i = constant_op.constant(0)
            m = array_ops.ones([2, 2])
            c = lambda i, j: math_ops.less(i, 2)

            def b(i, j):
                if False:
                    for i in range(10):
                        print('nop')
                new_i = math_ops.add(i, 1)
                new_j = array_ops.concat([j, j], 0)
                return [new_i, new_j]
            r = while_loop_tf.while_loop(c, b, [i, m], [i.get_shape(), tensor_shape.TensorShape([None, 2])])
            self.assertTrue(r[1].shape.is_compatible_with([8, 2]))

    @test_util.run_v1_only('b/120545219')
    def testWhileShapeInferenceBadShape(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            i = constant_op.constant(0)
            m = array_ops.ones([2, 2])
            c = lambda i, j: math_ops.less(i, 2)
            b = lambda i, j: [i + 1, array_ops.concat([j, j], 0)]
            with self.assertRaisesRegex(ValueError, '.*\\(2, 2\\).*\\(4, 2\\) after one iteration\\. To allow the shape to vary across iterations, use the `shape_invariants` argument of tf.while_loop to specify a less-specific shape\\.'):
                while_loop_tf.while_loop(c, b, [i, m])

    def testWhileShapeInferenceSparseTensor(self):
        if False:
            for i in range(10):
                print('nop')
        values = constant_op.constant([2.0, 4.0], name='values')
        indices = constant_op.constant([[0], [3]], dtype=dtypes.int64, name='indices')
        shape = constant_op.constant([10], dtype=dtypes.int64, name='dense_shape')
        i = constant_op.constant(0)
        x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)

        def c(i, _):
            if False:
                return 10
            return i < 10

        def b1(i, x):
            if False:
                print('Hello World!')
            return [i + 1, sparse_tensor.SparseTensor(x.indices, x.values * 2.0, x.dense_shape)]

        def b2(i, x):
            if False:
                i = 10
                return i + 15
            return [i + 1, sparse_ops.sparse_add(x, sparse_tensor.SparseTensor(indices=math_ops.cast(array_ops.fill([1, 1], i), dtypes.int64), values=array_ops.fill([1], 1.0), dense_shape=x.dense_shape))]

        def b3(i, x):
            if False:
                for i in range(10):
                    print('nop')
            return [i + 1, sparse_tensor.SparseTensor(array_ops.concat([x.indices, [[i], [i]]], axis=1), x.values * 2.0, array_ops.concat([x.dense_shape, [10]], axis=0))]

        def check_shapes(r, indices, values, dense_shape):
            if False:
                print('Hello World!')
            self.assertTrue(r.indices.shape.is_compatible_with(indices))
            self.assertTrue(r.values.shape.is_compatible_with(values))
            self.assertTrue(r.dense_shape.shape.is_compatible_with(dense_shape))
        (_, r) = while_loop_tf.while_loop(c, b1, [i, x])
        check_shapes(r, indices=[None, 1], values=[None], dense_shape=[1])
        (_, r) = while_loop_tf.while_loop(c, b2, [i, x])
        check_shapes(r, indices=[None, 1], values=[None], dense_shape=[1])
        (_, r) = while_loop_tf.while_loop(c, b1, [i, x], [i.get_shape(), tensor_shape.TensorShape([None])])
        check_shapes(r, indices=[None, None], values=[None], dense_shape=[None])
        (_, r) = while_loop_tf.while_loop(c, b3, [i, x], [i.get_shape(), tensor_shape.TensorShape([None])])
        check_shapes(r, indices=[None, None], values=[None], dense_shape=[None])
        (_, r) = while_loop_tf.while_loop(c, b1, [i, x], [i.get_shape(), tensor_shape.TensorShape(None)])
        check_shapes(r, indices=[None, None], values=[None], dense_shape=[None])
        (_, r) = while_loop_tf.while_loop(c, b3, [i, x], [i.get_shape(), tensor_shape.TensorShape(None)])
        check_shapes(r, indices=[None, None], values=[None], dense_shape=[None])

    @test_util.disable_control_flow_v2('b/131265085')
    @test_util.run_v1_only('b/131265085')
    def testWhileBadShapeSparseTensor(self):
        if False:
            for i in range(10):
                print('nop')
        values = constant_op.constant([2.0, 4.0], name='values')
        indices = constant_op.constant([[0], [3]], dtype=dtypes.int64, name='indices')
        shape = constant_op.constant([10], dtype=dtypes.int64, name='dense_shape')
        i = constant_op.constant(0)
        x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)
        c = lambda i, _: i < 10
        b1 = lambda i, x: [i + 1, x]

        def b2(i, x):
            if False:
                return 10
            return [i + 1, sparse_tensor.SparseTensor(array_ops.concat([x.indices, [[i], [i]]], axis=1), x.values * 2.0, array_ops.concat([x.dense_shape, [10]], axis=0))]
        with self.assertRaisesRegex(ValueError, 'is not compatible with'):
            while_loop_tf.while_loop(c, b1, [i, x], [i.get_shape(), tensor_shape.TensorShape([5])])
        with self.assertRaises(ValueError):
            while_loop_tf.while_loop(c, b2, [i, x])

    def testWhileShapeInferenceIndexedSlices(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            values = constant_op.constant([[2.0, 4.0], [3.0, 5.0]], name='values')
            indices = constant_op.constant([0, 3], name='indices')
            shape = constant_op.constant([10, 2], name='dense_shape')
            i = constant_op.constant(0)
            x = indexed_slices.IndexedSlices(values, indices, dense_shape=shape)

            def c(i, _):
                if False:
                    print('Hello World!')
                return i < 10

            def b(i, x):
                if False:
                    for i in range(10):
                        print('nop')
                return [i + 1, indexed_slices.IndexedSlices(x.values * 2.0, x.indices, x.dense_shape)]
            (_, r) = while_loop_tf.while_loop(c, b, [i, x])
            self.assertEqual(r.dense_shape.get_shape()[0], 2)
            self.assertEqual(r.values.get_shape(), tensor_shape.TensorShape([2, 2]))
            (_, r) = while_loop_tf.while_loop(c, b, [i, x], [i.get_shape(), tensor_shape.TensorShape([None, 2])])
            self.assertEqual(r.dense_shape.get_shape()[0], 2)
            self.assertTrue(r.values.get_shape().is_compatible_with([None, 2]))

    @test_util.disable_control_flow_v2('b/131265085')
    @test_util.run_v1_only('b/131265085')
    def testWhileBadShapeIndexedSlices(self):
        if False:
            for i in range(10):
                print('nop')
        values = constant_op.constant([2.0, 4.0], name='values')
        indices = constant_op.constant([[0], [3]], dtype=dtypes.int64, name='indices')
        shape = constant_op.constant([10], dtype=dtypes.int64, name='dense_shape')
        i = constant_op.constant(0)
        x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)
        c = lambda i, _: 10
        b = lambda i, x: [i + 1, x]
        with self.assertRaisesRegex(ValueError, 'is not compatible with'):
            while_loop_tf.while_loop(c, b, [i, x], [i.get_shape(), tensor_shape.TensorShape([5])])

    def testWhileShapeInferenceRaggedTensor(self):
        if False:
            for i in range(10):
                print('nop')
        i = constant_op.constant(0)
        x = ragged_factory_ops.constant([[1, 2], [3], [4, 5, 6]])
        c = lambda i, _: i < 10

        def b1(i, x):
            if False:
                print('Hello World!')
            return [i + 1, array_ops.concat([x, x], axis=1)]

        def b2(i, x):
            if False:
                for i in range(10):
                    print('nop')
            return [i + 1, array_ops.concat([x, x], axis=0)]

        def check_shapes(r, values, splits):
            if False:
                print('Hello World!')
            self.assertTrue(r.values.shape.is_compatible_with(values))
            self.assertTrue(r.row_splits.shape.is_compatible_with(splits))
        (_, r) = while_loop_tf.while_loop(c, b1, [i, x])
        check_shapes(r, values=[None], splits=[4])
        if not context.executing_eagerly():
            with self.assertRaises(ValueError):
                (_, r) = while_loop_tf.while_loop(c, b2, [i, x])
        (_, r) = while_loop_tf.while_loop(c, b1, [i, x], [i.get_shape(), tensor_shape.TensorShape([None, None])])
        check_shapes(r, values=[None], splits=[None])
        (_, r) = while_loop_tf.while_loop(c, b1, [i, x], [i.get_shape(), ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)])
        check_shapes(r, values=[None], splits=[None])
        (_, r) = while_loop_tf.while_loop(c, b2, [i, x], [i.get_shape(), ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)])
        check_shapes(r, values=[None], splits=[None])

    def testWhileShapeInferenceRaggedTensorRaggedRank2(self):
        if False:
            return 10
        i = constant_op.constant(0)
        x = ragged_factory_ops.constant([[[1, 2], [3], [4, 5, 6]], [[], [8, 9, 10]]])
        c = lambda i, _: i < 10

        def b(i, x):
            if False:
                i = 10
                return i + 15
            return [i + 1, array_ops.concat([x, x[..., i:i + 1]], axis=-1)]
        (_, r) = while_loop_tf.while_loop(c, b, [i, x])
        self.assertEqual(r.row_splits.shape.as_list(), [3])
        self.assertIn(r.values.row_splits.shape.as_list(), ([6], [None]))
        self.assertIn(r.values.values.shape.as_list(), ([49], [None]))

    def testWhileShapeInvariantTensorSpec(self):
        if False:
            i = 10
            return i + 15
        i = constant_op.constant(0)
        x = constant_op.constant([1])
        c = lambda i, _: i < 10
        b = lambda i, x: (i + 1, array_ops_stack.stack([x, x]))
        shape_invariants = [tensor_lib.TensorSpec([], dtype=dtypes.int32), tensor_lib.TensorSpec(None, dtype=dtypes.int32)]
        while_loop_tf.while_loop(c, b, [i, x], shape_invariants)

    @test_util.build_as_function_and_v1_graph
    def testWhileShapeInvariantWrongTypeSpecType(self):
        if False:
            for i in range(10):
                print('nop')
        c = lambda i, _: i < 10
        b = lambda i, x: (i + 1, x)
        i = constant_op.constant(0)
        x = sparse_tensor.SparseTensor([[0]], [1.0], [10])
        shape_invariants = [tensor_lib.TensorSpec([], dtype=dtypes.int32), sparse_tensor.SparseTensorSpec([None])]
        while_loop_tf.while_loop(c, b, [i, x], shape_invariants)
        x2 = constant_op.constant([1])
        with self.assertRaises(TypeError):
            while_loop_tf.while_loop(c, b, [i, x2], shape_invariants)
        x3 = ragged_factory_ops.constant([[1, 2], [3]])
        with self.assertRaises(TypeError):
            while_loop_tf.while_loop(c, b, [i, x3], shape_invariants)
        i2 = constant_op.constant(0.0)
        with self.assertRaises(TypeError):
            while_loop_tf.while_loop(c, b, [i2, x], shape_invariants)

    @test_util.build_as_function_and_v1_graph
    def testWhileShapeInvariantBadType(self):
        if False:
            while True:
                i = 10
        i = constant_op.constant(0)
        x = constant_op.constant([1])
        c = lambda i, _: i < 10
        b = lambda i, x: (i + 1, x)
        with self.assertRaises((ValueError, TypeError)):
            while_loop_tf.while_loop(c, b, [i, x], ['foo', 'bar'])

    def _testNestedWhile_1(self, use_gpu):
        if False:
            print('Hello World!')
        with self.cached_session(use_gpu=use_gpu):
            n = constant_op.constant(0)

            def cpu_sum(s):
                if False:
                    for i in range(10):
                        print('nop')
                c = lambda i, s: math_ops.less(i, 10)

                def b(i, s):
                    if False:
                        print('Hello World!')
                    i1 = math_ops.add(i, 1)
                    with ops.device('/cpu:0'):
                        s1 = math_ops.add(i, s)
                    return (i1, s1)
                (_, r_s) = while_loop_tf.while_loop(c, b, [n, s])
                return r_s
            c = lambda x: math_ops.less(x, 200)
            b = lambda x: math_ops.add(x, cpu_sum(n))
            r = while_loop_tf.while_loop(c, b, [n])
            self.assertEqual(225, self.evaluate(r))

    def testNestedWhile_1(self):
        if False:
            return 10
        self._testNestedWhile_1(use_gpu=False)
        self._testNestedWhile_1(use_gpu=True)

    def _testNestedWhile_2(self, use_gpu):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session(use_gpu=use_gpu):
            s0 = constant_op.constant(2.0)

            def inner_loop(s):
                if False:
                    return 10
                c = lambda s: math_ops.less(s, 20.0)

                def b(s):
                    if False:
                        i = 10
                        return i + 15
                    s1 = math_ops.add(s, s)
                    return s1
                r_s = while_loop_tf.while_loop(c, b, [s], parallel_iterations=1)
                return r_s
            outer_c = lambda x: math_ops.less(x, 3000.0)

            def outer_b(x):
                if False:
                    return 10
                x = logging_ops.Print(x, [x])
                x = inner_loop(x)
                with ops.device('/cpu:0'):
                    x = math_ops.square(x)
                return x
            r = while_loop_tf.while_loop(outer_c, outer_b, [s0], parallel_iterations=1)
            self.assertEqual(1048576.0, self.evaluate(r))

    def testNestedWhile_2(self):
        if False:
            while True:
                i = 10
        self._testNestedWhile_2(use_gpu=False)
        self._testNestedWhile_2(use_gpu=True)

    @test_util.run_v1_only('b/120545219')
    def testWhileWithControl_1(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            n = constant_op.constant(0)
            r = constant_op.constant(0)
            condition = lambda n_, r_: math_ops.less(n_, 10)

            def body(n_, r_):
                if False:
                    i = 10
                    return i + 15
                n_ = math_ops.add(n_, 1)
                with r_.graph.control_dependencies([r_]):
                    r_ = constant_op.constant(12)
                return [n_, r_]
            res = while_loop_tf.while_loop(condition, body, [n, r], parallel_iterations=1)
            self.assertAllEqual(12, res[1])

    @test_util.run_deprecated_v1
    def testWhileWithControl_2(self):
        if False:
            return 10
        with self.cached_session():
            r = constant_op.constant(0)
            condition = lambda r_: math_ops.less(r_, 10)

            def body(r_):
                if False:
                    print('Hello World!')
                with r_.graph.control_dependencies([r_]):
                    r_ = constant_op.constant(12)
                return [r_]
            res = while_loop_tf.while_loop(condition, body, [r], parallel_iterations=1)
            self.assertAllEqual(12, self.evaluate(res))

    @test_util.run_v1_only('b/120545219')
    def testWhileWithControl_3(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            b = array_ops.placeholder(dtypes.bool)
            c = constant_op.constant(1)
            x0 = constant_op.constant(0)
            with ops.control_dependencies([b]):
                r = while_loop_tf.while_loop(lambda x: x < 10, lambda x: x + c, [x0])
            self.assertEqual(10, sess.run(r, {b: True}))

    @test_util.run_v1_only('b/120545219')
    def testWhileWithControl_4(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            b = array_ops.placeholder(dtypes.bool)
            c = constant_op.constant(1)
            x0 = constant_op.constant(0)
            with ops.control_dependencies([b]):
                r = while_loop_tf.while_loop(lambda x: x < 10, lambda x: x + array_ops.identity(c), [x0])
            self.assertEqual(10, sess.run(r, {b: True}))

    @test_util.run_v1_only('b/120545219')
    def testWhileWithControl_5(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            b = array_ops.placeholder(dtypes.bool)
            c = constant_op.constant(1)
            x0 = constant_op.constant(0)

            def body(x):
                if False:
                    for i in range(10):
                        print('nop')
                with ops.control_dependencies([b]):
                    return x + c
            r = while_loop_tf.while_loop(lambda x: x < 10, body, [x0])
            self.assertEqual(10, sess.run(r, {b: True}))

    def testWhileCondWithControl(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            const_true = lambda : constant_op.constant(True)
            const_false = lambda : constant_op.constant(False)
            cond = lambda i: tf_cond.cond(i > 0, const_true, const_false)
            body = lambda i: tf_cond.cond(i > 0, lambda : i - 1, lambda : i)
            with ops.control_dependencies([control_flow_ops.no_op()]):
                loop = while_loop_tf.while_loop(cond, body, (constant_op.constant(5),))
            self.assertEqual(0, self.evaluate(loop))

    @test_util.disable_control_flow_v2('b/113324949 (ref vars)')
    @test_util.run_v1_only('b/120545219')
    def testWhileCondWithControl_1(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            v = variable_scope.get_variable('v', [], initializer=init_ops.constant_initializer(2))
            i0 = constant_op.constant(0)
            with ops.control_dependencies([i0]):

                def loop_condition(i):
                    if False:
                        i = 10
                        return i + 15
                    return i < 4

                def loop_body(i):
                    if False:
                        print('Hello World!')
                    some_cond = tf_cond.cond(constant_op.constant(True), lambda : state_ops.assign(v, math_ops.square(v)), lambda : v)
                    with ops.control_dependencies([some_cond]):
                        return i + 1
            r = while_loop_tf.while_loop(loop_condition, loop_body, (i0,))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(4, self.evaluate(r))
            self.assertAllClose(65536.0, self.evaluate(v))

    @test_util.disable_control_flow_v2('b/113324949 (ref vars)')
    @test_util.run_v1_only('b/120545219')
    def testWhileCondExitControl(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            v = variables.Variable(1)

            def false_branch():
                if False:
                    for i in range(10):
                        print('nop')
                cond = lambda i: i < 100

                def body(i):
                    if False:
                        for i in range(10):
                            print('nop')
                    x = state_ops.assign(v, i)
                    return x + 1
                loop = while_loop_tf.while_loop(cond, body, [0])
                with ops.control_dependencies([loop]):
                    return constant_op.constant(6.0)
            r = tf_cond.cond(constant_op.constant(False), lambda : constant_op.constant(1.0), false_branch)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(6.0, self.evaluate(r))
            self.assertEqual(99, self.evaluate(v))

    def testCondWhile_1(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            n = ops.convert_to_tensor(0, name='n')
            c = lambda x: math_ops.less(x, 10)
            b = lambda x: math_ops.add(x, 1)
            r = tf_cond.cond(math_ops.less(0, 1), lambda : while_loop_tf.while_loop(c, b, [n]), lambda : n)
            self.assertAllEqual(10, self.evaluate(r))

    def testCondWhile_2(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            n = ops.convert_to_tensor(0)
            c = lambda x: math_ops.less(x, 10)
            b = lambda x: math_ops.add(x, 1)
            r = tf_cond.cond(math_ops.less(1, 0), lambda : math_ops.add(n, 1), lambda : while_loop_tf.while_loop(c, b, [n]))
            self.assertAllEqual(10, self.evaluate(r))

    def _testCondWhile_3(self, use_gpu):
        if False:
            print('Hello World!')
        with self.cached_session(use_gpu=use_gpu) as sess:
            p = array_ops.placeholder(dtypes.bool)
            n = constant_op.constant(0.0)

            def c(x):
                if False:
                    print('Hello World!')
                return math_ops.less(x, 10.0)

            def b(x):
                if False:
                    for i in range(10):
                        print('nop')
                with ops.device('/cpu:0'):
                    x1 = math_ops.add(x, 1.0)
                return x1
            r = tf_cond.cond(p, lambda : while_loop_tf.while_loop(c, b, [n]), lambda : math_ops.multiply(n, 2.0))
            r1 = gradients_impl.gradients(r, [n])
            self.assertEqual(10.0, sess.run(r, {p: True}))
            self.assertEqual([1.0], sess.run(r1, {p: True}))
            self.assertEqual(0.0, sess.run(r, {p: False}))
            self.assertEqual([2.0], sess.run(r1, {p: False}))

    @test_util.run_deprecated_v1
    def testCondWhile_3(self):
        if False:
            i = 10
            return i + 15
        self._testCondWhile_3(use_gpu=False)
        self._testCondWhile_3(use_gpu=True)

    def testWhileCond_1(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            i = ops.convert_to_tensor(0, name='i')
            n = ops.convert_to_tensor(10, name='n')
            one = ops.convert_to_tensor(1, name='one')
            c = lambda x: math_ops.less(x, n)
            b = lambda x: tf_cond.cond(constant_op.constant(True), lambda : math_ops.add(x, one), lambda : math_ops.subtract(x, one))
            r = while_loop_tf.while_loop(c, b, [i])
            self.assertAllEqual(10, self.evaluate(r))

    def testWhileCond_2(self):
        if False:
            return 10
        with self.cached_session():
            n = ops.convert_to_tensor(0, name='n')
            c = lambda x: math_ops.less(x, 10)
            b = lambda x: tf_cond.cond(constant_op.constant(True), lambda : math_ops.add(x, 1), lambda : n)
            r = while_loop_tf.while_loop(c, b, [n])
            self.assertAllEqual(10, self.evaluate(r))

    def testWhileCond_3(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            n = ops.convert_to_tensor(0)
            c = lambda x: math_ops.less(x, 10)
            b = lambda x: tf_cond.cond(math_ops.less(0, 1), lambda : math_ops.add(x, 1), lambda : math_ops.subtract(x, 1))
            r = while_loop_tf.while_loop(c, b, [n])
            self.assertAllEqual(10, self.evaluate(r))

    @test_util.run_deprecated_v1
    def testWhileCondGradMultiDevice(self):
        if False:
            return 10
        config = config_pb2.ConfigProto(device_count={'CPU': 2}, allow_soft_placement=True)
        with self.cached_session(config=config) as sess:
            pred = array_ops.placeholder(dtypes.bool, [])
            x_init = constant_op.constant(1.0)
            with ops.device('/cpu:0'):
                z = while_loop_tf.while_loop(lambda i, _: i < 3, lambda i, x: (i + 1, tf_cond.cond(pred, lambda : x * 2.0, lambda : 10.0)), [0, x_init])
            with ops.device('/cpu:1'):
                grad = gradients_impl.gradients(z, x_init)[0]
            with ops.device('/cpu:0'):
                grad_grad = gradients_impl.gradients(grad, x_init)[0]
            self.assertEqual(sess.run(grad, {pred: True}), 8.0)
            self.assertEqual(sess.run(grad, {pred: False}), 0.0)
            if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
                return
            self.assertEqual(sess.run(grad_grad, {pred: True}), 0.0)
            self.assertEqual(sess.run(grad_grad, {pred: False}), 0.0)

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_deprecated_v1
    def testWhileUpdateVariable_1(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            select = variables.Variable([3.0, 4.0, 5.0])
            n = constant_op.constant(0)

            def loop_iterator(j):
                if False:
                    print('Hello World!')
                return math_ops.less(j, 3)

            def loop_body(j):
                if False:
                    i = 10
                    return i + 15
                ns = state_ops.scatter_update(select, j, 10.0)
                nj = math_ops.add(j, 1)
                op = control_flow_ops.group(ns)
                nj = control_flow_ops.with_dependencies([op], nj)
                return [nj]
            r = while_loop_tf.while_loop(loop_iterator, loop_body, [n], parallel_iterations=1)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(3, self.evaluate(r))
            result = self.evaluate(select)
            self.assertAllClose(np.array([10.0, 10.0, 10.0]), result)

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileUpdateVariable_2(self):
        if False:
            return 10
        with self.cached_session():
            select1 = variables.Variable([3.0, 4.0, 5.0])
            select2 = variables.Variable([3.0, 4.0, 5.0])
            n = constant_op.constant(0)

            def loop_iterator(j):
                if False:
                    while True:
                        i = 10
                return math_ops.less(j, 3)

            def loop_body(j):
                if False:
                    print('Hello World!')
                ns1 = state_ops.scatter_update(select1, j, 10.0)
                ns2 = state_ops.scatter_update(select2, j, 10.0)
                nj = math_ops.add(j, 1)
                op = control_flow_ops.group(ns1, ns2)
                nj = control_flow_ops.with_dependencies([op], nj)
                return [nj]
            r = while_loop_tf.while_loop(loop_iterator, loop_body, [n], parallel_iterations=1)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(3, self.evaluate(r))
            result1 = self.evaluate(select1)
            self.assertAllClose(np.array([10.0, 10.0, 10.0]), result1)
            result2 = self.evaluate(select2)
            self.assertAllClose(np.array([10.0, 10.0, 10.0]), result2)

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileUpdateVariable_3(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            select = variables.Variable([3.0, 4.0, 5.0])
            n = constant_op.constant(0)

            def loop_iterator(j, _):
                if False:
                    i = 10
                    return i + 15
                return math_ops.less(j, 3)

            def loop_body(j, _):
                if False:
                    i = 10
                    return i + 15
                ns = state_ops.scatter_update(select, j, 10.0)
                nj = math_ops.add(j, 1)
                return [nj, ns]
            r = while_loop_tf.while_loop(loop_iterator, loop_body, [n, array_ops.identity(select)], parallel_iterations=1)
            self.evaluate(variables.global_variables_initializer())
            result = r[1]
        self.assertAllClose(np.array([10.0, 10.0, 10.0]), result)

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileUpdateVariable_4(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            var_a = variables.Variable(0, name='a')
            var_b = variables.Variable(0, name='b')
            self.evaluate(variables.global_variables_initializer())
            c = constant_op.constant(0, name='c')
            asn1 = state_ops.assign_add(var_a, 1, name='a_add')

            def pred(i):
                if False:
                    print('Hello World!')
                return math_ops.less(i, 10)

            def loop_body(i):
                if False:
                    for i in range(10):
                        print('nop')
                asn2 = state_ops.assign_add(var_b, asn1, name='b_add')
                with ops.control_dependencies([asn2]):
                    ni = math_ops.add(i, 1, name='i_add')
                return ni
            lpa = while_loop_tf.while_loop(pred, loop_body, [c], parallel_iterations=1)
            self.assertEqual(0, self.evaluate(var_b))
            self.evaluate(lpa)
            self.assertEqual(10, self.evaluate(var_b))

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileUpdateVariable_5(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            var_a = variables.Variable(0, name='a')
            var_b = variables.Variable(0, name='b')
            self.evaluate(variables.global_variables_initializer())

            def pred(_):
                if False:
                    print('Hello World!')
                return math_ops.less(var_b, 10)

            def loop_body(i):
                if False:
                    return 10
                asn1 = state_ops.assign_add(var_a, constant_op.constant(1), name='a_add')
                asn2 = state_ops.assign_add(var_b, constant_op.constant(1), name='b_add')
                with ops.control_dependencies([asn1, asn2]):
                    inc_b = array_ops.identity(var_b)
                return inc_b
            lpa = while_loop_tf.while_loop(pred, loop_body, [var_b], parallel_iterations=1, name='loop')
            self.assertEqual(0, self.evaluate(var_b))
            self.evaluate(lpa)
            self.assertEqual(10, self.evaluate(var_a))
            self.assertEqual(10, self.evaluate(var_b))

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileUpdateVariable_6(self):
        if False:
            return 10
        with self.cached_session():
            var_a = variables.Variable(0, name='a')
            var_b = variables.Variable(0, name='b')
            c = constant_op.constant(0)
            self.evaluate(variables.global_variables_initializer())

            def pred(i):
                if False:
                    i = 10
                    return i + 15
                return math_ops.less(i, 10)

            def loop_body(i):
                if False:
                    while True:
                        i = 10
                asn1 = state_ops.assign_add(var_a, 1, name='a_add')
                with ops.control_dependencies([asn1]):
                    asn2 = state_ops.assign_add(var_b, var_a, name='b_add')
                with ops.control_dependencies([asn2]):
                    ni = math_ops.add(i, 1, name='i_add')
                    return ni
            lpa = while_loop_tf.while_loop(pred, loop_body, [c], parallel_iterations=1, name='loop')
            self.assertEqual(0, self.evaluate(var_b))
            self.evaluate(lpa)
            self.assertEqual(55, self.evaluate(var_b))
            self.assertEqual(10, self.evaluate(var_a))

    @test_util.run_v1_only('b/120545219')
    def testWhileQueue_1(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            q = data_flow_ops.FIFOQueue(-1, dtypes.int32)
            i = constant_op.constant(0)

            def c(i):
                if False:
                    return 10
                return math_ops.less(i, 10)

            def b(i):
                if False:
                    print('Hello World!')
                ni = math_ops.add(i, 1)
                ni = control_flow_ops.with_dependencies([q.enqueue((i,))], ni)
                return ni
            r = while_loop_tf.while_loop(c, b, [i], parallel_iterations=1)
            self.assertEqual([10], self.evaluate(r))
            for i in range(10):
                self.assertEqual([i], self.evaluate(q.dequeue()))

    @test_util.run_v1_only('b/120545219')
    def testWhileTimeOut(self):
        if False:
            while True:
                i = 10
        run_options = config_pb2.RunOptions(timeout_in_ms=1)
        with self.cached_session() as sess:
            n = constant_op.constant(0)
            c = lambda x: True
            b = lambda x: math_ops.add(x, 1)
            r = while_loop_tf.while_loop(c, b, [n])
            with self.assertRaises(errors_impl.DeadlineExceededError):
                sess.run(r, options=run_options)

    @test_util.disable_control_flow_v2('b/117119329 (stack)')
    @test_util.run_v1_only('b/120545219')
    def testWhileStack_1(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            s = gen_data_flow_ops.stack_v2(-1, dtypes.int32, stack_name='foo')
            i = constant_op.constant(0)

            def c(i):
                if False:
                    print('Hello World!')
                return math_ops.less(i, 10)

            def b(i):
                if False:
                    return 10
                ni = math_ops.add(i, 1)
                ni = control_flow_ops.with_dependencies([gen_data_flow_ops.stack_push_v2(s, i)], ni)
                return ni
            r = while_loop_tf.while_loop(c, b, [i], parallel_iterations=1)
            x = constant_op.constant(0)

            def c1(i, _):
                if False:
                    return 10
                return math_ops.greater(i, 0)

            def b1(i, x):
                if False:
                    return 10
                ni = math_ops.subtract(i, 1)
                nx = x + gen_data_flow_ops.stack_pop_v2(s, dtypes.int32)
                return [ni, nx]
            (_, rx) = while_loop_tf.while_loop(c1, b1, [r, x], [r.get_shape(), tensor_shape.unknown_shape()], parallel_iterations=1)
            self.assertEqual(45, self.evaluate(rx))

    def _testWhileGrad_ColocateGradients(self, colocate):
        if False:
            i = 10
            return i + 15
        gpu_dev_name = test.gpu_device_name() if test.is_gpu_available() else '/device:CPU:0'
        graph = ops.Graph()
        with graph.as_default():
            v = constant_op.constant(2.0, name='v')
            c = lambda v: math_ops.less(v, 100.0)

            def b(x):
                if False:
                    for i in range(10):
                        print('nop')
                with ops.device(gpu_dev_name):
                    return math_ops.square(x)
            loop = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
            r = gradients_impl.gradients(loop, v, colocate_gradients_with_ops=colocate)[0]
        r_ops = graph.get_operations()
        r_devices = [(op.name, op.device) for op in r_ops]
        self.assertTrue(any(('Square' in op.name for op in r_ops)))
        for (name, dev) in r_devices:
            if not colocate and name.endswith('Square'):
                self.assertTrue(gpu_dev_name in dev)
            elif colocate and 'Square' in name:
                self.assertTrue(gpu_dev_name in dev)
            else:
                self.assertFalse(gpu_dev_name in dev)
        with self.session(graph=graph) as sess:
            self.assertAllClose(1024.0, self.evaluate(r))

    @test_util.disable_control_flow_v2('b/116351701 (colocation)')
    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_ColocateGradients(self):
        if False:
            return 10
        self._testWhileGrad_ColocateGradients(colocate=False)
        self._testWhileGrad_ColocateGradients(colocate=True)

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_Square(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            v = constant_op.constant(2.0, name='v')
            c = lambda v: math_ops.less(v, 100.0)
            b = math_ops.square
            r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
            r = tf_cond.cond(math_ops.less(1, 2), lambda : r, lambda : v)
            r = gradients_impl.gradients(r, v)[0]
            self.assertAllClose(1024.0, self.evaluate(r))

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_Shape(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            x = array_ops.placeholder(dtypes.float32, shape=[None])
            v = constant_op.constant([2.0], name='v')
            n = constant_op.constant(0, name='n')
            c = lambda i, v: math_ops.less(i, 5)
            b = lambda i, v: [i + 1, math_ops.multiply(x, v)]
            r = while_loop_tf.while_loop(c, b, [n, v], [n.get_shape(), tensor_shape.unknown_shape()], parallel_iterations=1)
            r = gradients_impl.gradients(r[1], x)[0]
            self.assertEqual([None], r.get_shape().as_list())
            self.assertAllClose([810.0, 2560.0], r.eval(feed_dict={x: [3.0, 4.0]}))

    @test_util.run_deprecated_v1
    def testWhileGrad_BaseShape(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            x = array_ops.placeholder(dtypes.float32, [None])
            v0 = constant_op.constant([2.0, 2.0], name='v')
            c = lambda v: constant_op.constant(False)
            b = lambda v: math_ops.multiply(v, x)
            r = while_loop_tf.while_loop(c, b, [v0])
            y = math_ops.square(x)
            r = gradients_impl.gradients([r, y], x)[0]
            self.assertAllClose([2.0, 4.0], sess.run(r, feed_dict={x: [1.0, 2.0]}))

    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testWhileGradAfterSessionRun(self):
        if False:
            return 10
        v0 = constant_op.constant(2.0)
        r = while_loop_tf.while_loop(lambda _: True, lambda v: v * v, [v0], maximum_iterations=3)
        self.assertAllEqual(r, 256.0)
        grad = gradients_impl.gradients(r, v0)[0]
        self.assertAllClose(grad, 1024.0)

    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testNestedWhileGradAfterSessionRun(self):
        if False:
            return 10
        v0 = constant_op.constant(2.0)

        def body(v):
            if False:
                while True:
                    i = 10
            inner_v0 = constant_op.constant(1.0)
            return while_loop_tf.while_loop(lambda _: True, lambda x: x * v, [inner_v0], maximum_iterations=2)
        r = while_loop_tf.while_loop(lambda _: True, body, [v0], maximum_iterations=3)
        self.assertAllEqual(r, 256.0)
        grad = gradients_impl.gradients(r, v0)[0]
        self.assertAllClose(grad, 1024.0)

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_MultipleUses(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            v = constant_op.constant(2.0, name='v')
            c = lambda v: math_ops.less(v, 100.0)
            b = math_ops.square
            r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
            r = math_ops.multiply(r, r)
            r = gradients_impl.gradients(r, v)[0]
            self.assertEqual(524288.0, self.evaluate(r))

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_LoopAdd(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            v = constant_op.constant(2.0, name='v')
            c = lambda v: math_ops.less(v, 100.0)
            b = math_ops.square
            r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
            r = math_ops.add(r, r)
            r = gradients_impl.gradients(r, v)[0]
            self.assertAllClose(2048.0, self.evaluate(r))

    def _testWhileGrad_Mul(self, use_gpu, p_iters):
        if False:
            while True:
                i = 10
        with self.cached_session(use_gpu=use_gpu) as sess:
            a = constant_op.constant(3.0, name='a')
            v = constant_op.constant(2.0, name='v')
            c = lambda v: math_ops.less(v, 100.0)
            b = lambda v: math_ops.multiply(v, a)
            r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=p_iters)
            (grad_a, grad_v) = gradients_impl.gradients(r, [a, v])
            (grad_a_val, grad_v_val) = self.evaluate([grad_a, grad_v])
            self.assertAllClose(216.0, grad_a_val)
            self.assertAllClose(81.0, grad_v_val)

    @test_util.run_deprecated_v1
    def testWhileGrad_Mul(self):
        if False:
            print('Hello World!')
        self._testWhileGrad_Mul(use_gpu=False, p_iters=1)
        self._testWhileGrad_Mul(use_gpu=False, p_iters=10)
        self._testWhileGrad_Mul(use_gpu=True, p_iters=1)
        self._testWhileGrad_Mul(use_gpu=True, p_iters=10)

    def testWhileGradInControlDeps(self):
        if False:
            for i in range(10):
                print('nop')

        @eager_def_function.function
        def f():
            if False:
                return 10
            x_init = constant_op.constant(2.0)
            loop_cond = lambda i, x: math_ops.less(i, 2)
            loop_body = lambda i, x: [i + 1, x ** 2]
            (_, x) = while_loop_tf.while_loop(loop_cond, loop_body, [0, x_init])
            with ops.control_dependencies([x]):
                (grad,) = gradients_impl.gradients(x, x_init)
                return grad
        self.assertAllEqual(f(), 4.0 * 2.0 ** 3)

    @test_util.run_deprecated_v1
    def testTfFunctionInV1WhileLoop(self):
        if False:
            while True:
                i = 10
        config = opt_cfg()
        assert config.graph_options.optimizer_options.do_function_inlining
        with session.Session(config=config):

            @eager_def_function.function
            def loop_body(i):
                if False:
                    while True:
                        i = 10
                return i + 1.0
            loop_cond = lambda i: True
            x = while_loop_tf.while_loop(loop_cond, loop_body, [0.0], maximum_iterations=5)
            self.assertAllEqual(x, 5.0)

    def _testNestedWhileCondWhileGrad(self, use_gpu):
        if False:
            return 10
        with self.cached_session(use_gpu=use_gpu):
            v = constant_op.constant(1.0)

            def inner_loop(s):
                if False:
                    for i in range(10):
                        print('nop')
                z = constant_op.constant(0)
                c = lambda i, x: math_ops.less(i, 4)
                b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
                return while_loop_tf.while_loop(c, b, [z, s])
            c = lambda x: math_ops.less(x, 128.0)

            def b(x):
                if False:
                    while True:
                        i = 10
                return tf_cond.cond(constant_op.constant(True), lambda : math_ops.square(inner_loop(x)[1]), lambda : math_ops.multiply(x, 2.0))
            r = while_loop_tf.while_loop(c, b, [v])
            r = gradients_impl.gradients(r, v)[0]
            self.assertAllClose(512.0, self.evaluate(r))

    @test_util.run_deprecated_v1
    def testNestedWhileCondWhileGrad(self):
        if False:
            return 10
        self._testNestedWhileCondWhileGrad(use_gpu=False)

    @test_util.run_deprecated_v1
    def testNestedWhileCondWhileGradGpu(self):
        if False:
            while True:
                i = 10
        self._testNestedWhileCondWhileGrad(use_gpu=True)

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_Variable(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            a = variables.Variable(3.0)
            v = constant_op.constant(2.0, name='v')
            c = lambda v: math_ops.less(v, 100.0)
            b = lambda v: math_ops.multiply(v, a)
            r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
            r = gradients_impl.gradients(r, a)
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(216.0, r[0])

    @test_util.run_deprecated_v1
    def testWhileGrad_ResourceVariable(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            a = resource_variable_ops.ResourceVariable(3.0)
            v = constant_op.constant(2.0, name='v')
            c = lambda v: math_ops.less(v, 100.0)
            b = lambda v: math_ops.multiply(v, a)
            r = while_loop_tf.while_loop(c, b, [v], parallel_iterations=1)
            g = gradients_impl.gradients(r, a)
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(216.0, g[0])

    def testWhileGrad_EagerResourceVariable(self):
        if False:
            return 10
        with context.eager_mode():
            a = resource_variable_ops.ResourceVariable(np.ones([2, 2], dtype=np.float32))
            v = constant_op.constant(1.0)

            @eager_def_function.function
            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                r = while_loop_tf.while_loop(lambda i, _: i < 2, lambda i, x: (i + 1, x * math_ops.reduce_sum(a) * v), [0, 1.0])[1]
                return gradients_impl.gradients(r, [v])[0]
            self.assertEqual(self.evaluate(fn()), 32.0)

    def testWhileGrad_ResourceVarInFunctionCall(self):
        if False:
            i = 10
            return i + 15

        @eager_def_function.function
        def foo(x, var):
            if False:
                return 10
            return x + math_ops.reduce_sum(var.sparse_read([1, 3]))

        @eager_def_function.function
        def bar(var):
            if False:
                while True:
                    i = 10
            r = while_loop_tf.while_loop(lambda i, _: i < 2, lambda i, x: (i + 1, foo(x, var)), [0, 0.0])[1]
            return gradients_impl.gradients(r, var)[0]
        var = resource_variable_ops.ResourceVariable([1.0, 2.0, 3.0, 4.0])
        self.evaluate(variables.global_variables_initializer())
        grad = self.evaluate(bar(var))
        self.assertAllEqual(gradient_checker_v2._to_numpy(grad), [0.0, 2.0, 0.0, 2.0])

    def testWhileGrad_ResourceVarInNestedFunctionCall(self):
        if False:
            i = 10
            return i + 15

        @eager_def_function.function
        def foo(x, var):
            if False:
                i = 10
                return i + 15
            return x + math_ops.reduce_sum(var.sparse_read([1, 3]))

        @eager_def_function.function
        def foo2(x, var):
            if False:
                return 10
            return foo(x, var)

        @eager_def_function.function
        def bar(var):
            if False:
                print('Hello World!')
            r = while_loop_tf.while_loop(lambda i, _: i < 2, lambda i, x: (i + 1, foo2(x, var)), [0, 0.0])[1]
            return gradients_impl.gradients(r, var)[0]
        var = resource_variable_ops.ResourceVariable([1.0, 1.0, 1.0, 1.0])
        self.evaluate(variables.global_variables_initializer())
        grad = self.evaluate(bar(var))
        self.assertAllEqual(gradient_checker_v2._to_numpy(grad), [0.0, 2.0, 0.0, 2.0])

    def testWhileGrad_ResourceVarInLoopInFunctionCall(self):
        if False:
            for i in range(10):
                print('nop')
        if test.is_gpu_available():
            self.skipTest('b/128635252')

        @eager_def_function.function
        def foo(x, var):
            if False:
                for i in range(10):
                    print('nop')
            return while_loop_tf.while_loop(lambda j, _: j < 3, lambda j, y: (j + 1, y + math_ops.reduce_sum(var.sparse_read([1, 2]))), [0, x])[1]

        @eager_def_function.function
        def bar(var):
            if False:
                i = 10
                return i + 15
            r = while_loop_tf.while_loop(lambda i, _: i < 2, lambda i, x: (i + 1, foo(x, var)), [0, 0.0])[1]
            return gradients_impl.gradients(r, var)[0]
        var = resource_variable_ops.ResourceVariable([1.0, 1.0, 1.0, 1.0])
        self.evaluate(variables.global_variables_initializer())
        grad = self.evaluate(bar(var))
        self.assertAllEqual(gradient_checker_v2._to_numpy(grad), [0.0, 6.0, 6.0, 0.0])

    def testWhileCondGrad_ResourceVarInFunctionCall(self):
        if False:
            return 10

        @eager_def_function.function
        def foo(x, var):
            if False:
                i = 10
                return i + 15
            return x + var.sparse_read([1])[0]

        def body(i, x):
            if False:
                for i in range(10):
                    print('nop')
            return (i + 1, tf_cond.cond(math_ops.equal(i % 2, 0), lambda : foo(x, var1), lambda : foo(x, var2)))

        @eager_def_function.function
        def bar(var1, var2):
            if False:
                i = 10
                return i + 15
            r = while_loop_tf.while_loop(lambda i, _: i < 4, body, [0, 0.0])
            return gradients_impl.gradients(r, [var1, var2])
        var1 = resource_variable_ops.ResourceVariable([1.0, 2.0, 3.0])
        var2 = resource_variable_ops.ResourceVariable([4.0, 5.0])
        self.evaluate(variables.global_variables_initializer())
        grads = self.evaluate(bar(var1, var2))
        self.assertAllEqual(gradient_checker_v2._to_numpy(grads[0]), [0.0, 2.0, 0.0])
        self.assertAllEqual(gradient_checker_v2._to_numpy(grads[1]), [0.0, 2.0])

    @test_util.run_deprecated_v1
    def testWhileGrad_ResourceVarSparseRead(self):
        if False:
            print('Hello World!')
        var = resource_variable_ops.ResourceVariable(np.ones(5), dtype=dtypes.float32)
        r = while_loop_tf.while_loop(lambda i, _: i < 3, lambda i, x: (i + 1, x * math_ops.reduce_sum(var.sparse_read([1, 3]))), [0, constant_op.constant(1.0)])[1]
        grad = gradients_impl.gradients(r, var)[0]
        self.evaluate(variables.global_variables_initializer())
        grad_val = self.evaluate(grad)
        arr = gradient_checker_v2._to_numpy(grad_val)
        self.assertAllEqual(arr, [0.0, 12.0, 0.0, 12.0, 0.0])

    @test_util.run_deprecated_v1
    def testWhileGrad_MultiResourceVarSparseRead(self):
        if False:
            for i in range(10):
                print('nop')
        var1 = resource_variable_ops.ResourceVariable(np.ones(5), dtype=dtypes.float32)
        var2 = resource_variable_ops.ResourceVariable(np.ones(3), dtype=dtypes.float32)
        x1_init = constant_op.constant([0.0, 0.0])
        x2_init = constant_op.constant(1.0)
        x3_init = constant_op.constant(1.0)

        def body(i, unused_x1, x2, x3):
            if False:
                print('Hello World!')
            y1 = var1.sparse_read([1, 3])
            y2 = x2 * 2
            y3 = x3 * math_ops.reduce_sum(var2.sparse_read([0]))
            return (i + 1, y1, y2, y3)
        r = while_loop_tf.while_loop(lambda i, x1, x2, x3: i < 3, body, [0, x1_init, x2_init, x3_init])[1:]
        (var1_grad, var2_grad) = gradients_impl.gradients(r, [var1, var2])
        self.evaluate(variables.global_variables_initializer())
        var1_grad_val = self.evaluate(var1_grad)
        var2_grad_val = self.evaluate(var2_grad)
        self.assertAllEqual(gradient_checker_v2._to_numpy(var1_grad_val), [0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertAllEqual(gradient_checker_v2._to_numpy(var2_grad_val), [3.0, 0.0, 0.0])

    def testWhileGrad_Gather(self):
        if False:
            i = 10
            return i + 15

        @tf_function_in_tf2
        def fn():
            if False:
                i = 10
                return i + 15
            x = constant_op.constant([1.0, 1.0, 1.0, 1.0, 1.0])
            y = while_loop_tf.while_loop(lambda i, _: i < 3, lambda i, x: (i + 1, x + array_ops.gather(x, [0])), [0, x[:1]])[1]
            z = y * 3.0
            grad = gradients_impl.gradients(z, x)[0]
            return (y, grad)
        (y, grad) = fn()
        self.assertEqual(self.evaluate(y), 8.0)
        self.assertAllEqual(self.evaluate(grad), [24.0, 0.0, 0.0, 0.0, 0.0])

    def testWhileGrad_GatherNoFanOut(self):
        if False:
            print('Hello World!')

        @tf_function_in_tf2
        def fn():
            if False:
                i = 10
                return i + 15
            x = constant_op.constant([1.0, 1.0, 1.0, 1.0, 1.0])
            y = while_loop_tf.while_loop(lambda i, _: i < 3, lambda i, x: (i + 1, array_ops.gather(x, [0])), [0, x[:1]])[1]
            z = y * 3.0
            grad = gradients_impl.gradients(z, x)[0]
            return (y, grad)
        (y, grad) = fn()
        self.assertEqual(self.evaluate(y), 1.0)
        self.assertAllEqual(self.evaluate(grad), [3.0, 0.0, 0.0, 0.0, 0.0])

    @test_util.run_v1_only('b/120545219')
    def testWhileGradInCond(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            n = ops.convert_to_tensor(1.0, name='n')
            x = array_ops.placeholder(dtypes.float32, shape=None)
            c = lambda n: math_ops.less(n, 10.0)
            b = lambda n: math_ops.add(n, x)

            def fn1():
                if False:
                    for i in range(10):
                        print('nop')
                r = while_loop_tf.while_loop(c, b, [n], [tensor_shape.unknown_shape()])
                return gradients_impl.gradients(r, x)[0]
            r = tf_cond.cond(math_ops.less(1, 2), fn1, lambda : x)
            self.assertAllClose(9.0, r.eval(feed_dict={x: 1.0}))

    @test_util.disable_control_flow_v2('b/116340060')
    @test_util.run_v1_only('b/120545219')
    def testGradInWhileWrtInitialLoopVal(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            x = array_ops.placeholder(dtypes.float32, shape=(), name='x')
            y = x + 1

            def body(i, v):
                if False:
                    print('Hello World!')
                z = v * 2
                return (i + 1, gradients_impl.gradients(z, x)[0])
            with self.assertRaisesRegex(ValueError, "Cannot compute gradient inside while loop with respect to op 'x'. We do not support taking the gradient wrt or through the initial value of a loop variable. Gradients can be computed through loop invariants or wrt the input parameters to the loop body."):
                while_loop_tf.while_loop(lambda i, x: i < 3, body, [0, y])

    @test_util.run_v1_only('b/120545219')
    def testWhileGradInWhile(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            n = ops.convert_to_tensor(1.0, name='n')
            x = array_ops.placeholder(dtypes.float32, shape=None)
            c = lambda n: math_ops.less(n, 10.0)
            b = lambda n: math_ops.add(n, x)

            def b1(n):
                if False:
                    print('Hello World!')
                r = while_loop_tf.while_loop(c, b, [n], [tensor_shape.unknown_shape()])
                return gradients_impl.gradients(r, x)
            r = while_loop_tf.while_loop(lambda n: n < 6.0, b1, [n], [tensor_shape.unknown_shape()])
            self.assertAllClose(9.0, r.eval(feed_dict={x: 1.0}))

    @test_util.run_v1_only('b/120545219')
    def testCondGradInNestedWhiles(self):
        if False:
            return 10

        def outer_body(i, x):
            if False:
                for i in range(10):
                    print('nop')
            (_, x) = while_loop_tf.while_loop(lambda j, x: j < 3, inner_body, [0, 0.0])
            return (i + 1, x)

        def inner_body(j, x):
            if False:
                while True:
                    i = 10
            y = tf_cond.cond(math_ops.less(x, 1), lambda : 2 * x, lambda : x)
            return (j + 1, gradients_impl.gradients(y, x)[0])
        (i, x) = while_loop_tf.while_loop(lambda i, x: i < 3, outer_body, [0, 0.0])
        with self.cached_session() as sess:
            (i_val, x_val) = self.evaluate([i, x])
            self.assertEqual(i_val, 3)
            self.assertAllClose(x_val, 1.0)

    @test_util.run_gpu_only
    def testGpuResourceAccess(self):
        if False:
            return 10
        with ops.device(test.gpu_device_name()):
            var = resource_variable_ops.ResourceVariable(constant_op.constant(3.0))

        @eager_def_function.function
        def foo():
            if False:
                return 10
            return while_loop_tf.while_loop(lambda i, _: i < 3, lambda i, x: (i + 1, tf_cond.cond(constant_op.constant(True), lambda : x + var, lambda : x)), [0, 0.0])[1]
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(foo()), 9.0)

    def testNestedResourceAccess(self):
        if False:
            return 10
        var = resource_variable_ops.ResourceVariable(constant_op.constant(3.0))

        @eager_def_function.function
        def test_fn():
            if False:
                return 10
            x = constant_op.constant(0.0)
            r = while_loop_tf.while_loop(lambda i, y: i < 2, lambda i, y: (i + 1, y + tf_cond.cond(constant_op.constant(True), lambda : while_loop_tf.while_loop(lambda j, z: j < 3, lambda j, z: (j + 1, z + math_ops.square(var)), [0, y])[1], lambda : 0.0)), [0, x])[1]
            grad = gradients_impl.gradients(r, x)[0]
            return (r, grad)
        self.evaluate(variables.global_variables_initializer())
        (r, grad) = self.evaluate(test_fn())
        self.assertEqual(r, 81.0)
        if control_flow_util.ENABLE_CONTROL_FLOW_V2:
            self.assertEqual(grad, 4.0)

    def testWhile_NestedInput(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            named = collections.namedtuple('named', ('a', 'b'))
            loop_vars = [named(a=constant_op.constant(0.0), b=constant_op.constant(1.0)), (constant_op.constant(2.0), constant_op.constant(3.0)), constant_op.constant(4.0)]
            c = lambda lv0, _1, _2: lv0.a < 100.0

            def b(lv0, lv1, lv2):
                if False:
                    return 10
                lv0 = named(a=lv0.a + 1, b=lv0.b)
                lv1 = (lv1[0] + 1, lv1[1])
                lv2 += 2
                return [lv0, lv1, lv2]
            r = while_loop_tf.while_loop(c, b, loop_vars)
            self.assertTrue(isinstance(r, list))
            self.assertTrue(isinstance(r[0], named))
            self.assertTrue(isinstance(r[1], tuple))
            self.assertTrue(isinstance(r[2], tensor_lib.Tensor))
            r_flattened = nest.flatten(r)
            self.assertEqual([100.0, 1.0, 102.0, 3.0, 4.0 + 100 * 2.0], self.evaluate(r_flattened))

    @test_util.run_v1_only('b/120545219')
    def testWhile_NestedBadArityFails(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            named = collections.namedtuple('named', ('a', 'b'))
            loop_vars = [named(a=constant_op.constant(0.0), b=constant_op.constant(1.0)), (constant_op.constant(2.0), constant_op.constant(3.0)), constant_op.constant(4.0)]
            c = lambda lv0, _1, _2: lv0.a < 100.0

            def b(lv0, lv1, _):
                if False:
                    for i in range(10):
                        print('nop')
                return [lv0, lv1]
            with self.assertRaisesRegex(ValueError, 'the same number of elements'):
                while_loop_tf.while_loop(c, b, loop_vars)

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_ys_xs(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            x = constant_op.constant(3.0, name='x')
            y = constant_op.constant(2.0, name='y')
            c = lambda x, y: math_ops.less(x, 100.0)

            def b(x, y):
                if False:
                    i = 10
                    return i + 15
                y1 = math_ops.add(x, y)
                x1 = math_ops.multiply(x, y1)
                return (x1, y1)
            (rx, ry) = while_loop_tf.while_loop(c, b, [x, y], parallel_iterations=1)
            r = gradients_impl.gradients([rx, ry], x)
            self.assertAllClose(304.0, r[0])
            r = gradients_impl.gradients([rx, ry], y)
            self.assertAllClose(124.0, r[0])
            r = gradients_impl.gradients([rx], x)
            self.assertAllClose(295.0, r[0])
            r = gradients_impl.gradients([rx], y)
            self.assertAllClose(120.0, r[0])

    @test_util.run_deprecated_v1
    def testWhileGrad_Dependency(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            i = constant_op.constant(0, name='i')
            x = constant_op.constant(2.0, name='x')
            c = lambda i, x: math_ops.less(i, 10)

            def b(i, x):
                if False:
                    i = 10
                    return i + 15
                x = math_ops.multiply(x, 2.0)
                i = math_ops.add(i, 1)
                return (i, x)
            (ri, rx) = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=1)
            r = gradients_impl.gradients([ri, rx], x)
            self.assertAllClose(1024.0, r[0])
            r = gradients_impl.gradients([rx], x)
            self.assertAllClose(1024.0, r[0])

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_NoGradient(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            v = constant_op.constant(2.0, name='v')
            c = lambda v: math_ops.less(v, 100.0)
            b = math_ops.square
            r = while_loop_tf.while_loop(c, b, [v], back_prop=False)
            r = math_ops.add(r, v)
            r = gradients_impl.gradients(r, v)
            self.assertAllClose(1.0, r[0])

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_NoDependency(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            variable = variables.Variable(array_ops.ones([2, 3]))
            duration = array_ops.zeros([], dtype=dtypes.int32)

            def cond(duration, tensor, _):
                if False:
                    print('Hello World!')
                del tensor
                return duration < 10

            def body(duration, tensor, _):
                if False:
                    while True:
                        i = 10
                return (duration + 1, tensor, tensor)
            loop_vars = [duration, variable, variable]
            tensors = while_loop_tf.while_loop(cond=cond, body=body, loop_vars=loop_vars)
            cost = math_ops.reduce_sum(tensors[2])
            grad = gradients_impl.gradients(cost, [variable])
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(np.ones([2, 3]), sess.run(grad[0]))

    @test_util.run_deprecated_v1
    def testWhileGrad_Const(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            c0 = constant_op.constant(0.0, name='c0')
            c1 = constant_op.constant(1.0, name='c1')
            duration = constant_op.constant(0, name='t')

            def cond(duration, _):
                if False:
                    i = 10
                    return i + 15
                return duration < 1

            def body(duration, _):
                if False:
                    while True:
                        i = 10
                return (duration + 1, c1)
            loop_vars = [duration, c0]
            tensors = while_loop_tf.while_loop(cond=cond, body=body, loop_vars=loop_vars)
            cost = math_ops.reduce_sum(tensors[1])
            grad = gradients_impl.gradients(cost, [c0])
            self.assertAllClose(0.0, sess.run(grad[0]))

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_SerialTwoLoops(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            i = constant_op.constant(0, name='i')
            x = constant_op.constant(2.0, name='x')
            c = lambda i, x: math_ops.less(i, 5)

            def b(i, x):
                if False:
                    print('Hello World!')
                x = math_ops.multiply(x, 2.0)
                i = math_ops.add(i, 1)
                return (i, x)
            (_, rx) = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=1)
            (_, rx) = while_loop_tf.while_loop(c, b, [i, rx], parallel_iterations=1)
            r = gradients_impl.gradients([rx], x)
            self.assertAllClose(1024.0, r[0])

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_ParallelTwoLoops(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            i = constant_op.constant(0, name='i')
            x = constant_op.constant(2.0, name='x')
            c = lambda i, x: math_ops.less(i, 5)

            def b(i, x):
                if False:
                    i = 10
                    return i + 15
                x = math_ops.multiply(x, 2.0)
                i = math_ops.add(i, 1)
                return (i, x)
            (_, r1) = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=1)
            (_, r2) = while_loop_tf.while_loop(c, b, [i, x], parallel_iterations=1)
            rx = math_ops.add(r1, r2)
            r = gradients_impl.gradients([rx], x)
            self.assertAllClose(64.0, r[0])

    @test_util.run_v1_only('b/120545219')
    def testWhileGrad_OneOutputWithControlDependencyOnSecond(self):
        if False:
            return 10
        with self.cached_session():
            i = constant_op.constant(0, name='i')
            x = constant_op.constant(1.0, name='x')
            y = constant_op.constant(1.0, name='y')
            c = lambda i, *_: math_ops.less(i, 1, name='cond_less')

            def b(i, xi, yi):
                if False:
                    i = 10
                    return i + 15
                return (math_ops.add(i, 1, name='inc'), array_ops.identity(xi, name='xi'), math_ops.add(xi, yi, name='xi_plus_yi'))
            (_, x_f, y_f) = while_loop_tf.while_loop(c, b, [i, x, y])
            with ops.control_dependencies([x_f]):
                y_f_d = array_ops.identity(y_f, name='y_f_d')
            self.assertAllClose(2.0, self.evaluate(y_f_d))
            g = gradients_impl.gradients([y_f_d], [x])[0]
            self.assertTrue(g is not None)
            self.assertAllClose(1.0, self.evaluate(g))

    def _testNestedWhileGrad_Simple(self, use_gpu):
        if False:
            print('Hello World!')
        with self.cached_session(use_gpu=use_gpu):
            v = constant_op.constant(1.0)

            def inner_loop(s):
                if False:
                    for i in range(10):
                        print('nop')
                c = lambda x: math_ops.less(x, 4.0)
                b = lambda x: math_ops.multiply(x, 2.0)
                return while_loop_tf.while_loop(c, b, [s])
            c = lambda x: math_ops.less(x, 2.0)
            b = lambda x: math_ops.multiply(inner_loop(x), 2.0)
            r = while_loop_tf.while_loop(c, b, [v])
            r = gradients_impl.gradients(r, v)[0]
            self.assertAllClose(8.0, self.evaluate(r))

    @test_util.run_deprecated_v1
    def testNestedWhileGrad_Simple(self):
        if False:
            i = 10
            return i + 15
        self._testNestedWhileGrad_Simple(use_gpu=False)
        self._testNestedWhileGrad_Simple(use_gpu=True)

    @test_util.run_v1_only('b/120545219')
    def testNestedWhileGrad_SerialInner(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            v = constant_op.constant(1.0)

            def inner_loop1(s):
                if False:
                    while True:
                        i = 10
                z = constant_op.constant(0)
                c = lambda i, x: math_ops.less(i, 4)
                b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
                return while_loop_tf.while_loop(c, b, [z, s])

            def inner_loop2(s):
                if False:
                    while True:
                        i = 10
                z = constant_op.constant(0)
                c = lambda i, x: math_ops.less(i, 4)
                b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
                return while_loop_tf.while_loop(c, b, [z, s])
            c = lambda x: math_ops.less(x, 128.0)
            b = lambda x: inner_loop2(inner_loop1(x)[1])[1]
            r = while_loop_tf.while_loop(c, b, [v])
            r = gradients_impl.gradients(r, v)[0]
            self.assertAllClose(256.0, self.evaluate(r))

    @test_util.run_deprecated_v1
    def testNestedWhileGrad_ParallelInner(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            v = constant_op.constant(1.0)

            def inner_loop1(s):
                if False:
                    while True:
                        i = 10
                z = constant_op.constant(0)
                c = lambda i, x: math_ops.less(i, 4)
                b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
                return while_loop_tf.while_loop(c, b, [z, s])

            def inner_loop2(s):
                if False:
                    return 10
                z = constant_op.constant(0)
                c = lambda i, x: math_ops.less(i, 4)
                b = lambda i, x: [math_ops.add(i, 1), math_ops.multiply(x, 2.0)]
                return while_loop_tf.while_loop(c, b, [z, s])
            c = lambda x: math_ops.less(x, 128.0)
            b = lambda x: math_ops.multiply(inner_loop1(x)[1], inner_loop2(x)[1])
            r = while_loop_tf.while_loop(c, b, [v])
            r = gradients_impl.gradients(r, v)[0]
            self.assertAllClose(512.0, self.evaluate(r))

    @test_util.run_v1_only('b/120545219')
    def testNestedWhileGrad_ParallelIterations(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:

            def inner_loop(t):
                if False:
                    return 10
                fn = lambda n: n + math_ops.square(var)
                return map_fn.map_fn(fn=fn, elems=t, parallel_iterations=10)

            def outer_loop(inp):
                if False:
                    i = 10
                    return i + 15
                return map_fn.map_fn(fn=inner_loop, elems=inp, parallel_iterations=10)
            var = variables.Variable(constant_op.constant(3.0))
            inp = constant_op.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            res = outer_loop(inp)
            optimizer = adam.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(math_ops.reduce_mean(math_ops.square(res)))
            self.evaluate(variables.global_variables_initializer())
            self.evaluate(train_op)
            self.assertAllClose(2.999, var.read_value())

    def _testWhileCondGrad_Simple(self, use_gpu):
        if False:
            i = 10
            return i + 15
        with self.cached_session(use_gpu=use_gpu):
            v = ops.convert_to_tensor(2.0, name='v')
            n = ops.convert_to_tensor(100.0, name='n')
            one = ops.convert_to_tensor(1.0, name='one')
            c = lambda x: math_ops.less(x, n)
            b = lambda x: tf_cond.cond(constant_op.constant(True), lambda : math_ops.square(x), lambda : math_ops.subtract(x, one))
            r = while_loop_tf.while_loop(c, b, [v])
            r = gradients_impl.gradients(r, v)[0]
            self.assertAllClose(1024.0, self.evaluate(r))

    @test_util.run_deprecated_v1
    def testWhileCondGrad_Simple(self):
        if False:
            return 10
        self._testWhileCondGrad_Simple(use_gpu=False)
        self._testWhileCondGrad_Simple(use_gpu=True)

    @test_util.run_deprecated_v1
    def testWhileCondGrad_UnknownShape(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            v = array_ops.placeholder(dtypes.float32)
            n = ops.convert_to_tensor(100.0, name='n')
            one = ops.convert_to_tensor(1.0, name='one')
            c = lambda x: math_ops.less(x, n)
            b = lambda x: tf_cond.cond(constant_op.constant(True), lambda : math_ops.square(x), lambda : math_ops.subtract(x, one))
            r = while_loop_tf.while_loop(c, b, [v])
            r = gradients_impl.gradients(r, v)[0]
            r = sess.run(r, feed_dict={v: 2.0})
            self.assertAllClose(1024.0, r)

    @test_util.run_deprecated_v1
    def testWhileGrad_Concat(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            x = variable_scope.get_variable('x', initializer=[[1.0, 2.0]])
            i0 = constant_op.constant(0)
            h0 = array_ops.zeros([0, 2])

            def condition(i, _):
                if False:
                    print('Hello World!')
                return i < 2

            def body(i, h):
                if False:
                    while True:
                        i = 10
                return (i + 1, array_ops.concat([h, x], 0))
            (_, h) = while_loop_tf.while_loop(condition, body, [i0, h0], [i0.get_shape(), tensor_shape.TensorShape([None, 2])])
            s = math_ops.reduce_sum(h)
            optimizer = gradient_descent.GradientDescentOptimizer(0.01)
            op = optimizer.minimize(s)
            self.evaluate(variables.global_variables_initializer())
            self.evaluate(op)
            self.assertAllClose([[0.98000002, 1.98000002]], self.evaluate(x))

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileWithRefsWithGradients_1(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            x = variable_v1.VariableV1(0.0)._ref()
            i = constant_op.constant(0)
            c = lambda i, x: math_ops.less(i, 10)
            self.assertEqual(x.dtype, dtypes.float32_ref)

            def body(i, x):
                if False:
                    return 10
                self.assertEqual(x.dtype, dtypes.float32_ref)
                return [i + 1, gen_array_ops.ref_identity(x)]
            r = while_loop_tf.while_loop(c, body, [i, x], parallel_iterations=5)
            grad_ys = [variable_v1.VariableV1(73)._ref()]
            grad = gradients_impl.gradients([r[1]], [x], grad_ys=grad_ys)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(r[0].dtype, dtypes.int32)
            self.assertEqual(r[1].dtype, dtypes.float32_ref)
            (value_i, value_x, value_x_grad) = sess.run(r + grad)
        self.assertEqual(10, value_i)
        self.assertEqual(0, value_x)
        self.assertEqual(73, value_x_grad)

    @test_util.deprecated_graph_mode_only
    def testWhileGrad_IndexedSlices(self):
        if False:
            return 10
        with self.cached_session():
            values = constant_op.constant([2.0, 4.0], name='values')
            indices = constant_op.constant([0, 3], name='indices')
            shape = constant_op.constant([10], name='dense_shape')
            i = constant_op.constant(0)
            x = indexed_slices.IndexedSlices(values, indices, dense_shape=shape)

            def c(i, _):
                if False:
                    i = 10
                    return i + 15
                return i < 10

            def b(i, x):
                if False:
                    i = 10
                    return i + 15
                return [i + 1, indexed_slices.IndexedSlices(x.values * 2.0, x.indices, x.dense_shape)]
            (_, r) = while_loop_tf.while_loop(c, b, [i, x])
            r = gradients_impl.gradients(r.values, values)[0]
            self.assertAllClose(np.array([1024.0, 1024.0]), self.evaluate(r))

    @test_util.deprecated_graph_mode_only
    def testWhileGrad_SparseTensor(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            values = constant_op.constant([2.0, 4.0], name='values')
            indices = constant_op.constant([[0], [3]], dtype=dtypes.int64, name='indices')
            shape = constant_op.constant([10], dtype=dtypes.int64, name='dense_shape')
            i = constant_op.constant(0)
            x = sparse_tensor.SparseTensor(indices, values, dense_shape=shape)

            def c(i, _):
                if False:
                    while True:
                        i = 10
                return i < 10

            def b(i, x):
                if False:
                    while True:
                        i = 10
                return [i + 1, sparse_tensor.SparseTensor(x.indices, x.values * 2.0, x.dense_shape)]
            (_, r) = while_loop_tf.while_loop(c, b, [i, x])
            r = gradients_impl.gradients(r.values, values)[0]
            self.assertAllClose(np.array([1024.0, 1024.0]), self.evaluate(r))

    @test_util.deprecated_graph_mode_only
    def testCallGradInLoop(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            i0 = constant_op.constant(0)
            params = constant_op.constant(5.0)
            params_1 = math_ops.square(params)

            def c(i, _):
                if False:
                    while True:
                        i = 10
                return i < 10

            def b(i, x):
                if False:
                    i = 10
                    return i + 15
                data = constant_op.constant([1.0, 2.0, 3.0])
                data = math_ops.multiply(data, params_1)
                x1 = x + gradients_impl.gradients(data, params)[0]
                return (i + 1, x1)
            output_grad = while_loop_tf.while_loop(c, b, [i0, constant_op.constant(0.0)])
            self.assertAllClose(600.0, self.evaluate(output_grad)[1])

    @test_util.run_deprecated_v1
    def testWhileAndTensorArray(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            param = constant_op.constant(2.0)
            n0 = constant_op.constant(0)
            y0 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='elems')

            def c(i, _):
                if False:
                    i = 10
                    return i + 15
                return i < 10

            def b(i, y):
                if False:
                    while True:
                        i = 10
                return [i + 1, map_fn.map_fn(lambda x: math_ops.multiply(x, param), y)]
            r = while_loop_tf.while_loop(c, b, [n0, y0], parallel_iterations=1)
            r = gradients_impl.gradients(r, param)[0]
            self.assertAllClose(107520.0, self.evaluate(r))

    @test_util.run_deprecated_v1
    def testNestedWhileAndTensorArray(self):
        if False:
            return 10
        n = constant_op.constant(3.0)

        def Body(row, ta):
            if False:
                print('Hello World!')

            def InnerBody(row, col, ta):
                if False:
                    for i in range(10):
                        print('nop')
                ta = ta.write(math_ops.cast(n * (row - 1.0) + col - 1.0, dtypes.int32), row * col)
                return (row, col + 1.0, ta)
            ta = while_loop_tf.while_loop(lambda _, col, _1: col <= n, InnerBody, [row, constant_op.constant(1.0), ta], return_same_structure=False)[2]
            return (row + 1.0, ta)
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=9)
        ta = while_loop_tf.while_loop(lambda row, _: row <= n, Body, [constant_op.constant(1.0), ta], return_same_structure=False)[1]
        output = array_ops.reshape(ta.stack(), [3, 3])
        self.assertAllEqual(self.evaluate(output), [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])

    @test_util.run_deprecated_v1
    def testWhileGrad_StopGrad(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            x = constant_op.constant(3.0, name='x')
            y = constant_op.constant(2.0, name='y')
            c = lambda x, y: math_ops.less(x, 100.0)

            def b(x, y):
                if False:
                    print('Hello World!')
                y1 = math_ops.square(y)
                x1 = math_ops.add(math_ops.square(x), y1)
                return (x1, y1)
            (rx, ry) = while_loop_tf.while_loop(c, b, [x, y])
            r = gradients_impl.gradients(rx, y)[0]
            self.assertEqual(136.0, self.evaluate(r))
            r = gradients_impl.gradients(ry, y)[0]
            self.assertEqual(32.0, self.evaluate(r))
            r = gradients_impl.gradients(array_ops.stop_gradient(rx), y)[0]
            self.assertEqual(r, None)
            r = gradients_impl.gradients(array_ops.stop_gradient(ry), y)[0]
            self.assertEqual(r, None)
            r = gradients_impl.gradients(array_ops.stop_gradient(math_ops.square(rx)), y)[0]
            self.assertEqual(r, None)
            r = gradients_impl.gradients(array_ops.stop_gradient(math_ops.add(rx, ry)), x)[0]
            self.assertEqual(r, None)
            r = gradients_impl.gradients(array_ops.stop_gradient(math_ops.add(rx, ry)), y)[0]
            self.assertEqual(r, None)
            r = gradients_impl.gradients(math_ops.add(rx, ry), y)[0]
            self.assertEqual(168.0, self.evaluate(r))
            r = gradients_impl.gradients(math_ops.add(rx, array_ops.stop_gradient(ry)), y)[0]
            self.assertEqual(136.0, self.evaluate(r))
            r = gradients_impl.gradients(math_ops.add(array_ops.stop_gradient(rx), ry), y)[0]
            self.assertEqual(32.0, self.evaluate(r))

    @test_util.run_deprecated_v1
    def testWhileGrad_StopGradInside(self):
        if False:
            return 10
        with self.cached_session():
            x = constant_op.constant(3.0, name='x')
            y = constant_op.constant(2.0, name='y')
            c = lambda x, y: math_ops.less(x, 100.0)

            def b(x, y):
                if False:
                    while True:
                        i = 10
                y1 = array_ops.stop_gradient(math_ops.square(y))
                x1 = math_ops.add(math_ops.square(x), y1)
                return (x1, y1)
            (rx, _) = while_loop_tf.while_loop(c, b, [x, y])
            r = gradients_impl.gradients(rx, y)[0]
            self.assertAllClose(0.0, self.evaluate(r))
            r = gradients_impl.gradients(rx, x)[0]
            self.assertAllClose(156.0, self.evaluate(r))

    @test_util.run_deprecated_v1
    def testWhileGrad_StopGradInsideNoShape(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            x = array_ops.placeholder(dtypes.float32)
            y = array_ops.placeholder(dtypes.float32)
            c = lambda x, y: math_ops.less(math_ops.reduce_sum(x), 100.0)

            def b(x, y):
                if False:
                    i = 10
                    return i + 15
                y1 = array_ops.stop_gradient(math_ops.square(y, name='stopped'))
                x1 = math_ops.add(math_ops.square(x), y1)
                return (x1, y1)
            (rx, _) = while_loop_tf.while_loop(c, b, [x, y])
            grad_y = gradients_impl.gradients(rx, y)[0]
            grad_x = gradients_impl.gradients(rx, x)[0]
            feed_dict = {x: [3.0, 4.0], y: [2.0, 3.0]}
            self.assertAllClose([0.0, 0.0], sess.run(grad_y, feed_dict=feed_dict))
            self.assertAllClose([156.0, 400.0], sess.run(grad_x, feed_dict=feed_dict))
            name = 'gradients/while/stopped_grad'
            all_ops = x.graph.get_operations()
            self.assertFalse(any((name in op.name for op in all_ops)))

    @test_util.run_deprecated_v1
    def testWhileGradGradFail(self):
        if False:
            print('Hello World!')
        theta = variables.Variable(initial_value=1.0)

        def fn(prev, x):
            if False:
                print('Hello World!')
            return prev + x * theta
        result = functional_ops.scan(fn, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        grad_theta = gradients_impl.gradients(result, theta)
        if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
            with self.assertRaisesRegex(TypeError, 'Second-order gradient'):
                gradients_impl.gradients(grad_theta, theta)
        grad_theta_stopped = array_ops.stop_gradient(grad_theta)
        gradients_impl.gradients(grad_theta_stopped, theta)

    @test_util.run_deprecated_v1
    def testStopGradOnWhileGrad(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            x = constant_op.constant(2.0, name='x')
            y = constant_op.constant(2.0, name='y')
            c = lambda x: math_ops.less(x, 100.0)
            b = lambda x: math_ops.multiply(x, y)
            rx = while_loop_tf.while_loop(c, b, [x])
            rg = gradients_impl.gradients(rx, y)[0]
            rg = array_ops.stop_gradient(rg)
            r = math_ops.add(math_ops.square(y), rx)
            r = math_ops.add(r, rg)
            r = gradients_impl.gradients(r, y)[0]
            self.assertEqual(388.0, self.evaluate(r))

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_deprecated_v1
    def testWhileGradientWithNontrainablePath1(self):
        if False:
            return 10
        q = variables.Variable([7.0, 8.0])

        def cond(_, y):
            if False:
                for i in range(10):
                    print('nop')
            del y
            return False

        def body(x, _):
            if False:
                return 10
            return (x, math_ops.cast(x, dtypes.float32) + math_ops.reduce_sum(q))
        (_, y) = while_loop_tf.while_loop(cond, body, (math_ops.argmin(q), 0.0))
        (dy_dq,) = gradients_impl.gradients(y, q)
        self.assertIsNotNone(dy_dq)
        with self.cached_session() as sess:
            self.evaluate(q.initializer)
            self.assertAllClose([0.0, 0.0], self.evaluate(dy_dq))

    @test_util.disable_control_flow_v2('b/113324949 (RefVariable)')
    @test_util.run_v1_only('b/120545219')
    def testWhileGradientWithNontrainablePath2(self):
        if False:
            for i in range(10):
                print('nop')
        q = variables.Variable([7.0, 8.0])

        def cond(_, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.equal(y, 0.0)

        def body(x, _):
            if False:
                print('Hello World!')
            zero = constant_op.constant(0, dtype=dtypes.int64)
            return (zero, math_ops.cast(x, dtypes.float32) + math_ops.reduce_sum(q))
        (_, y) = while_loop_tf.while_loop(cond, body, (math_ops.argmin(q), 0.0))
        (dy_dq,) = gradients_impl.gradients(y, q)
        self.assertIsNotNone(dy_dq)
        with self.cached_session() as sess:
            self.evaluate(q.initializer)
            self.assertAllClose([1.0, 1.0], self.evaluate(dy_dq))

    @test_util.run_v1_only('b/120545219')
    def testIssue16504(self):
        if False:
            i = 10
            return i + 15
        c = constant_op.constant(np.arange(100), dtype=dtypes.float32)
        w = variables.Variable(initial_value=np.ones(100), dtype=dtypes.float32) / 100
        k = variables.Variable(0, dtype=dtypes.int32)
        chg_w = constant_op.constant(np.inf, dtype=dtypes.float32)

        def cond(k, _, chg_w):
            if False:
                print('Hello World!')
            return math_ops.logical_and(k < 10, chg_w > 0.001)

        def body(k, w, chg_w):
            if False:
                i = 10
                return i + 15
            (grad,) = gradients_impl.gradients(-math_ops.reduce_sum(w * c), w)
            w_n = w * math_ops.exp(-0.1 * grad)
            w_n /= math_ops.reduce_sum(w_n)
            chg_w = math_ops.reduce_sum(math_ops.abs(w_n - w)) / math_ops.reduce_sum(math_ops.abs(w))
            return (k + 1, w_n, chg_w)
        (_, w, _) = while_loop_tf.while_loop(cond, body, [k, w, chg_w])
        (grad,) = gradients_impl.gradients(w, c)
        self.assertIsNotNone(grad)

    @test_util.run_v1_only('b/120545219')
    def testStopGradMultiFlows(self):
        if False:
            print('Hello World!')
        with self.cached_session():

            def body(i, y, r):
                if False:
                    print('Hello World!')
                x = variable_scope.get_variable('x', shape=(), dtype=dtypes.float32, initializer=init_ops.ones_initializer())
                y *= x
                return [i + 1, y, r + math_ops.reduce_sum(y)]
            i0 = constant_op.constant(0)
            y0 = array_ops.ones(5)
            r0 = constant_op.constant(0.0)
            cond = lambda i, y, r: i < 1
            (_, _, r) = while_loop_tf.while_loop(cond, body, [i0, y0, r0], back_prop=True)
            vars_ = variables.global_variables()
            grads = linalg_ops.norm(gradients_impl.gradients(r, vars_)[0])
            z = math_ops.add(r, array_ops.stop_gradient(math_ops.reduce_sum(grads)))
            result = gradients_impl.gradients(z, vars_)[0]
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(5.0, self.evaluate(result))

    @test_util.run_v1_only('b/120545219')
    def testOneValueCond(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            c = array_ops.placeholder(dtypes.int32, shape=[])
            one = ops.convert_to_tensor(1, name='one')
            two = ops.convert_to_tensor(2, name='two')
            p = math_ops.greater_equal(c, 1)
            i = tf_cond.cond(p, lambda : one, lambda : two)
            self.assertTrue(isinstance(i, tensor_lib.Tensor))
            self.assertEqual([1], i.eval(feed_dict={c: 2}))
            self.assertEqual([2], i.eval(feed_dict={c: 0}))

    @test_util.run_deprecated_v1
    def testExampleCond(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            x = ops.convert_to_tensor([-2.0, 2.0], name='x')
            d = array_ops.placeholder(dtypes.int32, shape=[])

            def l2():
                if False:
                    while True:
                        i = 10
                return math_ops.sqrt(math_ops.reduce_sum(math_ops.square(x)))

            def l1():
                if False:
                    while True:
                        i = 10
                return math_ops.reduce_sum(math_ops.abs(x))
            i = tf_cond.cond(math_ops.equal(d, 2), l2, l1)
            self.assertAllClose(4.0, i.eval(feed_dict={d: 1}))
            self.assertAllClose(2.0 * math.sqrt(2), i.eval(feed_dict={d: 2}))

    @test_util.run_v1_only('b/120545219')
    def testCase(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            x = constant_op.constant(1)
            y = constant_op.constant(2)
            z = constant_op.constant(3)
            f1 = lambda : constant_op.constant(17)
            f2 = lambda : constant_op.constant(23)
            f3 = lambda : constant_op.constant(-1)
            r1 = control_flow_case.case({x < y: f1, x > z: f2}, default=f3, exclusive=True)
            self.assertAllEqual(r1, 17)
            r2 = control_flow_case.case([(y > z, f1), (y > x, f2)], default=f3)
            self.assertAllEqual(r2, 23)
            r3 = control_flow_case.case([(x < y, f1), (x < y, f2)], default=f3)
            self.assertAllEqual(r3, 17)
            r4 = control_flow_case.case([(x < y, f1), (x < y, f2)], default=f3, exclusive=True)
            with self.assertRaisesOpError('Input error:'):
                self.evaluate(r4)
            r5 = control_flow_case.case({x > y: f1}, default=f3)
            self.assertAllEqual(r5, -1)
            ran_once = [False, False, False]

            def break_run_twice(ix):
                if False:
                    while True:
                        i = 10

                def _break():
                    if False:
                        i = 10
                        return i + 15
                    ran_once[ix] = True
                    return constant_op.constant(ix)
                return _break
            r6 = control_flow_case.case([(x < y, break_run_twice(0)), (x > y, break_run_twice(1))], default=lambda : constant_op.constant(2))
            self.assertAllEqual(r6, 0)

    @test_util.run_v1_only('b/120545219')
    def testCaseSideEffects(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            v0 = variables.Variable(-1)
            v1 = variables.Variable(-1)
            v2 = variables.Variable(-1)
            a = lambda : control_flow_ops.with_dependencies([state_ops.assign(v0, 0)], 0)
            b = lambda : control_flow_ops.with_dependencies([state_ops.assign(v1, 1)], 1)
            c = lambda : control_flow_ops.with_dependencies([state_ops.assign(v2, 2)], 2)
            x = constant_op.constant(1)
            y = constant_op.constant(2)
            r0 = control_flow_case.case(((x < y, a), (x > y, b)), default=c, exclusive=True)
            r1 = control_flow_case.case(((x > y, a), (x < y, b)), default=c, exclusive=True)
            r2 = control_flow_case.case(((x > y, a), (x > y, b)), default=c, exclusive=True)
            self.evaluate(variables.global_variables_initializer())
            self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1] * 3)
            self.assertEqual(2, self.evaluate(r2))
            self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1, -1, 2])
            self.evaluate(variables.global_variables_initializer())
            self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1] * 3)
            self.assertEqual(1, self.evaluate(r1))
            self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1, 1, -1])
            self.evaluate(variables.global_variables_initializer())
            self.assertAllEqual(self.evaluate([v0, v1, v2]), [-1] * 3)
            self.assertEqual(0, self.evaluate(r0))
            self.assertAllEqual(self.evaluate([v0, v1, v2]), [0, -1, -1])

    @test_util.disable_control_flow_v2('b/113324949 (ref vars)')
    @test_util.run_v1_only('b/120545219')
    def testOneOpCond(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            v = variables.Variable(0)
            c = ops.convert_to_tensor(0)
            one = ops.convert_to_tensor(1)
            two = ops.convert_to_tensor(2)
            p = math_ops.greater_equal(c, 1)

            def a():
                if False:
                    i = 10
                    return i + 15
                return state_ops.assign(v, one)

            def b():
                if False:
                    return 10
                return state_ops.assign(v, two)
            i = tf_cond.cond(p, a, b)
            self.assertTrue(isinstance(i, tensor_lib.Tensor))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(0, self.evaluate(v))
            self.assertEqual(1, i.eval(feed_dict={c.name: 2}))
            self.assertEqual(1, self.evaluate(v))
            self.assertEqual(2, i.eval(feed_dict={c.name: 0}))
            self.assertEqual(2, self.evaluate(v))

    @test_util.run_v1_only('b/120545219')
    def testWithOpsDependencies(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            v = variable_v1.VariableV1(0.0)
            c = constant_op.constant(10)
            with self.assertRaisesOpError('Attempting to use uninitialized value'):
                self.evaluate([c, v])
            real_v = control_flow_ops.with_dependencies(name='real_tensor', output_tensor=v._ref(), dependencies=[v.initializer])
            (c_val, real_v_val) = self.evaluate([c, real_v])
        self.assertAllEqual(10, c_val)
        self.assertAllClose(0.0, real_v_val)

    @test_util.run_v1_only('b/120545219')
    def testWithTensorDependencies(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            v = variable_v1.VariableV1(0.0)
            c1 = constant_op.constant(10)
            c2 = constant_op.constant(20)
            c1_with_init_v = control_flow_ops.with_dependencies(name='c1_with_init_v', output_tensor=c1, dependencies=[v.initializer])
            c2_with_c1_dep = control_flow_ops.with_dependencies(name='c2_with_c1_dep', output_tensor=c2, dependencies=[c1_with_init_v])
            with self.assertRaisesOpError('Attempting to use uninitialized value'):
                self.evaluate(v)
            self.assertAllEqual(20, self.evaluate(c2_with_c1_dep))
            self.assertAllClose(0.0, self.evaluate(v))

    @test_util.run_v1_only('b/120545219')
    def testWithIndexedSlicesDependencies(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            v = variable_v1.VariableV1(np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]).astype(np.float32))
            v_at_1 = indexed_slices.IndexedSlices(v, constant_op.constant([1]))
            gather_v_at_1 = array_ops.gather(v_at_1.values, v_at_1.indices)
            v_at_1_after_init = control_flow_ops.with_dependencies([v.initializer], v_at_1)
            gather_v_at_1_after_init = array_ops.gather(v_at_1_after_init.values, v_at_1_after_init.indices)
            with self.assertRaisesOpError('Attempting to use uninitialized value'):
                self.evaluate(gather_v_at_1)
            self.assertAllEqual([[10.0, 11.0]], self.evaluate(gather_v_at_1_after_init))
            self.assertAllClose([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], self.evaluate(v))

    def testDependenciesDevice(self):
        if False:
            return 10
        with ops.Graph().as_default():
            with ops.device('/job:ps'):
                vd = variable_v1.VariableV1([0.0])
            with_vd_dep = control_flow_ops.with_dependencies([vd.initializer], vd)
            self.assertTrue('/job:ps' in with_vd_dep.device)
            vnod = variable_v1.VariableV1([0.0])
            with_vnod_dep = control_flow_ops.with_dependencies([vnod.initializer], vnod)
            self.assertDeviceEqual(None, with_vnod_dep.device)
            vdef = variable_v1.VariableV1([0.0], name='vdef')
            with ops.device('/job:worker/device:GPU:1'):
                with_vdef_dep = control_flow_ops.with_dependencies([vdef.initializer], vdef)
                self.assertDeviceEqual('', with_vdef_dep.device)
                self.assertEqual([b'loc:@vdef'], with_vdef_dep.op.colocation_groups())

    @test_util.run_v1_only('b/120545219')
    def testGroup(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            v1 = variable_v1.VariableV1([0.0])
            v2 = variable_v1.VariableV1([1.0])
            init = control_flow_ops.group(v1.initializer, v2.initializer)
            with self.assertRaisesOpError('Attempting to use uninitialized value'):
                self.evaluate(v1)
            init.run()
            (v1_val, v2_val) = self.evaluate([v1, v2])
        self.assertAllClose([0.0], v1_val)
        self.assertAllClose([1.0], v2_val)

    @test_util.run_v1_only('b/120545219')
    def testGroupEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        op = control_flow_ops.group()
        self.assertEqual(op.type, 'NoOp')
        self.assertEqual(op.control_inputs, [])

    @test_util.run_deprecated_v1
    def testMergeShapes(self):
        if False:
            i = 10
            return i + 15
        p1 = array_ops.placeholder(dtypes.float32)
        p2 = array_ops.placeholder(dtypes.float32)
        p3 = array_ops.placeholder(dtypes.float32)
        (m, index) = control_flow_ops.merge([p1, p2, p3])
        self.assertIs(None, m.get_shape().ndims)
        self.assertEqual([], index.get_shape())
        p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
        p2 = array_ops.placeholder(dtypes.float32, shape=[1, 2, 3])
        (m, index) = control_flow_ops.merge([p1, p2])
        self.assertIs(None, m.get_shape().ndims)
        self.assertEqual([], index.get_shape())
        p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
        p2 = array_ops.placeholder(dtypes.float32, shape=[2, 1])
        (m, index) = control_flow_ops.merge([p1, p2])
        self.assertEqual([None, None], m.get_shape().as_list())
        self.assertEqual([], index.get_shape())
        p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
        p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
        (m, index) = control_flow_ops.merge([p1, p2])
        self.assertEqual([None, 2], m.get_shape().as_list())
        self.assertEqual([], index.get_shape())
        p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
        p2 = array_ops.placeholder(dtypes.float32, shape=[2, 2])
        (m, index) = control_flow_ops.merge([p1, p2])
        self.assertEqual([None, 2], m.get_shape().as_list())
        self.assertEqual([], index.get_shape())
        p1 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
        p2 = array_ops.placeholder(dtypes.float32, shape=[1, 2])
        (m, index) = control_flow_ops.merge([p1, p2])
        self.assertEqual([1, 2], m.get_shape().as_list())
        self.assertEqual([], index.get_shape())
        p1 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
        p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
        (m, index) = control_flow_ops.merge([p1, p2])
        self.assertEqual([None, 2], m.get_shape().as_list())
        self.assertEqual([], index.get_shape())
        p1 = array_ops.placeholder(dtypes.float32, shape=[None, None])
        p2 = array_ops.placeholder(dtypes.float32, shape=[None, None])
        (m, index) = control_flow_ops.merge([p1, p2])
        self.assertEqual([None, None], m.get_shape().as_list())
        self.assertEqual([], index.get_shape())

    @test_util.run_v1_only('b/120545219')
    def testRefSelect(self):
        if False:
            return 10
        index = array_ops.placeholder(dtypes.int32)
        p1 = array_ops.placeholder(dtypes.float32)
        p2 = array_ops.placeholder(dtypes.float32)
        p3 = array_ops.placeholder(dtypes.float32)
        v1 = variable_v1.VariableV1(p1, validate_shape=False)
        v2 = variable_v1.VariableV1(p2, validate_shape=False)
        v3 = variable_v1.VariableV1(p3, validate_shape=False)
        self.assertIs(None, v1.get_shape().ndims)
        s = control_flow_ops.ref_select(index, [v1, v2, v3])
        self.assertIs(None, s.get_shape().ndims)
        v1 = variable_v1.VariableV1([[1, 2]])
        v2 = variable_v1.VariableV1([[2], [1]])
        s = control_flow_ops.ref_select(index, [v1, v2])
        self.assertIs(None, s.get_shape().ndims)
        v1 = variable_v1.VariableV1([[1, 2]])
        v2 = variable_v1.VariableV1([[1, 2]])
        s = control_flow_ops.ref_select(index, [v1, v2])
        self.assertEqual([1, 2], s.get_shape())
        v1 = variable_v1.VariableV1([[1.0, 2.0]])
        p2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
        v2 = variable_v1.VariableV1(p2, validate_shape=False)
        s = control_flow_ops.ref_select(index, [v1, v2])
        self.assertEqual(None, s.get_shape())

    @test_util.run_deprecated_v1
    def testRunLoopTensor(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            tensor_list = []

            def condition(t):
                if False:
                    while True:
                        i = 10
                return t < constant_op.constant(5)

            def body(_):
                if False:
                    return 10
                tensor_list.append(constant_op.constant(5))
                return constant_op.constant(10)
            result = while_loop_tf.while_loop(condition, body, [constant_op.constant(4)])
            self.assertEqual(10, self.evaluate(result))
            with self.assertRaises(ValueError):
                sess.run(tensor_list[0])

    @test_util.run_v1_only('b/120545219')
    def testWhilePyFuncBasic(self):
        if False:
            while True:
                i = 10

        def func(x):
            if False:
                print('Hello World!')
            return np.square(x)
        with self.cached_session():
            r = while_loop_tf.while_loop(lambda i, v: i < 4, lambda i, v: [i + 1, script_ops.py_func(func, [v], [dtypes.float32])[0]], [constant_op.constant(0), constant_op.constant(2.0, dtypes.float32)], [tensor_shape.unknown_shape(), tensor_shape.unknown_shape()])
            self.assertEqual(self.evaluate(r[1]), 65536.0)

    @test_util.run_v1_only('b/120545219')
    def testWhileFuncBasic(self):
        if False:
            return 10

        @function.Defun(dtypes.float32)
        def func(x):
            if False:
                print('Hello World!')
            return math_ops.square(math_ops.square(x))
        with self.cached_session():
            x = constant_op.constant(2.0, dtypes.float32)
            r = while_loop_tf.while_loop(lambda i, v: i < 2, lambda i, v: [i + 1, func(v)], [constant_op.constant(0), x], [tensor_shape.unknown_shape(), tensor_shape.unknown_shape()])
            grad = gradients_impl.gradients(r, x)[0]
            self.assertEqual(self.evaluate(r[1]), 65536.0)
            self.assertEqual(self.evaluate(grad), 524288.0)
            if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
                self.assertEqual(len([op for op in x.graph.get_operations() if op.type == 'StackV2']), 1)

    @test_util.run_v1_only('b/120545219')
    def testQIntSwitchMerge(self):
        if False:
            return 10
        with self.cached_session(force_gpu=test.is_gpu_available()) as sess:
            constant_qint = constant_op.constant(np.array([42]), dtypes.qint8)
            cond = constant_op.constant(True, dtypes.bool)
            (v_f, v_t) = control_flow_ops.switch(constant_qint, cond)
            result = control_flow_ops.merge([v_f, v_t])
            self.evaluate(result)

    @test_util.run_v1_only('b/120545219')
    def testQIntRefSwitchMerge(self):
        if False:
            return 10
        with self.cached_session(use_gpu=test.is_gpu_available()) as sess:
            var_qint = gen_state_ops.variable(shape=[1], dtype=dtypes.qint8, name='v', container='', shared_name='')
            assign_op = state_ops.assign(var_qint, constant_op.constant(np.array([42]), dtypes.qint8))
            self.evaluate(assign_op)
            cond = constant_op.constant(True, dtypes.bool)
            (v_f, v_t) = control_flow_ops.ref_switch(var_qint, cond)
            result = control_flow_ops.ref_merge([v_f, v_t])
            self.evaluate(result)

    @test_util.run_v1_only('b/120545219')
    def testUInt64SwitchMerge(self):
        if False:
            print('Hello World!')
        with self.cached_session(force_gpu=test.is_gpu_available()) as sess:
            constant_uint64 = constant_op.constant(np.array([42]), dtypes.uint64)
            cond = constant_op.constant(True, dtypes.bool)
            (v_f, v_t) = control_flow_ops.switch(constant_uint64, cond)
            result = control_flow_ops.merge([v_f, v_t])
            self.evaluate(result)

    def testSwitchEagerMode(self):
        if False:
            print('Hello World!')
        if not context.executing_eagerly():
            return
        input_data = [1, 2, 3, 4]
        (vf, vt) = control_flow_ops.switch(input_data, False)
        self.assertAllEqual(vf, input_data)
        self.assertAllEqual(vt, [])

    @test_util.run_deprecated_v1
    def testQIntArgAndRet(self):
        if False:
            return 10

        @function.Defun(dtypes.qint8)
        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            return x
        with self.cached_session(force_gpu=test.is_gpu_available()) as sess:
            qint = constant_op.constant(np.array([42]), dtypes.qint8)
            result = func(qint)
            self.evaluate(result)

    def testSparseIdentity(self):
        if False:
            return 10
        st1 = sparse_tensor.SparseTensor([[0, 5]], ['x'], [10, 10])
        st2 = control_flow_ops._Identity(st1)
        self.assertAllEqual(st1.indices, st2.indices)
        self.assertAllEqual(st1.values, st2.values)
        self.assertAllEqual(st1.dense_shape, st2.dense_shape)

    def testSparseEnterExit(self):
        if False:
            return 10
        st1 = sparse_tensor.SparseTensor([[0, 5]], ['x'], [10, 10])
        st2 = control_flow_ops._Enter(st1, 'foo_1')
        st3 = control_flow_ops.exit(st2)
        self.assertAllEqual(st1.indices, st3.indices)
        self.assertAllEqual(st1.values, st3.values)
        self.assertAllEqual(st1.dense_shape, st3.dense_shape)

    def _buildWhileWithShapeInvariants(self, shape_invariants):
        if False:
            for i in range(10):
                print('nop')
        r = constant_op.constant([1, 2])

        def cond(_):
            if False:
                while True:
                    i = 10
            return False

        def body(_):
            if False:
                print('Hello World!')
            return constant_op.constant([1])
        return while_loop_tf.while_loop(cond, body, [r], shape_invariants=shape_invariants)

    def testWhileOutputShapeWithShapeInvariantsUnknownRank(self):
        if False:
            while True:
                i = 10

        @eager_def_function.function
        def runTest():
            if False:
                while True:
                    i = 10
            while_output = self._buildWhileWithShapeInvariants([tensor_shape.TensorShape(None)])
            self.assertIsNone(while_output.shape.rank)
        runTest()

    def testWhileOutputShapeWithShapeInvariantsPartialShape(self):
        if False:
            print('Hello World!')

        @eager_def_function.function
        def runTest():
            if False:
                print('Hello World!')
            while_output = self._buildWhileWithShapeInvariants([tensor_shape.TensorShape([None])])
            self.assertAllEqual(while_output.shape.as_list(), [None])
        runTest()

    def testFunctionInWhile(self):
        if False:
            return 10

        @eager_def_function.function
        def body(x):
            if False:
                return 10
            return x + 1
        r = while_loop_tf.while_loop(lambda x: x < 5, body, [0])
        self.assertAllEqual(r, 5.0)

class ControlFlowContextCheckTest(test.TestCase):

    def _getWhileTensor(self):
        if False:
            while True:
                i = 10
        'Creates and returns a tensor from a while context.'
        tensor = []

        def body(i):
            if False:
                return 10
            if not tensor:
                tensor.append(constant_op.constant(1))
            return i + tensor[0]
        while_loop_tf.while_loop(lambda i: i < 10, body, [0])
        return tensor[0]

    def _getCondTensor(self):
        if False:
            return 10
        cond_tensor = []

        def true_fn():
            if False:
                return 10
            if not cond_tensor:
                cond_tensor.append(constant_op.constant(1))
            return cond_tensor[0]
        tf_cond.cond(math_ops.less(1, 2), true_fn, lambda : constant_op.constant(0))
        return cond_tensor[0]

    @test_util.run_v1_only('b/120545219')
    def testInvalidContext(self):
        if False:
            return 10
        while_tensor = self._getWhileTensor()
        with self.assertRaisesRegex(ValueError, "Cannot use 'while/Const_1' as input to 'Add' because 'while/Const_1' is in a while loop. See info log for more details."):
            math_ops.add(1, while_tensor)

    @test_util.run_v1_only('b/120545219')
    def testInvalidContextInCond(self):
        if False:
            i = 10
            return i + 15
        while_tensor = self._getWhileTensor()
        with self.assertRaisesRegex(ValueError, "Cannot use 'while/Const_1' as input to 'cond/Add' because 'while/Const_1' is in a while loop. See info log for more details."):
            tf_cond.cond(math_ops.less(1, 2), lambda : math_ops.add(1, while_tensor), lambda : constant_op.constant(0))

    @test_util.run_v1_only('b/120545219')
    def testInvalidContextInWhile(self):
        if False:
            print('Hello World!')
        while_tensor = self._getWhileTensor()
        with self.assertRaisesRegex(ValueError, "Cannot use 'while/Const_1' as input to 'while_1/Add' because they are in different while loops. See info log for more details."):
            while_loop_tf.while_loop(lambda i: i < 10, lambda x: math_ops.add(1, while_tensor), [0])
        with self.assertRaisesRegex(ValueError, "Cannot use 'while/Const_1' as input to 'while_2/NextIteration' because they are in different while loops. See info log for more details."):
            while_loop_tf.while_loop(lambda i: i < 10, lambda i: while_tensor, [0])

    def testValidCondContext(self):
        if False:
            while True:
                i = 10
        cond_tensor = self._getCondTensor()
        math_ops.add(1, cond_tensor)

    def testValidCondContextBranches(self):
        if False:
            for i in range(10):
                print('nop')
        cond_tensor = []

        def branch_fn():
            if False:
                while True:
                    i = 10
            if not cond_tensor:
                cond_tensor.append(constant_op.constant(1))
            return cond_tensor[0]
        tf_cond.cond(math_ops.less(1, 2), branch_fn, branch_fn)

    @test_util.run_v1_only('b/120545219')
    def testValidWhileContext(self):
        if False:
            print('Hello World!')

        def body(_):
            if False:
                while True:
                    i = 10
            c = constant_op.constant(1)
            return while_loop_tf.while_loop(lambda i: i < 3, lambda i: i + c, [0])
        while_loop_tf.while_loop(lambda i: i < 5, body, [0])

    @test_util.run_v1_only('b/120545219')
    def testValidNestedContexts(self):
        if False:
            print('Hello World!')

        def body(_):
            if False:
                return 10
            cond_tensor = self._getCondTensor()
            return tf_cond.cond(math_ops.less(1, 2), lambda : while_loop_tf.while_loop(lambda i: i < 3, lambda i: i + cond_tensor, [0]), lambda : constant_op.constant(0))
        while_loop_tf.while_loop(lambda i: i < 5, body, [0])

    @test_util.run_v1_only('b/120545219')
    def testInvalidNestedContexts(self):
        if False:
            return 10

        def true_fn():
            if False:
                return 10
            while_tensor = self._getWhileTensor()
            return while_loop_tf.while_loop(lambda i: i < 3, lambda i: i + while_tensor, [0])
        with self.assertRaisesRegex(ValueError, "Cannot use 'cond/while/Const_1' as input to 'cond/while_1/add' because they are in different while loops. See info log for more details."):
            tf_cond.cond(math_ops.less(1, 2), true_fn, lambda : constant_op.constant(0))

class TupleTest(test.TestCase):

    @test_util.run_v1_only('b/120545219')
    def testTensors(self):
        if False:
            for i in range(10):
                print('nop')
        for v1_first in [True, False]:
            with self.cached_session():
                v1 = variable_v1.VariableV1([1.0])
                add1 = math_ops.add(control_flow_ops.with_dependencies([v1.initializer], v1._ref()), 2.0)
                v2 = variable_v1.VariableV1([10.0])
                add2 = math_ops.add(control_flow_ops.with_dependencies([v2.initializer], v2._ref()), 20.0)
                (t1, _, t2) = control_flow_ops.tuple([add1, None, add2])
                with self.assertRaisesOpError('Attempting to use uninitialized value'):
                    self.evaluate(v1)
                with self.assertRaisesOpError('Attempting to use uninitialized value'):
                    self.evaluate(v2)
                if v1_first:
                    self.assertAllClose([3.0], self.evaluate(t1))
                    self.assertAllClose([10.0], self.evaluate(v2))
                else:
                    self.assertAllClose([30.0], self.evaluate(t2))
                    self.assertAllClose([1.0], self.evaluate(v1))

    @test_util.run_v1_only('b/120545219')
    def testIndexedSlices(self):
        if False:
            i = 10
            return i + 15
        for v1_first in [True, False]:
            with self.cached_session():
                v1 = variable_v1.VariableV1(np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]).astype(np.float32))
                v1_at_1 = indexed_slices.IndexedSlices(control_flow_ops.with_dependencies([v1.initializer], v1._ref()), constant_op.constant([1]))
                v2 = variable_v1.VariableV1(np.array([[0.1, 1.1], [10.1, 11.1], [20.1, 21.1]]).astype(np.float32))
                v2_at_1 = indexed_slices.IndexedSlices(control_flow_ops.with_dependencies([v2.initializer], v2._ref()), constant_op.constant([1]))
                (st1, st2) = control_flow_ops.tuple([v1_at_1, v2_at_1])
                g1 = array_ops.gather(st1.values, st1.indices)
                g2 = array_ops.gather(st2.values, st2.indices)
                with self.assertRaisesOpError('Attempting to use uninitialized value'):
                    self.evaluate(v1)
                with self.assertRaisesOpError('Attempting to use uninitialized value'):
                    self.evaluate(v2)
                if v1_first:
                    self.assertAllClose([[10.0, 11.0]], self.evaluate(g1))
                    self.assertAllClose([[0.1, 1.1], [10.1, 11.1], [20.1, 21.1]], self.evaluate(v2))
                else:
                    self.assertAllClose([[10.1, 11.1]], self.evaluate(g2))
                    self.assertAllClose([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], self.evaluate(v1))

    def testAcceptTensorsAsControlInputs(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            var = variable_v1.VariableV1(0)
            assign = state_ops.assign(var, 1)
            (t,) = control_flow_ops.tuple([constant_op.constant(0)], control_inputs=[assign])
            self.evaluate(t)
            self.assertEqual(1, self.evaluate(var))

class AssertTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testGuardedAssertDoesNotCopyWhenTrue(self):
        if False:
            while True:
                i = 10
        if test_util.is_gpu_available():
            self.skipTest('b/128646478 fails in opensource')
        with self.session() as sess:
            with ops.device(test.gpu_device_name()):
                value = constant_op.constant(1.0)
            with ops.device('/cpu:0'):
                true = constant_op.constant(True)
                guarded_assert = control_flow_assert.Assert(true, [value], name='guarded')
                unguarded_assert = gen_logging_ops._assert(true, [value], name='unguarded')
            opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            guarded_metadata = config_pb2.RunMetadata()
            sess.run(guarded_assert, options=opts, run_metadata=guarded_metadata)
            unguarded_metadata = config_pb2.RunMetadata()
            sess.run(unguarded_assert, options=opts, run_metadata=unguarded_metadata)
            guarded_nodestat_names = [n.node_name for d in guarded_metadata.step_stats.dev_stats for n in d.node_stats]
            unguarded_nodestat_names = [n.node_name for d in unguarded_metadata.step_stats.dev_stats for n in d.node_stats]
            guarded_memcpy_nodestat_names = [n for n in guarded_nodestat_names if 'MEMCPYDtoH' in n]
            unguarded_memcpy_nodestat_names = [n for n in unguarded_nodestat_names if 'MEMCPYDtoH' in n]
            if 'GPU' in [d.device_type for d in device_lib.list_local_devices()]:
                self.assertLess(0, len(unguarded_memcpy_nodestat_names), str(unguarded_nodestat_names))
            self.assertEqual([], guarded_memcpy_nodestat_names)

class WhileOpBenchmark(test.Benchmark):
    """Evaluate the performance of while_loop op."""

    def _getInitVariables(self):
        if False:
            while True:
                i = 10
        batch_size = 10
        image_size = 256
        kernel_size = 3
        depth = 16
        init_step = constant_op.constant(-1)
        image = variable_scope.get_variable('image', initializer=random_ops.random_normal([batch_size, image_size, image_size, depth], dtype=dtypes.float32, stddev=0.1))
        kernel = variable_scope.get_variable('weights', initializer=random_ops.truncated_normal([kernel_size, kernel_size, depth, depth], dtype=dtypes.float32, stddev=0.1))
        return (init_step, image, kernel)

    def _runOneBenchmark(self, default_device, num_iters=10, static_unroll=False, steps=10):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate the while loop performance.\n\n    Args:\n      default_device: The default device to run all ops except the loop_body.\n        loop_body is always run on GPU.\n      num_iters: Number of iterations to run.\n      static_unroll: If true, run unrolled version; otherwise, run while_loop.\n      steps: Total number of repeated steps to run the loop.\n\n    Returns:\n      The duration of the run in seconds.\n    '

        def loop_body(i, x):
            if False:
                return 10
            with ops.device('/gpu:0'):
                nx = nn_ops.conv2d(input=x, filter=kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', name='conv2d')
                ni = math_ops.add(i, 1)
                return (ni, nx)
        ops.reset_default_graph()
        with session.Session() as sess, ops.device(default_device):
            (i, x, kernel) = self._getInitVariables()
            self.evaluate(variables.global_variables_initializer())
            if static_unroll:
                for _ in range(steps):
                    (i, x) = loop_body(i, x)
            else:
                (i, x) = while_loop_tf.while_loop(lambda i, _: i < steps, loop_body, [i, x], parallel_iterations=steps, swap_memory=True)
            r = math_ops.reduce_sum(x)
            (dx, dk) = gradients_impl.gradients(r, [x, kernel])
            r = control_flow_ops.group(dx, dk)
            for _ in range(3):
                self.evaluate(r)
            start_time = time.time()
            for _ in range(num_iters):
                self.evaluate(r)
            return (time.time() - start_time) / num_iters

    def benchmarkWhileOpCrossDevicePlacement(self):
        if False:
            return 10
        iters = 10
        duration = self._runOneBenchmark('cpu', iters, static_unroll=False)
        self.report_benchmark(name='while_op_cross_device', iters=iters, wall_time=duration)

    def benchmarkWhileOpSameDevicePlacement(self):
        if False:
            for i in range(10):
                print('nop')
        iters = 10
        duration = self._runOneBenchmark('gpu', iters, static_unroll=False)
        self.report_benchmark(name='while_op_same_device', iters=iters, wall_time=duration)

    def benchmarkWhileOpUnrollCrossDevicePlacement(self):
        if False:
            while True:
                i = 10
        iters = 10
        duration = self._runOneBenchmark('cpu', iters, static_unroll=True)
        self.report_benchmark(name='unroll_cross_device_cpu', iters=iters, wall_time=duration)

    def benchmarkWhileOpUnrollSameDevicePlacement(self):
        if False:
            i = 10
            return i + 15
        iters = 10
        duration = self._runOneBenchmark('gpu', iters, static_unroll=True)
        self.report_benchmark(name='unroll_same_device', iters=iters, wall_time=duration)

@test_util.with_control_flow_v2
class EagerTest(test.TestCase):

    def testCond(self):
        if False:
            return 10
        with context.eager_mode():
            pred = math_ops.less(1, 2)
            fn1 = lambda : [constant_op.constant(10)]
            fn2 = lambda : [constant_op.constant(20)]
            r = tf_cond.cond(pred, fn1, fn2)
            self.assertAllEqual(r.numpy(), 10)
            self.assertFalse(isinstance(r, list))

    def DISABLED_testCondInDefun(self):
        if False:
            i = 10
            return i + 15
        with context.eager_mode():

            @eager_def_function.function
            def foo(pred):
                if False:
                    return 10
                fn1 = lambda : (constant_op.constant(10), constant_op.constant(100))
                fn2 = lambda : (constant_op.constant(20), constant_op.constant(200))
                return tf_cond.cond(constant_op.constant(pred), fn1, fn2)
            r = foo(True)
            self.assertAllEqual(r[0].numpy(), 10)
            self.assertNotIsInstance(r, list)
            r = foo(False)
            self.assertAllEqual(r[0].numpy(), 20)
            self.assertFalse(isinstance(r, list))

    def testWhileLoop(self):
        if False:
            for i in range(10):
                print('nop')
        with context.eager_mode():
            tensor = constant_op.constant([1, 2, 3, 4, 5])
            self.assertAllEqual(isum(tensor).numpy(), [46, 47, 48, 49, 50])

    def testWhileLoopWithMaxIterations(self):
        if False:
            print('Hello World!')
        with context.eager_mode():
            tensor = constant_op.constant([1, 2, 3, 4, 5])
            self.assertAllEqual(isum(tensor, maximum_iterations=3).numpy(), [1 + 3, 2 + 3, 3 + 3, 4 + 3, 5 + 3])

    @test_util.run_v1_only('b/120545219')
    def testWhileWithMaximumIterationsAndSingleArgument(self):
        if False:
            return 10
        with context.eager_mode():
            tensor = constant_op.constant(0)
            r = while_loop_tf.while_loop(lambda i: i < 3, lambda i: i + 1, [tensor], maximum_iterations=1)
            self.assertEqual(1, r.numpy())

    def testWithDependencies(self):
        if False:
            i = 10
            return i + 15
        with context.eager_mode():
            t1 = constant_op.constant(1)
            t2 = constant_op.constant(2)
            t3 = control_flow_ops.with_dependencies(t1, t2)
            self.assertAllEqual(t2.numpy(), t3.numpy())

    def testTuple(self):
        if False:
            return 10
        with context.eager_mode():
            t1 = constant_op.constant(1)
            t2 = constant_op.constant(2)
            (tup1, tup2) = control_flow_ops.tuple([t1, t2])
            self.assertAllEqual(t1.numpy(), tup1.numpy())
            self.assertAllEqual(t2.numpy(), tup2.numpy())

    @test_util.run_v1_only('b/120545219')
    def testCase(self):
        if False:
            for i in range(10):
                print('nop')
        with context.eager_mode():
            x = constant_op.constant(1)
            y = constant_op.constant(2)
            z = constant_op.constant(3)
            f1 = lambda : constant_op.constant(17)
            f2 = lambda : constant_op.constant(23)
            f3 = lambda : constant_op.constant(-1)
            r1 = control_flow_case.case([(x < y, f1), (x > z, f2)], default=f3, exclusive=True)
            self.assertAllEqual(r1.numpy(), 17)
if __name__ == '__main__':
    test.main()