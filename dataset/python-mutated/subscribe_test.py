"""Tests for tf.subscribe."""
import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import subscribe
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import googletest

class SubscribeTest(test_util.TensorFlowTestCase):

    def _ExpectSubscribedIdentities(self, container):
        if False:
            i = 10
            return i + 15
        'Convenience function to test a container of subscribed identities.'
        self.assertTrue(all((subscribe._is_subscribed_identity(x) for x in container)))

    @test_util.run_deprecated_v1
    def testSideEffect(self):
        if False:
            print('Hello World!')
        a = constant_op.constant(1)
        b = constant_op.constant(1)
        c = math_ops.add(a, b)
        with ops.control_dependencies([c]):
            d = constant_op.constant(42)
        n = math_ops.negative(c)
        shared = []

        def sub(t):
            if False:
                i = 10
                return i + 15
            shared.append(t)
            return t
        c0 = c
        self.assertTrue(c0.op in d.op.control_inputs)
        c = subscribe.subscribe(c, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        self.assertFalse(c0.op in d.op.control_inputs)
        self.assertTrue(c.op in d.op.control_inputs)
        with self.cached_session() as sess:
            c_out = self.evaluate([c])
            n_out = self.evaluate([n])
            d_out = self.evaluate([d])
        self.assertEqual(n_out, [-2])
        self.assertEqual(c_out, [2])
        self.assertEqual(d_out, [42])
        self.assertEqual(shared, [2, 2, 2])

    @test_util.run_deprecated_v1
    def testSupportedTypes(self):
        if False:
            while True:
                i = 10
        'Confirm that supported types are correctly detected and handled.'
        a = constant_op.constant(1)
        b = constant_op.constant(1)
        c = math_ops.add(a, b)

        def sub(t):
            if False:
                print('Hello World!')
            return t
        subscribed = subscribe.subscribe((a, b), lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        self.assertIsInstance(subscribed, tuple)
        self._ExpectSubscribedIdentities(subscribed)
        subscribed = subscribe.subscribe([a, b], lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        self.assertIsInstance(subscribed, list)
        self._ExpectSubscribedIdentities(subscribed)
        subscribed = subscribe.subscribe({'first': a, 'second': b}, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        self.assertIsInstance(subscribed, dict)
        self._ExpectSubscribedIdentities(subscribed.values())
        TensorPair = collections.namedtuple('TensorPair', ['first', 'second'])
        pair = TensorPair(a, b)
        subscribed = subscribe.subscribe(pair, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        self.assertIsInstance(subscribed, TensorPair)
        self._ExpectSubscribedIdentities(subscribed)
        with self.assertRaisesRegex(TypeError, 'has invalid type'):
            subscribe.subscribe(c.name, lambda t: script_ops.py_func(sub, [t], [t.dtype]))

    @test_util.run_deprecated_v1
    def testCaching(self):
        if False:
            i = 10
            return i + 15
        'Confirm caching of control output is recalculated between calls.'
        a = constant_op.constant(1)
        b = constant_op.constant(2)
        with ops.control_dependencies([a]):
            c = constant_op.constant(42)
        shared = {}

        def sub(t):
            if False:
                while True:
                    i = 10
            shared[t] = shared.get(t, 0) + 1
            return t
        a = subscribe.subscribe(a, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        with ops.control_dependencies([b]):
            d = constant_op.constant(11)
        b = subscribe.subscribe(b, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        with self.cached_session() as sess:
            c_out = self.evaluate([c])
            d_out = self.evaluate([d])
        self.assertEqual(c_out, [42])
        self.assertEqual(d_out, [11])
        self.assertEqual(shared, {2: 1, 1: 1})

    @test_util.run_deprecated_v1
    def testIsSubscribedIdentity(self):
        if False:
            return 10
        'Confirm subscribed identity ops are correctly detected.'
        a = constant_op.constant(1)
        b = constant_op.constant(2)
        c = math_ops.add(a, b)
        idop = array_ops.identity(c)
        c_sub = subscribe.subscribe(c, [])
        self.assertFalse(subscribe._is_subscribed_identity(a))
        self.assertFalse(subscribe._is_subscribed_identity(c))
        self.assertFalse(subscribe._is_subscribed_identity(idop))
        self.assertTrue(subscribe._is_subscribed_identity(c_sub))

    @test_util.run_deprecated_v1
    def testSubscribeExtend(self):
        if False:
            i = 10
            return i + 15
        'Confirm side effect are correctly added for different input types.'
        a = constant_op.constant(1)
        b = constant_op.constant(2)
        c = math_ops.add(a, b)
        shared = {}

        def sub(t, name):
            if False:
                while True:
                    i = 10
            shared[name] = shared.get(name, 0) + 1
            return t
        sub_graph1 = lambda t: sub(t, 'graph1')
        c_sub = subscribe.subscribe(c, lambda t: script_ops.py_func(sub_graph1, [t], [t.dtype]))
        sub_graph2 = lambda t: sub(t, 'graph2')
        c_sub2 = subscribe.subscribe(c_sub, lambda t: script_ops.py_func(sub_graph2, [t], [t.dtype]))
        sub_graph3 = lambda t: sub(t, 'graph3')
        c_sub3 = subscribe.subscribe(c, lambda t: script_ops.py_func(sub_graph3, [t], [t.dtype]))
        graph_ops = ops.get_default_graph().get_operations()
        name_prefix = c.op.name + '/subscription/Identity'
        identity_ops = [op for op in graph_ops if op.name.startswith(name_prefix)]
        self.assertEqual(1, len(identity_ops))
        self.assertIs(c_sub, c_sub2)
        self.assertIs(c_sub, c_sub3)
        with self.cached_session() as sess:
            self.evaluate([c_sub])
        self.assertIn('graph1', shared)
        self.assertIn('graph2', shared)
        self.assertIn('graph3', shared)

    @test_util.run_v1_only('b/120545219')
    def testSubscribeVariable(self):
        if False:
            i = 10
            return i + 15
        'Confirm that variables can be subscribed.'
        v1 = variable_v1.VariableV1(0.0)
        v2 = variable_v1.VariableV1(4.0)
        add = math_ops.add(v1, v2)
        assign_v1 = v1.assign(3.0)
        shared = []

        def sub(t):
            if False:
                while True:
                    i = 10
            shared.append(t)
            return t
        v1_sub = subscribe.subscribe(v1, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        self.assertTrue(subscribe._is_subscribed_identity(v1_sub))
        with self.cached_session() as sess:
            self.evaluate([v1.initializer])
            self.evaluate([v2.initializer])
            self.evaluate([add])
            self.assertEqual(1, len(shared))
            self.evaluate([assign_v1])
            self.assertEqual(1, len(shared))
            self.evaluate([add])
            self.assertEqual(2, len(shared))
            self.assertEqual([0.0, 3.0], shared)

    @test_util.run_v1_only('b/120545219')
    def testResourceType(self):
        if False:
            while True:
                i = 10
        "Confirm that subscribe correctly handles tensors with 'resource' type."
        tensor_array = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name='test', size=3, infer_shape=False)
        writer = tensor_array.write(0, [[4.0, 5.0]])
        reader = writer.read(0)
        shared = []

        def sub(t):
            if False:
                while True:
                    i = 10
            shared.append(t)
            return t
        tensor_array_sub = subscribe.subscribe(tensor_array.handle, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        self.assertIs(tensor_array_sub, tensor_array.handle)
        self.assertFalse(subscribe._is_subscribed_identity(tensor_array.handle))
        with self.cached_session() as sess:
            self.evaluate([reader])
        self.assertEqual(0, len(shared))

    @test_util.run_deprecated_v1
    def testMultipleOutputs(self):
        if False:
            while True:
                i = 10
        'Handle subscriptions to multiple outputs from the same op.'
        sparse_tensor_1 = sparse_tensor.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
        sparse_tensor_2 = sparse_tensor.SparseTensor(indices=[[0, 0], [1, 2]], values=[2, 3], dense_shape=[3, 4])
        sparse_add = sparse_ops.sparse_add(sparse_tensor_1, sparse_tensor_2)
        self.assertEqual(3, len(sparse_add.op.outputs))
        c1 = constant_op.constant(1)
        with ops.control_dependencies(sparse_add.op.outputs):
            neg = -c1
        shared = []

        def sub(t):
            if False:
                print('Hello World!')
            shared.append(t)
            return t
        subscribe.subscribe(sparse_add.op.outputs, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        with self.cached_session() as sess:
            self.evaluate([neg])
        self.assertEqual(3, len(shared))

    @test_util.run_deprecated_v1
    def test_subscribe_tensors_on_different_devices(self):
        if False:
            while True:
                i = 10
        'Side effect ops are added with the same device of the subscribed op.'
        c1 = constant_op.constant(10)
        c2 = constant_op.constant(20)
        with ops.device('cpu:0'):
            add = math_ops.add(c1, c2)
        with ops.device('cpu:1'):
            mul = math_ops.multiply(c1, c2)

        def sub(t):
            if False:
                for i in range(10):
                    print('nop')
            return t
        add_sub = subscribe.subscribe(add, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        mul_sub = subscribe.subscribe(mul, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        self.assertNotEqual(add_sub.device, mul_sub.device)
        self.assertEqual(add.device, add_sub.device)
        self.assertEqual(mul.device, mul_sub.device)

    @test_util.run_v1_only('b/120545219')
    def test_subscribe_tensors_within_control_flow_context(self):
        if False:
            i = 10
            return i + 15
        'Side effect ops are added with the same control flow context.'
        c1 = constant_op.constant(10)
        c2 = constant_op.constant(20)
        x1 = math_ops.add(c1, c2)
        x2 = math_ops.multiply(c1, c2)
        cond = tf_cond.cond(x1 < x2, lambda : math_ops.add(c1, c2, name='then'), lambda : math_ops.subtract(c1, c2, name='else'), name='cond')
        branch = ops.get_default_graph().get_tensor_by_name('cond/then:0')

        def context(tensor):
            if False:
                for i in range(10):
                    print('nop')
            return tensor.op._get_control_flow_context()
        self.assertIs(context(x1), context(x2))
        self.assertIsNot(context(x1), context(branch))
        results = []

        def sub(tensor):
            if False:
                print('Hello World!')
            results.append(tensor)
            return tensor
        tensors = [x1, branch, x2]
        subscriptions = subscribe.subscribe(tensors, lambda t: script_ops.py_func(sub, [t], [t.dtype]))
        for (tensor, subscription) in zip(tensors, subscriptions):
            self.assertIs(context(tensor), context(subscription))
        self.assertIs(context(subscriptions[0]), context(subscriptions[2]))
        self.assertIsNot(context(subscriptions[0]), context(subscriptions[1]))
        with self.cached_session() as sess:
            self.evaluate(cond)
        self.assertEqual(3, len(results))
if __name__ == '__main__':
    googletest.main()