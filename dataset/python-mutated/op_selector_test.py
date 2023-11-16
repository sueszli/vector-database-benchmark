"""Tests for op_selector.py."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.platform import test

class SelectTest(test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.graph = ops_lib.Graph()
        with self.graph.as_default():
            self.a = constant_op.constant([1.0, 1.0], shape=[2], name='a')
            with ops_lib.name_scope('foo'):
                self.b = constant_op.constant([2.0, 2.0], shape=[2], name='b')
                self.c = math_ops.add(self.a, self.b, name='c')
                self.d = constant_op.constant([3.0, 3.0], shape=[2], name='d')
                with ops_lib.name_scope('bar'):
                    self.e = math_ops.add(self.c, self.d, name='e')
                    self.f = math_ops.add(self.c, self.d, name='f')
                    self.g = math_ops.add(self.c, self.a, name='g')
                    with ops_lib.control_dependencies([self.c.op]):
                        self.h = math_ops.add(self.f, self.g, name='h')

    def test_is_iterable(self):
        if False:
            while True:
                i = 10
        'Test for is_iterable.'
        self.assertTrue(op_selector.is_iterable([0, 1, 2]))
        self.assertFalse(op_selector.is_iterable(3))

    def test_unique_graph(self):
        if False:
            while True:
                i = 10
        'Test for check_graphs and get_unique_graph.'
        g0 = ops_lib.Graph()
        with g0.as_default():
            a0 = constant_op.constant(1)
            b0 = constant_op.constant(2)
        g1 = ops_lib.Graph()
        with g1.as_default():
            a1 = constant_op.constant(1)
            b1 = constant_op.constant(2)
        self.assertIsNone(op_selector.check_graphs(a0, b0))
        with self.assertRaises(ValueError):
            op_selector.check_graphs(a0, b0, a1, b1)
        self.assertEqual(op_selector.get_unique_graph([a0, b0]), g0)
        with self.assertRaises(ValueError):
            op_selector.get_unique_graph([a0, b0, a1, b1])

    def test_unique_graph_func_graph(self):
        if False:
            print('Hello World!')
        'Test for get_unique_graph with FuncGraph.'
        outer = ops_lib.Graph()
        with outer.as_default():
            k1 = constant_op.constant(1)
            inner = func_graph.FuncGraph('inner')
            inner._graph_key = outer._graph_key
            with inner.as_default():
                k2 = constant_op.constant(2)
        unique_graph = op_selector.get_unique_graph([k1, k2])
        self.assertEqual(unique_graph._graph_key, inner._graph_key)

    def test_make_list_of_op(self):
        if False:
            i = 10
            return i + 15
        'Test for make_list_of_op.'
        g0 = ops_lib.Graph()
        with g0.as_default():
            a0 = constant_op.constant(1)
            b0 = constant_op.constant(2)
        self.assertEqual(len(op_selector.make_list_of_op(g0)), 2)
        self.assertEqual(len(op_selector.make_list_of_op((a0.op, b0.op))), 2)

    def test_make_list_of_t(self):
        if False:
            i = 10
            return i + 15
        'Test for make_list_of_t.'
        g0 = ops_lib.Graph()
        with g0.as_default():
            a0 = constant_op.constant(1)
            b0 = constant_op.constant(2)
            c0 = math_ops.add(a0, b0)
        self.assertEqual(len(op_selector.make_list_of_t(g0)), 3)
        self.assertEqual(len(op_selector.make_list_of_t((a0, b0))), 2)
        self.assertEqual(len(op_selector.make_list_of_t((a0, a0.op, b0), ignore_ops=True)), 2)

    def test_get_generating_consuming(self):
        if False:
            i = 10
            return i + 15
        'Test for get_generating_ops and get_consuming_ops.'
        g0 = ops_lib.Graph()
        with g0.as_default():
            a0 = constant_op.constant(1)
            b0 = constant_op.constant(2)
            c0 = math_ops.add(a0, b0)
        self.assertEqual(len(op_selector.get_generating_ops([a0, b0])), 2)
        self.assertEqual(len(op_selector.get_consuming_ops([a0, b0])), 1)
        self.assertEqual(len(op_selector.get_generating_ops([c0])), 1)
        self.assertEqual(op_selector.get_consuming_ops([c0]), [])

    def test_backward_walk_ops(self):
        if False:
            return 10
        seed_ops = [self.h.op]
        within_ops = [x.op for x in [self.a, self.b, self.c, self.d, self.e, self.f, self.h]]
        within_ops_fn = lambda op: op not in (self.c.op,)
        stop_at_ts = (self.f,)
        with self.graph.as_default():
            ops = op_selector.get_backward_walk_ops(seed_ops, inclusive=True, within_ops=within_ops, within_ops_fn=within_ops_fn, stop_at_ts=stop_at_ts)
            self.assertEqual(set(ops), set([self.h.op]))
            ops = op_selector.get_backward_walk_ops(seed_ops, inclusive=False, within_ops=within_ops, within_ops_fn=within_ops_fn, stop_at_ts=stop_at_ts)
            self.assertEqual(set(ops), set())
            ops = op_selector.get_backward_walk_ops(seed_ops, inclusive=True, within_ops=within_ops, within_ops_fn=within_ops_fn)
            self.assertEqual(set(ops), set([self.d.op, self.f.op, self.h.op]))
            ops = op_selector.get_backward_walk_ops(seed_ops, inclusive=True, within_ops=within_ops)
            self.assertEqual(set(ops), set([self.a.op, self.b.op, self.c.op, self.d.op, self.f.op, self.h.op]))
            ops = op_selector.get_backward_walk_ops(seed_ops, inclusive=True)
            self.assertEqual(set(ops), set([self.a.op, self.b.op, self.c.op, self.d.op, self.f.op, self.g.op, self.h.op]))
if __name__ == '__main__':
    test.main()