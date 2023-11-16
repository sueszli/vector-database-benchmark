"""Tests for cfg module."""
import gast
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.platform import test

class CountingVisitor(cfg.GraphVisitor):

    def __init__(self, graph):
        if False:
            while True:
                i = 10
        super(CountingVisitor, self).__init__(graph)
        self.counts = {}

    def init_state(self, _):
        if False:
            return 10
        return None

    def visit_node(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.counts[node.ast_node] = self.counts.get(node.ast_node, 0) + 1
        return False

class GraphVisitorTest(test.TestCase):

    def _build_cfg(self, fn):
        if False:
            return 10
        (node, _) = parser.parse_entity(fn, future_features=())
        cfgs = cfg.build(node)
        return (cfgs, node)

    def test_basic_coverage_forward(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                return 10
            while a > 0:
                a = 1
                break
                return a
            a = 2
        (graphs, node) = self._build_cfg(test_fn)
        (graph,) = graphs.values()
        visitor = CountingVisitor(graph)
        visitor.visit_forward()
        self.assertEqual(visitor.counts[node.args], 1)
        self.assertEqual(visitor.counts[node.body[0].test], 1)
        self.assertEqual(visitor.counts[node.body[0].body[0]], 1)
        self.assertEqual(visitor.counts[node.body[0].body[1]], 1)
        self.assertNotIn(node.body[0].body[2], visitor.counts)
        self.assertEqual(visitor.counts[node.body[1]], 1)

    def test_basic_coverage_reverse(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            while a > 0:
                a = 1
                break
                return a
            a = 2
        (graphs, node) = self._build_cfg(test_fn)
        (graph,) = graphs.values()
        visitor = CountingVisitor(graph)
        visitor.visit_reverse()
        self.assertEqual(visitor.counts[node.args], 1)
        self.assertEqual(visitor.counts[node.body[0].test], 1)
        self.assertEqual(visitor.counts[node.body[0].body[0]], 1)
        self.assertEqual(visitor.counts[node.body[0].body[1]], 1)
        self.assertEqual(visitor.counts[node.body[0].body[2]], 1)
        self.assertEqual(visitor.counts[node.body[1]], 1)

class AstToCfgTest(test.TestCase):

    def _build_cfg(self, fn):
        if False:
            print('Hello World!')
        (node, _) = parser.parse_entity(fn, future_features=())
        cfgs = cfg.build(node)
        return cfgs

    def _repr_set(self, node_set):
        if False:
            return 10
        return frozenset((repr(n) for n in node_set))

    def _as_set(self, elements):
        if False:
            for i in range(10):
                print('nop')
        if elements is None:
            return frozenset()
        elif isinstance(elements, str):
            return frozenset((elements,))
        else:
            return frozenset(elements)

    def assertGraphMatches(self, graph, edges):
        if False:
            return 10
        'Tests whether the CFG contains the specified edges.'
        for (prev, node_repr, next_) in edges:
            matched = False
            for cfg_node in graph.index.values():
                if repr(cfg_node) == node_repr:
                    if self._as_set(prev) == frozenset(map(repr, cfg_node.prev)) and self._as_set(next_) == frozenset(map(repr, cfg_node.next)):
                        matched = True
                        break
            if not matched:
                self.fail('match failed for node "%s" in graph:\n%s' % (node_repr, graph))

    def assertGraphEnds(self, graph, entry_repr, exit_reprs):
        if False:
            i = 10
            return i + 15
        'Tests whether the CFG has the specified entry and exits.'
        self.assertEqual(repr(graph.entry), entry_repr)
        self.assertSetEqual(frozenset(map(repr, graph.exit)), frozenset(exit_reprs))

    def assertStatementEdges(self, graph, edges):
        if False:
            while True:
                i = 10
        'Tests whether the CFG contains the specified statement edges.'
        for (prev_node_reprs, node_repr, next_node_reprs) in edges:
            matched = False
            partial_matches = []
            self.assertSetEqual(frozenset(graph.stmt_next.keys()), frozenset(graph.stmt_prev.keys()))
            for stmt_ast_node in graph.stmt_next:
                ast_repr = '%s:%s' % (stmt_ast_node.__class__.__name__, stmt_ast_node.lineno)
                if ast_repr == node_repr:
                    actual_next = frozenset(map(repr, graph.stmt_next[stmt_ast_node]))
                    actual_prev = frozenset(map(repr, graph.stmt_prev[stmt_ast_node]))
                    partial_matches.append((actual_prev, node_repr, actual_next))
                    if self._as_set(prev_node_reprs) == actual_prev and self._as_set(next_node_reprs) == actual_next:
                        matched = True
                        break
            if not matched:
                self.fail('edges mismatch for %s: %s' % (node_repr, partial_matches))

    def test_straightline(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            a += 1
            a = 2
            a = 3
            return
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((None, 'a', 'a += 1'), ('a += 1', 'a = 2', 'a = 3'), ('a = 2', 'a = 3', 'return'), ('a = 3', 'return', None)))
        self.assertGraphEnds(graph, 'a', ('return',))

    def test_straightline_no_return(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a, b):
            if False:
                i = 10
                return i + 15
            a = b + 1
            a += max(a)
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((None, 'a, b', 'a = (b + 1)'), ('a = (b + 1)', 'a += max(a)', None)))
        self.assertGraphEnds(graph, 'a, b', ('a += max(a)',))

    def test_unreachable_code(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                while True:
                    i = 10
            return
            a += 1
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((None, 'a', 'return'), ('a', 'return', None), (None, 'a += 1', None)))
        self.assertGraphEnds(graph, 'a', ('return', 'a += 1'))

    def test_if_straightline(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            if a > 0:
                a = 1
            else:
                a += -1
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((None, 'a', '(a > 0)'), ('(a > 0)', 'a = 1', None), ('(a > 0)', 'a += (- 1)', None)))
        self.assertStatementEdges(graph, (('a', 'If:2', None),))
        self.assertGraphEnds(graph, 'a', ('a = 1', 'a += (- 1)'))

    def test_branch_nested(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                print('Hello World!')
            if a > 0:
                if a > 1:
                    a = 1
                else:
                    a = 2
            elif a > 2:
                a = 3
            else:
                a = 4
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((None, 'a', '(a > 0)'), ('a', '(a > 0)', ('(a > 1)', '(a > 2)')), ('(a > 0)', '(a > 1)', ('a = 1', 'a = 2')), ('(a > 1)', 'a = 1', None), ('(a > 1)', 'a = 2', None), ('(a > 0)', '(a > 2)', ('a = 3', 'a = 4')), ('(a > 2)', 'a = 3', None), ('(a > 2)', 'a = 4', None)))
        self.assertStatementEdges(graph, (('a', 'If:2', None), ('(a > 0)', 'If:3', None), ('(a > 0)', 'If:8', None)))
        self.assertGraphEnds(graph, 'a', ('a = 1', 'a = 2', 'a = 3', 'a = 4'))

    def test_branch_straightline_unbalanced(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                return 10
            if a > 0:
                a = 1
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((None, 'a', '(a > 0)'), ('a', '(a > 0)', 'a = 1'), ('(a > 0)', 'a = 1', None)))
        self.assertStatementEdges(graph, (('a', 'If:2', None),))
        self.assertGraphEnds(graph, 'a', ('(a > 0)', 'a = 1'))

    def test_branch_return(self):
        if False:
            return 10

        def test_fn(a):
            if False:
                while True:
                    i = 10
            if a > 0:
                return
            else:
                a = 1
            a = 2
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', '(a > 0)', ('return', 'a = 1')), ('(a > 0)', 'a = 1', 'a = 2'), ('(a > 0)', 'return', None), ('a = 1', 'a = 2', None)))
        self.assertStatementEdges(graph, (('a', 'If:2', 'a = 2'),))
        self.assertGraphEnds(graph, 'a', ('a = 2', 'return'))

    def test_branch_raise(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a):
            if False:
                return 10
            if a > 0:
                raise a
            else:
                a = 1
            a = 2
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', '(a > 0)', ('raise a', 'a = 1')), ('(a > 0)', 'a = 1', 'a = 2'), ('(a > 0)', 'raise a', None), ('a = 1', 'a = 2', None)))
        self.assertStatementEdges(graph, (('a', 'If:2', 'a = 2'),))
        self.assertGraphEnds(graph, 'a', ('a = 2', 'raise a'))

    def test_branch_return_minimal(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                while True:
                    i = 10
            if a > 0:
                return
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', '(a > 0)', 'return'), ('(a > 0)', 'return', None)))
        self.assertStatementEdges(graph, (('a', 'If:2', None),))
        self.assertGraphEnds(graph, 'a', ('(a > 0)', 'return'))

    def test_while_straightline(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                while True:
                    i = 10
            while a > 0:
                a = 1
            a = 2
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), '(a > 0)', ('a = 1', 'a = 2')), ('(a > 0)', 'a = 1', '(a > 0)'), ('(a > 0)', 'a = 2', None)))
        self.assertStatementEdges(graph, (('a', 'While:2', 'a = 2'),))
        self.assertGraphEnds(graph, 'a', ('a = 2',))

    def test_while_else_straightline(self):
        if False:
            return 10

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            while a > 0:
                a = 1
            else:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), '(a > 0)', ('a = 1', 'a = 2')), ('(a > 0)', 'a = 1', '(a > 0)'), ('(a > 0)', 'a = 2', 'a = 3'), ('a = 2', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'While:2', 'a = 3'),))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_while_else_continue(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            while a > 0:
                if a > 1:
                    continue
                else:
                    a = 0
                a = 1
            else:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'continue', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')), ('(a > 0)', '(a > 1)', ('continue', 'a = 0')), ('(a > 1)', 'continue', '(a > 0)'), ('a = 0', 'a = 1', '(a > 0)'), ('(a > 0)', 'a = 2', 'a = 3'), ('a = 2', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'While:2', 'a = 3'), ('(a > 0)', 'If:3', ('a = 1', '(a > 0)'))))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_while_else_break(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            while a > 0:
                if a > 1:
                    break
                a = 1
            else:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')), ('(a > 0)', '(a > 1)', ('break', 'a = 1')), ('(a > 1)', 'break', 'a = 3'), ('(a > 1)', 'a = 1', '(a > 0)'), ('(a > 0)', 'a = 2', 'a = 3'), (('break', 'a = 2'), 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'While:2', 'a = 3'), ('(a > 0)', 'If:3', ('a = 1', 'a = 3'))))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_while_else_return(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a):
            if False:
                return 10
            while a > 0:
                if a > 1:
                    return
                a = 1
            else:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), '(a > 0)', ('(a > 1)', 'a = 2')), ('(a > 0)', '(a > 1)', ('return', 'a = 1')), ('(a > 1)', 'return', None), ('(a > 1)', 'a = 1', '(a > 0)'), ('(a > 0)', 'a = 2', 'a = 3'), ('a = 2', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'While:2', 'a = 3'), ('(a > 0)', 'If:3', 'a = 1')))
        self.assertGraphEnds(graph, 'a', ('a = 3', 'return'))

    def test_while_nested_straightline(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                while True:
                    i = 10
            while a > 0:
                while a > 1:
                    a = 1
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')), (('(a > 0)', 'a = 1'), '(a > 1)', ('a = 1', 'a = 2')), ('(a > 1)', 'a = 1', '(a > 1)'), ('(a > 1)', 'a = 2', '(a > 0)'), ('(a > 0)', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'While:2', 'a = 3'), ('(a > 0)', 'While:3', 'a = 2')))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_while_nested_continue(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                while True:
                    i = 10
            while a > 0:
                while a > 1:
                    if a > 3:
                        continue
                    a = 1
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')), (('(a > 0)', 'continue', 'a = 1'), '(a > 1)', ('(a > 3)', 'a = 2')), ('(a > 1)', '(a > 3)', ('continue', 'a = 1')), ('(a > 3)', 'continue', '(a > 1)'), ('(a > 3)', 'a = 1', '(a > 1)'), ('(a > 1)', 'a = 2', '(a > 0)'), ('(a > 0)', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'While:2', 'a = 3'), ('(a > 0)', 'While:3', 'a = 2'), ('(a > 1)', 'If:4', ('a = 1', '(a > 1)'))))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_while_nested_break(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a):
            if False:
                while True:
                    i = 10
            while a > 0:
                while a > 1:
                    if a > 2:
                        break
                    a = 1
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 2'), '(a > 0)', ('(a > 1)', 'a = 3')), (('(a > 0)', 'a = 1'), '(a > 1)', ('(a > 2)', 'a = 2')), ('(a > 1)', '(a > 2)', ('break', 'a = 1')), ('(a > 2)', 'break', 'a = 2'), ('(a > 2)', 'a = 1', '(a > 1)'), (('(a > 1)', 'break'), 'a = 2', '(a > 0)'), ('(a > 0)', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'While:2', 'a = 3'), ('(a > 0)', 'While:3', 'a = 2'), ('(a > 1)', 'If:4', ('a = 1', 'a = 2'))))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_for_straightline(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                return 10
            for a in range(0, a):
                a = 1
            a = 2
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), 'range(0, a)', ('a = 1', 'a = 2')), ('range(0, a)', 'a = 1', 'range(0, a)'), ('range(0, a)', 'a = 2', None)))
        self.assertStatementEdges(graph, (('a', 'For:2', 'a = 2'),))
        self.assertGraphEnds(graph, 'a', ('a = 2',))

    def test_for_else_straightline(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                return 10
            for a in range(0, a):
                a = 1
            else:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), 'range(0, a)', ('a = 1', 'a = 2')), ('range(0, a)', 'a = 1', 'range(0, a)'), ('range(0, a)', 'a = 2', 'a = 3'), ('a = 2', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'For:2', 'a = 3'),))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_for_else_continue(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            for a in range(0, a):
                if a > 1:
                    continue
                else:
                    a = 0
                a = 1
            else:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'continue', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')), ('range(0, a)', '(a > 1)', ('continue', 'a = 0')), ('(a > 1)', 'continue', 'range(0, a)'), ('(a > 1)', 'a = 0', 'a = 1'), ('a = 0', 'a = 1', 'range(0, a)'), ('range(0, a)', 'a = 2', 'a = 3'), ('a = 2', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'For:2', 'a = 3'), ('range(0, a)', 'If:3', ('a = 1', 'range(0, a)'))))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_for_else_break(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            for a in range(0, a):
                if a > 1:
                    break
                a = 1
            else:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')), ('range(0, a)', '(a > 1)', ('break', 'a = 1')), ('(a > 1)', 'break', 'a = 3'), ('(a > 1)', 'a = 1', 'range(0, a)'), ('range(0, a)', 'a = 2', 'a = 3'), (('break', 'a = 2'), 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'For:2', 'a = 3'), ('range(0, a)', 'If:3', ('a = 1', 'a = 3'))))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_for_else_return(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                while True:
                    i = 10
            for a in range(0, a):
                if a > 1:
                    return
                a = 1
            else:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), 'range(0, a)', ('(a > 1)', 'a = 2')), ('range(0, a)', '(a > 1)', ('return', 'a = 1')), ('(a > 1)', 'return', None), ('(a > 1)', 'a = 1', 'range(0, a)'), ('range(0, a)', 'a = 2', 'a = 3'), ('a = 2', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'For:2', 'a = 3'), ('range(0, a)', 'If:3', 'a = 1')))
        self.assertGraphEnds(graph, 'a', ('a = 3', 'return'))

    def test_for_nested_straightline(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a):
            if False:
                return 10
            for a in range(0, a):
                for b in range(1, a):
                    b += 1
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')), (('range(0, a)', 'b += 1'), 'range(1, a)', ('b += 1', 'a = 2')), ('range(1, a)', 'b += 1', 'range(1, a)'), ('range(1, a)', 'a = 2', 'range(0, a)'), ('range(0, a)', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'For:2', 'a = 3'), ('range(0, a)', 'For:3', 'a = 2')))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_for_nested_continue(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            for a in range(0, a):
                for b in range(1, a):
                    if a > 3:
                        continue
                    b += 1
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')), (('range(0, a)', 'continue', 'b += 1'), 'range(1, a)', ('(a > 3)', 'a = 2')), ('range(1, a)', '(a > 3)', ('continue', 'b += 1')), ('(a > 3)', 'continue', 'range(1, a)'), ('(a > 3)', 'b += 1', 'range(1, a)'), ('range(1, a)', 'a = 2', 'range(0, a)'), ('range(0, a)', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'For:2', 'a = 3'), ('range(0, a)', 'For:3', 'a = 2'), ('range(1, a)', 'If:4', ('b += 1', 'range(1, a)'))))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_for_nested_break(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                while True:
                    i = 10
            for a in range(0, a):
                for b in range(1, a):
                    if a > 2:
                        break
                    b += 1
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 2'), 'range(0, a)', ('range(1, a)', 'a = 3')), (('range(0, a)', 'b += 1'), 'range(1, a)', ('(a > 2)', 'a = 2')), ('range(1, a)', '(a > 2)', ('break', 'b += 1')), ('(a > 2)', 'break', 'a = 2'), ('(a > 2)', 'b += 1', 'range(1, a)'), (('range(1, a)', 'break'), 'a = 2', 'range(0, a)'), ('range(0, a)', 'a = 3', None)))
        self.assertStatementEdges(graph, (('a', 'For:2', 'a = 3'), ('range(0, a)', 'For:3', 'a = 2'), ('range(1, a)', 'If:4', ('b += 1', 'a = 2'))))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_complex(self):
        if False:
            return 10

        def test_fn(a):
            if False:
                print('Hello World!')
            b = 0
            while a > 0:
                for b in range(0, a):
                    if a > 2:
                        break
                    if a > 3:
                        if a > 4:
                            continue
                        else:
                            max(a)
                            break
                    b += 1
                else:
                    return a
                a = 2
            for a in range(1, a):
                return b
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('b = 0', 'a = 2'), '(a > 0)', ('range(0, a)', 'range(1, a)')), (('(a > 0)', 'continue', 'b += 1'), 'range(0, a)', ('(a > 2)', 'return a')), ('range(0, a)', '(a > 2)', ('(a > 3)', 'break')), ('(a > 2)', 'break', 'a = 2'), ('(a > 2)', '(a > 3)', ('(a > 4)', 'b += 1')), ('(a > 3)', '(a > 4)', ('continue', 'max(a)')), ('(a > 4)', 'max(a)', 'break'), ('max(a)', 'break', 'a = 2'), ('(a > 4)', 'continue', 'range(0, a)'), ('(a > 3)', 'b += 1', 'range(0, a)'), ('range(0, a)', 'return a', None), ('break', 'a = 2', '(a > 0)'), ('(a > 0)', 'range(1, a)', ('return b', 'a = 3')), ('range(1, a)', 'return b', None), ('range(1, a)', 'a = 3', None)))
        self.assertStatementEdges(graph, (('b = 0', 'While:3', 'range(1, a)'), ('(a > 0)', 'For:4', 'a = 2'), ('range(0, a)', 'If:5', ('(a > 3)', 'a = 2')), ('(a > 2)', 'If:7', ('b += 1', 'a = 2', 'range(0, a)')), ('(a > 3)', 'If:8', ('a = 2', 'range(0, a)')), ('(a > 0)', 'For:17', 'a = 3')))
        self.assertGraphEnds(graph, 'a', ('a = 3', 'return a', 'return b'))

    def test_finally_straightline(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a):
            if False:
                return 10
            try:
                a += 1
            finally:
                a = 2
            a = 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'a += 1', 'a = 2'), ('a += 1', 'a = 2', 'a = 3'), ('a = 2', 'a = 3', None)))
        self.assertGraphEnds(graph, 'a', ('a = 3',))

    def test_return_finally(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return a
            finally:
                a = 1
            a = 2
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'return a', 'a = 1'), ('return a', 'a = 1', None), (None, 'a = 2', None)))
        self.assertGraphEnds(graph, 'a', ('a = 2', 'a = 1'))

    def test_break_finally(self):
        if False:
            return 10

        def test_fn(a):
            if False:
                print('Hello World!')
            while a > 0:
                try:
                    break
                finally:
                    a = 1
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', '(a > 0)', 'break'), ('(a > 0)', 'break', 'a = 1'), ('break', 'a = 1', None)))
        self.assertGraphEnds(graph, 'a', ('(a > 0)', 'a = 1'))

    def test_continue_finally(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                print('Hello World!')
            while a > 0:
                try:
                    continue
                finally:
                    a = 1
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, ((('a', 'a = 1'), '(a > 0)', 'continue'), ('(a > 0)', 'continue', 'a = 1'), ('continue', 'a = 1', '(a > 0)')))
        self.assertGraphEnds(graph, 'a', ('(a > 0)',))

    def test_with_straightline(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            with max(a) as b:
                a = 0
                return b
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'max(a)', 'a = 0'), ('max(a)', 'a = 0', 'return b'), ('a = 0', 'return b', None)))
        self.assertGraphEnds(graph, 'a', ('return b',))

    def test_lambda_basic(self):
        if False:
            return 10

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            a = lambda b: a + b
            return a
        graphs = self._build_cfg(test_fn)
        for (k, v) in graphs.items():
            if isinstance(k, gast.Lambda):
                lam_graph = v
            else:
                fn_graph = v
        self.assertGraphMatches(fn_graph, (('a', '(lambda b: (a + b))', 'a = (lambda b: (a + b))'), ('(lambda b: (a + b))', 'a = (lambda b: (a + b))', 'return a'), ('a = (lambda b: (a + b))', 'return a', None)))
        self.assertGraphEnds(fn_graph, 'a', ('return a',))
        self.assertGraphMatches(lam_graph, (('b', '(a + b)', None),))
        self.assertGraphEnds(lam_graph, 'b', ('(a + b)',))

    def test_lambda_in_return(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                return 10
            return lambda b: a + b
        graphs = self._build_cfg(test_fn)
        for (k, v) in graphs.items():
            if isinstance(k, gast.Lambda):
                lam_graph = v
            else:
                fn_graph = v
        self.assertGraphMatches(fn_graph, (('a', '(lambda b: (a + b))', 'return (lambda b: (a + b))'), ('(lambda b: (a + b))', 'return (lambda b: (a + b))', None)))
        self.assertGraphEnds(fn_graph, 'a', ('return (lambda b: (a + b))',))
        self.assertGraphMatches(lam_graph, (('b', '(a + b)', None),))
        self.assertGraphEnds(lam_graph, 'b', ('(a + b)',))

    def test_lambda_in_while_loop_test(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                print('Hello World!')
            while (lambda b: a + b)(a):
                pass
        graphs = self._build_cfg(test_fn)
        for (k, v) in graphs.items():
            if isinstance(k, gast.Lambda):
                lam_graph = v
            else:
                fn_graph = v
        self.assertGraphMatches(fn_graph, (('a', '(lambda b: (a + b))', '(lambda b: (a + b))(a)'), (('(lambda b: (a + b))', 'pass'), '(lambda b: (a + b))(a)', 'pass'), ('(lambda b: (a + b))(a)', 'pass', '(lambda b: (a + b))(a)')))
        self.assertGraphEnds(fn_graph, 'a', ('(lambda b: (a + b))(a)',))
        self.assertGraphMatches(lam_graph, (('b', '(a + b)', None),))
        self.assertGraphEnds(lam_graph, 'b', ('(a + b)',))

    def test_lambda_in_for_loop_test(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            for _ in (lambda b: a + b)(a):
                pass
        graphs = self._build_cfg(test_fn)
        for (k, v) in graphs.items():
            if isinstance(k, gast.Lambda):
                lam_graph = v
            else:
                fn_graph = v
        self.assertGraphMatches(fn_graph, (('a', '(lambda b: (a + b))', '(lambda b: (a + b))(a)'), (('(lambda b: (a + b))', 'pass'), '(lambda b: (a + b))(a)', 'pass'), ('(lambda b: (a + b))(a)', 'pass', '(lambda b: (a + b))(a)')))
        self.assertGraphEnds(fn_graph, 'a', ('(lambda b: (a + b))(a)',))
        self.assertGraphMatches(lam_graph, (('b', '(a + b)', None),))
        self.assertGraphEnds(lam_graph, 'b', ('(a + b)',))

    def test_pass(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                return 10
            pass
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'pass', None),))
        self.assertGraphEnds(graph, 'a', ('pass',))

    def test_try_finally(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                while True:
                    i = 10
            try:
                a = 1
            finally:
                a = 2
            return a
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'a = 1', 'a = 2'), ('a = 1', 'a = 2', 'return a'), ('a = 2', 'return a', None)))
        self.assertStatementEdges(graph, (('a', 'Try:2', 'return a'),))
        self.assertGraphEnds(graph, 'a', ('return a',))

    def test_try_except_single_bare(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                while True:
                    i = 10
            try:
                a = 1
                a = 2
            except:
                a = 3
            return a
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'a = 1', 'a = 2'), ('a = 2', 'a = 3', 'return a'), (('a = 2', 'a = 3'), 'return a', None)))
        self.assertStatementEdges(graph, (('a', 'Try:2', 'return a'), ('a = 2', 'ExceptHandler:5', 'return a')))
        self.assertGraphEnds(graph, 'a', ('return a',))

    def test_try_except_single(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            try:
                a = 1
                a = 2
            except Exception1:
                a = 3
            return a
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'a = 1', 'a = 2'), ('a = 2', 'a = 3', 'return a'), (('a = 2', 'a = 3'), 'return a', None)))
        self.assertStatementEdges(graph, (('a', 'Try:2', 'return a'), ('a = 2', 'ExceptHandler:5', 'return a')))
        self.assertGraphEnds(graph, 'a', ('return a',))

    def test_try_except_single_aliased(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                print('Hello World!')
            try:
                a = 1
            except Exception1 as e:
                a = 2
            return a
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'a = 1', ('a = 2', 'return a')), (('a = 1', 'a = 2'), 'return a', None)))
        self.assertStatementEdges(graph, (('a', 'Try:2', 'return a'), ('a = 1', 'ExceptHandler:4', 'return a')))
        self.assertGraphEnds(graph, 'a', ('return a',))

    def test_try_except_single_tuple_aliased(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                while True:
                    i = 10
            try:
                a = 1
            except (Exception1, Exception2) as e:
                a = 2
            return a
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'a = 1', ('a = 2', 'return a')), (('a = 1', 'a = 2'), 'return a', None)))
        self.assertStatementEdges(graph, (('a', 'Try:2', 'return a'), ('a = 1', 'ExceptHandler:4', 'return a')))
        self.assertGraphEnds(graph, 'a', ('return a',))

    def test_try_except_multiple(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            try:
                a = 1
            except Exception1:
                a = 2
            except Exception2:
                a = 3
            return a
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'a = 1', ('a = 2', 'a = 3', 'return a')), (('a = 1', 'a = 2', 'a = 3'), 'return a', None)))
        self.assertStatementEdges(graph, (('a', 'Try:2', 'return a'), ('a = 1', 'ExceptHandler:4', 'return a'), ('a = 1', 'ExceptHandler:6', 'return a')))
        self.assertGraphEnds(graph, 'a', ('return a',))

    def test_try_except_finally(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                while True:
                    i = 10
            try:
                a = 1
            except Exception1:
                a = 2
            except Exception2:
                a = 3
            finally:
                a = 4
            return a
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'a = 1', ('a = 2', 'a = 3', 'a = 4')), (('a = 1', 'a = 2', 'a = 3'), 'a = 4', 'return a'), ('a = 4', 'return a', None)))
        self.assertStatementEdges(graph, (('a', 'Try:2', 'return a'), ('a = 1', 'ExceptHandler:4', 'a = 4'), ('a = 1', 'ExceptHandler:6', 'a = 4')))
        self.assertGraphEnds(graph, 'a', ('return a',))

    def test_try_in_if(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                return 10
            try:
                if a > 0:
                    a = 1
                else:
                    a = 2
            except Exception1:
                a = 3
            a = 4
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', '(a > 0)', ('a = 1', 'a = 2')), ('(a > 0)', 'a = 1', ('a = 3', 'a = 4')), ('(a > 0)', 'a = 2', ('a = 3', 'a = 4')), (('a = 1', 'a = 2'), 'a = 3', 'a = 4'), (('a = 1', 'a = 2', 'a = 3'), 'a = 4', None)))
        self.assertStatementEdges(graph, (('a', 'Try:2', 'a = 4'), ('a', 'If:3', ('a = 3', 'a = 4')), (('a = 1', 'a = 2'), 'ExceptHandler:7', 'a = 4')))
        self.assertGraphEnds(graph, 'a', ('a = 4',))

    def test_try_in_if_all_branches_exit(self):
        if False:
            print('Hello World!')

        def test_fn(a, b):
            if False:
                while True:
                    i = 10
            try:
                if a > 0:
                    raise b
                else:
                    return 0
            except b:
                return 1
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a, b', '(a > 0)', ('raise b', 'return 0')), ('(a > 0)', 'raise b', 'return 1'), ('(a > 0)', 'return 0', None), ('raise b', 'return 1', None)))
        self.assertStatementEdges(graph, (('a, b', 'Try:2', None), ('a, b', 'If:3', 'return 1'), ('raise b', 'ExceptHandler:7', None)))
        self.assertGraphEnds(graph, 'a, b', ('return 0', 'return 1', 'raise b'))

    def test_raise_exits(self):
        if False:
            while True:
                i = 10

        def test_fn(a, b):
            if False:
                while True:
                    i = 10
            raise b
            return a
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a, b', 'raise b', None), (None, 'return a', None)))
        self.assertGraphEnds(graph, 'a, b', ('raise b', 'return a'))

    def test_raise_triggers_enclosing_finally(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                print('Hello World!')
            try:
                try:
                    raise a
                    return 1
                finally:
                    b = 1
                return 2
            finally:
                b = 2
            return b
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'raise a', 'b = 1'), (('raise a', 'return 1'), 'b = 1', 'b = 2'), (None, 'return 1', 'b = 1'), (None, 'return 2', 'b = 2'), (('return 2', 'b = 1'), 'b = 2', None), (None, 'return b', None)))
        self.assertGraphEnds(graph, 'a', ('return b', 'b = 2'))

    def test_raise_adds_finally_sortcuts(self):
        if False:
            return 10

        def test_fn(a):
            if False:
                return 10
            try:
                try:
                    if a > 0:
                        raise a
                    c = 1
                finally:
                    b = 1
                c = 2
            finally:
                b = 2
            return (b, c)
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', '(a > 0)', ('raise a', 'c = 1')), ('(a > 0)', 'raise a', 'b = 1'), ('(a > 0)', 'c = 1', 'b = 1'), (('raise a', 'c = 1'), 'b = 1', ('c = 2', 'b = 2')), ('b = 1', 'c = 2', 'b = 2'), (('b = 1', 'c = 2'), 'b = 2', 'return (b, c)'), ('b = 2', 'return (b, c)', None)))
        self.assertGraphEnds(graph, 'a', ('return (b, c)', 'b = 2'))

    def test_raise_exits_via_except(self):
        if False:
            print('Hello World!')

        def test_fn(a, b):
            if False:
                print('Hello World!')
            try:
                raise b
            except a:
                c = 1
            except b:
                c = 2
            finally:
                c += 3
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a, b', 'raise b', ('c = 1', 'c = 2', 'c += 3')), ('raise b', 'c = 1', 'c += 3'), ('raise b', 'c = 2', 'c += 3'), (('raise b', 'c = 1', 'c = 2'), 'c += 3', None)))
        self.assertGraphEnds(graph, 'a, b', ('c += 3',))

    def test_list_comprehension(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            c = [b for b in a]
            return c
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a', 'c = [b for b in a]', 'return c'), ('c = [b for b in a]', 'return c', None)))
        self.assertGraphEnds(graph, 'a', ('return c',))

    def test_class_definition_empty(self):
        if False:
            print('Hello World!')

        def test_fn(a, b):
            if False:
                i = 10
                return i + 15

            class C(a(b)):
                pass
            return C
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a, b', 'class C', 'return C'), ('class C', 'return C', None)))
        self.assertGraphEnds(graph, 'a, b', ('return C',))

    def test_class_definition_with_members(self):
        if False:
            print('Hello World!')

        def test_fn(a, b):
            if False:
                return 10

            class C(a(b)):
                d = 1
            return C
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('a, b', 'class C', 'return C'), ('class C', 'return C', None)))
        self.assertGraphEnds(graph, 'a, b', ('return C',))

    def test_import(self):
        if False:
            while True:
                i = 10

        def test_fn():
            if False:
                print('Hello World!')
            from a import b
            return b
        (graph,) = self._build_cfg(test_fn).values()
        self.assertGraphMatches(graph, (('', 'from a import b', 'return b'), ('from a import b', 'return b', None)))
        self.assertGraphEnds(graph, '', ('return b',))
if __name__ == '__main__':
    test.main()