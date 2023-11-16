import unittest
import apache_beam as beam
from apache_beam.dataframe import expressions
from apache_beam.runners.interactive.caching.expression_cache import ExpressionCache

class ExpressionCacheTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._pcollection_cache = {}
        self._computed_cache = set()
        self._pipeline = beam.Pipeline()
        self.cache = ExpressionCache(self._pcollection_cache, self._computed_cache)

    def create_trace(self, expr):
        if False:
            return 10
        trace = [expr]
        for input in expr.args():
            trace += self.create_trace(input)
        return trace

    def mock_cache(self, expr):
        if False:
            while True:
                i = 10
        pcoll = beam.PCollection(self._pipeline)
        self._pcollection_cache[expr._id] = pcoll
        self._computed_cache.add(pcoll)

    def assertTraceTypes(self, expr, expected):
        if False:
            while True:
                i = 10
        actual_types = [type(e).__name__ for e in self.create_trace(expr)]
        expected_types = [e.__name__ for e in expected]
        self.assertListEqual(actual_types, expected_types)

    def test_only_replaces_cached(self):
        if False:
            print('Hello World!')
        in_expr = expressions.ConstantExpression(0)
        comp_expr = expressions.ComputedExpression('test', lambda x: x, [in_expr])
        expected_trace = [expressions.ComputedExpression, expressions.ConstantExpression]
        self.assertTraceTypes(comp_expr, expected_trace)
        self.cache.replace_with_cached(comp_expr)
        self.assertTraceTypes(comp_expr, expected_trace)
        self.mock_cache(in_expr)
        replaced = self.cache.replace_with_cached(comp_expr)
        expected_trace = [expressions.ComputedExpression, expressions.PlaceholderExpression]
        self.assertTraceTypes(comp_expr, expected_trace)
        self.assertIn(in_expr._id, replaced)

    def test_only_replaces_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        arg_0_expr = expressions.ConstantExpression(0)
        ident_val = expressions.ComputedExpression('ident', lambda x: x, [arg_0_expr])
        arg_1_expr = expressions.ConstantExpression(1)
        comp_expr = expressions.ComputedExpression('add', lambda x, y: x + y, [ident_val, arg_1_expr])
        self.mock_cache(ident_val)
        replaced = self.cache.replace_with_cached(comp_expr)
        expected_trace = [expressions.ComputedExpression, expressions.PlaceholderExpression, expressions.ConstantExpression]
        self.assertTraceTypes(comp_expr, expected_trace)
        self.assertIn(ident_val._id, replaced)
        self.assertNotIn(arg_0_expr, self.create_trace(comp_expr))

    def test_only_caches_same_input(self):
        if False:
            while True:
                i = 10
        arg_0_expr = expressions.ConstantExpression(0)
        ident_val = expressions.ComputedExpression('ident', lambda x: x, [arg_0_expr])
        comp_expr = expressions.ComputedExpression('add', lambda x, y: x + y, [ident_val, arg_0_expr])
        self.mock_cache(arg_0_expr)
        replaced = self.cache.replace_with_cached(comp_expr)
        expected_trace = [expressions.ComputedExpression, expressions.ComputedExpression, expressions.PlaceholderExpression, expressions.PlaceholderExpression]
        actual_trace = self.create_trace(comp_expr)
        unique_placeholders = set((t for t in actual_trace if isinstance(t, expressions.PlaceholderExpression)))
        self.assertTraceTypes(comp_expr, expected_trace)
        self.assertTrue(all((e == replaced[arg_0_expr._id] for e in unique_placeholders)))
        self.assertIn(arg_0_expr._id, replaced)
if __name__ == '__main__':
    unittest.main()