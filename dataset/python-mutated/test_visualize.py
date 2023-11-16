from __future__ import annotations
import os
import pytest
import ibis
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
graphviz = pytest.importorskip('graphviz')
import ibis.expr.visualize as viz
from ibis.expr import api
pytestmark = pytest.mark.skipif(int(os.environ.get('CONDA_BUILD', 0)) == 1, reason='CONDA_BUILD defined')

def key(node):
    if False:
        print('Hello World!')
    return str(hash(node))

@pytest.mark.parametrize('expr_func', [lambda t: t.a, lambda t: t.a + t.b, lambda t: t.a + t.b > 3 ** t.a, lambda t: t[(t.a + t.b * 2 * t.b / t.b ** 3 > 4) & (t.b > 5)], lambda t: t[(t.a + t.b * 2 * t.b / t.b ** 3 > 4) & (t.b > 5)].group_by('c').aggregate(amean=lambda f: f.a.mean(), bsum=lambda f: f.b.sum())])
def test_exprs(alltypes, expr_func):
    if False:
        i = 10
        return i + 15
    expr = expr_func(alltypes)
    graph = viz.to_graph(expr)
    assert key(alltypes.op()) in graph.source
    assert key(expr.op()) in graph.source

def test_custom_expr():
    if False:
        for i in range(10):
            print('nop')

    class MyExpr(ir.Expr):
        pass

    class MyExprNode(ops.Node):
        foo: ops.Value[dt.String, ds.Any]
        bar: ops.Value[dt.Numeric, ds.Any]

        def to_expr(self):
            if False:
                return 10
            return MyExpr(self)
    op = MyExprNode('Hello!', 42.3)
    expr = op.to_expr()
    graph = viz.to_graph(expr)
    assert key(expr.op()) in graph.source

def test_custom_expr_with_not_implemented_type():
    if False:
        print('Hello World!')

    class MyExpr(ir.Expr):

        def type(self):
            if False:
                i = 10
                return i + 15
            raise NotImplementedError

        def schema(self):
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError

    class MyExprNode(ops.Node):
        foo: ops.Value[dt.String, ds.Any]
        bar: ops.Value[dt.Numeric, ds.Any]

        def to_expr(self):
            if False:
                i = 10
                return i + 15
            return MyExpr(self)
    op = MyExprNode('Hello!', 42.3)
    expr = op.to_expr()
    graph = viz.to_graph(expr)
    assert key(expr.op()) in graph.source

@pytest.mark.parametrize('how', ['inner', 'left', 'right', 'outer'])
def test_join(how):
    if False:
        for i in range(10):
            print('nop')
    left = ibis.table([('a', 'int64'), ('b', 'string')])
    right = ibis.table([('b', 'string'), ('c', 'int64')])
    joined = left.join(right, left.b == right.b, how=how)
    result = joined[left.a, right.c]
    graph = viz.to_graph(result)
    assert key(result.op()) in graph.source

def test_order_by():
    if False:
        return 10
    t = ibis.table([('a', 'int64'), ('b', 'string'), ('c', 'int32')])
    expr = t.group_by(t.b).aggregate(sum_a=t.a.sum().cast('double')).order_by('b')
    graph = viz.to_graph(expr)
    assert key(expr.op()) in graph.source

def test_optional_graphviz_repr(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr(ibis.options, 'graphviz_repr', True)
    t = ibis.table([('a', 'int64'), ('b', 'string'), ('c', 'int32')])
    expr = t.group_by(t.b).aggregate(sum_a=t.a.sum().cast('double')).order_by('b')
    assert expr._repr_png_() is not None
    ibis.options.graphviz_repr = False
    assert expr._repr_png_() is None
    ibis.options.graphviz_repr = True
    assert expr._repr_png_() is not None

def test_between():
    if False:
        return 10
    t = ibis.table([('a', 'int64'), ('b', 'string'), ('c', 'int32')])
    expr = t.a.between(1, 1)
    (lower_bound, upper_bound) = expr.op().args[1:]
    graph = viz.to_graph(expr)
    source = graph.source
    assert key(lower_bound) in source
    assert key(upper_bound) in source

def test_asof_join():
    if False:
        return 10
    left = ibis.table([('time', 'int32'), ('value', 'double')])
    right = ibis.table([('time', 'int32'), ('value2', 'double')])
    right = right.mutate(foo=1)
    joined = api.asof_join(left, right, 'time')
    result = joined[left, right.foo]
    graph = viz.to_graph(result)
    assert key(result.op()) in graph.source

def test_filter():
    if False:
        print('Hello World!')
    t = ibis.table([('a', 'int64'), ('b', 'string'), ('c', 'int32')])
    expr = t.filter((t.a == 1) & (t.b == 'x'))
    graph = viz.to_graph(expr, label_edges=True)
    assert 'predicates[0]' in graph.source
    assert 'predicates[1]' in graph.source

def test_html_escape(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(ibis.options, 'graphviz_repr', True)
    expr = ibis.table([('<a & b>', ibis.expr.datatypes.Array('string'))])
    assert expr._repr_png_() is not None
    expr = ibis.array([1, 2, 3])
    assert expr._repr_png_() is not None
    expr = ibis.array([1, 2, 3]).name('COL')
    assert expr._repr_png_() is not None