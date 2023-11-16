from __future__ import annotations
import pytest
pytest
from bokeh.models.expressions import CumSum
from bokeh.models.transforms import Dodge
from tests.support.util.api import verify_all
import bokeh.core.property.vectorization as bcpv
ALL = ('Expr', 'Field', 'Value', 'expr', 'field', 'value')

def test_value_function() -> None:
    if False:
        return 10
    transform = Dodge()
    assert bcpv.value('foo') == bcpv.Value(value='foo')
    assert bcpv.value('foo', transform) == bcpv.Value(value='foo', transform=transform)
    assert bcpv.value('foo', transform=transform) == bcpv.Value(value='foo', transform=transform)

def test_field_function() -> None:
    if False:
        for i in range(10):
            print('nop')
    transform = Dodge()
    assert bcpv.field('foo') == bcpv.Field(field='foo')
    assert bcpv.field('foo', transform) == bcpv.Field(field='foo', transform=transform)
    assert bcpv.field('foo', transform=transform) == bcpv.Field(field='foo', transform=transform)

def test_expr_function() -> None:
    if False:
        for i in range(10):
            print('nop')
    expr = CumSum(field='foo')
    transform = Dodge()
    assert bcpv.expr(expr) == bcpv.Expr(expr=expr)
    assert bcpv.expr(expr, transform) == bcpv.Expr(expr=expr, transform=transform)
    assert bcpv.expr(expr, transform=transform) == bcpv.Expr(expr=expr, transform=transform)
Test___all__ = verify_all(bcpv, ALL)