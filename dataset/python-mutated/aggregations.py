"""Execution rules for Aggregatons - mostly TODO.

- ops.Aggregation
- ops.Any
- ops.All
"""
from __future__ import annotations
import functools
import operator
from typing import TYPE_CHECKING
import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import ibis.expr.operations as ops
from ibis.backends.base.df.scope import Scope
from ibis.backends.dask import aggcontext as agg_ctx
from ibis.backends.dask.core import execute
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import coerce_to_output, safe_concat
if TYPE_CHECKING:
    from ibis.backends.base.df.timecontext import TimeContext

@execute_node.register(ops.Aggregation, dd.DataFrame)
def execute_aggregation_dataframe(op, data, scope=None, timecontext: TimeContext | None=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    assert op.metrics, 'no metrics found during aggregation execution'
    if op.sort_keys:
        raise NotImplementedError('sorting on aggregations not yet implemented')
    if op.predicates:
        predicate = functools.reduce(operator.and_, (execute(p, scope=scope, timecontext=timecontext, **kwargs) for p in op.predicates))
        data = data.loc[predicate]
    columns = {}
    if op.by:
        grouping_keys = [key.name if isinstance(key, ops.TableColumn) else execute(key, scope=scope, timecontext=timecontext, **kwargs).rename(key.name) for key in op.by]
        source = data.groupby(grouping_keys)
    else:
        source = data
    scope = scope.merge_scope(Scope({op.table: source}, timecontext))
    pieces = []
    for metric in op.metrics:
        piece = execute(metric, scope=scope, timecontext=timecontext, **kwargs)
        piece = coerce_to_output(piece, metric)
        pieces.append(piece)
    result = safe_concat(pieces)
    if op.by:
        result = result.reset_index()
    result.columns = [columns.get(c, c) for c in result.columns]
    if op.having:
        if not op.by:
            raise ValueError('Filtering out aggregation values is not allowed without at least one grouping key')
        predicate = functools.reduce(operator.and_, (execute(having, scope=scope, timecontext=timecontext, **kwargs) for having in op.having))
        assert len(predicate) == len(result), 'length of predicate does not match length of DataFrame'
        result = result.loc[predicate.values]
    return result

@execute_node.register((ops.Any, ops.All), dd.Series, (dd.Series, type(None)))
def execute_any_all_series(op, data, mask, aggcontext=None, **kwargs):
    if False:
        while True:
            i = 10
    if mask is not None:
        data = data.loc[mask]
    name = type(op).__name__.lower()
    if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
        result = aggcontext.agg(data, name)
    else:
        result = aggcontext.agg(data, operator.methodcaller(name))
    return result

@execute_node.register((ops.Any, ops.All), ddgb.SeriesGroupBy, type(None))
def execute_any_all_series_group_by(op, data, mask, aggcontext=None, **kwargs):
    if False:
        while True:
            i = 10
    name = type(op).__name__.lower()
    if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
        result = aggcontext.agg(data, name)
    else:
        result = aggcontext.agg(data, operator.methodcaller(name))
    return result