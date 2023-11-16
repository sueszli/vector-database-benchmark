"""A module providing various functionality from the top-level pandas namespace.
"""
import re
from typing import Mapping
import pandas as pd
from apache_beam.dataframe import expressions
from apache_beam.dataframe import frame_base
from apache_beam.dataframe import partitionings

def _call_on_first_arg(name):
    if False:
        i = 10
        return i + 15

    def wrapper(target, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if isinstance(target, frame_base.DeferredBase):
            return getattr(target, name)(*args, **kwargs)
        else:
            return getattr(pd, name)(target, *args, **kwargs)
    return staticmethod(wrapper)

def _maybe_wrap_constant_expr(res):
    if False:
        return 10
    if type(res) in frame_base.DeferredBase._pandas_type_map:
        return frame_base.DeferredBase.wrap(expressions.ConstantExpression(res, res[0:0]))
    else:
        return res

def _defer_to_pandas(name):
    if False:
        for i in range(10):
            print('nop')
    func = getattr(pd, name)

    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        res = func(*args, **kwargs)
        return _maybe_wrap_constant_expr(res)
    return staticmethod(wrapper)

def _defer_to_pandas_maybe_elementwise(name):
    if False:
        return 10
    ' Same as _defer_to_pandas, except it handles DeferredBase args, assuming\n  the function can be processed elementwise. '
    func = getattr(pd, name)

    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        if any((isinstance(arg, frame_base.DeferredBase) for arg in args + tuple(kwargs.values()))):
            return frame_base._elementwise_function(func, name)(*args, **kwargs)
        res = func(*args, **kwargs)
        return _maybe_wrap_constant_expr(res)
    return staticmethod(wrapper)

def _is_top_level_function(o):
    if False:
        print('Hello World!')
    return callable(o) and (not isinstance(o, type)) and hasattr(o, '__name__') and re.match('[a-z].*', o.__name__)

class DeferredPandasModule(object):
    array = _defer_to_pandas('array')
    bdate_range = _defer_to_pandas('bdate_range')

    @staticmethod
    @frame_base.args_to_kwargs(pd)
    @frame_base.populate_defaults(pd)
    def concat(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort, copy):
        if False:
            i = 10
            return i + 15
        if ignore_index:
            raise NotImplementedError('concat(ignore_index)')
        if levels:
            raise NotImplementedError('concat(levels)')
        if isinstance(objs, Mapping):
            if keys is None:
                keys = list(objs.keys())
            objs = [objs[k] for k in keys]
        else:
            objs = list(objs)
        if keys is None:
            preserves_partitioning = partitionings.Arbitrary()
        else:
            nlevels = min((o._expr.proxy().index.nlevels for o in objs))
            preserves_partitioning = partitionings.Index([i for i in range(1, nlevels + 1)])
        deferred_none = expressions.ConstantExpression(None)
        exprs = [deferred_none if o is None else o._expr for o in objs]
        if axis in (1, 'columns'):
            required_partitioning = partitionings.Index()
        elif verify_integrity:
            required_partitioning = partitionings.Index()
        else:
            required_partitioning = partitionings.Arbitrary()
        return frame_base.DeferredBase.wrap(expressions.ComputedExpression('concat', lambda *objs: pd.concat(objs, axis=axis, join=join, ignore_index=ignore_index, keys=keys, levels=levels, names=names, verify_integrity=verify_integrity), exprs, requires_partition_by=required_partitioning, preserves_partition_by=preserves_partitioning))
    date_range = _defer_to_pandas('date_range')
    describe_option = _defer_to_pandas('describe_option')
    factorize = _call_on_first_arg('factorize')
    get_option = _defer_to_pandas('get_option')
    interval_range = _defer_to_pandas('interval_range')
    isna = _call_on_first_arg('isna')
    isnull = _call_on_first_arg('isnull')
    json_normalize = _defer_to_pandas('json_normalize')
    melt = _call_on_first_arg('melt')
    merge = _call_on_first_arg('merge')
    melt = _call_on_first_arg('melt')
    merge_ordered = frame_base.wont_implement_method(pd, 'merge_ordered', reason='order-sensitive')
    notna = _call_on_first_arg('notna')
    notnull = _call_on_first_arg('notnull')
    option_context = _defer_to_pandas('option_context')
    period_range = _defer_to_pandas('period_range')
    pivot = _call_on_first_arg('pivot')
    pivot_table = _call_on_first_arg('pivot_table')
    set_eng_float_format = _defer_to_pandas('set_eng_float_format')
    show_versions = _defer_to_pandas('show_versions')
    test = frame_base.wont_implement_method(pd, 'test', explanation='because it is an internal pandas testing utility.')
    timedelta_range = _defer_to_pandas('timedelta_range')
    to_pickle = frame_base.wont_implement_method(pd, 'to_pickle', reason='order-sensitive')
    to_datetime = _defer_to_pandas_maybe_elementwise('to_datetime')
    notna = _call_on_first_arg('notna')

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name.startswith('read_'):

            def func(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                raise frame_base.WontImplementError('Use p | apache_beam.dataframe.io.%s' % name)
            return func
        res = getattr(pd, name)
        if _is_top_level_function(res):
            return frame_base.not_implemented_method(name, base_type=pd)
        else:
            return res
pd_wrapper = DeferredPandasModule()