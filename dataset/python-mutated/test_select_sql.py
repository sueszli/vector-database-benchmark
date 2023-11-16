from __future__ import annotations
import pytest
from pytest import param
import ibis
from ibis import _
from ibis.backends.base.sql.compiler import Compiler
from ibis.tests.sql.conftest import get_query, to_sql
from ibis.tests.util import assert_decompile_roundtrip

@pytest.mark.parametrize('expr_fn', [param(lambda star1, **_: star1.aggregate([star1['f'].sum().name('total')], [star1['foo_id']]), id='agg_explicit_column'), param(lambda star1, **_: star1.aggregate([star1['f'].sum().name('total')], ['foo_id', 'bar_id']), id='agg_string_columns'), param(lambda star1, **_: star1.order_by('f'), id='single_column'), param(lambda star1, **_: star1.limit(10), id='limit_simple'), param(lambda star1, **_: star1.limit(10, offset=5), id='limit_with_offset'), param(lambda star1, **_: star1[star1.f > 0].limit(10), id='filter_then_limit'), param(lambda star1, **_: star1.limit(10)[lambda x: x.f > 0], id='limit_then_filter'), param(lambda star1, **_: star1.count(), id='aggregate_table_count_metric'), param(lambda star1, **_: star1.view(), id='self_reference_simple'), param(lambda t, **_: t, id='test_physical_table_reference_translate')])
def test_select_sql(alltypes, star1, expr_fn, snapshot):
    if False:
        for i in range(10):
            print('nop')
    expr = expr_fn(t=alltypes, star1=star1)
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_nameless_table(snapshot):
    if False:
        while True:
            i = 10
    nameless = ibis.table([('key', 'string')])
    assert to_sql(nameless) == f'SELECT t0.*\nFROM {nameless.op().name} t0'
    expr = ibis.table([('key', 'string')], name='baz')
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_simple_joins(star1, star2, snapshot):
    if False:
        while True:
            i = 10
    t1 = star1
    t2 = star2
    pred = t1['foo_id'] == t2['foo_id']
    pred2 = t1['bar_id'] == t2['foo_id']
    expr = t1.inner_join(t2, [pred])[[t1]]
    snapshot.assert_match(to_sql(expr), 'inner.sql')
    expr = t1.left_join(t2, [pred])[[t1]]
    snapshot.assert_match(to_sql(expr), 'left.sql')
    expr = t1.outer_join(t2, [pred])[[t1]]
    snapshot.assert_match(to_sql(expr), 'outer.sql')
    expr = t1.inner_join(t2, [pred, pred2])[[t1]]
    snapshot.assert_match(to_sql(expr), 'inner_two_preds.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_multiple_joins(star1, star2, star3, snapshot):
    if False:
        i = 10
        return i + 15
    t1 = star1
    t2 = star2
    t3 = star3
    predA = t1['foo_id'] == t2['foo_id']
    predB = t1['bar_id'] == t3['bar_id']
    expr = t1.left_join(t2, [predA]).inner_join(t3, [predB]).select([t1, t2['value1'], t3['value2']])
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_join_between_joins(snapshot):
    if False:
        print('Hello World!')
    t1 = ibis.table([('key1', 'string'), ('key2', 'string'), ('value1', 'double')], 'first')
    t2 = ibis.table([('key1', 'string'), ('value2', 'double')], 'second')
    t3 = ibis.table([('key2', 'string'), ('key3', 'string'), ('value3', 'double')], 'third')
    t4 = ibis.table([('key3', 'string'), ('value4', 'double')], 'fourth')
    left = t1.inner_join(t2, [('key1', 'key1')])[t1, t2.value2]
    right = t3.inner_join(t4, [('key3', 'key3')])[t3, t4.value4]
    joined = left.inner_join(right, [('key2', 'key2')])
    exprs = [left, right.value3, right.value4]
    expr = joined.select(exprs)
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)

def test_join_just_materialized(nation, region, customer, snapshot):
    if False:
        i = 10
        return i + 15
    t1 = nation
    t2 = region
    t3 = customer
    expr = t1.inner_join(t2, t1.n_regionkey == t2.r_regionkey).inner_join(t3, t1.n_nationkey == t3.c_nationkey)
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_semi_join(star1, star2, snapshot):
    if False:
        while True:
            i = 10
    expr = star1.semi_join(star2, [star1.foo_id == star2.foo_id])[[star1]]
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_anti_join(star1, star2, snapshot):
    if False:
        for i in range(10):
            print('nop')
    expr = star1.anti_join(star2, [star1.foo_id == star2.foo_id])[[star1]]
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_where_no_pushdown_possible(star1, star2, snapshot):
    if False:
        print('Hello World!')
    t1 = star1
    t2 = star2
    joined = t1.inner_join(t2, [t1.foo_id == t2.foo_id])[t1, (t1.f - t2.value1).name('diff')]
    expr = joined[joined.diff > 1]
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_where_with_between(alltypes, snapshot):
    if False:
        i = 10
        return i + 15
    t = alltypes
    expr = t.filter([t.a > 0, t.f.between(0, 1)])
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_where_analyze_scalar_op(functional_alltypes, snapshot):
    if False:
        for i in range(10):
            print('nop')
    table = functional_alltypes
    expr = table.filter([table.timestamp_col < ibis.timestamp('2010-01-01') + ibis.interval(months=3), table.timestamp_col < ibis.now() + ibis.interval(days=10)]).count()
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)

def test_bug_duplicated_where(airlines, snapshot):
    if False:
        return 10
    table = airlines
    t = table['arrdelay', 'dest']
    expr = t.group_by('dest').mutate(dest_avg=t.arrdelay.mean(), dev=t.arrdelay - t.arrdelay.mean())
    tmp1 = expr[expr.dev.notnull()]
    tmp2 = tmp1.order_by(ibis.desc('dev'))
    expr = tmp2.limit(10)
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_aggregate_having(star1, snapshot):
    if False:
        print('Hello World!')
    t1 = star1
    total = t1.f.sum().name('total')
    metrics = [total]
    e1 = t1.aggregate(metrics, by=['foo_id'], having=[total > 10])
    snapshot.assert_match(to_sql(e1), 'explicit.sql')
    e2 = t1.aggregate(metrics, by=['foo_id'], having=[t1.count() > 100])
    snapshot.assert_match(to_sql(e2), 'inline.sql')

def test_aggregate_count_joined(con, snapshot):
    if False:
        print('Hello World!')
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    expr = region.inner_join(nation, region.r_regionkey == nation.n_regionkey).select([nation, region.r_name.name('region')]).count()
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_fuse_projections(snapshot):
    if False:
        i = 10
        return i + 15
    table = ibis.table([('foo', 'int32'), ('bar', 'int64'), ('value', 'double')], name='tbl')
    f1 = (table['foo'] + table['bar']).name('baz')
    pred = table['value'] > 0
    table2 = table[table, f1]
    table2_filtered = table2[pred]
    f2 = (table2['foo'] * 2).name('qux')
    table3 = table2.select([table2, f2])
    snapshot.assert_match(to_sql(table3), 'project.sql')
    table3_filtered = table2_filtered.select([table2, f2])
    snapshot.assert_match(to_sql(table3_filtered), 'project_filter.sql')
    assert_decompile_roundtrip(table3_filtered, snapshot, check_equality=False)

def test_projection_filter_fuse(projection_fuse_filter, snapshot):
    if False:
        for i in range(10):
            print('nop')
    (expr1, expr2, expr3) = projection_fuse_filter
    sql1 = Compiler.to_sql(expr1)
    sql2 = Compiler.to_sql(expr2)
    assert sql1 == sql2
    snapshot.assert_match(to_sql(expr3), 'out.sql')

def test_bug_project_multiple_times(customer, nation, region, snapshot):
    if False:
        while True:
            i = 10
    joined = customer.inner_join(nation, [customer.c_nationkey == nation.n_nationkey]).inner_join(region, [nation.n_regionkey == region.r_regionkey])
    proj1 = [customer, nation.n_name, region.r_name]
    step1 = joined[proj1]
    topk_by = step1.c_acctbal.cast('double').sum()
    proj_exprs = [step1.c_name, step1.r_name, step1.n_name]
    step2 = step1.semi_join(step1.n_name.topk(10, by=topk_by), 'n_name')
    expr = step2.select(proj_exprs)
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_aggregate_projection_subquery(alltypes, snapshot):
    if False:
        i = 10
        return i + 15
    t = alltypes
    proj = t[t.f > 0][t, (t.a + t.b).name('foo')]

    def agg(x):
        if False:
            while True:
                i = 10
        return x.aggregate([x.foo.sum().name('foo total')], by=['g'])
    filtered = proj[proj.g == 'bar']
    snapshot.assert_match(to_sql(proj), 'proj.sql')
    snapshot.assert_match(to_sql(filtered), 'filtered.sql')
    snapshot.assert_match(to_sql(agg(filtered)), 'agg_filtered.sql')
    snapshot.assert_match(to_sql(agg(proj[proj.foo < 10])), 'agg_filtered2.sql')

def test_double_nested_subquery_no_aliases(snapshot):
    if False:
        print('Hello World!')
    t = ibis.table([('key1', 'string'), ('key2', 'string'), ('key3', 'string'), ('value', 'double')], 'foo_table')
    agg1 = t.aggregate([t.value.sum().name('total')], by=['key1', 'key2', 'key3'])
    agg2 = agg1.aggregate([agg1.total.sum().name('total')], by=['key1', 'key2'])
    expr = agg2.aggregate([agg2.total.sum().name('total')], by=['key1'])
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_aggregate_projection_alias_bug(star1, star2, snapshot):
    if False:
        return 10
    t1 = star1
    t2 = star2
    what = t1.inner_join(t2, [t1.foo_id == t2.foo_id])[[t1, t2.value1]]
    expr = what.aggregate([what.value1.sum().name('total')], by=[what.foo_id])
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_subquery_in_union(alltypes, snapshot):
    if False:
        for i in range(10):
            print('nop')
    t = alltypes
    expr1 = t.group_by(['a', 'g']).aggregate(t.f.sum().name('metric'))
    expr2 = expr1.view()
    join1 = expr1.join(expr2, expr1.g == expr2.g)[[expr1]]
    join2 = join1.view()
    expr = join1.union(join2)
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)

def test_limit_with_self_join(functional_alltypes, snapshot):
    if False:
        return 10
    t = functional_alltypes
    t2 = t.view()
    expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_topk_predicate_pushdown_bug(nation, customer, region, snapshot):
    if False:
        return 10
    cplusgeo = customer.inner_join(nation, [customer.c_nationkey == nation.n_nationkey]).inner_join(region, [nation.n_regionkey == region.r_regionkey])[customer, nation.n_name, region.r_name]
    expr = cplusgeo.semi_join(cplusgeo.n_name.topk(10, by=cplusgeo.c_acctbal.sum()), 'n_name')
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_topk_analysis_bug(snapshot):
    if False:
        i = 10
        return i + 15
    airlines = ibis.table([('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')], 'airlines')
    dests = ('ORD', 'JFK', 'SFO')
    t = airlines[airlines.dest.isin(dests)]
    expr = t.semi_join(t.dest.topk(10, by=t.arrdelay.mean()), 'dest').group_by('origin').count()
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_topk_to_aggregate(snapshot):
    if False:
        while True:
            i = 10
    t = ibis.table([('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')], 'airlines')
    expr = t.dest.topk(10, by=t.arrdelay.mean())
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_bool_bool(snapshot):
    if False:
        return 10
    t = ibis.table([('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')], 'airlines')
    x = ibis.literal(True)
    expr = t[(t.dest.cast('int64') == 0) == x]
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_case_in_projection(alltypes, snapshot):
    if False:
        return 10
    t = alltypes
    expr = t.g.case().when('foo', 'bar').when('baz', 'qux').else_('default').end()
    expr2 = ibis.case().when(t.g == 'foo', 'bar').when(t.g == 'baz', t.g).end()
    expr = t[expr.name('col1'), expr2.name('col2'), t]
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot, check_equality=False)

def test_identifier_quoting(snapshot):
    if False:
        print('Hello World!')
    data = ibis.table([('date', 'int32'), ('explain', 'string')], 'table')
    expr = data[data.date.name('else'), data.explain.name('join')]
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_scalar_subquery_different_table(foo, bar, snapshot):
    if False:
        print('Hello World!')
    expr = foo[foo.y > bar.x.max()]
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_exists_subquery_repr(t1, t2):
    if False:
        i = 10
        return i + 15
    cond = t1.key1 == t2.key1
    expr = t1[cond.any()]
    stmt = get_query(expr)
    repr(stmt.where[0])

def test_filter_inside_exists(snapshot):
    if False:
        for i in range(10):
            print('nop')
    events = ibis.table([('session_id', 'int64'), ('user_id', 'int64'), ('event_type', 'int32'), ('ts', 'timestamp')], 'events')
    purchases = ibis.table([('item_id', 'int64'), ('user_id', 'int64'), ('price', 'double'), ('ts', 'timestamp')], 'purchases')
    filt = purchases.ts > '2015-08-15'
    cond = (events.user_id == purchases[filt].user_id).any()
    expr = events[cond]
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_order_by_on_limit_yield_subquery(functional_alltypes, snapshot):
    if False:
        while True:
            i = 10
    t = functional_alltypes
    expr = t.group_by('string_col').aggregate([t.count().name('nrows')]).limit(5).order_by('string_col')
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_join_with_limited_table(star1, star2, snapshot):
    if False:
        while True:
            i = 10
    limited = star1.limit(100)
    expr = limited.inner_join(star2, [limited.foo_id == star2.foo_id])[[limited]]
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_multiple_limits(functional_alltypes, snapshot):
    if False:
        i = 10
        return i + 15
    t = functional_alltypes
    expr = t.limit(20).limit(10)
    stmt = get_query(expr)
    assert stmt.limit.n == 10
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_join_filtered_tables_no_pushdown(snapshot):
    if False:
        print('Hello World!')
    tbl_a = ibis.table([('year', 'int32'), ('month', 'int32'), ('day', 'int32'), ('value_a', 'double')], 'a')
    tbl_b = ibis.table([('year', 'int32'), ('month', 'int32'), ('day', 'int32'), ('value_b', 'double')], 'b')
    tbl_a_filter = tbl_a.filter([tbl_a.year == 2016, tbl_a.month == 2, tbl_a.day == 29])
    tbl_b_filter = tbl_b.filter([tbl_b.year == 2016, tbl_b.month == 2, tbl_b.day == 29])
    joined = tbl_a_filter.left_join(tbl_b_filter, ['year', 'month', 'day'])
    result = joined[tbl_a_filter.value_a, tbl_b_filter.value_b]
    join_op = result.op().table
    assert join_op.left == tbl_a_filter.op()
    assert join_op.right == tbl_b_filter.op()
    snapshot.assert_match(to_sql(result), 'out.sql')

def test_loj_subquery_filter_handling(snapshot):
    if False:
        while True:
            i = 10
    left = ibis.table([('id', 'int32'), ('desc', 'string')], 'foo')
    right = ibis.table([('id', 'int32'), ('desc', 'string')], 'bar')
    left = left[left.id < 2]
    right = right[right.id < 3]
    joined = left.left_join(right, ['id', 'desc'])
    expr = joined[[left[name].name('left_' + name) for name in left.columns] + [right[name].name('right_' + name) for name in right.columns]]
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_startswith(startswith, snapshot):
    if False:
        for i in range(10):
            print('nop')
    expr = startswith.name('tmp')
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_endswith(endswith, snapshot):
    if False:
        return 10
    expr = endswith.name('tmp')
    snapshot.assert_match(to_sql(expr), 'out.sql')
    assert_decompile_roundtrip(expr, snapshot)

def test_filter_predicates(snapshot):
    if False:
        i = 10
        return i + 15
    table = ibis.table([('color', 'string')], name='t')
    predicates = [lambda x: x.color.lower().like('%de%'), lambda x: x.color.lower().contains('de'), lambda x: x.color.lower().rlike('.*ge.*')]
    expr = table
    for pred in predicates:
        filtered = expr.filter(pred(expr))
        projected = filtered.select([expr])
        expr = projected
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_join_projection_subquery_bug(nation, region, customer, snapshot):
    if False:
        print('Hello World!')
    geo = nation.inner_join(region, [('n_regionkey', 'r_regionkey')])[nation.n_nationkey, nation.n_name.name('nation'), region.r_name.name('region')]
    expr = geo.inner_join(customer, [('n_nationkey', 'c_nationkey')])[customer, geo]
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_where_with_join(star1, star2, snapshot):
    if False:
        for i in range(10):
            print('nop')
    t1 = star1
    t2 = star2
    e1 = t1.inner_join(t2, [t1.foo_id == t2.foo_id]).select([t1, t2.value1, t2.value3]).filter([t1.f > 0, t2.value3 < 1000])
    snapshot.assert_match(to_sql(e1), 'out.sql')
    assert_decompile_roundtrip(e1, snapshot)

def test_subquery_used_for_self_join(con, snapshot):
    if False:
        while True:
            i = 10
    t = con.table('alltypes')
    agged = t.aggregate([t.f.sum().name('total')], by=['g', 'a', 'b'])
    view = agged.view()
    metrics = [(agged.total - view.total).max().name('metric')]
    expr = agged.inner_join(view, [agged.a == view.b]).aggregate(metrics, by=[agged.g])
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_subquery_factor_correlated_subquery(con, snapshot):
    if False:
        i = 10
        return i + 15
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    customer = con.table('tpch_customer')
    orders = con.table('tpch_orders')
    fields_of_interest = [customer, region.r_name.name('region'), orders.o_totalprice.name('amount'), orders.o_orderdate.cast('timestamp').name('odate')]
    tpch = region.join(nation, region.r_regionkey == nation.n_regionkey).join(customer, customer.c_nationkey == nation.n_nationkey).join(orders, orders.o_custkey == customer.c_custkey)[fields_of_interest]
    t2 = tpch.view()
    conditional_avg = t2[t2.region == tpch.region].amount.mean()
    amount_filter = tpch.amount > conditional_avg
    expr = tpch[amount_filter].limit(10)
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_self_join_subquery_distinct_equal(con, snapshot):
    if False:
        for i in range(10):
            print('nop')
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    j1 = region.join(nation, region.r_regionkey == nation.n_regionkey)[region, nation]
    j2 = region.join(nation, region.r_regionkey == nation.n_regionkey)[region, nation].view()
    expr = j1.join(j2, j1.r_regionkey == j2.r_regionkey)[j1.r_name, j2.n_name]
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_tpch_self_join_failure(con, snapshot):
    if False:
        return 10
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    customer = con.table('tpch_customer')
    orders = con.table('tpch_orders')
    fields_of_interest = [region.r_name.name('region'), nation.n_name.name('nation'), orders.o_totalprice.name('amount'), orders.o_orderdate.cast('timestamp').name('odate')]
    joined_all = region.join(nation, region.r_regionkey == nation.n_regionkey).join(customer, customer.c_nationkey == nation.n_nationkey).join(orders, orders.o_custkey == customer.c_custkey)[fields_of_interest]
    year = joined_all.odate.year().name('year')
    total = joined_all.amount.sum().cast('double').name('total')
    annual_amounts = joined_all.group_by(['region', year]).aggregate(total)
    current = annual_amounts
    prior = annual_amounts.view()
    yoy_change = (current.total - prior.total).name('yoy_change')
    yoy = current.join(prior, current.year == prior.year - 1)[current.region, current.year, yoy_change]
    snapshot.assert_match(to_sql(yoy), 'out.sql')

def test_subquery_in_filter_predicate(star1, snapshot):
    if False:
        while True:
            i = 10
    t1 = star1
    pred = t1.f > t1.f.mean()
    expr = t1[pred]
    snapshot.assert_match(to_sql(expr), 'expr.sql')
    pred2 = t1.f > t1[t1.foo_id == 'foo'].f.mean()
    expr2 = t1[pred2]
    snapshot.assert_match(to_sql(expr2), 'expr2.sql')

def test_filter_subquery_derived_reduction(star1, snapshot):
    if False:
        return 10
    t1 = star1
    pred3 = t1.f > t1[t1.foo_id == 'foo'].f.mean().log()
    pred4 = t1.f > t1[t1.foo_id == 'foo'].f.mean().log() + 1
    expr3 = t1[pred3]
    expr4 = t1[pred4]
    snapshot.assert_match(to_sql(expr3), 'expr3.sql')
    snapshot.assert_match(to_sql(expr4), 'expr4.sql')

def test_topk_operation(snapshot):
    if False:
        return 10
    table = ibis.table([('foo', 'string'), ('bar', 'string'), ('city', 'string'), ('v1', 'double'), ('v2', 'double')], 'tbl')
    e1 = table.semi_join(table.city.topk(10, by=table.v2.mean()), 'city')
    snapshot.assert_match(to_sql(e1), 'e1.sql')
    e2 = table.semi_join(table.city.topk(10), 'city')
    snapshot.assert_match(to_sql(e2), 'e2.sql')

def self_reference_limit_exists(con, snapshot):
    if False:
        print('Hello World!')
    alltypes = con.table('functional_alltypes')
    t = alltypes.limit(100)
    t2 = t.view()
    expr = t[-(t.string_col == t2.string_col).any()]
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_limit_cte_extract(con, snapshot):
    if False:
        return 10
    alltypes = con.table('functional_alltypes')
    t = alltypes.limit(100)
    t2 = t.view()
    expr = t.join(t2).select(t)
    snapshot.assert_match(to_sql(expr), 'out.sql')

def test_filter_self_join_analysis_bug(snapshot):
    if False:
        i = 10
        return i + 15
    purchases = ibis.table([('region', 'string'), ('kind', 'string'), ('user', 'int64'), ('amount', 'double')], 'purchases')
    metric = purchases.amount.sum().name('total')
    agged = purchases.group_by(['region', 'kind']).aggregate(metric)
    left = agged[agged.kind == 'foo']
    right = agged[agged.kind == 'bar']
    joined = left.join(right, left.region == right.region)
    result = joined[left.region, (left.total - right.total).name('diff')]
    snapshot.assert_match(to_sql(result), 'result.sql')

def test_sort_then_group_by_propagates_keys(snapshot):
    if False:
        i = 10
        return i + 15
    t = ibis.table(schema={'a': 'string', 'b': 'int64'}, name='t')
    result = t.order_by('a').b.value_counts()
    snapshot.assert_match(to_sql(result), 'result1.sql')
    result = t.order_by('b').b.value_counts()
    snapshot.assert_match(to_sql(result), 'result2.sql')

def test_incorrect_predicate_pushdown(snapshot):
    if False:
        return 10
    t = ibis.table({'x': int}, name='t')
    result = t.mutate(x=_.x + 1).filter(_.x > 1)
    snapshot.assert_match(to_sql(result), 'result.sql')

def test_incorrect_predicate_pushdown_with_literal(snapshot):
    if False:
        print('Hello World!')
    t = ibis.table(dict(a='int'), name='t')
    expr = t.mutate(a=ibis.literal(1)).filter(lambda t: t.a > 1)
    snapshot.assert_match(to_sql(expr), 'result.sql')

def test_complex_union(snapshot):
    if False:
        i = 10
        return i + 15

    def compute(t):
        if False:
            for i in range(10):
                print('nop')
        return t.select('diag', 'status').mutate(diag=_.diag + 1).mutate(diag=_.diag.cast('int32'))
    schema = ibis.schema(dict(diag='int64', status='string'))
    t1 = compute(ibis.table(schema, name='aids2_one'))
    t2 = compute(ibis.table(schema, name='aids2_two'))
    u = ibis.union(t1, t2)
    snapshot.assert_match(to_sql(u), 'result.sql')

def test_chain_limit_doesnt_collapse(snapshot):
    if False:
        print('Hello World!')
    t = ibis.table([('foo', 'string'), ('bar', 'string'), ('city', 'string'), ('v1', 'double'), ('v2', 'double')], 'tbl')
    expr = t.city.topk(10)[-5:]
    snapshot.assert_match(to_sql(expr), 'result.sql')