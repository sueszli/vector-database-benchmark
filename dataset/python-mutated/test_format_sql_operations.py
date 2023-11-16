from __future__ import annotations
import ibis
from ibis import _

def test_format_sql_query_result(con, snapshot):
    if False:
        i = 10
        return i + 15
    t = con.table('airlines')
    query = '\n        SELECT carrier, mean(arrdelay) AS avg_arrdelay\n        FROM airlines\n        GROUP BY 1\n        ORDER BY 2 DESC\n    '
    schema = ibis.schema({'carrier': 'string', 'avg_arrdelay': 'double'})
    with con.set_query_schema(query, schema):
        expr = t.sql(query)
        expr = expr.op().copy(name='foo').to_expr()
    expr = expr.mutate(island=_.carrier.lower(), avg_arrdelay=_.avg_arrdelay.round(1))
    snapshot.assert_match(repr(expr), 'repr.txt')

def test_memoize_database_table(con, snapshot):
    if False:
        print('Hello World!')
    table = con.table('test1')
    table2 = con.table('test2')
    filter_pred = table['f'] > 0
    table3 = table[filter_pred]
    join_pred = table3['g'] == table2['key']
    joined = table2.inner_join(table3, [join_pred])
    met1 = (table3['f'] - table2['value']).mean().name('foo')
    expr = joined.aggregate([met1, table3['f'].sum().name('bar')], by=[table3['g'], table2['key']])
    result = repr(expr)
    assert result.count('test1') == 1
    assert result.count('test2') == 1
    snapshot.assert_match(result, 'repr.txt')

def test_memoize_insert_sort_key(con, snapshot):
    if False:
        print('Hello World!')
    table = con.table('airlines')
    t = table['arrdelay', 'dest']
    expr = t.group_by('dest').mutate(dest_avg=t.arrdelay.mean(), dev=t.arrdelay - t.arrdelay.mean())
    worst = expr[expr.dev.notnull()].order_by(ibis.desc('dev')).limit(10)
    result = repr(worst)
    assert result.count('airlines') == 1
    snapshot.assert_match(result, 'repr.txt')