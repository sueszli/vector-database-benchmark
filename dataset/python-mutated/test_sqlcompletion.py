from pgcli.packages.sqlcompletion import suggest_type, Special, Database, Schema, Table, Column, View, Keyword, FromClauseItem, Function, Datatype, Alias, JoinCondition, Join
from pgcli.packages.parseutils.tables import TableReference
import pytest

def cols_etc(table, schema=None, alias=None, is_function=False, parent=None, last_keyword=None):
    if False:
        return 10
    'Returns the expected select-clause suggestions for a single-table\n    select.'
    return {Column(table_refs=(TableReference(schema, table, alias, is_function),), qualifiable=True), Function(schema=parent), Keyword(last_keyword)}

def test_select_suggests_cols_with_visible_table_scope():
    if False:
        print('Hello World!')
    suggestions = suggest_type('SELECT  FROM tabl', 'SELECT ')
    assert set(suggestions) == cols_etc('tabl', last_keyword='SELECT')

def test_select_suggests_cols_with_qualified_table_scope():
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type('SELECT  FROM sch.tabl', 'SELECT ')
    assert set(suggestions) == cols_etc('tabl', 'sch', last_keyword='SELECT')

def test_cte_does_not_crash():
    if False:
        i = 10
        return i + 15
    sql = 'WITH CTE AS (SELECT F.* FROM Foo F WHERE F.Bar > 23) SELECT C.* FROM CTE C WHERE C.FooID BETWEEN 123 AND 234;'
    for i in range(len(sql)):
        suggestions = suggest_type(sql[:i + 1], sql[:i + 1])

@pytest.mark.parametrize('expression', ['SELECT * FROM "tabl" WHERE '])
def test_where_suggests_columns_functions_quoted_table(expression):
    if False:
        i = 10
        return i + 15
    expected = cols_etc('tabl', alias='"tabl"', last_keyword='WHERE')
    suggestions = suggest_type(expression, expression)
    assert expected == set(suggestions)

@pytest.mark.parametrize('expression', ['INSERT INTO OtherTabl(ID, Name) SELECT * FROM tabl WHERE ', 'INSERT INTO OtherTabl SELECT * FROM tabl WHERE ', 'SELECT * FROM tabl WHERE ', 'SELECT * FROM tabl WHERE (', 'SELECT * FROM tabl WHERE foo = ', 'SELECT * FROM tabl WHERE bar OR ', 'SELECT * FROM tabl WHERE foo = 1 AND ', 'SELECT * FROM tabl WHERE (bar > 10 AND ', 'SELECT * FROM tabl WHERE (bar AND (baz OR (qux AND (', 'SELECT * FROM tabl WHERE 10 < ', 'SELECT * FROM tabl WHERE foo BETWEEN ', 'SELECT * FROM tabl WHERE foo BETWEEN foo AND '])
def test_where_suggests_columns_functions(expression):
    if False:
        while True:
            i = 10
    suggestions = suggest_type(expression, expression)
    assert set(suggestions) == cols_etc('tabl', last_keyword='WHERE')

@pytest.mark.parametrize('expression', ['SELECT * FROM tabl WHERE foo IN (', 'SELECT * FROM tabl WHERE foo IN (bar, '])
def test_where_in_suggests_columns(expression):
    if False:
        while True:
            i = 10
    suggestions = suggest_type(expression, expression)
    assert set(suggestions) == cols_etc('tabl', last_keyword='WHERE')

@pytest.mark.parametrize('expression', ['SELECT 1 AS ', 'SELECT 1 FROM tabl AS '])
def test_after_as(expression):
    if False:
        return 10
    suggestions = suggest_type(expression, expression)
    assert set(suggestions) == set()

def test_where_equals_any_suggests_columns_or_keywords():
    if False:
        return 10
    text = 'SELECT * FROM tabl WHERE foo = ANY('
    suggestions = suggest_type(text, text)
    assert set(suggestions) == cols_etc('tabl', last_keyword='WHERE')

def test_lparen_suggests_cols_and_funcs():
    if False:
        return 10
    suggestion = suggest_type('SELECT MAX( FROM tbl', 'SELECT MAX(')
    assert set(suggestion) == {Column(table_refs=((None, 'tbl', None, False),), qualifiable=True), Function(schema=None), Keyword('(')}

def test_select_suggests_cols_and_funcs():
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type('SELECT ', 'SELECT ')
    assert set(suggestions) == {Column(table_refs=(), qualifiable=True), Function(schema=None), Keyword('SELECT')}

@pytest.mark.parametrize('expression', ['INSERT INTO ', 'COPY ', 'UPDATE ', 'DESCRIBE '])
def test_suggests_tables_views_and_schemas(expression):
    if False:
        print('Hello World!')
    suggestions = suggest_type(expression, expression)
    assert set(suggestions) == {Table(schema=None), View(schema=None), Schema()}

@pytest.mark.parametrize('expression', ['SELECT * FROM '])
def test_suggest_tables_views_schemas_and_functions(expression):
    if False:
        while True:
            i = 10
    suggestions = suggest_type(expression, expression)
    assert set(suggestions) == {FromClauseItem(schema=None), Schema()}

@pytest.mark.parametrize('expression', ['SELECT * FROM foo JOIN bar on bar.barid = foo.barid JOIN ', 'SELECT * FROM foo JOIN bar USING (barid) JOIN '])
def test_suggest_after_join_with_two_tables(expression):
    if False:
        print('Hello World!')
    suggestions = suggest_type(expression, expression)
    tables = tuple([(None, 'foo', None, False), (None, 'bar', None, False)])
    assert set(suggestions) == {FromClauseItem(schema=None, table_refs=tables), Join(tables, None), Schema()}

@pytest.mark.parametrize('expression', ['SELECT * FROM foo JOIN ', 'SELECT * FROM foo JOIN bar'])
def test_suggest_after_join_with_one_table(expression):
    if False:
        return 10
    suggestions = suggest_type(expression, expression)
    tables = ((None, 'foo', None, False),)
    assert set(suggestions) == {FromClauseItem(schema=None, table_refs=tables), Join(((None, 'foo', None, False),), None), Schema()}

@pytest.mark.parametrize('expression', ['INSERT INTO sch.', 'COPY sch.', 'DESCRIBE sch.'])
def test_suggest_qualified_tables_and_views(expression):
    if False:
        print('Hello World!')
    suggestions = suggest_type(expression, expression)
    assert set(suggestions) == {Table(schema='sch'), View(schema='sch')}

@pytest.mark.parametrize('expression', ['UPDATE sch.'])
def test_suggest_qualified_aliasable_tables_and_views(expression):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(expression, expression)
    assert set(suggestions) == {Table(schema='sch'), View(schema='sch')}

@pytest.mark.parametrize('expression', ['SELECT * FROM sch.', 'SELECT * FROM sch."', 'SELECT * FROM sch."foo', 'SELECT * FROM "sch".', 'SELECT * FROM "sch"."'])
def test_suggest_qualified_tables_views_and_functions(expression):
    if False:
        return 10
    suggestions = suggest_type(expression, expression)
    assert set(suggestions) == {FromClauseItem(schema='sch')}

@pytest.mark.parametrize('expression', ['SELECT * FROM foo JOIN sch.'])
def test_suggest_qualified_tables_views_functions_and_joins(expression):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(expression, expression)
    tbls = tuple([(None, 'foo', None, False)])
    assert set(suggestions) == {FromClauseItem(schema='sch', table_refs=tbls), Join(tbls, 'sch')}

def test_truncate_suggests_tables_and_schemas():
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type('TRUNCATE ', 'TRUNCATE ')
    assert set(suggestions) == {Table(schema=None), Schema()}

def test_truncate_suggests_qualified_tables():
    if False:
        while True:
            i = 10
    suggestions = suggest_type('TRUNCATE sch.', 'TRUNCATE sch.')
    assert set(suggestions) == {Table(schema='sch')}

@pytest.mark.parametrize('text', ['SELECT DISTINCT ', 'INSERT INTO foo SELECT DISTINCT '])
def test_distinct_suggests_cols(text):
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type(text, text)
    assert set(suggestions) == {Column(table_refs=(), local_tables=(), qualifiable=True), Function(schema=None), Keyword('DISTINCT')}

@pytest.mark.parametrize('text, text_before, last_keyword', [('SELECT DISTINCT FROM tbl x JOIN tbl1 y', 'SELECT DISTINCT', 'SELECT'), ('SELECT * FROM tbl x JOIN tbl1 y ORDER BY ', 'SELECT * FROM tbl x JOIN tbl1 y ORDER BY ', 'ORDER BY')])
def test_distinct_and_order_by_suggestions_with_aliases(text, text_before, last_keyword):
    if False:
        return 10
    suggestions = suggest_type(text, text_before)
    assert set(suggestions) == {Column(table_refs=(TableReference(None, 'tbl', 'x', False), TableReference(None, 'tbl1', 'y', False)), local_tables=(), qualifiable=True), Function(schema=None), Keyword(last_keyword)}

@pytest.mark.parametrize('text, text_before', [('SELECT DISTINCT x. FROM tbl x JOIN tbl1 y', 'SELECT DISTINCT x.'), ('SELECT * FROM tbl x JOIN tbl1 y ORDER BY x.', 'SELECT * FROM tbl x JOIN tbl1 y ORDER BY x.')])
def test_distinct_and_order_by_suggestions_with_alias_given(text, text_before):
    if False:
        print('Hello World!')
    suggestions = suggest_type(text, text_before)
    assert set(suggestions) == {Column(table_refs=(TableReference(None, 'tbl', 'x', False),), local_tables=(), qualifiable=False), Table(schema='x'), View(schema='x'), Function(schema='x')}

def test_function_arguments_with_alias_given():
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type('SELECT avg(x. FROM tbl x, tbl2 y', 'SELECT avg(x.')
    assert set(suggestions) == {Column(table_refs=(TableReference(None, 'tbl', 'x', False),), local_tables=(), qualifiable=False), Table(schema='x'), View(schema='x'), Function(schema='x')}

def test_col_comma_suggests_cols():
    if False:
        while True:
            i = 10
    suggestions = suggest_type('SELECT a, b, FROM tbl', 'SELECT a, b,')
    assert set(suggestions) == {Column(table_refs=((None, 'tbl', None, False),), qualifiable=True), Function(schema=None), Keyword('SELECT')}

def test_table_comma_suggests_tables_and_schemas():
    if False:
        while True:
            i = 10
    suggestions = suggest_type('SELECT a, b FROM tbl1, ', 'SELECT a, b FROM tbl1, ')
    assert set(suggestions) == {FromClauseItem(schema=None), Schema()}

def test_into_suggests_tables_and_schemas():
    if False:
        for i in range(10):
            print('nop')
    suggestion = suggest_type('INSERT INTO ', 'INSERT INTO ')
    assert set(suggestion) == {Table(schema=None), View(schema=None), Schema()}

@pytest.mark.parametrize('text', ['INSERT INTO abc (', 'INSERT INTO abc () SELECT * FROM hij;'])
def test_insert_into_lparen_suggests_cols(text):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(text, 'INSERT INTO abc (')
    assert suggestions == (Column(table_refs=((None, 'abc', None, False),), context='insert'),)

def test_insert_into_lparen_partial_text_suggests_cols():
    if False:
        return 10
    suggestions = suggest_type('INSERT INTO abc (i', 'INSERT INTO abc (i')
    assert suggestions == (Column(table_refs=((None, 'abc', None, False),), context='insert'),)

def test_insert_into_lparen_comma_suggests_cols():
    if False:
        return 10
    suggestions = suggest_type('INSERT INTO abc (id,', 'INSERT INTO abc (id,')
    assert suggestions == (Column(table_refs=((None, 'abc', None, False),), context='insert'),)

def test_partially_typed_col_name_suggests_col_names():
    if False:
        print('Hello World!')
    suggestions = suggest_type('SELECT * FROM tabl WHERE col_n', 'SELECT * FROM tabl WHERE col_n')
    assert set(suggestions) == cols_etc('tabl', last_keyword='WHERE')

def test_dot_suggests_cols_of_a_table_or_schema_qualified_table():
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type('SELECT tabl. FROM tabl', 'SELECT tabl.')
    assert set(suggestions) == {Column(table_refs=((None, 'tabl', None, False),)), Table(schema='tabl'), View(schema='tabl'), Function(schema='tabl')}

@pytest.mark.parametrize('sql', ['SELECT t1. FROM tabl1 t1', 'SELECT t1. FROM tabl1 t1, tabl2 t2', 'SELECT t1. FROM "tabl1" t1', 'SELECT t1. FROM "tabl1" t1, "tabl2" t2'])
def test_dot_suggests_cols_of_an_alias(sql):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(sql, 'SELECT t1.')
    assert set(suggestions) == {Table(schema='t1'), View(schema='t1'), Column(table_refs=((None, 'tabl1', 't1', False),)), Function(schema='t1')}

@pytest.mark.parametrize('sql', ['SELECT * FROM tabl1 t1 WHERE t1.', 'SELECT * FROM tabl1 t1, tabl2 t2 WHERE t1.', 'SELECT * FROM "tabl1" t1 WHERE t1.', 'SELECT * FROM "tabl1" t1, tabl2 t2 WHERE t1.'])
def test_dot_suggests_cols_of_an_alias_where(sql):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(sql, sql)
    assert set(suggestions) == {Table(schema='t1'), View(schema='t1'), Column(table_refs=((None, 'tabl1', 't1', False),)), Function(schema='t1')}

def test_dot_col_comma_suggests_cols_or_schema_qualified_table():
    if False:
        print('Hello World!')
    suggestions = suggest_type('SELECT t1.a, t2. FROM tabl1 t1, tabl2 t2', 'SELECT t1.a, t2.')
    assert set(suggestions) == {Column(table_refs=((None, 'tabl2', 't2', False),)), Table(schema='t2'), View(schema='t2'), Function(schema='t2')}

@pytest.mark.parametrize('expression', ['SELECT * FROM (', 'SELECT * FROM foo WHERE EXISTS (', 'SELECT * FROM foo WHERE bar AND NOT EXISTS ('])
def test_sub_select_suggests_keyword(expression):
    if False:
        i = 10
        return i + 15
    suggestion = suggest_type(expression, expression)
    assert suggestion == (Keyword(),)

@pytest.mark.parametrize('expression', ['SELECT * FROM (S', 'SELECT * FROM foo WHERE EXISTS (S', 'SELECT * FROM foo WHERE bar AND NOT EXISTS (S'])
def test_sub_select_partial_text_suggests_keyword(expression):
    if False:
        print('Hello World!')
    suggestion = suggest_type(expression, expression)
    assert suggestion == (Keyword(),)

def test_outer_table_reference_in_exists_subquery_suggests_columns():
    if False:
        print('Hello World!')
    q = 'SELECT * FROM foo f WHERE EXISTS (SELECT 1 FROM bar WHERE f.'
    suggestions = suggest_type(q, q)
    assert set(suggestions) == {Column(table_refs=((None, 'foo', 'f', False),)), Table(schema='f'), View(schema='f'), Function(schema='f')}

@pytest.mark.parametrize('expression', ['SELECT * FROM (SELECT * FROM '])
def test_sub_select_table_name_completion(expression):
    if False:
        for i in range(10):
            print('nop')
    suggestion = suggest_type(expression, expression)
    assert set(suggestion) == {FromClauseItem(schema=None), Schema()}

@pytest.mark.parametrize('expression', ['SELECT * FROM foo WHERE EXISTS (SELECT * FROM ', 'SELECT * FROM foo WHERE bar AND NOT EXISTS (SELECT * FROM '])
def test_sub_select_table_name_completion_with_outer_table(expression):
    if False:
        i = 10
        return i + 15
    suggestion = suggest_type(expression, expression)
    tbls = tuple([(None, 'foo', None, False)])
    assert set(suggestion) == {FromClauseItem(schema=None, table_refs=tbls), Schema()}

def test_sub_select_col_name_completion():
    if False:
        while True:
            i = 10
    suggestions = suggest_type('SELECT * FROM (SELECT  FROM abc', 'SELECT * FROM (SELECT ')
    assert set(suggestions) == {Column(table_refs=((None, 'abc', None, False),), qualifiable=True), Function(schema=None), Keyword('SELECT')}

@pytest.mark.xfail
def test_sub_select_multiple_col_name_completion():
    if False:
        while True:
            i = 10
    suggestions = suggest_type('SELECT * FROM (SELECT a, FROM abc', 'SELECT * FROM (SELECT a, ')
    assert set(suggestions) == cols_etc('abc')

def test_sub_select_dot_col_name_completion():
    if False:
        print('Hello World!')
    suggestions = suggest_type('SELECT * FROM (SELECT t. FROM tabl t', 'SELECT * FROM (SELECT t.')
    assert set(suggestions) == {Column(table_refs=((None, 'tabl', 't', False),)), Table(schema='t'), View(schema='t'), Function(schema='t')}

@pytest.mark.parametrize('join_type', ('', 'INNER', 'LEFT', 'RIGHT OUTER'))
@pytest.mark.parametrize('tbl_alias', ('', 'foo'))
def test_join_suggests_tables_and_schemas(tbl_alias, join_type):
    if False:
        return 10
    text = f'SELECT * FROM abc {tbl_alias} {join_type} JOIN '
    suggestion = suggest_type(text, text)
    tbls = tuple([(None, 'abc', tbl_alias or None, False)])
    assert set(suggestion) == {FromClauseItem(schema=None, table_refs=tbls), Schema(), Join(tbls, None)}

def test_left_join_with_comma():
    if False:
        return 10
    text = 'select * from foo f left join bar b,'
    suggestions = suggest_type(text, text)
    tbls = tuple([(None, 'foo', 'f', False)])
    assert set(suggestions) == {FromClauseItem(schema=None, table_refs=tbls), Schema()}

@pytest.mark.parametrize('sql', ['SELECT * FROM abc a JOIN def d ON a.', 'SELECT * FROM abc a JOIN def d ON a.id = d.id AND a.'])
def test_join_alias_dot_suggests_cols1(sql):
    if False:
        print('Hello World!')
    suggestions = suggest_type(sql, sql)
    tables = ((None, 'abc', 'a', False), (None, 'def', 'd', False))
    assert set(suggestions) == {Column(table_refs=((None, 'abc', 'a', False),)), Table(schema='a'), View(schema='a'), Function(schema='a'), JoinCondition(table_refs=tables, parent=(None, 'abc', 'a', False))}

@pytest.mark.parametrize('sql', ['SELECT * FROM abc a JOIN def d ON a.id = d.', 'SELECT * FROM abc a JOIN def d ON a.id = d.id AND a.id2 = d.'])
def test_join_alias_dot_suggests_cols2(sql):
    if False:
        i = 10
        return i + 15
    suggestion = suggest_type(sql, sql)
    assert set(suggestion) == {Column(table_refs=((None, 'def', 'd', False),)), Table(schema='d'), View(schema='d'), Function(schema='d')}

@pytest.mark.parametrize('sql', ['select a.x, b.y from abc a join bcd b on ', 'select a.x, b.y\nfrom abc a\njoin bcd b on\n', 'select a.x, b.y\nfrom abc a\njoin bcd b\non ', 'select a.x, b.y from abc a join bcd b on a.id = b.id OR '])
def test_on_suggests_aliases_and_join_conditions(sql):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(sql, sql)
    tables = ((None, 'abc', 'a', False), (None, 'bcd', 'b', False))
    assert set(suggestions) == {JoinCondition(table_refs=tables, parent=None), Alias(aliases=('a', 'b'))}

@pytest.mark.parametrize('sql', ['select abc.x, bcd.y from abc join bcd on abc.id = bcd.id AND ', 'select abc.x, bcd.y from abc join bcd on '])
def test_on_suggests_tables_and_join_conditions(sql):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(sql, sql)
    tables = ((None, 'abc', None, False), (None, 'bcd', None, False))
    assert set(suggestions) == {JoinCondition(table_refs=tables, parent=None), Alias(aliases=('abc', 'bcd'))}

@pytest.mark.parametrize('sql', ['select a.x, b.y from abc a join bcd b on a.id = ', 'select a.x, b.y from abc a join bcd b on a.id = b.id AND a.id2 = '])
def test_on_suggests_aliases_right_side(sql):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(sql, sql)
    assert suggestions == (Alias(aliases=('a', 'b')),)

@pytest.mark.parametrize('sql', ['select abc.x, bcd.y from abc join bcd on abc.id = bcd.id and ', 'select abc.x, bcd.y from abc join bcd on '])
def test_on_suggests_tables_and_join_conditions_right_side(sql):
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type(sql, sql)
    tables = ((None, 'abc', None, False), (None, 'bcd', None, False))
    assert set(suggestions) == {JoinCondition(table_refs=tables, parent=None), Alias(aliases=('abc', 'bcd'))}

@pytest.mark.parametrize('text', ('select * from abc inner join def using (', 'select * from abc inner join def using (col1, ', 'insert into hij select * from abc inner join def using (', 'insert into hij(x, y, z)\n    select * from abc inner join def using (col1, ', 'insert into hij (a,b,c)\n    select * from abc inner join def using (col1, '))
def test_join_using_suggests_common_columns(text):
    if False:
        i = 10
        return i + 15
    tables = ((None, 'abc', None, False), (None, 'def', None, False))
    assert set(suggest_type(text, text)) == {Column(table_refs=tables, require_last_table=True)}

def test_suggest_columns_after_multiple_joins():
    if False:
        i = 10
        return i + 15
    sql = 'select * from t1\n            inner join t2 ON\n              t1.id = t2.t1_id\n            inner join t3 ON\n              t2.id = t3.'
    suggestions = suggest_type(sql, sql)
    assert Column(table_refs=((None, 't3', None, False),)) in set(suggestions)

def test_2_statements_2nd_current():
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type('select * from a; select * from ', 'select * from a; select * from ')
    assert set(suggestions) == {FromClauseItem(schema=None), Schema()}
    suggestions = suggest_type('select * from a; select  from b', 'select * from a; select ')
    assert set(suggestions) == {Column(table_refs=((None, 'b', None, False),), qualifiable=True), Function(schema=None), Keyword('SELECT')}
    suggestions = suggest_type('select * from; select * from ', 'select * from; select * from ')
    assert set(suggestions) == {FromClauseItem(schema=None), Schema()}

def test_2_statements_1st_current():
    if False:
        print('Hello World!')
    suggestions = suggest_type('select * from ; select * from b', 'select * from ')
    assert set(suggestions) == {FromClauseItem(schema=None), Schema()}
    suggestions = suggest_type('select  from a; select * from b', 'select ')
    assert set(suggestions) == cols_etc('a', last_keyword='SELECT')

def test_3_statements_2nd_current():
    if False:
        print('Hello World!')
    suggestions = suggest_type('select * from a; select * from ; select * from c', 'select * from a; select * from ')
    assert set(suggestions) == {FromClauseItem(schema=None), Schema()}
    suggestions = suggest_type('select * from a; select  from b; select * from c', 'select * from a; select ')
    assert set(suggestions) == cols_etc('b', last_keyword='SELECT')

@pytest.mark.parametrize('text', ['\nCREATE OR REPLACE FUNCTION func() RETURNS setof int AS $$\nSELECT  FROM foo;\nSELECT 2 FROM bar;\n$$ language sql;\n    ', 'create function func2(int, varchar)\nRETURNS text\nlanguage sql AS\n$func$\nSELECT 2 FROM bar;\nSELECT  FROM foo;\n$func$\n    ', '\nCREATE OR REPLACE FUNCTION func() RETURNS setof int AS $func$\nSELECT 3 FROM foo;\nSELECT 2 FROM bar;\n$$ language sql;\ncreate function func2(int, varchar)\nRETURNS text\nlanguage sql AS\n$func$\nSELECT 2 FROM bar;\nSELECT  FROM foo;\n$func$\n    ', '\nSELECT * FROM baz;\nCREATE OR REPLACE FUNCTION func() RETURNS setof int AS $func$\nSELECT  FROM foo;\nSELECT 2 FROM bar;\n$$ language sql;\ncreate function func2(int, varchar)\nRETURNS text\nlanguage sql AS\n$func$\nSELECT 3 FROM bar;\nSELECT  FROM foo;\n$func$\nSELECT * FROM qux;\n    '])
def test_statements_in_function_body(text):
    if False:
        while True:
            i = 10
    suggestions = suggest_type(text, text[:text.find('  ') + 1])
    assert set(suggestions) == {Column(table_refs=((None, 'foo', None, False),), qualifiable=True), Function(schema=None), Keyword('SELECT')}
functions = ['\nCREATE OR REPLACE FUNCTION func() RETURNS setof int AS $$\nSELECT 1 FROM foo;\nSELECT 2 FROM bar;\n$$ language sql;\n    ', "\ncreate function func2(int, varchar)\nRETURNS text\nlanguage sql AS\n'\nSELECT 2 FROM bar;\nSELECT 1 FROM foo;\n';\n    "]

@pytest.mark.parametrize('text', functions)
def test_statements_with_cursor_after_function_body(text):
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type(text, text[:text.find('; ') + 1])
    assert set(suggestions) == {Keyword(), Special()}

@pytest.mark.parametrize('text', functions)
def test_statements_with_cursor_before_function_body(text):
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type(text, '')
    assert set(suggestions) == {Keyword(), Special()}

def test_create_db_with_template():
    if False:
        while True:
            i = 10
    suggestions = suggest_type('create database foo with template ', 'create database foo with template ')
    assert set(suggestions) == {Database()}

@pytest.mark.parametrize('initial_text', ('', '    ', '\t \t', '\n'))
def test_specials_included_for_initial_completion(initial_text):
    if False:
        print('Hello World!')
    suggestions = suggest_type(initial_text, initial_text)
    assert set(suggestions) == {Keyword(), Special()}

def test_drop_schema_qualified_table_suggests_only_tables():
    if False:
        for i in range(10):
            print('nop')
    text = 'DROP TABLE schema_name.table_name'
    suggestions = suggest_type(text, text)
    assert suggestions == (Table(schema='schema_name'),)

@pytest.mark.parametrize('text', (',', '  ,', 'sel ,'))
def test_handle_pre_completion_comma_gracefully(text):
    if False:
        i = 10
        return i + 15
    suggestions = suggest_type(text, text)
    assert iter(suggestions)

def test_drop_schema_suggests_schemas():
    if False:
        for i in range(10):
            print('nop')
    sql = 'DROP SCHEMA '
    assert suggest_type(sql, sql) == (Schema(),)

@pytest.mark.parametrize('text', ['SELECT x::', 'SELECT x::y', 'SELECT (x + y)::'])
def test_cast_operator_suggests_types(text):
    if False:
        print('Hello World!')
    assert set(suggest_type(text, text)) == {Datatype(schema=None), Table(schema=None), Schema()}

@pytest.mark.parametrize('text', ['SELECT foo::bar.', 'SELECT foo::bar.baz', 'SELECT (x + y)::bar.'])
def test_cast_operator_suggests_schema_qualified_types(text):
    if False:
        for i in range(10):
            print('nop')
    assert set(suggest_type(text, text)) == {Datatype(schema='bar'), Table(schema='bar')}

def test_alter_column_type_suggests_types():
    if False:
        print('Hello World!')
    q = 'ALTER TABLE foo ALTER COLUMN bar TYPE '
    assert set(suggest_type(q, q)) == {Datatype(schema=None), Table(schema=None), Schema()}

@pytest.mark.parametrize('text', ['CREATE TABLE foo (bar ', 'CREATE TABLE foo (bar DOU', 'CREATE TABLE foo (bar INT, baz ', 'CREATE TABLE foo (bar INT, baz TEXT, qux ', 'CREATE FUNCTION foo (bar ', 'CREATE FUNCTION foo (bar INT, baz ', 'SELECT * FROM foo() AS bar (baz ', 'SELECT * FROM foo() AS bar (baz INT, qux ', 'CREATE TABLE foo (dt d'])
def test_identifier_suggests_types_in_parentheses(text):
    if False:
        while True:
            i = 10
    assert set(suggest_type(text, text)) == {Datatype(schema=None), Table(schema=None), Schema()}

@pytest.mark.parametrize('text', ['SELECT foo ', 'SELECT foo FROM bar ', 'SELECT foo AS bar ', 'SELECT foo bar ', 'SELECT * FROM foo AS bar ', 'SELECT * FROM foo bar ', 'SELECT foo FROM (SELECT bar '])
def test_alias_suggests_keywords(text):
    if False:
        print('Hello World!')
    suggestions = suggest_type(text, text)
    assert suggestions == (Keyword(),)

def test_invalid_sql():
    if False:
        print('Hello World!')
    text = 'selt *'
    suggestions = suggest_type(text, text)
    assert suggestions == (Keyword(),)

@pytest.mark.parametrize('text', ['SELECT * FROM foo where created > now() - ', 'select * from foo where bar '])
def test_suggest_where_keyword(text):
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type(text, text)
    assert set(suggestions) == cols_etc('foo', last_keyword='WHERE')

@pytest.mark.parametrize('text, before, expected', [('\\ns abc SELECT ', 'SELECT ', [Column(table_refs=(), qualifiable=True), Function(schema=None), Keyword('SELECT')]), ('\\ns abc SELECT foo ', 'SELECT foo ', (Keyword(),)), ('\\ns abc SELECT t1. FROM tabl1 t1', 'SELECT t1.', [Table(schema='t1'), View(schema='t1'), Column(table_refs=((None, 'tabl1', 't1', False),)), Function(schema='t1')])])
def test_named_query_completion(text, before, expected):
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type(text, before)
    assert set(expected) == set(suggestions)

def test_select_suggests_fields_from_function():
    if False:
        while True:
            i = 10
    suggestions = suggest_type('SELECT  FROM func()', 'SELECT ')
    assert set(suggestions) == cols_etc('func', is_function=True, last_keyword='SELECT')

@pytest.mark.parametrize('sql', ['('])
def test_leading_parenthesis(sql):
    if False:
        i = 10
        return i + 15
    suggest_type(sql, sql)

@pytest.mark.parametrize('sql', ['select * from "', 'select * from "foo'])
def test_ignore_leading_double_quotes(sql):
    if False:
        for i in range(10):
            print('nop')
    suggestions = suggest_type(sql, sql)
    assert FromClauseItem(schema=None) in set(suggestions)

@pytest.mark.parametrize('sql', ['ALTER TABLE foo ALTER COLUMN ', 'ALTER TABLE foo ALTER COLUMN bar', 'ALTER TABLE foo DROP COLUMN ', 'ALTER TABLE foo DROP COLUMN bar'])
def test_column_keyword_suggests_columns(sql):
    if False:
        return 10
    suggestions = suggest_type(sql, sql)
    assert set(suggestions) == {Column(table_refs=((None, 'foo', None, False),))}

def test_handle_unrecognized_kw_generously():
    if False:
        for i in range(10):
            print('nop')
    sql = 'SELECT * FROM sessions WHERE session = 1 AND '
    suggestions = suggest_type(sql, sql)
    expected = Column(table_refs=((None, 'sessions', None, False),), qualifiable=True)
    assert expected in set(suggestions)

@pytest.mark.parametrize('sql', ['ALTER ', 'ALTER TABLE foo ALTER '])
def test_keyword_after_alter(sql):
    if False:
        i = 10
        return i + 15
    assert Keyword('ALTER') in set(suggest_type(sql, sql))