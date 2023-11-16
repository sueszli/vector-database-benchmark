from typing import Optional
import pytest
import sqlparse
from pytest_mock import MockerFixture
from sqlalchemy import text
from sqlparse.sql import Identifier, Token, TokenList
from sqlparse.tokens import Name
from superset.exceptions import QueryClauseValidationException
from superset.sql_parse import add_table_name, extract_table_references, get_rls_for_table, has_table_query, insert_rls_as_subquery, insert_rls_in_predicate, ParsedQuery, sanitize_clause, strip_comments_from_sql, Table

def extract_tables(query: str) -> set[Table]:
    if False:
        return 10
    '\n    Helper function to extract tables referenced in a query.\n    '
    return ParsedQuery(query).tables

def test_table() -> None:
    if False:
        print('Hello World!')
    '\n    Test the ``Table`` class and its string conversion.\n\n    Special characters in the table, schema, or catalog name should be escaped correctly.\n    '
    assert str(Table('tbname')) == 'tbname'
    assert str(Table('tbname', 'schemaname')) == 'schemaname.tbname'
    assert str(Table('tbname', 'schemaname', 'catalogname')) == 'catalogname.schemaname.tbname'
    assert str(Table('table.name', 'schema/name', 'catalog\nname')) == 'catalog%0Aname.schema%2Fname.table%2Ename'

def test_extract_tables() -> None:
    if False:
        return 10
    '\n    Test that referenced tables are parsed correctly from the SQL.\n    '
    assert extract_tables('SELECT * FROM tbname') == {Table('tbname')}
    assert extract_tables('SELECT * FROM tbname foo') == {Table('tbname')}
    assert extract_tables('SELECT * FROM tbname AS foo') == {Table('tbname')}
    assert extract_tables('SELECT * FROM tb_name') == {Table('tb_name')}
    assert extract_tables('SELECT * FROM "tbname"') == {Table('tbname')}
    assert extract_tables('SELECT * FROM "tb_name" WHERE city = "Lübeck"') == {Table('tb_name')}
    assert extract_tables('SELECT field1, field2 FROM tb_name') == {Table('tb_name')}
    assert extract_tables('SELECT t1.f1, t2.f2 FROM t1, t2') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT a.date, a.field FROM left_table a LIMIT 10') == {Table('left_table')}
    assert extract_tables('FROM t1 SELECT field') == {Table('t1')}

def test_extract_tables_subselect() -> None:
    if False:
        print('Hello World!')
    '\n    Test that tables inside subselects are parsed correctly.\n    '
    assert extract_tables("\nSELECT sub.*\nFROM (\n    SELECT *\n        FROM s1.t1\n        WHERE day_of_week = 'Friday'\n    ) sub, s2.t2\nWHERE sub.resolution = 'NONE'\n") == {Table('t1', 's1'), Table('t2', 's2')}
    assert extract_tables("\nSELECT sub.*\nFROM (\n    SELECT *\n    FROM s1.t1\n    WHERE day_of_week = 'Friday'\n) sub\nWHERE sub.resolution = 'NONE'\n") == {Table('t1', 's1')}
    assert extract_tables('\nSELECT * FROM t1\nWHERE s11 > ANY (\n    SELECT COUNT(*) /* no hint */ FROM t2\n    WHERE NOT EXISTS (\n        SELECT * FROM t3\n        WHERE ROW(5*t2.s1,77)=(\n            SELECT 50,11*s1 FROM t4\n        )\n    )\n)\n') == {Table('t1'), Table('t2'), Table('t3'), Table('t4')}

def test_extract_tables_select_in_expression() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test that parser works with ``SELECT``s used as expressions.\n    '
    assert extract_tables('SELECT f1, (SELECT count(1) FROM t2) FROM t1') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT f1, (SELECT count(1) FROM t2) as f2 FROM t1') == {Table('t1'), Table('t2')}

def test_extract_tables_parenthesis() -> None:
    if False:
        print('Hello World!')
    '\n    Test that parenthesis are parsed correctly.\n    '
    assert extract_tables('SELECT f1, (x + y) AS f2 FROM t1') == {Table('t1')}

def test_extract_tables_with_schema() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that schemas are parsed correctly.\n    '
    assert extract_tables('SELECT * FROM schemaname.tbname') == {Table('tbname', 'schemaname')}
    assert extract_tables('SELECT * FROM "schemaname"."tbname"') == {Table('tbname', 'schemaname')}
    assert extract_tables('SELECT * FROM "schemaname"."tbname" foo') == {Table('tbname', 'schemaname')}
    assert extract_tables('SELECT * FROM "schemaname"."tbname" AS foo') == {Table('tbname', 'schemaname')}

def test_extract_tables_union() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test that ``UNION`` queries work as expected.\n    '
    assert extract_tables('SELECT * FROM t1 UNION SELECT * FROM t2') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT * FROM t1 UNION ALL SELECT * FROM t2') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT * FROM t1 INTERSECT ALL SELECT * FROM t2') == {Table('t1'), Table('t2')}

def test_extract_tables_select_from_values() -> None:
    if False:
        print('Hello World!')
    '\n    Test that selecting from values returns no tables.\n    '
    assert extract_tables('SELECT * FROM VALUES (13, 42)') == set()

def test_extract_tables_select_array() -> None:
    if False:
        return 10
    '\n    Test that queries selecting arrays work as expected.\n    '
    assert extract_tables('\nSELECT ARRAY[1, 2, 3] AS my_array\nFROM t1 LIMIT 10\n') == {Table('t1')}

def test_extract_tables_select_if() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test that queries with an ``IF`` work as expected.\n    '
    assert extract_tables('\nSELECT IF(CARDINALITY(my_array) >= 3, my_array[3], NULL)\nFROM t1 LIMIT 10\n') == {Table('t1')}

def test_extract_tables_with_catalog() -> None:
    if False:
        return 10
    '\n    Test that catalogs are parsed correctly.\n    '
    assert extract_tables('SELECT * FROM catalogname.schemaname.tbname') == {Table('tbname', 'schemaname', 'catalogname')}

def test_extract_tables_illdefined() -> None:
    if False:
        while True:
            i = 10
    '\n    Test that ill-defined tables return an empty set.\n    '
    assert extract_tables('SELECT * FROM schemaname.') == set()
    assert extract_tables('SELECT * FROM catalogname.schemaname.') == set()
    assert extract_tables('SELECT * FROM catalogname..') == set()
    assert extract_tables('SELECT * FROM catalogname..tbname') == set()

def test_extract_tables_show_tables_from() -> None:
    if False:
        print('Hello World!')
    '\n    Test ``SHOW TABLES FROM``.\n    '
    assert extract_tables("SHOW TABLES FROM s1 like '%order%'") == set()

def test_extract_tables_show_columns_from() -> None:
    if False:
        print('Hello World!')
    '\n    Test ``SHOW COLUMNS FROM``.\n    '
    assert extract_tables('SHOW COLUMNS FROM t1') == {Table('t1')}

def test_extract_tables_where_subquery() -> None:
    if False:
        print('Hello World!')
    '\n    Test that tables in a ``WHERE`` subquery are parsed correctly.\n    '
    assert extract_tables('\nSELECT name\nFROM t1\nWHERE regionkey = (SELECT max(regionkey) FROM t2)\n') == {Table('t1'), Table('t2')}
    assert extract_tables('\nSELECT name\nFROM t1\nWHERE regionkey IN (SELECT regionkey FROM t2)\n') == {Table('t1'), Table('t2')}
    assert extract_tables('\nSELECT name\nFROM t1\nWHERE regionkey EXISTS (SELECT regionkey FROM t2)\n') == {Table('t1'), Table('t2')}

def test_extract_tables_describe() -> None:
    if False:
        return 10
    '\n    Test ``DESCRIBE``.\n    '
    assert extract_tables('DESCRIBE t1') == {Table('t1')}

def test_extract_tables_show_partitions() -> None:
    if False:
        print('Hello World!')
    '\n    Test ``SHOW PARTITIONS``.\n    '
    assert extract_tables("\nSHOW PARTITIONS FROM orders\nWHERE ds >= '2013-01-01' ORDER BY ds DESC\n") == {Table('orders')}

def test_extract_tables_join() -> None:
    if False:
        print('Hello World!')
    '\n    Test joins.\n    '
    assert extract_tables('SELECT t1.*, t2.* FROM t1 JOIN t2 ON t1.a = t2.a;') == {Table('t1'), Table('t2')}
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nJOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n    FROM right_table\n) b\nON a.date = b.date\n') == {Table('left_table'), Table('right_table')}
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nLEFT INNER JOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n    FROM right_table\n) b\nON a.date = b.date\n') == {Table('left_table'), Table('right_table')}
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nRIGHT OUTER JOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n    FROM right_table\n) b\nON a.date = b.date\n') == {Table('left_table'), Table('right_table')}
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nFULL OUTER JOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n        FROM right_table\n) b\nON a.date = b.date\n') == {Table('left_table'), Table('right_table')}

def test_extract_tables_semi_join() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test ``LEFT SEMI JOIN``.\n    '
    assert extract_tables('\nSELECT a.date, b.name\nFROM left_table a\nLEFT SEMI JOIN (\n    SELECT\n        CAST((b.year) as VARCHAR) date,\n        name\n    FROM right_table\n) b\nON a.data = b.date\n') == {Table('left_table'), Table('right_table')}

def test_extract_tables_combinations() -> None:
    if False:
        return 10
    '\n    Test a complex case with nested queries.\n    '
    assert extract_tables('\nSELECT * FROM t1\nWHERE s11 > ANY (\n    SELECT * FROM t1 UNION ALL SELECT * FROM (\n        SELECT t6.*, t3.* FROM t6 JOIN t3 ON t6.a = t3.a\n    ) tmp_join\n    WHERE NOT EXISTS (\n        SELECT * FROM t3\n        WHERE ROW(5*t3.s1,77)=(\n            SELECT 50,11*s1 FROM t4\n        )\n    )\n)\n') == {Table('t1'), Table('t3'), Table('t4'), Table('t6')}
    assert extract_tables('\nSELECT * FROM (\n    SELECT * FROM (\n        SELECT * FROM (\n            SELECT * FROM EmployeeS\n        ) AS S1\n    ) AS S2\n) AS S3\n') == {Table('EmployeeS')}

def test_extract_tables_with() -> None:
    if False:
        return 10
    '\n    Test ``WITH``.\n    '
    assert extract_tables('\nWITH\n    x AS (SELECT a FROM t1),\n    y AS (SELECT a AS b FROM t2),\n    z AS (SELECT b AS c FROM t3)\nSELECT c FROM z\n') == {Table('t1'), Table('t2'), Table('t3')}
    assert extract_tables('\nWITH\n    x AS (SELECT a FROM t1),\n    y AS (SELECT a AS b FROM x),\n    z AS (SELECT b AS c FROM y)\nSELECT c FROM z\n') == {Table('t1')}

def test_extract_tables_reusing_aliases() -> None:
    if False:
        print('Hello World!')
    '\n    Test that the parser follows aliases.\n    '
    assert extract_tables("\nwith q1 as ( select key from q2 where key = '5'),\nq2 as ( select key from src where key = '5')\nselect * from (select key from q1) a\n") == {Table('src')}

def test_extract_tables_multistatement() -> None:
    if False:
        print('Hello World!')
    '\n    Test that the parser works with multiple statements.\n    '
    assert extract_tables('SELECT * FROM t1; SELECT * FROM t2') == {Table('t1'), Table('t2')}
    assert extract_tables('SELECT * FROM t1; SELECT * FROM t2;') == {Table('t1'), Table('t2')}

def test_extract_tables_complex() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test a few complex queries.\n    '
    assert extract_tables('\nSELECT sum(m_examples) AS "sum__m_example"\nFROM (\n    SELECT\n        COUNT(DISTINCT id_userid) AS m_examples,\n        some_more_info\n    FROM my_b_table b\n    JOIN my_t_table t ON b.ds=t.ds\n    JOIN my_l_table l ON b.uid=l.uid\n    WHERE\n        b.rid IN (\n            SELECT other_col\n            FROM inner_table\n        )\n        AND l.bla IN (\'x\', \'y\')\n    GROUP BY 2\n    ORDER BY 2 ASC\n) AS "meh"\nORDER BY "sum__m_example" DESC\nLIMIT 10;\n') == {Table('my_l_table'), Table('my_b_table'), Table('my_t_table'), Table('inner_table')}
    assert extract_tables('\nSELECT *\nFROM table_a AS a, table_b AS b, table_c as c\nWHERE a.id = b.id and b.id = c.id\n') == {Table('table_a'), Table('table_b'), Table('table_c')}
    assert extract_tables('\nSELECT somecol AS somecol\nFROM (\n    WITH bla AS (\n        SELECT col_a\n        FROM a\n        WHERE\n            1=1\n            AND column_of_choice NOT IN (\n                SELECT interesting_col\n                FROM b\n            )\n    ),\n    rb AS (\n        SELECT yet_another_column\n        FROM (\n            SELECT a\n            FROM c\n            GROUP BY the_other_col\n        ) not_table\n        LEFT JOIN bla foo\n        ON foo.prop = not_table.bad_col0\n        WHERE 1=1\n        GROUP BY\n            not_table.bad_col1 ,\n            not_table.bad_col2 ,\n        ORDER BY not_table.bad_col_3 DESC ,\n            not_table.bad_col4 ,\n            not_table.bad_col5\n    )\n    SELECT random_col\n    FROM d\n    WHERE 1=1\n    UNION ALL SELECT even_more_cols\n    FROM e\n    WHERE 1=1\n    UNION ALL SELECT lets_go_deeper\n    FROM f\n    WHERE 1=1\n    WHERE 2=2\n    GROUP BY last_col\n    LIMIT 50000\n)\n') == {Table('a'), Table('b'), Table('c'), Table('d'), Table('e'), Table('f')}

def test_extract_tables_mixed_from_clause() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test that the parser handles a ``FROM`` clause with table and subselect.\n    '
    assert extract_tables('\nSELECT *\nFROM table_a AS a, (select * from table_b) AS b, table_c as c\nWHERE a.id = b.id and b.id = c.id\n') == {Table('table_a'), Table('table_b'), Table('table_c')}

def test_extract_tables_nested_select() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test that the parser handles selects inside functions.\n    '
    assert extract_tables('\nselect (extractvalue(1,concat(0x7e,(select GROUP_CONCAT(TABLE_NAME)\nfrom INFORMATION_SCHEMA.COLUMNS\nWHERE TABLE_SCHEMA like "%bi%"),0x7e)));\n') == {Table('COLUMNS', 'INFORMATION_SCHEMA')}
    assert extract_tables('\nselect (extractvalue(1,concat(0x7e,(select GROUP_CONCAT(COLUMN_NAME)\nfrom INFORMATION_SCHEMA.COLUMNS\nWHERE TABLE_NAME="bi_achievement_daily"),0x7e)));\n') == {Table('COLUMNS', 'INFORMATION_SCHEMA')}

def test_extract_tables_complex_cte_with_prefix() -> None:
    if False:
        return 10
    '\n    Test that the parser handles CTEs with prefixes.\n    '
    assert extract_tables('\nWITH CTE__test (SalesPersonID, SalesOrderID, SalesYear)\nAS (\n    SELECT SalesPersonID, SalesOrderID, YEAR(OrderDate) AS SalesYear\n    FROM SalesOrderHeader\n    WHERE SalesPersonID IS NOT NULL\n)\nSELECT SalesPersonID, COUNT(SalesOrderID) AS TotalSales, SalesYear\nFROM CTE__test\nGROUP BY SalesYear, SalesPersonID\nORDER BY SalesPersonID, SalesYear;\n') == {Table('SalesOrderHeader')}

def test_extract_tables_identifier_list_with_keyword_as_alias() -> None:
    if False:
        print('Hello World!')
    '\n    Test that aliases that are keywords are parsed correctly.\n    '
    assert extract_tables('\nWITH\n    f AS (SELECT * FROM foo),\n    match AS (SELECT * FROM f)\nSELECT * FROM match\n') == {Table('foo')}

def test_update() -> None:
    if False:
        while True:
            i = 10
    '\n    Test that ``UPDATE`` is not detected as ``SELECT``.\n    '
    assert ParsedQuery('UPDATE t1 SET col1 = NULL').is_select() is False

def test_set() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that ``SET`` is detected correctly.\n    '
    query = ParsedQuery("\n-- comment\nSET hivevar:desc='Legislators';\n")
    assert query.is_set() is True
    assert query.is_select() is False
    assert ParsedQuery("set hivevar:desc='bla'").is_set() is True
    assert ParsedQuery('SELECT 1').is_set() is False

def test_show() -> None:
    if False:
        print('Hello World!')
    '\n    Test that ``SHOW`` is detected correctly.\n    '
    query = ParsedQuery('\n-- comment\nSHOW LOCKS test EXTENDED;\n-- comment\n')
    assert query.is_show() is True
    assert query.is_select() is False
    assert ParsedQuery('SHOW TABLES').is_show() is True
    assert ParsedQuery('shOw TABLES').is_show() is True
    assert ParsedQuery('show TABLES').is_show() is True
    assert ParsedQuery('SELECT 1').is_show() is False

def test_is_explain() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test that ``EXPLAIN`` is detected correctly.\n    '
    assert ParsedQuery('EXPLAIN SELECT 1').is_explain() is True
    assert ParsedQuery('EXPLAIN SELECT 1').is_select() is False
    assert ParsedQuery('\n-- comment\nEXPLAIN select * from table\n-- comment 2\n').is_explain() is True
    assert ParsedQuery("\n-- comment\nEXPLAIN select * from table\nwhere col1 = 'something'\n-- comment 2\n\n-- comment 3\nEXPLAIN select * from table\nwhere col1 = 'something'\n-- comment 4\n").is_explain() is True
    assert ParsedQuery('\n-- This is a comment\n    -- this is another comment but with a space in the front\nEXPLAIN SELECT * FROM TABLE\n').is_explain() is True
    assert ParsedQuery('\n/* This is a comment\n     with stars instead */\nEXPLAIN SELECT * FROM TABLE\n').is_explain() is True
    assert ParsedQuery("\n-- comment\nselect * from table\nwhere col1 = 'something'\n-- comment 2\n").is_explain() is False

def test_is_valid_ctas() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if a query is a valid CTAS.\n\n    A valid CTAS has a ``SELECT`` as its last statement.\n    '
    assert ParsedQuery('SELECT * FROM table', strip_comments=True).is_valid_ctas() is True
    assert ParsedQuery('\n-- comment\nSELECT * FROM table\n-- comment 2\n', strip_comments=True).is_valid_ctas() is True
    assert ParsedQuery('\n-- comment\nSET @value = 42;\nSELECT @value as foo;\n-- comment 2\n', strip_comments=True).is_valid_ctas() is True
    assert ParsedQuery('\n-- comment\nEXPLAIN SELECT * FROM table\n-- comment 2\n', strip_comments=True).is_valid_ctas() is False
    assert ParsedQuery('\nSELECT * FROM table;\nINSERT INTO TABLE (foo) VALUES (42);\n', strip_comments=True).is_valid_ctas() is False

def test_is_valid_cvas() -> None:
    if False:
        while True:
            i = 10
    '\n    Test if a query is a valid CVAS.\n\n    A valid CVAS has a single ``SELECT`` statement.\n    '
    assert ParsedQuery('SELECT * FROM table', strip_comments=True).is_valid_cvas() is True
    assert ParsedQuery('\n-- comment\nSELECT * FROM table\n-- comment 2\n', strip_comments=True).is_valid_cvas() is True
    assert ParsedQuery('\n-- comment\nSET @value = 42;\nSELECT @value as foo;\n-- comment 2\n', strip_comments=True).is_valid_cvas() is False
    assert ParsedQuery('\n-- comment\nEXPLAIN SELECT * FROM table\n-- comment 2\n', strip_comments=True).is_valid_cvas() is False
    assert ParsedQuery('\nSELECT * FROM table;\nINSERT INTO TABLE (foo) VALUES (42);\n', strip_comments=True).is_valid_cvas() is False

def test_is_select_cte_with_comments() -> None:
    if False:
        while True:
            i = 10
    '\n    Some CTES with comments are not correctly identified as SELECTS.\n    '
    sql = ParsedQuery('WITH blah AS\n  (SELECT * FROM core_dev.manager_team),\n\nblah2 AS\n  (SELECT * FROM core_dev.manager_workspace)\n\nSELECT * FROM blah\nINNER JOIN blah2 ON blah2.team_id = blah.team_id')
    assert sql.is_select()
    sql = ParsedQuery('WITH blah AS\n/*blahblahbalh*/\n  (SELECT * FROM core_dev.manager_team),\n--blahblahbalh\n\nblah2 AS\n  (SELECT * FROM core_dev.manager_workspace)\n\nSELECT * FROM blah\nINNER JOIN blah2 ON blah2.team_id = blah.team_id')
    assert sql.is_select()

def test_cte_is_select() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Some CTEs are not correctly identified as SELECTS.\n    '
    sql = ParsedQuery('WITH foo AS(\nSELECT\n  FLOOR(__time TO WEEK) AS "week",\n  name,\n  COUNT(DISTINCT user_id) AS "unique_users"\nFROM "druid"."my_table"\nGROUP BY 1,2\n)\nSELECT\n  f.week,\n  f.name,\n  f.unique_users\nFROM foo f')
    assert sql.is_select()

def test_cte_is_select_lowercase() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Some CTEs with lowercase select are not correctly identified as SELECTS.\n    '
    sql = ParsedQuery('WITH foo AS(\nselect\n  FLOOR(__time TO WEEK) AS "week",\n  name,\n  COUNT(DISTINCT user_id) AS "unique_users"\nFROM "druid"."my_table"\nGROUP BY 1,2\n)\nselect\n  f.week,\n  f.name,\n  f.unique_users\nFROM foo f')
    assert sql.is_select()

def test_cte_insert_is_not_select() -> None:
    if False:
        return 10
    '\n    Some CTEs with lowercase select are not correctly identified as SELECTS.\n    '
    sql = ParsedQuery('WITH foo AS(\n        INSERT INTO foo (id) VALUES (1) RETURNING 1\n        ) select * FROM foo f')
    assert sql.is_select() is False

def test_cte_delete_is_not_select() -> None:
    if False:
        while True:
            i = 10
    '\n    Some CTEs with lowercase select are not correctly identified as SELECTS.\n    '
    sql = ParsedQuery('WITH foo AS(\n        DELETE FROM foo RETURNING *\n        ) select * FROM foo f')
    assert sql.is_select() is False

def test_cte_is_not_select_lowercase() -> None:
    if False:
        return 10
    '\n    Some CTEs with lowercase select are not correctly identified as SELECTS.\n    '
    sql = ParsedQuery('WITH foo AS(\n        insert into foo (id) values (1) RETURNING 1\n        ) select * FROM foo f')
    assert sql.is_select() is False

def test_cte_with_multiple_selects() -> None:
    if False:
        return 10
    sql = ParsedQuery('WITH a AS ( select * from foo1 ), b as (select * from foo2) SELECT * FROM a;')
    assert sql.is_select()

def test_cte_with_multiple_with_non_select() -> None:
    if False:
        for i in range(10):
            print('nop')
    sql = ParsedQuery('WITH a AS (\n        select * from foo1\n        ), b as (\n        update foo2 set id=2\n        ) SELECT * FROM a')
    assert sql.is_select() is False
    sql = ParsedQuery('WITH a AS (\n         update foo2 set name=2\n         ),\n        b as (\n        select * from foo1\n        ) SELECT * FROM a')
    assert sql.is_select() is False
    sql = ParsedQuery('WITH a AS (\n         update foo2 set name=2\n         ),\n        b as (\n        update foo1 set name=2\n        ) SELECT * FROM a')
    assert sql.is_select() is False
    sql = ParsedQuery('WITH a AS (\n        INSERT INTO foo (id) VALUES (1)\n        ),\n        b as (\n        select 1\n        ) SELECT * FROM a')
    assert sql.is_select() is False

def test_unknown_select() -> None:
    if False:
        print('Hello World!')
    '\n    Test that `is_select` works when sqlparse fails to identify the type.\n    '
    sql = 'WITH foo AS(SELECT 1) SELECT 1'
    assert sqlparse.parse(sql)[0].get_type() == 'SELECT'
    assert ParsedQuery(sql).is_select()
    sql = 'WITH foo AS(SELECT 1) INSERT INTO my_table (a) VALUES (1)'
    assert sqlparse.parse(sql)[0].get_type() == 'INSERT'
    assert not ParsedQuery(sql).is_select()
    sql = 'WITH foo AS(SELECT 1) DELETE FROM my_table'
    assert sqlparse.parse(sql)[0].get_type() == 'DELETE'
    assert not ParsedQuery(sql).is_select()

def test_get_query_with_new_limit_comment() -> None:
    if False:
        while True:
            i = 10
    '\n    Test that limit is applied correctly.\n    '
    query = ParsedQuery('SELECT * FROM birth_names -- SOME COMMENT')
    assert query.set_or_update_query_limit(1000) == 'SELECT * FROM birth_names -- SOME COMMENT\nLIMIT 1000'

def test_get_query_with_new_limit_comment_with_limit() -> None:
    if False:
        return 10
    '\n    Test that limits in comments are ignored.\n    '
    query = ParsedQuery('SELECT * FROM birth_names -- SOME COMMENT WITH LIMIT 555')
    assert query.set_or_update_query_limit(1000) == 'SELECT * FROM birth_names -- SOME COMMENT WITH LIMIT 555\nLIMIT 1000'

def test_get_query_with_new_limit_lower() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that lower limits are not replaced.\n    '
    query = ParsedQuery('SELECT * FROM birth_names LIMIT 555')
    assert query.set_or_update_query_limit(1000) == 'SELECT * FROM birth_names LIMIT 555'

def test_get_query_with_new_limit_upper() -> None:
    if False:
        print('Hello World!')
    '\n    Test that higher limits are replaced.\n    '
    query = ParsedQuery('SELECT * FROM birth_names LIMIT 2000')
    assert query.set_or_update_query_limit(1000) == 'SELECT * FROM birth_names LIMIT 1000'

def test_basic_breakdown_statements() -> None:
    if False:
        while True:
            i = 10
    '\n    Test that multiple statements are parsed correctly.\n    '
    query = ParsedQuery('\nSELECT * FROM birth_names;\nSELECT * FROM birth_names LIMIT 1;\n')
    assert query.get_statements() == ['SELECT * FROM birth_names', 'SELECT * FROM birth_names LIMIT 1']

def test_messy_breakdown_statements() -> None:
    if False:
        return 10
    '\n    Test the messy multiple statements are parsed correctly.\n    '
    query = ParsedQuery('\nSELECT 1;\t\n\n\n  \t\n\t\nSELECT 2;\nSELECT * FROM birth_names;;;\nSELECT * FROM birth_names LIMIT 1\n')
    assert query.get_statements() == ['SELECT 1', 'SELECT 2', 'SELECT * FROM birth_names', 'SELECT * FROM birth_names LIMIT 1']

def test_sqlparse_formatting():
    if False:
        while True:
            i = 10
    '\n    Test that ``from_unixtime`` is formatted correctly.\n    '
    assert sqlparse.format("SELECT extract(HOUR from from_unixtime(hour_ts) AT TIME ZONE 'America/Los_Angeles') from table", reindent=True) == "SELECT extract(HOUR\n               from from_unixtime(hour_ts) AT TIME ZONE 'America/Los_Angeles')\nfrom table"

def test_strip_comments_from_sql() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test that comments are stripped out correctly.\n    '
    assert strip_comments_from_sql('SELECT col1, col2 FROM table1') == 'SELECT col1, col2 FROM table1'
    assert strip_comments_from_sql('SELECT col1, col2 FROM table1\n-- comment') == 'SELECT col1, col2 FROM table1\n'
    assert strip_comments_from_sql("SELECT '--abc' as abc, col2 FROM table1\n") == "SELECT '--abc' as abc, col2 FROM table1"

def test_sanitize_clause_valid():
    if False:
        while True:
            i = 10
    assert sanitize_clause('col = 1') == 'col = 1'
    assert sanitize_clause('1=\t\n1') == '1=\t\n1'
    assert sanitize_clause('(col = 1)') == '(col = 1)'
    assert sanitize_clause('(col1 = 1) AND (col2 = 2)') == '(col1 = 1) AND (col2 = 2)'
    assert sanitize_clause("col = 'abc' -- comment") == "col = 'abc' -- comment\n"
    assert sanitize_clause("col = 'col1 = 1) AND (col2 = 2'") == "col = 'col1 = 1) AND (col2 = 2'"
    assert sanitize_clause("col = 'select 1; select 2'") == "col = 'select 1; select 2'"
    assert sanitize_clause("col = 'abc -- comment'") == "col = 'abc -- comment'"

def test_sanitize_clause_closing_unclosed():
    if False:
        return 10
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('col1 = 1) AND (col2 = 2)')

def test_sanitize_clause_unclosed():
    if False:
        print('Hello World!')
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('(col1 = 1) AND (col2 = 2')

def test_sanitize_clause_closing_and_unclosed():
    if False:
        return 10
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('col1 = 1) AND (col2 = 2')

def test_sanitize_clause_closing_and_unclosed_nested():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('(col1 = 1)) AND ((col2 = 2)')

def test_sanitize_clause_multiple():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(QueryClauseValidationException):
        sanitize_clause('TRUE; SELECT 1')

def test_sqlparse_issue_652():
    if False:
        while True:
            i = 10
    stmt = sqlparse.parse("foo = '\\' AND bar = 'baz'")[0]
    assert len(stmt.tokens) == 5
    assert str(stmt.tokens[0]) == "foo = '\\'"

@pytest.mark.parametrize('sql,expected', [('SELECT * FROM table', True), ('SELECT a FROM (SELECT 1 AS a) JOIN (SELECT * FROM table)', True), ('(SELECT COUNT(DISTINCT name) AS foo FROM    birth_names)', True), ('COUNT(*)', False), ('SELECT a FROM (SELECT 1 AS a)', False), ('SELECT a FROM (SELECT 1 AS a) JOIN table', True), ('SELECT * FROM (SELECT 1 AS foo, 2 AS bar) ORDER BY foo ASC, bar', False), ('SELECT * FROM other_table', True), ('extract(HOUR from from_unixtime(hour_ts)', False), ('(SELECT * FROM table)', True), ('(SELECT COUNT(DISTINCT name) from birth_names)', True), ("(SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%user%' LIMIT 1)", True), ("(SELECT table_name FROM /**/ information_schema.tables WHERE table_name LIKE '%user%' LIMIT 1)", True)])
def test_has_table_query(sql: str, expected: bool) -> None:
    if False:
        print('Hello World!')
    '\n    Test if a given statement queries a table.\n\n    This is used to prevent ad-hoc metrics from querying unauthorized tables, bypassing\n    row-level security.\n    '
    statement = sqlparse.parse(sql)[0]
    assert has_table_query(statement) == expected

@pytest.mark.parametrize('sql,table,rls,expected', [('SELECT * FROM some_table WHERE 1=1', 'some_table', 'id=42', 'SELECT * FROM (SELECT * FROM some_table WHERE some_table.id=42) AS some_table WHERE 1=1'), ('SELECT * FROM table WHERE 1=1', 'table', 'id=42', 'SELECT * FROM (SELECT * FROM table WHERE table.id=42) AS table WHERE 1=1'), ('SELECT * FROM table WHERE 1=1', 'other_table', 'id=42', 'SELECT * FROM table WHERE 1=1'), ('SELECT * FROM other_table WHERE 1=1', 'table', 'id=42', 'SELECT * FROM other_table WHERE 1=1'), ('SELECT * FROM table JOIN other_table ON table.id = other_table.id', 'other_table', 'id=42', 'SELECT * FROM table JOIN (SELECT * FROM other_table WHERE other_table.id=42) AS other_table ON table.id = other_table.id'), ('SELECT * FROM (SELECT * FROM other_table)', 'other_table', 'id=42', 'SELECT * FROM (SELECT * FROM (SELECT * FROM other_table WHERE other_table.id=42) AS other_table)'), ('SELECT * FROM table UNION ALL SELECT * FROM other_table', 'table', 'id=42', 'SELECT * FROM (SELECT * FROM table WHERE table.id=42) AS table UNION ALL SELECT * FROM other_table'), ('SELECT * FROM table UNION ALL SELECT * FROM other_table', 'other_table', 'id=42', 'SELECT * FROM table UNION ALL SELECT * FROM (SELECT * FROM other_table WHERE other_table.id=42) AS other_table'), ('SELECT * FROM schema.table_name', 'table_name', 'id=42', 'SELECT * FROM (SELECT * FROM schema.table_name WHERE table_name.id=42) AS table_name'), ('SELECT * FROM schema.table_name', 'schema.table_name', 'id=42', 'SELECT * FROM (SELECT * FROM schema.table_name WHERE schema.table_name.id=42) AS table_name'), ('SELECT * FROM table_name', 'schema.table_name', 'id=42', 'SELECT * FROM (SELECT * FROM table_name WHERE schema.table_name.id=42) AS table_name'), ('SELECT a.*, b.* FROM tbl_a AS a INNER JOIN tbl_b AS b ON a.col = b.col', 'tbl_a', 'id=42', 'SELECT a.*, b.* FROM (SELECT * FROM tbl_a WHERE tbl_a.id=42) AS a INNER JOIN tbl_b AS b ON a.col = b.col'), ('SELECT a.*, b.* FROM tbl_a a INNER JOIN tbl_b b ON a.col = b.col', 'tbl_a', 'id=42', 'SELECT a.*, b.* FROM (SELECT * FROM tbl_a WHERE tbl_a.id=42) AS a INNER JOIN tbl_b b ON a.col = b.col')])
def test_insert_rls_as_subquery(mocker: MockerFixture, sql: str, table: str, rls: str, expected: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Insert into a statement a given RLS condition associated with a table.\n    '
    condition = sqlparse.parse(rls)[0]
    add_table_name(condition, table)

    def get_rls_for_table(candidate: Token, database_id: int, default_schema: str) -> Optional[TokenList]:
        if False:
            print('Hello World!')
        '\n        Return the RLS ``condition`` if ``candidate`` matches ``table``.\n        '
        if not isinstance(candidate, Identifier):
            candidate = Identifier([Token(Name, candidate.value)])
        candidate_table = ParsedQuery.get_table(candidate)
        if not candidate_table:
            return None
        candidate_table_name = f'{candidate_table.schema}.{candidate_table.table}' if candidate_table.schema else candidate_table.table
        for (left, right) in zip(candidate_table_name.split('.')[::-1], table.split('.')[::-1]):
            if left != right:
                return None
        return condition
    mocker.patch('superset.sql_parse.get_rls_for_table', new=get_rls_for_table)
    statement = sqlparse.parse(sql)[0]
    assert str(insert_rls_as_subquery(token_list=statement, database_id=1, default_schema='my_schema')).strip() == expected.strip()

@pytest.mark.parametrize('sql,table,rls,expected', [('SELECT * FROM some_table WHERE 1=1', 'some_table', 'id=42', 'SELECT * FROM some_table WHERE ( 1=1) AND some_table.id=42'), ('SELECT * FROM some_table WHERE TRUE OR FALSE', 'some_table', '1=0', 'SELECT * FROM some_table WHERE ( TRUE OR FALSE) AND 1=0'), ('SELECT * FROM table WHERE 1=1', 'table', 'id=42', 'SELECT * FROM table WHERE ( 1=1) AND table.id=42'), ('SELECT * FROM table WHERE 1=1', 'other_table', 'id=42', 'SELECT * FROM table WHERE 1=1'), ('SELECT * FROM other_table WHERE 1=1', 'table', 'id=42', 'SELECT * FROM other_table WHERE 1=1'), ('SELECT * FROM table', 'table', 'id=42', 'SELECT * FROM table WHERE table.id=42'), ('SELECT * FROM some_table', 'some_table', 'id=42', 'SELECT * FROM some_table WHERE some_table.id=42'), ('SELECT * FROM table ORDER BY id', 'table', 'id=42', 'SELECT * FROM table  WHERE table.id=42 ORDER BY id'), ('SELECT * FROM some_table;', 'some_table', 'id=42', 'SELECT * FROM some_table WHERE some_table.id=42 ;'), ('SELECT * FROM some_table       ;', 'some_table', 'id=42', 'SELECT * FROM some_table        WHERE some_table.id=42 ;'), ('SELECT * FROM some_table       ', 'some_table', 'id=42', 'SELECT * FROM some_table        WHERE some_table.id=42'), ('SELECT * FROM table WHERE 1=1 AND table.id=42', 'table', 'id=42', 'SELECT * FROM table WHERE ( 1=1 AND table.id=42) AND table.id=42'), ('SELECT * FROM table JOIN other_table ON table.id = other_table.id AND other_table.id=42', 'other_table', 'id=42', 'SELECT * FROM table JOIN other_table ON other_table.id=42 AND ( table.id = other_table.id AND other_table.id=42 )'), ('SELECT * FROM table WHERE 1=1 AND id=42', 'table', 'id=42', 'SELECT * FROM table WHERE ( 1=1 AND id=42) AND table.id=42'), ('SELECT * FROM table JOIN other_table ON table.id = other_table.id', 'other_table', 'id=42', 'SELECT * FROM table JOIN other_table ON other_table.id=42 AND ( table.id = other_table.id )'), ('SELECT * FROM table JOIN other_table ON table.id = other_table.id WHERE 1=1', 'other_table', 'id=42', 'SELECT * FROM table JOIN other_table ON other_table.id=42 AND ( table.id = other_table.id  ) WHERE 1=1'), ('SELECT * FROM (SELECT * FROM other_table)', 'other_table', 'id=42', 'SELECT * FROM (SELECT * FROM other_table WHERE other_table.id=42 )'), ('SELECT * FROM table UNION ALL SELECT * FROM other_table', 'table', 'id=42', 'SELECT * FROM table  WHERE table.id=42 UNION ALL SELECT * FROM other_table'), ('SELECT * FROM table UNION ALL SELECT * FROM other_table', 'other_table', 'id=42', 'SELECT * FROM table UNION ALL SELECT * FROM other_table WHERE other_table.id=42'), ('SELECT * FROM schema.table_name', 'table_name', 'id=42', 'SELECT * FROM schema.table_name WHERE table_name.id=42'), ('SELECT * FROM schema.table_name', 'schema.table_name', 'id=42', 'SELECT * FROM schema.table_name WHERE schema.table_name.id=42'), ('SELECT * FROM table_name', 'schema.table_name', 'id=42', 'SELECT * FROM table_name WHERE schema.table_name.id=42')])
def test_insert_rls_in_predicate(mocker: MockerFixture, sql: str, table: str, rls: str, expected: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Insert into a statement a given RLS condition associated with a table.\n    '
    condition = sqlparse.parse(rls)[0]
    add_table_name(condition, table)

    def get_rls_for_table(candidate: Token, database_id: int, default_schema: str) -> Optional[TokenList]:
        if False:
            return 10
        '\n        Return the RLS ``condition`` if ``candidate`` matches ``table``.\n        '
        for (left, right) in zip(str(candidate).split('.')[::-1], table.split('.')[::-1]):
            if left != right:
                return None
        return condition
    mocker.patch('superset.sql_parse.get_rls_for_table', new=get_rls_for_table)
    statement = sqlparse.parse(sql)[0]
    assert str(insert_rls_in_predicate(token_list=statement, database_id=1, default_schema='my_schema')).strip() == expected.strip()

@pytest.mark.parametrize('rls,table,expected', [('id=42', 'users', 'users.id=42'), ('users.id=42', 'users', 'users.id=42'), ('schema.users.id=42', 'users', 'schema.users.id=42'), ('false', 'users', 'false')])
def test_add_table_name(rls: str, table: str, expected: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    condition = sqlparse.parse(rls)[0]
    add_table_name(condition, table)
    assert str(condition) == expected

def test_get_rls_for_table(mocker: MockerFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests for ``get_rls_for_table``.\n    '
    candidate = Identifier([Token(Name, 'some_table')])
    db = mocker.patch('superset.db')
    dataset = db.session.query().filter().one_or_none()
    dataset.__str__.return_value = 'some_table'
    dataset.get_sqla_row_level_filters.return_value = [text('organization_id = 1')]
    assert str(get_rls_for_table(candidate, 1, 'public')) == 'some_table.organization_id = 1'
    dataset.get_sqla_row_level_filters.return_value = [text('organization_id = 1'), text("foo = 'bar'")]
    assert str(get_rls_for_table(candidate, 1, 'public')) == "some_table.organization_id = 1 AND some_table.foo = 'bar'"
    dataset.get_sqla_row_level_filters.return_value = []
    assert get_rls_for_table(candidate, 1, 'public') is None

def test_extract_table_references(mocker: MockerFixture) -> None:
    if False:
        return 10
    '\n    Test the ``extract_table_references`` helper function.\n    '
    assert extract_table_references('SELECT 1', 'trino') == set()
    assert extract_table_references('SELECT 1 FROM some_table', 'trino') == {Table(table='some_table', schema=None, catalog=None)}
    assert extract_table_references('SELECT {{ jinja }} FROM some_table', 'trino') == {Table(table='some_table', schema=None, catalog=None)}
    assert extract_table_references('SELECT 1 FROM some_catalog.some_schema.some_table', 'trino') == {Table(table='some_table', schema='some_schema', catalog='some_catalog')}
    assert extract_table_references('SELECT 1 FROM `some_catalog`.`some_schema`.`some_table`', 'mysql') == {Table(table='some_table', schema='some_schema', catalog='some_catalog')}
    assert extract_table_references('SELECT 1 FROM "some_catalog".some_schema."some_table"', 'trino') == {Table(table='some_table', schema='some_schema', catalog='some_catalog')}
    assert extract_table_references('SELECT * FROM some_table JOIN other_table ON some_table.id = other_table.id', 'trino') == {Table(table='some_table', schema=None, catalog=None), Table(table='other_table', schema=None, catalog=None)}
    logger = mocker.patch('superset.sql_parse.logger')
    sql = 'SELECT * FROM table UNION ALL SELECT * FROM other_table'
    assert extract_table_references(sql, 'trino') == {Table(table='other_table', schema=None, catalog=None)}
    logger.warning.assert_called_once()
    logger = mocker.patch('superset.migrations.shared.utils.logger')
    sql = 'SELECT * FROM table UNION ALL SELECT * FROM other_table'
    assert extract_table_references(sql, 'trino', show_warning=False) == {Table(table='other_table', schema=None, catalog=None)}
    logger.warning.assert_not_called()

def test_is_select() -> None:
    if False:
        return 10
    '\n    Test `is_select`.\n    '
    assert not ParsedQuery('SELECT 1; DROP DATABASE superset').is_select()
    assert ParsedQuery('with base as(select id from table1 union all select id from table2) select * from base').is_select()
    assert ParsedQuery('\nWITH t AS (\n    SELECT 1 UNION ALL SELECT 2\n)\nSELECT * FROM t').is_select()