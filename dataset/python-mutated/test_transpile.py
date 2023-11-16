import os
import unittest
from unittest import mock
from sqlglot import parse_one, transpile
from sqlglot.errors import ErrorLevel, ParseError, UnsupportedError
from tests.helpers import assert_logger_contains, load_sql_fixture_pairs, load_sql_fixtures

class TestTranspile(unittest.TestCase):
    file_dir = os.path.dirname(__file__)
    fixtures_dir = os.path.join(file_dir, 'fixtures')
    maxDiff = None

    def validate(self, sql, target, **kwargs):
        if False:
            i = 10
            return i + 15
        self.assertEqual(transpile(sql, **kwargs)[0], target)

    def test_weird_chars(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(transpile('0Êß')[0], '0 AS Êß')

    def test_alias(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(transpile('SELECT SUM(y) KEEP')[0], 'SELECT SUM(y) AS KEEP')
        self.assertEqual(transpile('SELECT 1 overwrite')[0], 'SELECT 1 AS overwrite')
        self.assertEqual(transpile('SELECT 1 is')[0], 'SELECT 1 AS is')
        self.assertEqual(transpile('SELECT 1 current_time')[0], 'SELECT 1 AS current_time')
        self.assertEqual(transpile('SELECT 1 current_timestamp')[0], 'SELECT 1 AS current_timestamp')
        self.assertEqual(transpile('SELECT 1 current_date')[0], 'SELECT 1 AS current_date')
        self.assertEqual(transpile('SELECT 1 current_datetime')[0], 'SELECT 1 AS current_datetime')
        self.assertEqual(transpile('SELECT 1 row')[0], 'SELECT 1 AS row')
        self.assertEqual(transpile('SELECT 1 FROM a.b.table1 t UNPIVOT((c3) FOR c4 IN (a, b))')[0], 'SELECT 1 FROM a.b.table1 AS t UNPIVOT((c3) FOR c4 IN (a, b))')
        for key in ('union', 'over', 'from', 'join'):
            with self.subTest(f'alias {key}'):
                self.validate(f'SELECT x AS {key}', f'SELECT x AS {key}')
                self.validate(f'SELECT x "{key}"', f'SELECT x AS "{key}"')
                with self.assertRaises(ParseError):
                    self.validate(f'SELECT x {key}', '')

    def test_unary(self):
        if False:
            for i in range(10):
                print('nop')
        self.validate('+++1', '1')
        self.validate('+-1', '-1')
        self.validate('+- - -1', '- - -1')

    def test_paren(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ParseError):
            transpile('1 + (2 + 3')
            transpile('select f(')

    def test_some(self):
        if False:
            for i in range(10):
                print('nop')
        self.validate('SELECT * FROM x WHERE a = SOME (SELECT 1)', 'SELECT * FROM x WHERE a = ANY (SELECT 1)')

    def test_leading_comma(self):
        if False:
            i = 10
            return i + 15
        self.validate('SELECT FOO, BAR, BAZ', 'SELECT\n    FOO\n  , BAR\n  , BAZ', leading_comma=True, pretty=True)
        self.validate('SELECT FOO, /*x*/\nBAR, /*y*/\nBAZ', 'SELECT\n    FOO /* x */\n  , BAR /* y */\n  , BAZ', leading_comma=True, pretty=True)
        self.validate('SELECT FOO, BAR, BAZ', 'SELECT FOO, BAR, BAZ', leading_comma=True)

    def test_space(self):
        if False:
            while True:
                i = 10
        self.validate('SELECT MIN(3)>MIN(2)', 'SELECT MIN(3) > MIN(2)')
        self.validate('SELECT MIN(3)>=MIN(2)', 'SELECT MIN(3) >= MIN(2)')
        self.validate('SELECT 1>0', 'SELECT 1 > 0')
        self.validate('SELECT 3>=3', 'SELECT 3 >= 3')

    def test_comments(self):
        if False:
            print('Hello World!')
        self.validate('SELECT * FROM t1\n/*x*/\nUNION ALL SELECT * FROM t2', 'SELECT * FROM t1 /* x */ UNION ALL SELECT * FROM t2')
        self.validate('SELECT * FROM t1\n/*x*/\nINTERSECT ALL SELECT * FROM t2', 'SELECT * FROM t1 /* x */ INTERSECT ALL SELECT * FROM t2')
        self.validate('SELECT\n  foo\n/* comments */\n;', 'SELECT foo /* comments */')
        self.validate('SELECT * FROM a INNER /* comments */ JOIN b', 'SELECT * FROM a /* comments */ INNER JOIN b')
        self.validate('SELECT * FROM a LEFT /* comment 1 */ OUTER /* comment 2 */ JOIN b', 'SELECT * FROM a /* comment 1 */ /* comment 2 */ LEFT OUTER JOIN b')
        self.validate('SELECT CASE /* test */ WHEN a THEN b ELSE c END', 'SELECT CASE WHEN a THEN b ELSE c END /* test */')
        self.validate('SELECT 1 /*/2 */', 'SELECT 1 /* /2 */')
        self.validate('SELECT */*comment*/', 'SELECT * /* comment */')
        self.validate('SELECT * FROM table /*comment 1*/ /*comment 2*/', 'SELECT * FROM table /* comment 1 */ /* comment 2 */')
        self.validate('SELECT 1 FROM foo -- comment', 'SELECT 1 FROM foo /* comment */')
        self.validate('SELECT --+5\nx FROM foo', '/* +5 */ SELECT x FROM foo')
        self.validate('SELECT --!5\nx FROM foo', '/* !5 */ SELECT x FROM foo')
        self.validate('SELECT 1 /* inline */ FROM foo -- comment', 'SELECT 1 /* inline */ FROM foo /* comment */')
        self.validate('SELECT FUN(x) /*x*/, [1,2,3] /*y*/', 'SELECT FUN(x) /* x */, ARRAY(1, 2, 3) /* y */')
        self.validate('\n            SELECT 1 -- comment\n            FROM foo -- comment\n            ', 'SELECT 1 /* comment */ FROM foo /* comment */')
        self.validate('\n            SELECT 1 /* big comment\n             like this */\n            FROM foo -- comment\n            ', 'SELECT 1 /* big comment\n             like this */ FROM foo /* comment */')
        self.validate('select x from foo --       x', 'SELECT x FROM foo /*       x */')
        self.validate('select x, --\n            from foo', 'SELECT x FROM foo')
        self.validate('\n-- comment 1\n-- comment 2\n-- comment 3\nSELECT * FROM foo\n            ', '/* comment 1 */ /* comment 2 */ /* comment 3 */ SELECT * FROM foo')
        self.validate('\n-- comment 1\n-- comment 2\n-- comment 3\nSELECT * FROM foo', '/* comment 1 */ /* comment 2 */ /* comment 3 */\nSELECT\n  *\nFROM foo', pretty=True)
        self.validate('\nSELECT * FROM tbl /*line1\nline2\nline3*/ /*another comment*/ where 1=1 -- comment at the end', 'SELECT * FROM tbl /* line1\nline2\nline3 */ /* another comment */ WHERE 1 = 1 /* comment at the end */')
        self.validate('\nSELECT * FROM tbl /*line1\nline2\nline3*/ /*another comment*/ where 1=1 -- comment at the end', 'SELECT\n  *\nFROM tbl /* line1\nline2\nline3 */ /* another comment */\nWHERE\n  1 = 1 /* comment at the end */', pretty=True)
        self.validate('\n            /* multi\n               line\n               comment\n            */\n            SELECT\n              tbl.cola /* comment 1 */ + tbl.colb /* comment 2 */,\n              CAST(x AS CHAR), # comment 3\n              y               -- comment 4\n            FROM\n              bar /* comment 5 */,\n              tbl #          comment 6\n            ', '/* multi\n               line\n               comment\n            */\nSELECT\n  tbl.cola /* comment 1 */ + tbl.colb /* comment 2 */,\n  CAST(x AS CHAR), /* comment 3 */\n  y /* comment 4 */\nFROM bar /* comment 5 */, tbl /*          comment 6 */', read='mysql', pretty=True)
        self.validate('\n            SELECT a FROM b\n            WHERE foo\n            -- comment 1\n            AND bar\n            -- comment 2\n            AND bla\n            -- comment 3\n            LIMIT 10\n            ;\n            ', 'SELECT a FROM b WHERE foo AND /* comment 1 */ bar AND /* comment 2 */ bla LIMIT 10 /* comment 3 */')
        self.validate('\n            SELECT a FROM b WHERE foo\n            -- comment 1\n            ', 'SELECT a FROM b WHERE foo /* comment 1 */')
        self.validate('\n            select a\n            -- from\n            from b\n            -- where\n            where foo\n            -- comment 1\n            and bar\n            -- comment 2\n            and bla\n            ', 'SELECT\n  a\n/* from */\nFROM b\n/* where */\nWHERE\n  foo /* comment 1 */ AND bar AND bla /* comment 2 */', pretty=True)
        self.validate('\n            -- test\n            WITH v AS (\n              SELECT\n                1 AS literal\n            )\n            SELECT\n              *\n            FROM v\n            ', '/* test */\nWITH v AS (\n  SELECT\n    1 AS literal\n)\nSELECT\n  *\nFROM v', pretty=True)
        self.validate('(/* 1 */ 1 ) /* 2 */', '(1) /* 1 */ /* 2 */')
        self.validate('select * from t where not a in (23) /*test*/ and b in (14)', 'SELECT * FROM t WHERE NOT a IN (23) /* test */ AND b IN (14)')
        self.validate('select * from t where a in (23) /*test*/ and b in (14)', 'SELECT * FROM t WHERE a IN (23) /* test */ AND b IN (14)')
        self.validate('select * from t where ((condition = 1)/*test*/)', 'SELECT * FROM t WHERE ((condition = 1) /* test */)')
        self.validate('SELECT 1 // hi this is a comment', 'SELECT 1 /* hi this is a comment */', read='snowflake')
        self.validate('-- comment\nDROP TABLE IF EXISTS foo', '/* comment */ DROP TABLE IF EXISTS foo')
        self.validate('\n            -- comment1\n            -- comment2\n\n            -- comment3\n            DROP TABLE IF EXISTS db.tba\n            ', '/* comment1 */ /* comment2 */ /* comment3 */\nDROP TABLE IF EXISTS db.tba', pretty=True)
        self.validate('\n            -- comment4\n            CREATE TABLE db.tba AS\n            SELECT a, b, c\n            FROM tb_01\n            WHERE\n            -- comment5\n              a = 1 AND b = 2 --comment6\n              -- and c = 1\n            -- comment7\n            ;\n            ', '/* comment4 */\nCREATE TABLE db.tba AS\nSELECT\n  a,\n  b,\n  c\nFROM tb_01\nWHERE\n  a /* comment5 */ = 1 AND b = 2 /* comment6 */ /* and c = 1 */ /* comment7 */', pretty=True)
        self.validate('\n            SELECT\n               -- This is testing comments\n                col,\n            -- 2nd testing comments\n            CASE WHEN a THEN b ELSE c END as d\n            FROM t\n            ', 'SELECT\n  col, /* This is testing comments */\n  CASE WHEN a THEN b ELSE c END /* 2nd testing comments */ AS d\nFROM t', pretty=True)
        self.validate('\n            SELECT * FROM a\n            -- comments\n            INNER JOIN b\n            ', 'SELECT\n  *\nFROM a\n/* comments */\nINNER JOIN b', pretty=True)
        self.validate('SELECT * FROM a LEFT /* comment 1 */ OUTER /* comment 2 */ JOIN b', 'SELECT\n  *\nFROM a\n/* comment 1 */ /* comment 2 */\nLEFT OUTER JOIN b', pretty=True)
        self.validate('SELECT\n  a /* sqlglot.meta case_sensitive */ -- noqa\nFROM tbl', 'SELECT\n  a /* sqlglot.meta case_sensitive */ /* noqa */\nFROM tbl', pretty=True)
        self.validate("\nSELECT\n  'hotel1' AS hotel,\n  *\nFROM dw_1_dw_1_1.exactonline_1.transactionlines\n/*\n    UNION ALL\n    SELECT\n      'Thon Partner Hotel Jølster' AS hotel,\n      name,\n      date,\n      CAST(identifier AS VARCHAR) AS identifier,\n      value\n    FROM d2o_889_oupjr_1348.public.accountvalues_forecast\n*/\nUNION ALL\nSELECT\n  'hotel2' AS hotel,\n  *\nFROM dw_1_dw_1_1.exactonline_2.transactionlines", "SELECT\n  'hotel1' AS hotel,\n  *\nFROM dw_1_dw_1_1.exactonline_1.transactionlines /*\n    UNION ALL\n    SELECT\n      'Thon Partner Hotel Jølster' AS hotel,\n      name,\n      date,\n      CAST(identifier AS VARCHAR) AS identifier,\n      value\n    FROM d2o_889_oupjr_1348.public.accountvalues_forecast\n*/\nUNION ALL\nSELECT\n  'hotel2' AS hotel,\n  *\nFROM dw_1_dw_1_1.exactonline_2.transactionlines", pretty=True)

    def test_types(self):
        if False:
            i = 10
            return i + 15
        self.validate('INT 1', 'CAST(1 AS INT)')
        self.validate("VARCHAR 'x' y", "CAST('x' AS VARCHAR) AS y")
        self.validate("STRING 'x' y", "CAST('x' AS TEXT) AS y")
        self.validate('x::INT', 'CAST(x AS INT)')
        self.validate('x::INTEGER', 'CAST(x AS INT)')
        self.validate('x::INT y', 'CAST(x AS INT) AS y')
        self.validate('x::INT AS y', 'CAST(x AS INT) AS y')
        self.validate('x::INT::BOOLEAN', 'CAST(CAST(x AS INT) AS BOOLEAN)')
        self.validate('interval::int', 'CAST(interval AS INT)')
        self.validate('x::user_defined_type', 'CAST(x AS user_defined_type)')
        self.validate('CAST(x::INT AS BOOLEAN)', 'CAST(CAST(x AS INT) AS BOOLEAN)')
        self.validate('CAST(x AS INT)::BOOLEAN', 'CAST(CAST(x AS INT) AS BOOLEAN)')
        with self.assertRaises(ParseError):
            transpile('x::z', read='duckdb')

    def test_not_range(self):
        if False:
            while True:
                i = 10
        self.validate('a NOT LIKE b', 'NOT a LIKE b')
        self.validate('a NOT BETWEEN b AND c', 'NOT a BETWEEN b AND c')
        self.validate('a NOT IN (1, 2)', 'NOT a IN (1, 2)')
        self.validate('a IS NOT NULL', 'NOT a IS NULL')
        self.validate("a LIKE TEXT 'y'", "a LIKE CAST('y' AS TEXT)")

    def test_extract(self):
        if False:
            print('Hello World!')
        self.validate("EXTRACT(day FROM '2020-01-01'::TIMESTAMP)", "EXTRACT(day FROM CAST('2020-01-01' AS TIMESTAMP))")
        self.validate("EXTRACT(timezone FROM '2020-01-01'::TIMESTAMP)", "EXTRACT(timezone FROM CAST('2020-01-01' AS TIMESTAMP))")
        self.validate("EXTRACT(year FROM '2020-01-01'::TIMESTAMP WITH TIME ZONE)", "EXTRACT(year FROM CAST('2020-01-01' AS TIMESTAMPTZ))")
        self.validate("extract(month from '2021-01-31'::timestamp without time zone)", "EXTRACT(month FROM CAST('2021-01-31' AS TIMESTAMP))")
        self.validate('extract(week from current_date + 2)', 'EXTRACT(week FROM CURRENT_DATE + 2)')
        self.validate('EXTRACT(minute FROM datetime1 - datetime2)', 'EXTRACT(minute FROM datetime1 - datetime2)')

    def test_if(self):
        if False:
            return 10
        self.validate('SELECT IF(a > 1, 1, 0) FROM foo', 'SELECT CASE WHEN a > 1 THEN 1 ELSE 0 END FROM foo')
        self.validate('SELECT IF a > 1 THEN b END', 'SELECT CASE WHEN a > 1 THEN b END')
        self.validate('SELECT IF a > 1 THEN b ELSE c END', 'SELECT CASE WHEN a > 1 THEN b ELSE c END')
        self.validate('SELECT IF(a > 1, 1) FROM foo', 'SELECT CASE WHEN a > 1 THEN 1 END FROM foo')

    def test_with(self):
        if False:
            i = 10
            return i + 15
        self.validate('WITH a AS (SELECT 1) WITH b AS (SELECT 2) SELECT *', 'WITH a AS (SELECT 1), b AS (SELECT 2) SELECT *')
        self.validate('WITH a AS (SELECT 1), WITH b AS (SELECT 2) SELECT *', 'WITH a AS (SELECT 1), b AS (SELECT 2) SELECT *')
        self.validate('WITH A(filter) AS (VALUES 1, 2, 3) SELECT * FROM A WHERE filter >= 2', 'WITH A(filter) AS (VALUES (1), (2), (3)) SELECT * FROM A WHERE filter >= 2')
        self.validate('SELECT BOOL_OR(a > 10) FROM (VALUES 1, 2, 15) AS T(a)', 'SELECT BOOL_OR(a > 10) FROM (VALUES (1), (2), (15)) AS T(a)', write='presto')

    def test_alter(self):
        if False:
            print('Hello World!')
        self.validate('ALTER TABLE integers ADD k INTEGER', 'ALTER TABLE integers ADD COLUMN k INT')
        self.validate('ALTER TABLE integers ALTER i TYPE VARCHAR', 'ALTER TABLE integers ALTER COLUMN i SET DATA TYPE VARCHAR')
        self.validate('ALTER TABLE integers ALTER i TYPE VARCHAR COLLATE foo USING bar', 'ALTER TABLE integers ALTER COLUMN i SET DATA TYPE VARCHAR COLLATE foo USING bar')

    def test_time(self):
        if False:
            return 10
        self.validate("INTERVAL '1 day'", "INTERVAL '1' day")
        self.validate("INTERVAL '1 days' * 5", "INTERVAL '1' days * 5")
        self.validate("5 * INTERVAL '1 day'", "5 * INTERVAL '1' day")
        self.validate('INTERVAL 1 day', "INTERVAL '1' day")
        self.validate('INTERVAL 2 months', "INTERVAL '2' months")
        self.validate("TIMESTAMP '2020-01-01'", "CAST('2020-01-01' AS TIMESTAMP)")
        self.validate("TIMESTAMP WITH TIME ZONE '2020-01-01'", "CAST('2020-01-01' AS TIMESTAMPTZ)")
        self.validate("TIMESTAMP(9) WITH TIME ZONE '2020-01-01'", "CAST('2020-01-01' AS TIMESTAMPTZ(9))")
        self.validate("TIMESTAMP WITHOUT TIME ZONE '2020-01-01'", "CAST('2020-01-01' AS TIMESTAMP)")
        self.validate("'2020-01-01'::TIMESTAMP", "CAST('2020-01-01' AS TIMESTAMP)")
        self.validate("'2020-01-01'::TIMESTAMP WITHOUT TIME ZONE", "CAST('2020-01-01' AS TIMESTAMP)")
        self.validate("'2020-01-01'::TIMESTAMP WITH TIME ZONE", "CAST('2020-01-01' AS TIMESTAMPTZ)")
        self.validate("timestamp with time zone '2025-11-20 00:00:00+00' AT TIME ZONE 'Africa/Cairo'", "CAST('2025-11-20 00:00:00+00' AS TIMESTAMPTZ) AT TIME ZONE 'Africa/Cairo'")
        self.validate("DATE '2020-01-01'", "CAST('2020-01-01' AS DATE)")
        self.validate("'2020-01-01'::DATE", "CAST('2020-01-01' AS DATE)")
        self.validate("STR_TO_TIME('x', 'y')", "STRPTIME('x', 'y')", write='duckdb')
        self.validate("STR_TO_UNIX('x', 'y')", "EPOCH(STRPTIME('x', 'y'))", write='duckdb')
        self.validate("TIME_TO_STR(x, 'y')", "STRFTIME(x, 'y')", write='duckdb')
        self.validate('TIME_TO_UNIX(x)', 'EPOCH(x)', write='duckdb')
        self.validate("UNIX_TO_STR(123, 'y')", "STRFTIME(TO_TIMESTAMP(123), 'y')", write='duckdb')
        self.validate('UNIX_TO_TIME(123)', 'TO_TIMESTAMP(123)', write='duckdb')
        self.validate("STR_TO_TIME(x, 'y')", "CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(x, 'y')) AS TIMESTAMP)", write='hive')
        self.validate("STR_TO_TIME(x, 'yyyy-MM-dd HH:mm:ss')", 'CAST(x AS TIMESTAMP)', write='hive')
        self.validate("STR_TO_TIME(x, 'yyyy-MM-dd')", 'CAST(x AS TIMESTAMP)', write='hive')
        self.validate("STR_TO_UNIX('x', 'y')", "UNIX_TIMESTAMP('x', 'y')", write='hive')
        self.validate("TIME_TO_STR(x, 'y')", "DATE_FORMAT(x, 'y')", write='hive')
        self.validate('TIME_STR_TO_TIME(x)', 'TIME_STR_TO_TIME(x)', write=None)
        self.validate('TIME_STR_TO_UNIX(x)', 'TIME_STR_TO_UNIX(x)', write=None)
        self.validate('TIME_TO_TIME_STR(x)', 'CAST(x AS TEXT)', write=None)
        self.validate("TIME_TO_STR(x, 'y')", "TIME_TO_STR(x, 'y')", write=None)
        self.validate('TIME_TO_UNIX(x)', 'TIME_TO_UNIX(x)', write=None)
        self.validate("UNIX_TO_STR(x, 'y')", "UNIX_TO_STR(x, 'y')", write=None)
        self.validate('UNIX_TO_TIME(x)', 'UNIX_TO_TIME(x)', write=None)
        self.validate('UNIX_TO_TIME_STR(x)', 'UNIX_TO_TIME_STR(x)', write=None)
        self.validate('TIME_STR_TO_DATE(x)', 'TIME_STR_TO_DATE(x)', write=None)
        self.validate('TIME_STR_TO_DATE(x)', 'TO_DATE(x)', write='hive')
        self.validate("UNIX_TO_STR(x, 'yyyy-MM-dd HH:mm:ss')", 'FROM_UNIXTIME(x)', write='hive')
        self.validate("STR_TO_UNIX(x, 'yyyy-MM-dd HH:mm:ss')", 'UNIX_TIMESTAMP(x)', write='hive')
        self.validate('IF(x > 1, x + 1)', 'IF(x > 1, x + 1)', write='presto')
        self.validate('IF(x > 1, 1 + 1)', 'IF(x > 1, 1 + 1)', write='hive')
        self.validate('IF(x > 1, 1, 0)', 'IF(x > 1, 1, 0)', write='hive')
        self.validate('TIME_TO_UNIX(x)', 'UNIX_TIMESTAMP(x)', write='hive')
        self.validate("UNIX_TO_STR(123, 'y')", "FROM_UNIXTIME(123, 'y')", write='hive')
        self.validate('UNIX_TO_TIME(123)', 'FROM_UNIXTIME(123)', write='hive')
        self.validate("STR_TO_TIME('x', 'y')", "DATE_PARSE('x', 'y')", write='presto')
        self.validate("STR_TO_UNIX('x', 'y')", "TO_UNIXTIME(DATE_PARSE('x', 'y'))", write='presto')
        self.validate("TIME_TO_STR(x, 'y')", "DATE_FORMAT(x, 'y')", write='presto')
        self.validate('TIME_TO_UNIX(x)', 'TO_UNIXTIME(x)', write='presto')
        self.validate("UNIX_TO_STR(123, 'y')", "DATE_FORMAT(FROM_UNIXTIME(123), 'y')", write='presto')
        self.validate('UNIX_TO_TIME(123)', 'FROM_UNIXTIME(123)', write='presto')
        self.validate("STR_TO_TIME('x', 'y')", "TO_TIMESTAMP('x', 'y')", write='spark')
        self.validate("STR_TO_UNIX('x', 'y')", "UNIX_TIMESTAMP('x', 'y')", write='spark')
        self.validate("TIME_TO_STR(x, 'y')", "DATE_FORMAT(x, 'y')", write='spark')
        self.validate('TIME_TO_UNIX(x)', 'UNIX_TIMESTAMP(x)', write='spark')
        self.validate("UNIX_TO_STR(123, 'y')", "FROM_UNIXTIME(123, 'y')", write='spark')
        self.validate('UNIX_TO_TIME(123)', 'CAST(FROM_UNIXTIME(123) AS TIMESTAMP)', write='spark')
        self.validate('CREATE TEMPORARY TABLE test AS SELECT 1', 'CREATE TEMPORARY VIEW test AS SELECT 1', write='spark2')

    @mock.patch('sqlglot.helper.logger')
    def test_index_offset(self, logger):
        if False:
            return 10
        self.validate('x[0]', 'x[1]', write='presto', identity=False)
        self.validate('x[1]', 'x[0]', read='presto', identity=False)
        logger.warning.assert_any_call('Applying array index offset (%s)', 1)
        logger.warning.assert_any_call('Applying array index offset (%s)', -1)
        self.validate('x[x - 1]', 'x[x - 1]', write='presto', identity=False)
        self.validate('x[array_size(y) - 1]', 'x[CARDINALITY(y) - 1 + 1]', write='presto', identity=False)
        self.validate('x[3 - 1]', 'x[3]', write='presto', identity=False)
        self.validate('MAP(a, b)[0]', 'MAP(a, b)[0]', write='presto', identity=False)

    def test_identify_lambda(self):
        if False:
            for i in range(10):
                print('nop')
        self.validate('x(y -> y)', 'X("y" -> "y")', identify=True)

    def test_identity(self):
        if False:
            print('Hello World!')
        self.assertEqual(transpile('')[0], '')
        for sql in load_sql_fixtures('identity.sql'):
            with self.subTest(sql):
                self.assertEqual(transpile(sql)[0], sql.strip())

    def test_normalize_name(self):
        if False:
            while True:
                i = 10
        self.assertEqual(transpile('cardinality(x)', read='presto', write='presto', normalize_functions='lower')[0], 'cardinality(x)')

    def test_partial(self):
        if False:
            return 10
        for sql in load_sql_fixtures('partial.sql'):
            with self.subTest(sql):
                self.assertEqual(transpile(sql, error_level=ErrorLevel.IGNORE)[0], sql.strip())

    def test_pretty(self):
        if False:
            i = 10
            return i + 15
        for (_, sql, pretty) in load_sql_fixture_pairs('pretty.sql'):
            with self.subTest(sql[:100]):
                generated = transpile(sql, pretty=True)[0]
                self.assertEqual(generated, pretty)
                self.assertEqual(parse_one(sql), parse_one(pretty))

    def test_pretty_line_breaks(self):
        if False:
            while True:
                i = 10
        self.assertEqual(transpile("SELECT '1\n2'", pretty=True)[0], "SELECT\n  '1\n2'")

    @mock.patch('sqlglot.parser.logger')
    def test_error_level(self, logger):
        if False:
            print('Hello World!')
        invalid = 'x + 1. ('
        expected_messages = ["Required keyword: 'expressions' missing for <class 'sqlglot.expressions.Aliases'>. Line 1, Col: 8.\n  x + 1. \x1b[4m(\x1b[0m", 'Expecting ). Line 1, Col: 8.\n  x + 1. \x1b[4m(\x1b[0m']
        expected_errors = [{'description': "Required keyword: 'expressions' missing for <class 'sqlglot.expressions.Aliases'>", 'line': 1, 'col': 8, 'start_context': 'x + 1. ', 'highlight': '(', 'end_context': '', 'into_expression': None}, {'description': 'Expecting )', 'line': 1, 'col': 8, 'start_context': 'x + 1. ', 'highlight': '(', 'end_context': '', 'into_expression': None}]
        transpile(invalid, error_level=ErrorLevel.WARN)
        for error in expected_messages:
            assert_logger_contains(error, logger)
        with self.assertRaises(ParseError) as ctx:
            transpile(invalid, error_level=ErrorLevel.IMMEDIATE)
        self.assertEqual(str(ctx.exception), expected_messages[0])
        self.assertEqual(ctx.exception.errors[0], expected_errors[0])
        with self.assertRaises(ParseError) as ctx:
            transpile(invalid, error_level=ErrorLevel.RAISE)
        self.assertEqual(str(ctx.exception), '\n\n'.join(expected_messages))
        self.assertEqual(ctx.exception.errors, expected_errors)
        more_than_max_errors = '(((('
        expected_messages = "Required keyword: 'this' missing for <class 'sqlglot.expressions.Paren'>. Line 1, Col: 4.\n  (((\x1b[4m(\x1b[0m\n\nExpecting ). Line 1, Col: 4.\n  (((\x1b[4m(\x1b[0m\n\nExpecting ). Line 1, Col: 4.\n  (((\x1b[4m(\x1b[0m\n\n... and 2 more"
        expected_errors = [{'description': "Required keyword: 'this' missing for <class 'sqlglot.expressions.Paren'>", 'line': 1, 'col': 4, 'start_context': '(((', 'highlight': '(', 'end_context': '', 'into_expression': None}, {'description': 'Expecting )', 'line': 1, 'col': 4, 'start_context': '(((', 'highlight': '(', 'end_context': '', 'into_expression': None}]
        expected_errors += [expected_errors[1]] * 3
        with self.assertRaises(ParseError) as ctx:
            transpile(more_than_max_errors, error_level=ErrorLevel.RAISE)
        self.assertEqual(str(ctx.exception), expected_messages)
        self.assertEqual(ctx.exception.errors, expected_errors)

    @mock.patch('sqlglot.generator.logger')
    def test_unsupported_level(self, logger):
        if False:
            while True:
                i = 10

        def unsupported(level):
            if False:
                return 10
            transpile('SELECT MAP(a, b), MAP(a, b), MAP(a, b), MAP(a, b)', read='presto', write='hive', unsupported_level=level)
        error = 'Cannot convert array columns into map.'
        unsupported(ErrorLevel.WARN)
        assert_logger_contains('\n'.join([error] * 4), logger, level='warning')
        with self.assertRaises(UnsupportedError) as ctx:
            unsupported(ErrorLevel.RAISE)
        self.assertEqual(str(ctx.exception).count(error), 3)
        with self.assertRaises(UnsupportedError) as ctx:
            unsupported(ErrorLevel.IMMEDIATE)
        self.assertEqual(str(ctx.exception).count(error), 1)