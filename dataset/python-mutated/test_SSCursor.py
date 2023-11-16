import pytest
from pymysql.tests import base
import pymysql.cursors
from pymysql.constants import CLIENT, ER

class TestSSCursor(base.PyMySQLTestCase):

    def test_SSCursor(self):
        if False:
            i = 10
            return i + 15
        affected_rows = 18446744073709551615
        conn = self.connect(client_flag=CLIENT.MULTI_STATEMENTS)
        data = [('America', '', 'America/Jamaica'), ('America', '', 'America/Los_Angeles'), ('America', '', 'America/Lima'), ('America', '', 'America/New_York'), ('America', '', 'America/Menominee'), ('America', '', 'America/Havana'), ('America', '', 'America/El_Salvador'), ('America', '', 'America/Costa_Rica'), ('America', '', 'America/Denver'), ('America', '', 'America/Detroit')]
        cursor = conn.cursor(pymysql.cursors.SSCursor)
        cursor.execute('CREATE TABLE tz_data (region VARCHAR(64), zone VARCHAR(64), name VARCHAR(64))')
        conn.begin()
        for i in data:
            cursor.execute('INSERT INTO tz_data VALUES (%s, %s, %s)', i)
            self.assertEqual(conn.affected_rows(), 1, 'affected_rows does not match')
        conn.commit()
        iter = 0
        cursor.execute('SELECT * FROM tz_data')
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            iter += 1
            self.assertEqual(cursor.rowcount, affected_rows, 'cursor.rowcount != %s' % str(affected_rows))
            self.assertEqual(cursor.rownumber, iter, 'cursor.rowcount != %s' % str(iter))
            self.assertEqual(row in data, True, 'Row not found in source data')
        cursor.execute('SELECT * FROM tz_data')
        self.assertEqual(len(cursor.fetchall()), len(data), 'fetchall failed. Number of rows does not match')
        cursor.execute('SELECT * FROM tz_data')
        self.assertEqual(len(cursor.fetchmany(2)), 2, 'fetchmany failed. Number of rows does not match')
        while True:
            res = cursor.fetchone()
            if res is None:
                break
        cursor.execute('UPDATE tz_data SET zone = %s', ['Foo'])
        conn.commit()
        self.assertEqual(cursor.rowcount, len(data), 'Update failed. affected_rows != %s' % str(len(data)))
        cursor.executemany('INSERT INTO tz_data VALUES (%s, %s, %s)', data)
        self.assertEqual(cursor.rowcount, len(data), 'executemany failed. cursor.rowcount != %s' % str(len(data)))
        cursor.execute('SELECT 1; SELECT 2; SELECT 3')
        self.assertListEqual(list(cursor), [(1,)])
        self.assertTrue(cursor.nextset())
        self.assertListEqual(list(cursor), [(2,)])
        self.assertTrue(cursor.nextset())
        self.assertListEqual(list(cursor), [(3,)])
        self.assertFalse(cursor.nextset())
        cursor.execute('DROP TABLE IF EXISTS tz_data')
        cursor.close()

    def test_execution_time_limit(self):
        if False:
            i = 10
            return i + 15
        conn = self.connect()
        self.safe_create_table(conn, 'test', 'create table test (data varchar(10))')
        with conn.cursor() as cur:
            cur.execute("insert into test (data) values ('row1'), ('row2'), ('row3'), ('row4'), ('row5')")
            conn.commit()
        db_type = self.get_mysql_vendor(conn)
        with conn.cursor(pymysql.cursors.SSCursor) as cur:
            if db_type == 'mysql':
                sql = 'SELECT /*+ MAX_EXECUTION_TIME(2000) */ data, sleep(0.01) FROM test'
            else:
                sql = 'SET STATEMENT max_statement_time=2 FOR SELECT data, sleep(0.01) FROM test'
            cur.execute(sql)
            self.assertEqual(cur.fetchall(), [('row1', 0), ('row2', 0), ('row3', 0), ('row4', 0), ('row5', 0)])
            if db_type == 'mysql':
                sql = 'SELECT /*+ MAX_EXECUTION_TIME(2000) */ data, sleep(0.01) FROM test'
            else:
                sql = 'SET STATEMENT max_statement_time=2 FOR SELECT data, sleep(0.01) FROM test'
            cur.execute(sql)
            self.assertEqual(cur.fetchone(), ('row1', 0))
            with pytest.warns(UserWarning):
                cur.execute('SELECT 1')
            self.assertEqual(cur.fetchone(), (1,))
            self.assertEqual(cur.fetchone(), None)
            if db_type == 'mysql':
                sql = 'SELECT /*+ MAX_EXECUTION_TIME(1) */ data, sleep(1) FROM test'
            else:
                sql = 'SET STATEMENT max_statement_time=0.001 FOR SELECT data, sleep(1) FROM test'
            with pytest.raises(pymysql.err.OperationalError) as cm:
                cur.execute(sql)
                cur.fetchall()
            if db_type == 'mysql':
                self.assertEqual(cm.value.args[0], ER.QUERY_TIMEOUT)
            else:
                self.assertEqual(cm.value.args[0], ER.STATEMENT_TIMEOUT)
            cur.execute('SELECT 1')
            self.assertEqual(cur.fetchone(), (1,))

    def test_warnings(self):
        if False:
            i = 10
            return i + 15
        con = self.connect()
        cur = con.cursor(pymysql.cursors.SSCursor)
        cur.execute('DROP TABLE IF EXISTS `no_exists_table`')
        self.assertEqual(cur.warning_count, 1)
        cur.execute('SHOW WARNINGS')
        w = cur.fetchone()
        self.assertEqual(w[1], ER.BAD_TABLE_ERROR)
        self.assertIn('no_exists_table', w[2])
        self.assertIsNone(cur.fetchone())
        cur.execute('SELECT 1')
        self.assertEqual(cur.fetchone(), (1,))
        self.assertIsNone(cur.fetchone())
        self.assertEqual(cur.warning_count, 0)
        cur.execute("SELECT CAST('abc' AS SIGNED)")
        rows = cur.fetchmany(2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(cur.warning_count, 1)
__all__ = ['TestSSCursor']
if __name__ == '__main__':
    import unittest
    unittest.main()