from . import dbapi20
import pymysql
from pymysql.tests import base

class test_MySQLdb(dbapi20.DatabaseAPI20Test):
    driver = pymysql
    connect_args = ()
    connect_kw_args = base.PyMySQLTestCase.databases[0].copy()
    connect_kw_args.update(dict(read_default_file='~/.my.cnf', charset='utf8', sql_mode='ANSI,STRICT_TRANS_TABLES,TRADITIONAL'))

    def test_setoutputsize(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_setoutputsize_basic(self):
        if False:
            print('Hello World!')
        pass
    "The tests on fetchone and fetchall and rowcount bogusly\n    test for an exception if the statement cannot return a\n    result set. MySQL always returns a result set; it's just that\n    some things return empty result sets."

    def test_fetchall(self):
        if False:
            print('Hello World!')
        con = self._connect()
        try:
            cur = con.cursor()
            self.assertRaises(self.driver.Error, cur.fetchall)
            self.executeDDL1(cur)
            for sql in self._populate():
                cur.execute(sql)
            cur.execute('select name from %sbooze' % self.table_prefix)
            rows = cur.fetchall()
            self.assertTrue(cur.rowcount in (-1, len(self.samples)))
            self.assertEqual(len(rows), len(self.samples), 'cursor.fetchall did not retrieve all rows')
            rows = [r[0] for r in rows]
            rows.sort()
            for i in range(0, len(self.samples)):
                self.assertEqual(rows[i], self.samples[i], 'cursor.fetchall retrieved incorrect rows')
            rows = cur.fetchall()
            self.assertEqual(len(rows), 0, 'cursor.fetchall should return an empty list if called after the whole result set has been fetched')
            self.assertTrue(cur.rowcount in (-1, len(self.samples)))
            self.executeDDL2(cur)
            cur.execute('select name from %sbarflys' % self.table_prefix)
            rows = cur.fetchall()
            self.assertTrue(cur.rowcount in (-1, 0))
            self.assertEqual(len(rows), 0, 'cursor.fetchall should return an empty list if a select query returns no rows')
        finally:
            con.close()

    def test_fetchone(self):
        if False:
            print('Hello World!')
        con = self._connect()
        try:
            cur = con.cursor()
            self.assertRaises(self.driver.Error, cur.fetchone)
            self.executeDDL1(cur)
            cur.execute('select name from %sbooze' % self.table_prefix)
            self.assertEqual(cur.fetchone(), None, 'cursor.fetchone should return None if a query retrieves no rows')
            self.assertTrue(cur.rowcount in (-1, 0))
            cur.execute("insert into %sbooze values ('Victoria Bitter')" % self.table_prefix)
            cur.execute('select name from %sbooze' % self.table_prefix)
            r = cur.fetchone()
            self.assertEqual(len(r), 1, 'cursor.fetchone should have retrieved a single row')
            self.assertEqual(r[0], 'Victoria Bitter', 'cursor.fetchone retrieved incorrect data')
            self.assertTrue(cur.rowcount in (-1, 1))
        finally:
            con.close()

    def test_rowcount(self):
        if False:
            print('Hello World!')
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            cur.execute("insert into %sbooze values ('Victoria Bitter')" % self.table_prefix)
            cur.execute('select name from %sbooze' % self.table_prefix)
            self.assertTrue(cur.rowcount in (-1, 1), 'cursor.rowcount should == number of rows returned, or set to -1 after executing a select statement')
            self.executeDDL2(cur)
        finally:
            con.close()

    def test_callproc(self):
        if False:
            print('Hello World!')
        pass

    def help_nextset_setUp(self, cur):
        if False:
            print('Hello World!')
        'Should create a procedure called deleteme\n        that returns two result sets, first the\n        number of rows in booze then "name from booze"\n        '
        sql = '\n           create procedure deleteme()\n           begin\n               select count(*) from %(tp)sbooze;\n               select name from %(tp)sbooze;\n           end\n        ' % dict(tp=self.table_prefix)
        cur.execute(sql)

    def help_nextset_tearDown(self, cur):
        if False:
            for i in range(10):
                print('nop')
        'If cleaning up is needed after nextSetTest'
        cur.execute('drop procedure deleteme')

    def test_nextset(self):
        if False:
            for i in range(10):
                print('nop')
        con = self._connect()
        try:
            cur = con.cursor()
            if not hasattr(cur, 'nextset'):
                return
            try:
                self.executeDDL1(cur)
                sql = self._populate()
                for sql in self._populate():
                    cur.execute(sql)
                self.help_nextset_setUp(cur)
                cur.callproc('deleteme')
                numberofrows = cur.fetchone()
                assert numberofrows[0] == len(self.samples)
                assert cur.nextset()
                names = cur.fetchall()
                assert len(names) == len(self.samples)
                s = cur.nextset()
                if s:
                    empty = cur.fetchall()
                    self.assertEqual(len(empty), 0, 'non-empty result set after other result sets')
            finally:
                self.help_nextset_tearDown(cur)
        finally:
            con.close()