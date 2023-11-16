from pymysql.tests import base
import pymysql.cursors
import datetime
import warnings

class TestDictCursor(base.PyMySQLTestCase):
    bob = {'name': 'bob', 'age': 21, 'DOB': datetime.datetime(1990, 2, 6, 23, 4, 56)}
    jim = {'name': 'jim', 'age': 56, 'DOB': datetime.datetime(1955, 5, 9, 13, 12, 45)}
    fred = {'name': 'fred', 'age': 100, 'DOB': datetime.datetime(1911, 9, 12, 1, 1, 1)}
    cursor_type = pymysql.cursors.DictCursor

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.conn = conn = self.connect()
        c = conn.cursor(self.cursor_type)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            c.execute('drop table if exists dictcursor')
            c.execute('CREATE TABLE dictcursor (name char(20), age int , DOB datetime)')
        data = [('bob', 21, '1990-02-06 23:04:56'), ('jim', 56, '1955-05-09 13:12:45'), ('fred', 100, '1911-09-12 01:01:01')]
        c.executemany('insert into dictcursor values (%s,%s,%s)', data)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        c = self.conn.cursor()
        c.execute('drop table dictcursor')
        super().tearDown()

    def _ensure_cursor_expired(self, cursor):
        if False:
            i = 10
            return i + 15
        pass

    def test_DictCursor(self):
        if False:
            while True:
                i = 10
        (bob, jim, fred) = (self.bob.copy(), self.jim.copy(), self.fred.copy())
        conn = self.conn
        c = conn.cursor(self.cursor_type)
        c.execute("update dictcursor set age=20 where name='bob'")
        bob['age'] = 20
        c.execute("SELECT * from dictcursor where name='bob'")
        r = c.fetchone()
        self.assertEqual(bob, r, 'fetchone via DictCursor failed')
        self._ensure_cursor_expired(c)
        c.execute("SELECT * from dictcursor where name='bob'")
        r = c.fetchall()
        self.assertEqual([bob], r, 'fetch a 1 row result via fetchall failed via DictCursor')
        c.execute("SELECT * from dictcursor where name='bob'")
        for r in c:
            self.assertEqual(bob, r, 'fetch a 1 row result via iteration failed via DictCursor')
        c.execute('SELECT * from dictcursor')
        r = c.fetchall()
        self.assertEqual([bob, jim, fred], r, 'fetchall failed via DictCursor')
        c.execute('SELECT * from dictcursor')
        r = list(c)
        self.assertEqual([bob, jim, fred], r, 'DictCursor should be iterable')
        c.execute('SELECT * from dictcursor')
        r = c.fetchmany(2)
        self.assertEqual([bob, jim], r, 'fetchmany failed via DictCursor')
        self._ensure_cursor_expired(c)

    def test_custom_dict(self):
        if False:
            print('Hello World!')

        class MyDict(dict):
            pass

        class MyDictCursor(self.cursor_type):
            dict_type = MyDict
        keys = ['name', 'age', 'DOB']
        bob = MyDict([(k, self.bob[k]) for k in keys])
        jim = MyDict([(k, self.jim[k]) for k in keys])
        fred = MyDict([(k, self.fred[k]) for k in keys])
        cur = self.conn.cursor(MyDictCursor)
        cur.execute("SELECT * FROM dictcursor WHERE name='bob'")
        r = cur.fetchone()
        self.assertEqual(bob, r, 'fetchone() returns MyDictCursor')
        self._ensure_cursor_expired(cur)
        cur.execute('SELECT * FROM dictcursor')
        r = cur.fetchall()
        self.assertEqual([bob, jim, fred], r, 'fetchall failed via MyDictCursor')
        cur.execute('SELECT * FROM dictcursor')
        r = list(cur)
        self.assertEqual([bob, jim, fred], r, 'list failed via MyDictCursor')
        cur.execute('SELECT * FROM dictcursor')
        r = cur.fetchmany(2)
        self.assertEqual([bob, jim], r, 'list failed via MyDictCursor')
        self._ensure_cursor_expired(cur)

class TestSSDictCursor(TestDictCursor):
    cursor_type = pymysql.cursors.SSDictCursor

    def _ensure_cursor_expired(self, cursor):
        if False:
            for i in range(10):
                print('nop')
        list(cursor.fetchall_unbuffered())
if __name__ == '__main__':
    import unittest
    unittest.main()