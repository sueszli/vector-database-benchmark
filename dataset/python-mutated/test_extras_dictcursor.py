import copy
import time
import pickle
import unittest
from datetime import timedelta
from functools import lru_cache
import psycopg2
import psycopg2.extras
from psycopg2.extras import NamedTupleConnection, NamedTupleCursor
from .testutils import ConnectingTestCase, skip_before_postgres, crdb_version, skip_if_crdb

class _DictCursorBase(ConnectingTestCase):

    def setUp(self):
        if False:
            return 10
        ConnectingTestCase.setUp(self)
        curs = self.conn.cursor()
        if crdb_version(self.conn) is not None:
            curs.execute("SET experimental_enable_temp_tables = 'on'")
        curs.execute('CREATE TEMPORARY TABLE ExtrasDictCursorTests (foo text)')
        curs.execute("INSERT INTO ExtrasDictCursorTests VALUES ('bar')")
        self.conn.commit()

    def _testIterRowNumber(self, curs):
        if False:
            return 10
        curs.itersize = 20
        curs.execute('select * from generate_series(1,10)')
        for (i, r) in enumerate(curs):
            self.assertEqual(i + 1, curs.rownumber)

    def _testNamedCursorNotGreedy(self, curs):
        if False:
            while True:
                i = 10
        curs.itersize = 2
        curs.execute('select clock_timestamp() as ts from generate_series(1,3)')
        recs = []
        for t in curs:
            time.sleep(0.01)
            recs.append(t)
        self.assert_(recs[1]['ts'] - recs[0]['ts'] < timedelta(seconds=0.005))
        self.assert_(recs[2]['ts'] - recs[1]['ts'] > timedelta(seconds=0.0099))

class ExtrasDictCursorTests(_DictCursorBase):
    """Test if DictCursor extension class works."""

    @skip_if_crdb('named cursor', version='< 22.1')
    def testDictConnCursorArgs(self):
        if False:
            i = 10
            return i + 15
        self.conn.close()
        self.conn = self.connect(connection_factory=psycopg2.extras.DictConnection)
        cur = self.conn.cursor()
        self.assert_(isinstance(cur, psycopg2.extras.DictCursor))
        self.assertEqual(cur.name, None)
        cur = self.conn.cursor('foo', cursor_factory=psycopg2.extras.NamedTupleCursor)
        self.assertEqual(cur.name, 'foo')
        self.assert_(isinstance(cur, psycopg2.extras.NamedTupleCursor))

    def testDictCursorWithPlainCursorFetchOne(self):
        if False:
            i = 10
            return i + 15
        self._testWithPlainCursor(lambda curs: curs.fetchone())

    def testDictCursorWithPlainCursorFetchMany(self):
        if False:
            return 10
        self._testWithPlainCursor(lambda curs: curs.fetchmany(100)[0])

    def testDictCursorWithPlainCursorFetchManyNoarg(self):
        if False:
            for i in range(10):
                print('nop')
        self._testWithPlainCursor(lambda curs: curs.fetchmany()[0])

    def testDictCursorWithPlainCursorFetchAll(self):
        if False:
            i = 10
            return i + 15
        self._testWithPlainCursor(lambda curs: curs.fetchall()[0])

    def testDictCursorWithPlainCursorIter(self):
        if False:
            print('Hello World!')

        def getter(curs):
            if False:
                for i in range(10):
                    print('nop')
            for row in curs:
                return row
        self._testWithPlainCursor(getter)

    def testUpdateRow(self):
        if False:
            i = 10
            return i + 15
        row = self._testWithPlainCursor(lambda curs: curs.fetchone())
        row['foo'] = 'qux'
        self.failUnless(row['foo'] == 'qux')
        self.failUnless(row[0] == 'qux')

    @skip_before_postgres(8, 0)
    def testDictCursorWithPlainCursorIterRowNumber(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        self._testIterRowNumber(curs)

    def _testWithPlainCursor(self, getter):
        if False:
            return 10
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        curs.execute('SELECT * FROM ExtrasDictCursorTests')
        row = getter(curs)
        self.failUnless(row['foo'] == 'bar')
        self.failUnless(row[0] == 'bar')
        return row

    def testDictCursorWithNamedCursorFetchOne(self):
        if False:
            for i in range(10):
                print('nop')
        self._testWithNamedCursor(lambda curs: curs.fetchone())

    def testDictCursorWithNamedCursorFetchMany(self):
        if False:
            i = 10
            return i + 15
        self._testWithNamedCursor(lambda curs: curs.fetchmany(100)[0])

    def testDictCursorWithNamedCursorFetchManyNoarg(self):
        if False:
            i = 10
            return i + 15
        self._testWithNamedCursor(lambda curs: curs.fetchmany()[0])

    def testDictCursorWithNamedCursorFetchAll(self):
        if False:
            i = 10
            return i + 15
        self._testWithNamedCursor(lambda curs: curs.fetchall()[0])

    def testDictCursorWithNamedCursorIter(self):
        if False:
            while True:
                i = 10

        def getter(curs):
            if False:
                for i in range(10):
                    print('nop')
            for row in curs:
                return row
        self._testWithNamedCursor(getter)

    @skip_if_crdb('greedy cursor')
    @skip_before_postgres(8, 2)
    def testDictCursorWithNamedCursorNotGreedy(self):
        if False:
            i = 10
            return i + 15
        curs = self.conn.cursor('tmp', cursor_factory=psycopg2.extras.DictCursor)
        self._testNamedCursorNotGreedy(curs)

    @skip_if_crdb('named cursor', version='< 22.1')
    @skip_before_postgres(8, 0)
    def testDictCursorWithNamedCursorIterRowNumber(self):
        if False:
            i = 10
            return i + 15
        curs = self.conn.cursor('tmp', cursor_factory=psycopg2.extras.DictCursor)
        self._testIterRowNumber(curs)

    @skip_if_crdb('named cursor', version='< 22.1')
    def _testWithNamedCursor(self, getter):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor('aname', cursor_factory=psycopg2.extras.DictCursor)
        curs.execute('SELECT * FROM ExtrasDictCursorTests')
        row = getter(curs)
        self.failUnless(row['foo'] == 'bar')
        self.failUnless(row[0] == 'bar')

    def testPickleDictRow(self):
        if False:
            while True:
                i = 10
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        curs.execute('select 10 as a, 20 as b')
        r = curs.fetchone()
        d = pickle.dumps(r)
        r1 = pickle.loads(d)
        self.assertEqual(r, r1)
        self.assertEqual(r[0], r1[0])
        self.assertEqual(r[1], r1[1])
        self.assertEqual(r['a'], r1['a'])
        self.assertEqual(r['b'], r1['b'])
        self.assertEqual(r._index, r1._index)

    def test_copy(self):
        if False:
            return 10
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        curs.execute("select 10 as foo, 'hi' as bar")
        rv = curs.fetchone()
        self.assertEqual(len(rv), 2)
        rv2 = copy.copy(rv)
        self.assertEqual(len(rv2), 2)
        self.assertEqual(len(rv), 2)
        rv3 = copy.deepcopy(rv)
        self.assertEqual(len(rv3), 2)
        self.assertEqual(len(rv), 2)

    def test_iter_methods(self):
        if False:
            i = 10
            return i + 15
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        curs.execute('select 10 as a, 20 as b')
        r = curs.fetchone()
        self.assert_(not isinstance(r.keys(), list))
        self.assertEqual(len(list(r.keys())), 2)
        self.assert_(not isinstance(r.values(), list))
        self.assertEqual(len(list(r.values())), 2)
        self.assert_(not isinstance(r.items(), list))
        self.assertEqual(len(list(r.items())), 2)

    def test_order(self):
        if False:
            return 10
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        curs.execute('select 5 as foo, 4 as bar, 33 as baz, 2 as qux')
        r = curs.fetchone()
        self.assertEqual(list(r), [5, 4, 33, 2])
        self.assertEqual(list(r.keys()), ['foo', 'bar', 'baz', 'qux'])
        self.assertEqual(list(r.values()), [5, 4, 33, 2])
        self.assertEqual(list(r.items()), [('foo', 5), ('bar', 4), ('baz', 33), ('qux', 2)])
        r1 = pickle.loads(pickle.dumps(r))
        self.assertEqual(list(r1), list(r))
        self.assertEqual(list(r1.keys()), list(r.keys()))
        self.assertEqual(list(r1.values()), list(r.values()))
        self.assertEqual(list(r1.items()), list(r.items()))

class ExtrasDictCursorRealTests(_DictCursorBase):

    def testRealMeansReal(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute('SELECT * FROM ExtrasDictCursorTests')
        row = curs.fetchone()
        self.assert_(isinstance(row, dict))

    def testDictCursorWithPlainCursorRealFetchOne(self):
        if False:
            while True:
                i = 10
        self._testWithPlainCursorReal(lambda curs: curs.fetchone())

    def testDictCursorWithPlainCursorRealFetchMany(self):
        if False:
            while True:
                i = 10
        self._testWithPlainCursorReal(lambda curs: curs.fetchmany(100)[0])

    def testDictCursorWithPlainCursorRealFetchManyNoarg(self):
        if False:
            i = 10
            return i + 15
        self._testWithPlainCursorReal(lambda curs: curs.fetchmany()[0])

    def testDictCursorWithPlainCursorRealFetchAll(self):
        if False:
            print('Hello World!')
        self._testWithPlainCursorReal(lambda curs: curs.fetchall()[0])

    def testDictCursorWithPlainCursorRealIter(self):
        if False:
            for i in range(10):
                print('nop')

        def getter(curs):
            if False:
                for i in range(10):
                    print('nop')
            for row in curs:
                return row
        self._testWithPlainCursorReal(getter)

    @skip_before_postgres(8, 0)
    def testDictCursorWithPlainCursorRealIterRowNumber(self):
        if False:
            while True:
                i = 10
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self._testIterRowNumber(curs)

    def _testWithPlainCursorReal(self, getter):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute('SELECT * FROM ExtrasDictCursorTests')
        row = getter(curs)
        self.failUnless(row['foo'] == 'bar')

    def testPickleRealDictRow(self):
        if False:
            while True:
                i = 10
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute('select 10 as a, 20 as b')
        r = curs.fetchone()
        d = pickle.dumps(r)
        r1 = pickle.loads(d)
        self.assertEqual(r, r1)
        self.assertEqual(r['a'], r1['a'])
        self.assertEqual(r['b'], r1['b'])

    def test_copy(self):
        if False:
            i = 10
            return i + 15
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute("select 10 as foo, 'hi' as bar")
        rv = curs.fetchone()
        self.assertEqual(len(rv), 2)
        rv2 = copy.copy(rv)
        self.assertEqual(len(rv2), 2)
        self.assertEqual(len(rv), 2)
        rv3 = copy.deepcopy(rv)
        self.assertEqual(len(rv3), 2)
        self.assertEqual(len(rv), 2)

    def testDictCursorRealWithNamedCursorFetchOne(self):
        if False:
            for i in range(10):
                print('nop')
        self._testWithNamedCursorReal(lambda curs: curs.fetchone())

    def testDictCursorRealWithNamedCursorFetchMany(self):
        if False:
            return 10
        self._testWithNamedCursorReal(lambda curs: curs.fetchmany(100)[0])

    def testDictCursorRealWithNamedCursorFetchManyNoarg(self):
        if False:
            i = 10
            return i + 15
        self._testWithNamedCursorReal(lambda curs: curs.fetchmany()[0])

    def testDictCursorRealWithNamedCursorFetchAll(self):
        if False:
            while True:
                i = 10
        self._testWithNamedCursorReal(lambda curs: curs.fetchall()[0])

    def testDictCursorRealWithNamedCursorIter(self):
        if False:
            for i in range(10):
                print('nop')

        def getter(curs):
            if False:
                i = 10
                return i + 15
            for row in curs:
                return row
        self._testWithNamedCursorReal(getter)

    @skip_if_crdb('greedy cursor')
    @skip_before_postgres(8, 2)
    def testDictCursorRealWithNamedCursorNotGreedy(self):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor('tmp', cursor_factory=psycopg2.extras.RealDictCursor)
        self._testNamedCursorNotGreedy(curs)

    @skip_if_crdb('named cursor', version='< 22.1')
    @skip_before_postgres(8, 0)
    def testDictCursorRealWithNamedCursorIterRowNumber(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor('tmp', cursor_factory=psycopg2.extras.RealDictCursor)
        self._testIterRowNumber(curs)

    @skip_if_crdb('named cursor', version='< 22.1')
    def _testWithNamedCursorReal(self, getter):
        if False:
            print('Hello World!')
        curs = self.conn.cursor('aname', cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute('SELECT * FROM ExtrasDictCursorTests')
        row = getter(curs)
        self.failUnless(row['foo'] == 'bar')

    def test_iter_methods(self):
        if False:
            return 10
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute('select 10 as a, 20 as b')
        r = curs.fetchone()
        self.assert_(not isinstance(r.keys(), list))
        self.assertEqual(len(list(r.keys())), 2)
        self.assert_(not isinstance(r.values(), list))
        self.assertEqual(len(list(r.values())), 2)
        self.assert_(not isinstance(r.items(), list))
        self.assertEqual(len(list(r.items())), 2)

    def test_order(self):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute('select 5 as foo, 4 as bar, 33 as baz, 2 as qux')
        r = curs.fetchone()
        self.assertEqual(list(r), ['foo', 'bar', 'baz', 'qux'])
        self.assertEqual(list(r.keys()), ['foo', 'bar', 'baz', 'qux'])
        self.assertEqual(list(r.values()), [5, 4, 33, 2])
        self.assertEqual(list(r.items()), [('foo', 5), ('bar', 4), ('baz', 33), ('qux', 2)])
        r1 = pickle.loads(pickle.dumps(r))
        self.assertEqual(list(r1), list(r))
        self.assertEqual(list(r1.keys()), list(r.keys()))
        self.assertEqual(list(r1.values()), list(r.values()))
        self.assertEqual(list(r1.items()), list(r.items()))

    def test_pop(self):
        if False:
            return 10
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute('select 1 as a, 2 as b, 3 as c')
        r = curs.fetchone()
        self.assertEqual(r.pop('b'), 2)
        self.assertEqual(list(r), ['a', 'c'])
        self.assertEqual(list(r.keys()), ['a', 'c'])
        self.assertEqual(list(r.values()), [1, 3])
        self.assertEqual(list(r.items()), [('a', 1), ('c', 3)])
        self.assertEqual(r.pop('b', None), None)
        self.assertRaises(KeyError, r.pop, 'b')

    def test_mod(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        curs.execute('select 1 as a, 2 as b, 3 as c')
        r = curs.fetchone()
        r['d'] = 4
        self.assertEqual(list(r), ['a', 'b', 'c', 'd'])
        self.assertEqual(list(r.keys()), ['a', 'b', 'c', 'd'])
        self.assertEqual(list(r.values()), [1, 2, 3, 4])
        self.assertEqual(list(r.items()), [('a', 1), ('b', 2), ('c', 3), ('d', 4)])
        assert r['a'] == 1
        assert r['b'] == 2
        assert r['c'] == 3
        assert r['d'] == 4

class NamedTupleCursorTest(ConnectingTestCase):

    def setUp(self):
        if False:
            return 10
        ConnectingTestCase.setUp(self)
        self.conn = self.connect(connection_factory=NamedTupleConnection)
        curs = self.conn.cursor()
        if crdb_version(self.conn) is not None:
            curs.execute("SET experimental_enable_temp_tables = 'on'")
        curs.execute('CREATE TEMPORARY TABLE nttest (i int, s text)')
        curs.execute("INSERT INTO nttest VALUES (1, 'foo')")
        curs.execute("INSERT INTO nttest VALUES (2, 'bar')")
        curs.execute("INSERT INTO nttest VALUES (3, 'baz')")
        self.conn.commit()

    @skip_if_crdb('named cursor', version='< 22.1')
    def test_cursor_args(self):
        if False:
            for i in range(10):
                print('nop')
        cur = self.conn.cursor('foo', cursor_factory=psycopg2.extras.DictCursor)
        self.assertEqual(cur.name, 'foo')
        self.assert_(isinstance(cur, psycopg2.extras.DictCursor))

    def test_fetchone(self):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor()
        curs.execute('select * from nttest order by 1')
        t = curs.fetchone()
        self.assertEqual(t[0], 1)
        self.assertEqual(t.i, 1)
        self.assertEqual(t[1], 'foo')
        self.assertEqual(t.s, 'foo')
        self.assertEqual(curs.rownumber, 1)
        self.assertEqual(curs.rowcount, 3)

    def test_fetchmany_noarg(self):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor()
        curs.arraysize = 2
        curs.execute('select * from nttest order by 1')
        res = curs.fetchmany()
        self.assertEqual(2, len(res))
        self.assertEqual(res[0].i, 1)
        self.assertEqual(res[0].s, 'foo')
        self.assertEqual(res[1].i, 2)
        self.assertEqual(res[1].s, 'bar')
        self.assertEqual(curs.rownumber, 2)
        self.assertEqual(curs.rowcount, 3)

    def test_fetchmany(self):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor()
        curs.execute('select * from nttest order by 1')
        res = curs.fetchmany(2)
        self.assertEqual(2, len(res))
        self.assertEqual(res[0].i, 1)
        self.assertEqual(res[0].s, 'foo')
        self.assertEqual(res[1].i, 2)
        self.assertEqual(res[1].s, 'bar')
        self.assertEqual(curs.rownumber, 2)
        self.assertEqual(curs.rowcount, 3)

    def test_fetchall(self):
        if False:
            while True:
                i = 10
        curs = self.conn.cursor()
        curs.execute('select * from nttest order by 1')
        res = curs.fetchall()
        self.assertEqual(3, len(res))
        self.assertEqual(res[0].i, 1)
        self.assertEqual(res[0].s, 'foo')
        self.assertEqual(res[1].i, 2)
        self.assertEqual(res[1].s, 'bar')
        self.assertEqual(res[2].i, 3)
        self.assertEqual(res[2].s, 'baz')
        self.assertEqual(curs.rownumber, 3)
        self.assertEqual(curs.rowcount, 3)

    def test_executemany(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor()
        curs.executemany('delete from nttest where i = %s', [(1,), (2,)])
        curs.execute('select * from nttest order by 1')
        res = curs.fetchall()
        self.assertEqual(1, len(res))
        self.assertEqual(res[0].i, 3)
        self.assertEqual(res[0].s, 'baz')

    def test_iter(self):
        if False:
            return 10
        curs = self.conn.cursor()
        curs.execute('select * from nttest order by 1')
        i = iter(curs)
        self.assertEqual(curs.rownumber, 0)
        t = next(i)
        self.assertEqual(t.i, 1)
        self.assertEqual(t.s, 'foo')
        self.assertEqual(curs.rownumber, 1)
        self.assertEqual(curs.rowcount, 3)
        t = next(i)
        self.assertEqual(t.i, 2)
        self.assertEqual(t.s, 'bar')
        self.assertEqual(curs.rownumber, 2)
        self.assertEqual(curs.rowcount, 3)
        t = next(i)
        self.assertEqual(t.i, 3)
        self.assertEqual(t.s, 'baz')
        self.assertRaises(StopIteration, next, i)
        self.assertEqual(curs.rownumber, 3)
        self.assertEqual(curs.rowcount, 3)

    def test_record_updated(self):
        if False:
            while True:
                i = 10
        curs = self.conn.cursor()
        curs.execute('select 1 as foo;')
        r = curs.fetchone()
        self.assertEqual(r.foo, 1)
        curs.execute('select 2 as bar;')
        r = curs.fetchone()
        self.assertEqual(r.bar, 2)
        self.assertRaises(AttributeError, getattr, r, 'foo')

    def test_no_result_no_surprise(self):
        if False:
            return 10
        curs = self.conn.cursor()
        curs.execute('update nttest set s = s')
        self.assertRaises(psycopg2.ProgrammingError, curs.fetchone)
        curs.execute('update nttest set s = s')
        self.assertRaises(psycopg2.ProgrammingError, curs.fetchall)

    def test_bad_col_names(self):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor()
        curs.execute('select 1 as "foo.bar_baz", 2 as "?column?", 3 as "3"')
        rv = curs.fetchone()
        self.assertEqual(rv.foo_bar_baz, 1)
        self.assertEqual(rv.f_column_, 2)
        self.assertEqual(rv.f3, 3)

    @skip_before_postgres(8)
    def test_nonascii_name(self):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor()
        curs.execute('select 1 as åhé')
        rv = curs.fetchone()
        self.assertEqual(getattr(rv, 'åhé'), 1)

    def test_minimal_generation(self):
        if False:
            for i in range(10):
                print('nop')
        f_orig = NamedTupleCursor._make_nt
        calls = [0]

        def f_patched(self_):
            if False:
                for i in range(10):
                    print('nop')
            calls[0] += 1
            return f_orig(self_)
        NamedTupleCursor._make_nt = f_patched
        try:
            curs = self.conn.cursor()
            curs.execute('select * from nttest order by 1')
            curs.fetchone()
            curs.fetchone()
            curs.fetchone()
            self.assertEqual(1, calls[0])
            curs.execute('select * from nttest order by 1')
            curs.fetchone()
            curs.fetchall()
            self.assertEqual(2, calls[0])
            curs.execute('select * from nttest order by 1')
            curs.fetchone()
            curs.fetchmany(1)
            self.assertEqual(3, calls[0])
        finally:
            NamedTupleCursor._make_nt = f_orig

    @skip_if_crdb('named cursor', version='< 22.1')
    @skip_before_postgres(8, 0)
    def test_named(self):
        if False:
            return 10
        curs = self.conn.cursor('tmp')
        curs.execute('select i from generate_series(0,9) i')
        recs = []
        recs.extend(curs.fetchmany(5))
        recs.append(curs.fetchone())
        recs.extend(curs.fetchall())
        self.assertEqual(list(range(10)), [t.i for t in recs])

    @skip_if_crdb('named cursor', version='< 22.1')
    def test_named_fetchone(self):
        if False:
            i = 10
            return i + 15
        curs = self.conn.cursor('tmp')
        curs.execute('select 42 as i')
        t = curs.fetchone()
        self.assertEqual(t.i, 42)

    @skip_if_crdb('named cursor', version='< 22.1')
    def test_named_fetchmany(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor('tmp')
        curs.execute('select 42 as i')
        recs = curs.fetchmany(10)
        self.assertEqual(recs[0].i, 42)

    @skip_if_crdb('named cursor', version='< 22.1')
    def test_named_fetchall(self):
        if False:
            print('Hello World!')
        curs = self.conn.cursor('tmp')
        curs.execute('select 42 as i')
        recs = curs.fetchall()
        self.assertEqual(recs[0].i, 42)

    @skip_if_crdb('greedy cursor')
    @skip_before_postgres(8, 2)
    def test_not_greedy(self):
        if False:
            for i in range(10):
                print('nop')
        curs = self.conn.cursor('tmp')
        curs.itersize = 2
        curs.execute('select clock_timestamp() as ts from generate_series(1,3)')
        recs = []
        for t in curs:
            time.sleep(0.01)
            recs.append(t)
        self.assert_(recs[1].ts - recs[0].ts < timedelta(seconds=0.005))
        self.assert_(recs[2].ts - recs[1].ts > timedelta(seconds=0.0099))

    @skip_if_crdb('named cursor', version='< 22.1')
    @skip_before_postgres(8, 0)
    def test_named_rownumber(self):
        if False:
            i = 10
            return i + 15
        curs = self.conn.cursor('tmp')
        curs.itersize = 4
        curs.execute('select * from generate_series(1,3)')
        for (i, t) in enumerate(curs):
            self.assertEqual(i + 1, curs.rownumber)

    def test_cache(self):
        if False:
            for i in range(10):
                print('nop')
        NamedTupleCursor._cached_make_nt.cache_clear()
        curs = self.conn.cursor()
        curs.execute('select 10 as a, 20 as b')
        r1 = curs.fetchone()
        curs.execute('select 10 as a, 20 as c')
        r2 = curs.fetchone()
        curs = self.conn.cursor()
        curs.execute('select 10 as a, 30 as b')
        r3 = curs.fetchone()
        self.assert_(type(r1) is type(r3))
        self.assert_(type(r1) is not type(r2))
        cache_info = NamedTupleCursor._cached_make_nt.cache_info()
        self.assertEqual(cache_info.hits, 1)
        self.assertEqual(cache_info.misses, 2)
        self.assertEqual(cache_info.currsize, 2)

    def test_max_cache(self):
        if False:
            return 10
        old_func = NamedTupleCursor._cached_make_nt
        NamedTupleCursor._cached_make_nt = lru_cache(8)(NamedTupleCursor._cached_make_nt.__wrapped__)
        try:
            recs = []
            curs = self.conn.cursor()
            for i in range(10):
                curs.execute(f'select 1 as f{i}')
                recs.append(curs.fetchone())
            curs.execute('select 1 as f9')
            rec = curs.fetchone()
            self.assert_(any((type(r) is type(rec) for r in recs)))
            curs.execute('select 1 as f0')
            rec = curs.fetchone()
            self.assert_(all((type(r) is not type(rec) for r in recs)))
        finally:
            NamedTupleCursor._cached_make_nt = old_func

def test_suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()