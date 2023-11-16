""" Python DB API 2.0 driver compliance unit test suite.

    This software is Public Domain and may be used without restrictions.

 "Now we have booze and barflies entering the discussion, plus rumours of
  DBAs on drugs... and I won't tell you what flashes through my mind each
  time I read the subject line with 'Anal Compliance' in it.  All around
  this is turning out to be a thoroughly unwholesome unit test."

    -- Ian Bicking
"""
__rcs_id__ = '$Id: dbapi20.py,v 1.11 2005/01/02 02:41:01 zenzen Exp $'
__version__ = '$Revision: 1.12 $'[11:-2]
__author__ = 'Stuart Bishop <stuart@stuartbishop.net>'
import unittest
import time
import sys

class DatabaseAPI20Test(unittest.TestCase):
    """ Test a database self.driver for DB API 2.0 compatibility.
        This implementation tests Gadfly, but the TestCase
        is structured so that other self.drivers can subclass this
        test case to ensure compliance with the DB-API. It is
        expected that this TestCase may be expanded in the future
        if ambiguities or edge conditions are discovered.

        The 'Optional Extensions' are not yet being tested.

        self.drivers should subclass this test, overriding setUp, tearDown,
        self.driver, connect_args and connect_kw_args. Class specification
        should be as follows:

        from . import dbapi20
        class mytest(dbapi20.DatabaseAPI20Test):
           [...]

        Don't 'from .dbapi20 import DatabaseAPI20Test', or you will
        confuse the unit tester - just 'from . import dbapi20'.
    """
    driver = None
    connect_args = ()
    connect_kw_args = {}
    table_prefix = 'dbapi20test_'
    ddl1 = 'create table %sbooze (name varchar(20))' % table_prefix
    ddl2 = 'create table %sbarflys (name varchar(20))' % table_prefix
    xddl1 = 'drop table %sbooze' % table_prefix
    xddl2 = 'drop table %sbarflys' % table_prefix
    lowerfunc = 'lower'

    def executeDDL1(self, cursor):
        if False:
            print('Hello World!')
        cursor.execute(self.ddl1)

    def executeDDL2(self, cursor):
        if False:
            print('Hello World!')
        cursor.execute(self.ddl2)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ' self.drivers should override this method to perform required setup\n            if any is necessary, such as creating the database.\n        '
        pass

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ' self.drivers should override this method to perform required cleanup\n            if any is necessary, such as deleting the test database.\n            The default drops the tables that may be created.\n        '
        con = self._connect()
        try:
            cur = con.cursor()
            for ddl in (self.xddl1, self.xddl2):
                try:
                    cur.execute(ddl)
                    con.commit()
                except self.driver.Error:
                    pass
        finally:
            con.close()

    def _connect(self):
        if False:
            while True:
                i = 10
        try:
            return self.driver.connect(*self.connect_args, **self.connect_kw_args)
        except AttributeError:
            self.fail('No connect method found in self.driver module')

    def test_connect(self):
        if False:
            i = 10
            return i + 15
        con = self._connect()
        con.close()

    def test_apilevel(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            apilevel = self.driver.apilevel
            self.assertEqual(apilevel, '2.0')
        except AttributeError:
            self.fail("Driver doesn't define apilevel")

    def test_threadsafety(self):
        if False:
            print('Hello World!')
        try:
            threadsafety = self.driver.threadsafety
            self.failUnless(threadsafety in (0, 1, 2, 3))
        except AttributeError:
            self.fail("Driver doesn't define threadsafety")

    def test_paramstyle(self):
        if False:
            print('Hello World!')
        try:
            paramstyle = self.driver.paramstyle
            self.failUnless(paramstyle in ('qmark', 'numeric', 'named', 'format', 'pyformat'))
        except AttributeError:
            self.fail("Driver doesn't define paramstyle")

    def test_Exceptions(self):
        if False:
            return 10
        self.failUnless(issubclass(self.driver.Warning, Exception))
        self.failUnless(issubclass(self.driver.Error, Exception))
        self.failUnless(issubclass(self.driver.InterfaceError, self.driver.Error))
        self.failUnless(issubclass(self.driver.DatabaseError, self.driver.Error))
        self.failUnless(issubclass(self.driver.OperationalError, self.driver.Error))
        self.failUnless(issubclass(self.driver.IntegrityError, self.driver.Error))
        self.failUnless(issubclass(self.driver.InternalError, self.driver.Error))
        self.failUnless(issubclass(self.driver.ProgrammingError, self.driver.Error))
        self.failUnless(issubclass(self.driver.NotSupportedError, self.driver.Error))

    def test_ExceptionsAsConnectionAttributes(self):
        if False:
            while True:
                i = 10
        con = self._connect()
        drv = self.driver
        self.failUnless(con.Warning is drv.Warning)
        self.failUnless(con.Error is drv.Error)
        self.failUnless(con.InterfaceError is drv.InterfaceError)
        self.failUnless(con.DatabaseError is drv.DatabaseError)
        self.failUnless(con.OperationalError is drv.OperationalError)
        self.failUnless(con.IntegrityError is drv.IntegrityError)
        self.failUnless(con.InternalError is drv.InternalError)
        self.failUnless(con.ProgrammingError is drv.ProgrammingError)
        self.failUnless(con.NotSupportedError is drv.NotSupportedError)

    def test_commit(self):
        if False:
            i = 10
            return i + 15
        con = self._connect()
        try:
            con.commit()
        finally:
            con.close()

    def test_rollback(self):
        if False:
            print('Hello World!')
        con = self._connect()
        if hasattr(con, 'rollback'):
            try:
                con.rollback()
            except self.driver.NotSupportedError:
                pass

    def test_cursor(self):
        if False:
            print('Hello World!')
        con = self._connect()
        try:
            cur = con.cursor()
        finally:
            con.close()

    def test_cursor_isolation(self):
        if False:
            for i in range(10):
                print('nop')
        con = self._connect()
        try:
            cur1 = con.cursor()
            cur2 = con.cursor()
            self.executeDDL1(cur1)
            cur1.execute("insert into %sbooze values ('Victoria Bitter')" % self.table_prefix)
            cur2.execute('select name from %sbooze' % self.table_prefix)
            booze = cur2.fetchall()
            self.assertEqual(len(booze), 1)
            self.assertEqual(len(booze[0]), 1)
            self.assertEqual(booze[0][0], 'Victoria Bitter')
        finally:
            con.close()

    def test_description(self):
        if False:
            print('Hello World!')
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            self.assertEqual(cur.description, None, 'cursor.description should be none after executing a statement that can return no rows (such as DDL)')
            cur.execute('select name from %sbooze' % self.table_prefix)
            self.assertEqual(len(cur.description), 1, 'cursor.description describes too many columns')
            self.assertEqual(len(cur.description[0]), 7, 'cursor.description[x] tuples must have 7 elements')
            self.assertEqual(cur.description[0][0].lower(), 'name', 'cursor.description[x][0] must return column name')
            self.assertEqual(cur.description[0][1], self.driver.STRING, 'cursor.description[x][1] must return column type. Got %r' % cur.description[0][1])
            self.executeDDL2(cur)
            self.assertEqual(cur.description, None, 'cursor.description not being set to None when executing no-result statements (eg. DDL)')
        finally:
            con.close()

    def test_rowcount(self):
        if False:
            return 10
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            self.assertEqual(cur.rowcount, -1, 'cursor.rowcount should be -1 after executing no-result statements')
            cur.execute("insert into %sbooze values ('Victoria Bitter')" % self.table_prefix)
            self.failUnless(cur.rowcount in (-1, 1), 'cursor.rowcount should == number or rows inserted, or set to -1 after executing an insert statement')
            cur.execute('select name from %sbooze' % self.table_prefix)
            self.failUnless(cur.rowcount in (-1, 1), 'cursor.rowcount should == number of rows returned, or set to -1 after executing a select statement')
            self.executeDDL2(cur)
            self.assertEqual(cur.rowcount, -1, 'cursor.rowcount not being reset to -1 after executing no-result statements')
        finally:
            con.close()
    lower_func = 'lower'

    def test_callproc(self):
        if False:
            for i in range(10):
                print('nop')
        con = self._connect()
        try:
            cur = con.cursor()
            if self.lower_func and hasattr(cur, 'callproc'):
                r = cur.callproc(self.lower_func, ('FOO',))
                self.assertEqual(len(r), 1)
                self.assertEqual(r[0], 'FOO')
                r = cur.fetchall()
                self.assertEqual(len(r), 1, 'callproc produced no result set')
                self.assertEqual(len(r[0]), 1, 'callproc produced invalid result set')
                self.assertEqual(r[0][0], 'foo', 'callproc produced invalid results')
        finally:
            con.close()

    def test_close(self):
        if False:
            while True:
                i = 10
        con = self._connect()
        try:
            cur = con.cursor()
        finally:
            con.close()
        self.assertRaises(self.driver.Error, self.executeDDL1, cur)
        self.assertRaises(self.driver.Error, con.commit)

    def test_execute(self):
        if False:
            while True:
                i = 10
        con = self._connect()
        try:
            cur = con.cursor()
            self._paraminsert(cur)
        finally:
            con.close()

    def _paraminsert(self, cur):
        if False:
            return 10
        self.executeDDL1(cur)
        cur.execute("insert into %sbooze values ('Victoria Bitter')" % self.table_prefix)
        self.failUnless(cur.rowcount in (-1, 1))
        if self.driver.paramstyle == 'qmark':
            cur.execute('insert into %sbooze values (?)' % self.table_prefix, ("Cooper's",))
        elif self.driver.paramstyle == 'numeric':
            cur.execute('insert into %sbooze values (:1)' % self.table_prefix, ("Cooper's",))
        elif self.driver.paramstyle == 'named':
            cur.execute('insert into %sbooze values (:beer)' % self.table_prefix, {'beer': "Cooper's"})
        elif self.driver.paramstyle == 'format':
            cur.execute('insert into %sbooze values (%%s)' % self.table_prefix, ("Cooper's",))
        elif self.driver.paramstyle == 'pyformat':
            cur.execute('insert into %sbooze values (%%(beer)s)' % self.table_prefix, {'beer': "Cooper's"})
        else:
            self.fail('Invalid paramstyle')
        self.failUnless(cur.rowcount in (-1, 1))
        cur.execute('select name from %sbooze' % self.table_prefix)
        res = cur.fetchall()
        self.assertEqual(len(res), 2, 'cursor.fetchall returned too few rows')
        beers = [res[0][0], res[1][0]]
        beers.sort()
        self.assertEqual(beers[0], "Cooper's", 'cursor.fetchall retrieved incorrect data, or data inserted incorrectly')
        self.assertEqual(beers[1], 'Victoria Bitter', 'cursor.fetchall retrieved incorrect data, or data inserted incorrectly')

    def test_executemany(self):
        if False:
            for i in range(10):
                print('nop')
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            largs = [("Cooper's",), ("Boag's",)]
            margs = [{'beer': "Cooper's"}, {'beer': "Boag's"}]
            if self.driver.paramstyle == 'qmark':
                cur.executemany('insert into %sbooze values (?)' % self.table_prefix, largs)
            elif self.driver.paramstyle == 'numeric':
                cur.executemany('insert into %sbooze values (:1)' % self.table_prefix, largs)
            elif self.driver.paramstyle == 'named':
                cur.executemany('insert into %sbooze values (:beer)' % self.table_prefix, margs)
            elif self.driver.paramstyle == 'format':
                cur.executemany('insert into %sbooze values (%%s)' % self.table_prefix, largs)
            elif self.driver.paramstyle == 'pyformat':
                cur.executemany('insert into %sbooze values (%%(beer)s)' % self.table_prefix, margs)
            else:
                self.fail('Unknown paramstyle')
            self.failUnless(cur.rowcount in (-1, 2), 'insert using cursor.executemany set cursor.rowcount to incorrect value %r' % cur.rowcount)
            cur.execute('select name from %sbooze' % self.table_prefix)
            res = cur.fetchall()
            self.assertEqual(len(res), 2, 'cursor.fetchall retrieved incorrect number of rows')
            beers = [res[0][0], res[1][0]]
            beers.sort()
            self.assertEqual(beers[0], "Boag's", 'incorrect data retrieved')
            self.assertEqual(beers[1], "Cooper's", 'incorrect data retrieved')
        finally:
            con.close()

    def test_fetchone(self):
        if False:
            return 10
        con = self._connect()
        try:
            cur = con.cursor()
            self.assertRaises(self.driver.Error, cur.fetchone)
            self.executeDDL1(cur)
            self.assertRaises(self.driver.Error, cur.fetchone)
            cur.execute('select name from %sbooze' % self.table_prefix)
            self.assertEqual(cur.fetchone(), None, 'cursor.fetchone should return None if a query retrieves no rows')
            self.failUnless(cur.rowcount in (-1, 0))
            cur.execute("insert into %sbooze values ('Victoria Bitter')" % self.table_prefix)
            self.assertRaises(self.driver.Error, cur.fetchone)
            cur.execute('select name from %sbooze' % self.table_prefix)
            r = cur.fetchone()
            self.assertEqual(len(r), 1, 'cursor.fetchone should have retrieved a single row')
            self.assertEqual(r[0], 'Victoria Bitter', 'cursor.fetchone retrieved incorrect data')
            self.assertEqual(cur.fetchone(), None, 'cursor.fetchone should return None if no more rows available')
            self.failUnless(cur.rowcount in (-1, 1))
        finally:
            con.close()
    samples = ['Carlton Cold', 'Carlton Draft', 'Mountain Goat', 'Redback', 'Victoria Bitter', 'XXXX']

    def _populate(self):
        if False:
            return 10
        ' Return a list of sql commands to setup the DB for the fetch\n            tests.\n        '
        populate = [f"insert into {self.table_prefix}booze values ('{s}')" for s in self.samples]
        return populate

    def test_fetchmany(self):
        if False:
            while True:
                i = 10
        con = self._connect()
        try:
            cur = con.cursor()
            self.assertRaises(self.driver.Error, cur.fetchmany, 4)
            self.executeDDL1(cur)
            for sql in self._populate():
                cur.execute(sql)
            cur.execute('select name from %sbooze' % self.table_prefix)
            r = cur.fetchmany()
            self.assertEqual(len(r), 1, 'cursor.fetchmany retrieved incorrect number of rows, default of arraysize is one.')
            cur.arraysize = 10
            r = cur.fetchmany(3)
            self.assertEqual(len(r), 3, 'cursor.fetchmany retrieved incorrect number of rows')
            r = cur.fetchmany(4)
            self.assertEqual(len(r), 2, 'cursor.fetchmany retrieved incorrect number of rows')
            r = cur.fetchmany(4)
            self.assertEqual(len(r), 0, 'cursor.fetchmany should return an empty sequence after results are exhausted')
            self.failUnless(cur.rowcount in (-1, 6))
            cur.arraysize = 4
            cur.execute('select name from %sbooze' % self.table_prefix)
            r = cur.fetchmany()
            self.assertEqual(len(r), 4, 'cursor.arraysize not being honoured by fetchmany')
            r = cur.fetchmany()
            self.assertEqual(len(r), 2)
            r = cur.fetchmany()
            self.assertEqual(len(r), 0)
            self.failUnless(cur.rowcount in (-1, 6))
            cur.arraysize = 6
            cur.execute('select name from %sbooze' % self.table_prefix)
            rows = cur.fetchmany()
            self.failUnless(cur.rowcount in (-1, 6))
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(rows), 6)
            rows = [r[0] for r in rows]
            rows.sort()
            for i in range(0, 6):
                self.assertEqual(rows[i], self.samples[i], 'incorrect data retrieved by cursor.fetchmany')
            rows = cur.fetchmany()
            self.assertEqual(len(rows), 0, 'cursor.fetchmany should return an empty sequence if called after the whole result set has been fetched')
            self.failUnless(cur.rowcount in (-1, 6))
            self.executeDDL2(cur)
            cur.execute('select name from %sbarflys' % self.table_prefix)
            r = cur.fetchmany()
            self.assertEqual(len(r), 0, 'cursor.fetchmany should return an empty sequence if query retrieved no rows')
            self.failUnless(cur.rowcount in (-1, 0))
        finally:
            con.close()

    def test_fetchall(self):
        if False:
            return 10
        con = self._connect()
        try:
            cur = con.cursor()
            self.assertRaises(self.driver.Error, cur.fetchall)
            self.executeDDL1(cur)
            for sql in self._populate():
                cur.execute(sql)
            self.assertRaises(self.driver.Error, cur.fetchall)
            cur.execute('select name from %sbooze' % self.table_prefix)
            rows = cur.fetchall()
            self.failUnless(cur.rowcount in (-1, len(self.samples)))
            self.assertEqual(len(rows), len(self.samples), 'cursor.fetchall did not retrieve all rows')
            rows = [r[0] for r in rows]
            rows.sort()
            for i in range(0, len(self.samples)):
                self.assertEqual(rows[i], self.samples[i], 'cursor.fetchall retrieved incorrect rows')
            rows = cur.fetchall()
            self.assertEqual(len(rows), 0, 'cursor.fetchall should return an empty list if called after the whole result set has been fetched')
            self.failUnless(cur.rowcount in (-1, len(self.samples)))
            self.executeDDL2(cur)
            cur.execute('select name from %sbarflys' % self.table_prefix)
            rows = cur.fetchall()
            self.failUnless(cur.rowcount in (-1, 0))
            self.assertEqual(len(rows), 0, 'cursor.fetchall should return an empty list if a select query returns no rows')
        finally:
            con.close()

    def test_mixedfetch(self):
        if False:
            i = 10
            return i + 15
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            for sql in self._populate():
                cur.execute(sql)
            cur.execute('select name from %sbooze' % self.table_prefix)
            rows1 = cur.fetchone()
            rows23 = cur.fetchmany(2)
            rows4 = cur.fetchone()
            rows56 = cur.fetchall()
            self.failUnless(cur.rowcount in (-1, 6))
            self.assertEqual(len(rows23), 2, 'fetchmany returned incorrect number of rows')
            self.assertEqual(len(rows56), 2, 'fetchall returned incorrect number of rows')
            rows = [rows1[0]]
            rows.extend([rows23[0][0], rows23[1][0]])
            rows.append(rows4[0])
            rows.extend([rows56[0][0], rows56[1][0]])
            rows.sort()
            for i in range(0, len(self.samples)):
                self.assertEqual(rows[i], self.samples[i], 'incorrect data retrieved or inserted')
        finally:
            con.close()

    def help_nextset_setUp(self, cur):
        if False:
            return 10
        ' Should create a procedure called deleteme\n            that returns two result sets, first the\n\t    number of rows in booze then "name from booze"\n        '
        raise NotImplementedError('Helper not implemented')

    def help_nextset_tearDown(self, cur):
        if False:
            return 10
        'If cleaning up is needed after nextSetTest'
        raise NotImplementedError('Helper not implemented')

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
                assert s is None, 'No more return sets, should return None'
            finally:
                self.help_nextset_tearDown(cur)
        finally:
            con.close()

    def test_nextset(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Drivers need to override this test')

    def test_arraysize(self):
        if False:
            print('Hello World!')
        con = self._connect()
        try:
            cur = con.cursor()
            self.failUnless(hasattr(cur, 'arraysize'), 'cursor.arraysize must be defined')
        finally:
            con.close()

    def test_setinputsizes(self):
        if False:
            print('Hello World!')
        con = self._connect()
        try:
            cur = con.cursor()
            cur.setinputsizes((25,))
            self._paraminsert(cur)
        finally:
            con.close()

    def test_setoutputsize_basic(self):
        if False:
            print('Hello World!')
        con = self._connect()
        try:
            cur = con.cursor()
            cur.setoutputsize(1000)
            cur.setoutputsize(2000, 0)
            self._paraminsert(cur)
        finally:
            con.close()

    def test_setoutputsize(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('Driver needed to override this test')

    def test_None(self):
        if False:
            return 10
        con = self._connect()
        try:
            cur = con.cursor()
            self.executeDDL1(cur)
            cur.execute('insert into %sbooze values (NULL)' % self.table_prefix)
            cur.execute('select name from %sbooze' % self.table_prefix)
            r = cur.fetchall()
            self.assertEqual(len(r), 1)
            self.assertEqual(len(r[0]), 1)
            self.assertEqual(r[0][0], None, 'NULL value not returned as None')
        finally:
            con.close()

    def test_Date(self):
        if False:
            while True:
                i = 10
        d1 = self.driver.Date(2002, 12, 25)
        d2 = self.driver.DateFromTicks(time.mktime((2002, 12, 25, 0, 0, 0, 0, 0, 0)))

    def test_Time(self):
        if False:
            return 10
        t1 = self.driver.Time(13, 45, 30)
        t2 = self.driver.TimeFromTicks(time.mktime((2001, 1, 1, 13, 45, 30, 0, 0, 0)))

    def test_Timestamp(self):
        if False:
            i = 10
            return i + 15
        t1 = self.driver.Timestamp(2002, 12, 25, 13, 45, 30)
        t2 = self.driver.TimestampFromTicks(time.mktime((2002, 12, 25, 13, 45, 30, 0, 0, 0)))

    def test_Binary(self):
        if False:
            for i in range(10):
                print('nop')
        b = self.driver.Binary(b'Something')
        b = self.driver.Binary(b'')

    def test_STRING(self):
        if False:
            return 10
        self.failUnless(hasattr(self.driver, 'STRING'), 'module.STRING must be defined')

    def test_BINARY(self):
        if False:
            for i in range(10):
                print('nop')
        self.failUnless(hasattr(self.driver, 'BINARY'), 'module.BINARY must be defined.')

    def test_NUMBER(self):
        if False:
            print('Hello World!')
        self.failUnless(hasattr(self.driver, 'NUMBER'), 'module.NUMBER must be defined.')

    def test_DATETIME(self):
        if False:
            for i in range(10):
                print('nop')
        self.failUnless(hasattr(self.driver, 'DATETIME'), 'module.DATETIME must be defined.')

    def test_ROWID(self):
        if False:
            for i in range(10):
                print('nop')
        self.failUnless(hasattr(self.driver, 'ROWID'), 'module.ROWID must be defined.')