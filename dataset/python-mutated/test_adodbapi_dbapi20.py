print('This module depends on the dbapi20 compliance tests created by Stuart Bishop')
print('(see db-sig mailing list history for info)')
import platform
import sys
import unittest
import dbapi20
import setuptestframework
testfolder = setuptestframework.maketemp()
if '--package' in sys.argv:
    pth = setuptestframework.makeadopackage(testfolder)
    sys.argv.remove('--package')
else:
    pth = setuptestframework.find_ado_path()
if pth not in sys.path:
    sys.path.insert(1, pth)
cleanup = setuptestframework.getcleanupfunction()
import adodbapi
import adodbapi.is64bit as is64bit
db = adodbapi
if '--verbose' in sys.argv:
    db.adodbapi.verbose = 3
print(adodbapi.version)
print('Tested with dbapi20 %s' % dbapi20.__version__)
try:
    onWindows = bool(sys.getwindowsversion())
except:
    onWindows = False
node = platform.node()
conn_kws = {}
host = 'testsql.2txt.us,1430'
instance = '%s\\SQLEXPRESS'
conn_kws['name'] = 'adotest'
conn_kws['user'] = 'adotestuser'
conn_kws['password'] = 'Sq1234567'
conn_kws['macro_auto_security'] = 'security'
if host is None:
    conn_kws['macro_getnode'] = ['host', instance]
else:
    conn_kws['host'] = host
conn_kws['provider'] = 'Provider=MSOLEDBSQL;DataTypeCompatibility=80;MARS Connection=True;'
connStr = '%(provider)s; %(security)s; Initial Catalog=%(name)s;Data Source=%(host)s'
if onWindows and node != 'z-PC':
    pass
elif node == 'xxx':
    _computername = '25.223.161.222'
    _databasename = 'adotest'
    _username = 'adotestuser'
    _password = '12345678'
    _driver = 'PostgreSQL Unicode'
    _provider = ''
    connStr = '%sDriver={%s};Server=%s;Database=%s;uid=%s;pwd=%s;' % (_provider, _driver, _computername, _databasename, _username, _password)
elif node == 'yyy':
    if is64bit.Python():
        driver = 'Microsoft.ACE.OLEDB.12.0'
    else:
        driver = 'Microsoft.Jet.OLEDB.4.0'
    testmdb = setuptestframework.makemdb(testfolder)
    connStr = 'Provider=%s;Data Source=%s' % (driver, testmdb)
else:
    conn_kws['proxy_host'] = '25.44.77.176'
    import adodbapi.remote
    db = adodbapi.remote
print('Using Connection String like=%s' % connStr)
print('Keywords=%s' % repr(conn_kws))

class test_adodbapi(dbapi20.DatabaseAPI20Test):
    driver = db
    connect_args = (connStr,)
    connect_kw_args = conn_kws

    def __init__(self, arg):
        if False:
            print('Hello World!')
        dbapi20.DatabaseAPI20Test.__init__(self, arg)

    def getTestMethodName(self):
        if False:
            i = 10
            return i + 15
        return self.id().split('.')[-1]

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        dbapi20.DatabaseAPI20Test.setUp(self)
        if self.getTestMethodName() == 'test_callproc':
            con = self._connect()
            engine = con.dbms_name
            if engine != 'MS Jet':
                sql = '\n                    create procedure templower\n                        @theData varchar(50)\n                    as\n                        select lower(@theData)\n                '
            else:
                sql = '\n                    create procedure templower\n                        (theData varchar(50))\n                    as\n                        select lower(theData);\n                '
            cur = con.cursor()
            try:
                cur.execute(sql)
                con.commit()
            except:
                pass
            cur.close()
            con.close()
            self.lower_func = 'templower'

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.getTestMethodName() == 'test_callproc':
            con = self._connect()
            cur = con.cursor()
            try:
                cur.execute('drop procedure templower')
            except:
                pass
            con.commit()
        dbapi20.DatabaseAPI20Test.tearDown(self)

    def help_nextset_setUp(self, cur):
        if False:
            while True:
                i = 10
        'Should create a procedure called deleteme'
        'that returns two result sets, first the number of rows in booze then "name from booze"'
        sql = '\n            create procedure deleteme as\n            begin\n                select count(*) from %sbooze\n                select name from %sbooze\n            end\n        ' % (self.table_prefix, self.table_prefix)
        cur.execute(sql)

    def help_nextset_tearDown(self, cur):
        if False:
            print('Hello World!')
        'If cleaning up is needed after nextSetTest'
        try:
            cur.execute('drop procedure deleteme')
        except:
            pass

    def test_nextset(self):
        if False:
            return 10
        con = self._connect()
        try:
            cur = con.cursor()
            stmts = [self.ddl1] + self._populate()
            for sql in stmts:
                cur.execute(sql)
            self.help_nextset_setUp(cur)
            cur.callproc('deleteme')
            numberofrows = cur.fetchone()
            assert numberofrows[0] == 6
            assert cur.nextset()
            names = cur.fetchall()
            assert len(names) == len(self.samples)
            s = cur.nextset()
            assert s is None, 'No more return sets, should return None'
        finally:
            try:
                self.help_nextset_tearDown(cur)
            finally:
                con.close()

    def test_setoutputsize(self):
        if False:
            print('Hello World!')
        pass
if __name__ == '__main__':
    unittest.main()
    cleanup(testfolder, None)