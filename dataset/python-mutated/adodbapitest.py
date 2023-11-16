""" Unit tests version 2.6.1.0 for adodbapi"""
'\n    adodbapi - A python DB API 2.0 interface to Microsoft ADO\n\n    Copyright (C) 2002  Henrik Ekelund\n\n    This library is free software; you can redistribute it and/or\n    modify it under the terms of the GNU Lesser General Public\n    License as published by the Free Software Foundation; either\n    version 2.1 of the License, or (at your option) any later version.\n\n    This library is distributed in the hope that it will be useful,\n    but WITHOUT ANY WARRANTY; without even the implied warranty of\n    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU\n    Lesser General Public License for more details.\n\n    You should have received a copy of the GNU Lesser General Public\n    License along with this library; if not, write to the Free Software\n    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA\n\n    Updates by Vernon Cole\n'
import copy
import datetime
import decimal
import random
import string
import sys
import unittest
try:
    import win32com.client
    win32 = True
except ImportError:
    win32 = False
import adodbapitestconfig as config
import tryconnection
import adodbapi
import adodbapi.apibase as api
try:
    import adodbapi.ado_consts as ado_consts
except ImportError:
    try:
        import ado_consts
    except ImportError:
        from adodbapi import ado_consts
long = int

def randomstring(length):
    if False:
        for i in range(10):
            print('nop')
    return ''.join([random.choice(string.ascii_letters) for n in range(32)])

class CommonDBTests(unittest.TestCase):
    """Self contained super-simple tests in easy syntax, should work on everything between mySQL and Oracle"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.engine = 'unknown'

    def getEngine(self):
        if False:
            for i in range(10):
                print('nop')
        return self.engine

    def getConnection(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def getCursor(self):
        if False:
            print('Hello World!')
        return self.getConnection().cursor()

    def testConnection(self):
        if False:
            while True:
                i = 10
        crsr = self.getCursor()
        assert crsr.__class__.__name__ == 'Cursor'

    def testErrorHandlerInherits(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.remote:
            conn = self.getConnection()
            mycallable = lambda connection, cursor, errorclass, errorvalue: 1
            conn.errorhandler = mycallable
            crsr = conn.cursor()
            assert crsr.errorhandler == mycallable, 'Error handler on crsr should be same as on connection'

    def testDefaultErrorHandlerConnection(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.remote:
            conn = self.getConnection()
            del conn.messages[:]
            try:
                conn.close()
                conn.commit()
            except:
                assert len(conn.messages) == 1
                assert len(conn.messages[0]) == 2
                assert conn.messages[0][0] == api.ProgrammingError

    def testOwnErrorHandlerConnection(self):
        if False:
            print('Hello World!')
        if self.remote:
            return
        mycallable = lambda connection, cursor, errorclass, errorvalue: 1
        conn = self.getConnection()
        conn.errorhandler = mycallable
        conn.close()
        conn.commit()
        assert len(conn.messages) == 0
        conn.errorhandler = None
        try:
            conn.close()
            conn.commit()
        except:
            pass
        assert len(conn.messages) > 0, 'Setting errorhandler to none  should bring back the standard error handler'

    def testDefaultErrorHandlerCursor(self):
        if False:
            return 10
        crsr = self.getConnection().cursor()
        if not self.remote:
            del crsr.messages[:]
            try:
                crsr.execute('SELECT abbtytddrf FROM dasdasd')
            except:
                assert len(crsr.messages) == 1
                assert len(crsr.messages[0]) == 2
                assert crsr.messages[0][0] == api.DatabaseError

    def testOwnErrorHandlerCursor(self):
        if False:
            while True:
                i = 10
        if self.remote:
            return
        mycallable = lambda connection, cursor, errorclass, errorvalue: 1
        crsr = self.getConnection().cursor()
        crsr.errorhandler = mycallable
        crsr.execute('SELECT abbtytddrf FROM dasdasd')
        assert len(crsr.messages) == 0
        crsr.errorhandler = None
        try:
            crsr.execute('SELECT abbtytddrf FROM dasdasd')
        except:
            pass
        assert len(crsr.messages) > 0, 'Setting errorhandler to none  should bring back the standard error handler'

    def testUserDefinedConversions(self):
        if False:
            i = 10
            return i + 15
        if self.remote:
            return
        try:
            duplicatingConverter = lambda aStringField: aStringField * 2
            assert duplicatingConverter('gabba') == 'gabbagabba'
            self.helpForceDropOnTblTemp()
            conn = self.getConnection()
            self.assertRaises(AttributeError, lambda x: conn.variantConversions[x], [2])
            if not self.remote:
                conn.variantConversions = copy.copy(api.variantConversions)
                crsr = conn.cursor()
                tabdef = 'CREATE TABLE xx_%s (fldData VARCHAR(100) NOT NULL, fld2 VARCHAR(20))' % config.tmp
                crsr.execute(tabdef)
                crsr.execute("INSERT INTO xx_%s(fldData,fld2) VALUES('gabba','booga')" % config.tmp)
                crsr.execute("INSERT INTO xx_%s(fldData,fld2) VALUES('hey','yo')" % config.tmp)
                conn.variantConversions[api.adoStringTypes] = duplicatingConverter
                crsr.execute('SELECT fldData,fld2 FROM xx_%s ORDER BY fldData' % config.tmp)
                rows = crsr.fetchall()
                row = rows[0]
                self.assertEqual(row[0], 'gabbagabba')
                row = rows[1]
                self.assertEqual(row[0], 'heyhey')
                self.assertEqual(row[1], 'yoyo')
                upcaseConverter = lambda aStringField: aStringField.upper()
                assert upcaseConverter('upThis') == 'UPTHIS'
                rows.converters[1] = upcaseConverter
                self.assertEqual(row[0], 'heyhey')
                self.assertEqual(row[1], 'YO')
        finally:
            try:
                del conn.variantConversions
            except:
                pass
            self.helpRollbackTblTemp()

    def testUserDefinedConversionForExactNumericTypes(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.remote and sys.version_info < (3, 0):
            oldconverter = adodbapi.variantConversions[ado_consts.adNumeric]
            try:
                adodbapi.variantConversions[ado_consts.adNumeric] = adodbapi.cvtFloat
                self.helpTestDataType('decimal(18,2)', 'NUMBER', 3.45, compareAlmostEqual=1)
                self.helpTestDataType('numeric(18,2)', 'NUMBER', 3.45, compareAlmostEqual=1)
                adodbapi.variantConversions[ado_consts.adNumeric] = adodbapi.cvtString
                self.helpTestDataType('numeric(18,2)', 'NUMBER', '3.45')
                adodbapi.variantConversions[ado_consts.adNumeric] = lambda x: '!!This function returns a funny unicode string %s!!' % x
                self.helpTestDataType('numeric(18,2)', 'NUMBER', '3.45', allowedReturnValues=['!!This function returns a funny unicode string 3.45!!'])
            finally:
                adodbapi.variantConversions[ado_consts.adNumeric] = oldconverter

    def helpTestDataType(self, sqlDataTypeString, DBAPIDataTypeString, pyData, pyDataInputAlternatives=None, compareAlmostEqual=None, allowedReturnValues=None):
        if False:
            i = 10
            return i + 15
        self.helpForceDropOnTblTemp()
        conn = self.getConnection()
        crsr = conn.cursor()
        tabdef = '\n            CREATE TABLE xx_%s (\n                fldId integer NOT NULL,\n                fldData ' % config.tmp + sqlDataTypeString + ')\n'
        crsr.execute(tabdef)
        crsr.execute('INSERT INTO xx_%s (fldId) VALUES (1)' % config.tmp)
        crsr.execute('SELECT fldId,fldData FROM xx_%s' % config.tmp)
        rs = crsr.fetchone()
        self.assertEqual(rs[1], None)
        assert rs[0] == 1
        descTuple = crsr.description[1]
        assert descTuple[0] in ['fldData', 'flddata'], 'was "%s" expected "%s"' % (descTuple[0], 'fldData')
        if DBAPIDataTypeString == 'STRING':
            assert descTuple[1] == api.STRING, 'was "%s" expected "%s"' % (descTuple[1], api.STRING.values)
        elif DBAPIDataTypeString == 'NUMBER':
            assert descTuple[1] == api.NUMBER, 'was "%s" expected "%s"' % (descTuple[1], api.NUMBER.values)
        elif DBAPIDataTypeString == 'BINARY':
            assert descTuple[1] == api.BINARY, 'was "%s" expected "%s"' % (descTuple[1], api.BINARY.values)
        elif DBAPIDataTypeString == 'DATETIME':
            assert descTuple[1] == api.DATETIME, 'was "%s" expected "%s"' % (descTuple[1], api.DATETIME.values)
        elif DBAPIDataTypeString == 'ROWID':
            assert descTuple[1] == api.ROWID, 'was "%s" expected "%s"' % (descTuple[1], api.ROWID.values)
        elif DBAPIDataTypeString == 'UUID':
            assert descTuple[1] == api.OTHER, 'was "%s" expected "%s"' % (descTuple[1], api.OTHER.values)
        else:
            raise NotImplementedError
        inputs = [pyData]
        if pyDataInputAlternatives:
            inputs.extend(pyDataInputAlternatives)
        inputs = set(inputs)
        fldId = 1
        for inParam in inputs:
            fldId += 1
            try:
                crsr.execute('INSERT INTO xx_%s (fldId,fldData) VALUES (?,?)' % config.tmp, (fldId, inParam))
            except:
                if self.remote:
                    for message in crsr.messages:
                        print(message)
                else:
                    conn.printADOerrors()
                raise
            crsr.execute('SELECT fldData FROM xx_%s WHERE ?=fldID' % config.tmp, [fldId])
            rs = crsr.fetchone()
            if allowedReturnValues:
                allowedTypes = tuple([type(aRV) for aRV in allowedReturnValues])
                assert isinstance(rs[0], allowedTypes), 'result type "%s" must be one of %s' % (type(rs[0]), allowedTypes)
            else:
                assert isinstance(rs[0], type(pyData)), 'result type "%s" must be instance of %s' % (type(rs[0]), type(pyData))
            if compareAlmostEqual and DBAPIDataTypeString == 'DATETIME':
                iso1 = adodbapi.dateconverter.DateObjectToIsoFormatString(rs[0])
                iso2 = adodbapi.dateconverter.DateObjectToIsoFormatString(pyData)
                self.assertEqual(iso1, iso2)
            elif compareAlmostEqual:
                s = float(pyData)
                v = float(rs[0])
                assert abs(v - s) / s < 1e-05, 'Values not almost equal recvd=%s, expected=%f' % (rs[0], s)
            elif allowedReturnValues:
                ok = False
                self.assertTrue(rs[0] in allowedReturnValues, 'Value "%s" not in %s' % (repr(rs[0]), allowedReturnValues))
            else:
                self.assertEqual(rs[0], pyData, 'Values are not equal recvd="%s", expected="%s"' % (rs[0], pyData))

    def testDataTypeFloat(self):
        if False:
            i = 10
            return i + 15
        self.helpTestDataType('real', 'NUMBER', 3.45, compareAlmostEqual=True)
        self.helpTestDataType('float', 'NUMBER', 1.79e+37, compareAlmostEqual=True)

    def testDataTypeDecmal(self):
        if False:
            return 10
        self.helpTestDataType('decimal(18,2)', 'NUMBER', 3.45, allowedReturnValues=['3.45', '3,45', decimal.Decimal('3.45')])
        self.helpTestDataType('numeric(18,2)', 'NUMBER', 3.45, allowedReturnValues=['3.45', '3,45', decimal.Decimal('3.45')])
        self.helpTestDataType('decimal(20,2)', 'NUMBER', 444444444444444444, allowedReturnValues=['444444444444444444.00', '444444444444444444,00', decimal.Decimal('444444444444444444')])
        if self.getEngine() == 'MSSQL':
            self.helpTestDataType('uniqueidentifier', 'UUID', '{71A4F49E-39F3-42B1-A41E-48FF154996E6}', allowedReturnValues=['{71A4F49E-39F3-42B1-A41E-48FF154996E6}'])

    def testDataTypeMoney(self):
        if False:
            return 10
        if self.getEngine() == 'MySQL':
            self.helpTestDataType('DECIMAL(20,4)', 'NUMBER', decimal.Decimal('-922337203685477.5808'))
        elif self.getEngine() == 'PostgreSQL':
            self.helpTestDataType('money', 'NUMBER', decimal.Decimal('-922337203685477.5808'), compareAlmostEqual=True, allowedReturnValues=[-922337203685477.6, decimal.Decimal('-922337203685477.5808')])
        else:
            self.helpTestDataType('smallmoney', 'NUMBER', decimal.Decimal('214748.02'))
            self.helpTestDataType('money', 'NUMBER', decimal.Decimal('-922337203685477.5808'))

    def testDataTypeInt(self):
        if False:
            i = 10
            return i + 15
        if self.getEngine() != 'PostgreSQL':
            self.helpTestDataType('tinyint', 'NUMBER', 115)
        self.helpTestDataType('smallint', 'NUMBER', -32768)
        if self.getEngine() not in ['ACCESS', 'PostgreSQL']:
            self.helpTestDataType('bit', 'NUMBER', 1)
        if self.getEngine() in ['MSSQL', 'PostgreSQL']:
            self.helpTestDataType('bigint', 'NUMBER', 3000000000, allowedReturnValues=[3000000000, int(3000000000)])
        self.helpTestDataType('int', 'NUMBER', 2147483647)

    def testDataTypeChar(self):
        if False:
            for i in range(10):
                print('nop')
        for sqlDataType in ('char(6)', 'nchar(6)'):
            self.helpTestDataType(sqlDataType, 'STRING', 'spam  ', allowedReturnValues=['spam', 'spam', 'spam  ', 'spam  '])

    def testDataTypeVarChar(self):
        if False:
            return 10
        if self.getEngine() == 'MySQL':
            stringKinds = ['varchar(10)', 'text']
        elif self.getEngine() == 'PostgreSQL':
            stringKinds = ['varchar(10)', 'text', 'character varying']
        else:
            stringKinds = ['varchar(10)', 'nvarchar(10)', 'text', 'ntext']
        for sqlDataType in stringKinds:
            self.helpTestDataType(sqlDataType, 'STRING', 'spam', ['spam'])

    def testDataTypeDate(self):
        if False:
            for i in range(10):
                print('nop')
        if self.getEngine() == 'PostgreSQL':
            dt = 'timestamp'
        else:
            dt = 'datetime'
        self.helpTestDataType(dt, 'DATETIME', adodbapi.Date(2002, 10, 28), compareAlmostEqual=True)
        if self.getEngine() not in ['MySQL', 'PostgreSQL']:
            self.helpTestDataType('smalldatetime', 'DATETIME', adodbapi.Date(2002, 10, 28), compareAlmostEqual=True)
        if tag != 'pythontime' and self.getEngine() not in ['MySQL', 'PostgreSQL']:
            self.helpTestDataType(dt, 'DATETIME', adodbapi.Timestamp(2002, 10, 28, 12, 15, 1), compareAlmostEqual=True)

    def testDataTypeBinary(self):
        if False:
            print('Hello World!')
        binfld = b'\x07\x00\xe2@*'
        arv = [binfld, adodbapi.Binary(binfld), bytes(binfld)]
        if self.getEngine() == 'PostgreSQL':
            self.helpTestDataType('bytea', 'BINARY', adodbapi.Binary(binfld), allowedReturnValues=arv)
        else:
            self.helpTestDataType('binary(5)', 'BINARY', adodbapi.Binary(binfld), allowedReturnValues=arv)
            self.helpTestDataType('varbinary(100)', 'BINARY', adodbapi.Binary(binfld), allowedReturnValues=arv)
            if self.getEngine() != 'MySQL':
                self.helpTestDataType('image', 'BINARY', adodbapi.Binary(binfld), allowedReturnValues=arv)

    def helpRollbackTblTemp(self):
        if False:
            i = 10
            return i + 15
        self.helpForceDropOnTblTemp()

    def helpForceDropOnTblTemp(self):
        if False:
            i = 10
            return i + 15
        conn = self.getConnection()
        with conn.cursor() as crsr:
            try:
                crsr.execute('DROP TABLE xx_%s' % config.tmp)
                if not conn.autocommit:
                    conn.commit()
            except:
                pass

    def helpCreateAndPopulateTableTemp(self, crsr):
        if False:
            return 10
        tabdef = '\n            CREATE TABLE xx_%s (\n                fldData INTEGER\n            )\n            ' % config.tmp
        try:
            crsr.execute(tabdef)
        except api.DatabaseError:
            self.helpForceDropOnTblTemp()
            crsr.execute(tabdef)
        for i in range(9):
            crsr.execute('INSERT INTO xx_%s (fldData) VALUES (%i)' % (config.tmp, i))

    def testFetchAll(self):
        if False:
            for i in range(10):
                print('nop')
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('SELECT fldData FROM xx_%s' % config.tmp)
        rs = crsr.fetchall()
        assert len(rs) == 9
        i = 3
        for row in rs[3:-2]:
            assert row[0] == i
            i += 1
        self.helpRollbackTblTemp()

    def testPreparedStatement(self):
        if False:
            for i in range(10):
                print('nop')
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.prepare('SELECT fldData FROM xx_%s' % config.tmp)
        crsr.execute(crsr.command)
        rs = crsr.fetchall()
        assert len(rs) == 9
        assert rs[2][0] == 2
        self.helpRollbackTblTemp()

    def testWrongPreparedStatement(self):
        if False:
            for i in range(10):
                print('nop')
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.prepare('SELECT * FROM nowhere')
        crsr.execute('SELECT fldData FROM xx_%s' % config.tmp)
        rs = crsr.fetchall()
        assert len(rs) == 9
        assert rs[2][0] == 2
        self.helpRollbackTblTemp()

    def testIterator(self):
        if False:
            for i in range(10):
                print('nop')
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('SELECT fldData FROM xx_%s' % config.tmp)
        for (i, row) in enumerate(crsr):
            assert row[0] == i
        self.helpRollbackTblTemp()

    def testExecuteMany(self):
        if False:
            return 10
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        seq_of_values = [(111,), (222,)]
        crsr.executemany('INSERT INTO xx_%s (fldData) VALUES (?)' % config.tmp, seq_of_values)
        if crsr.rowcount == -1:
            print(self.getEngine() + ' Provider does not support rowcount (on .executemany())')
        else:
            self.assertEqual(crsr.rowcount, 2)
        crsr.execute('SELECT fldData FROM xx_%s' % config.tmp)
        rs = crsr.fetchall()
        assert len(rs) == 11
        self.helpRollbackTblTemp()

    def testRowCount(self):
        if False:
            while True:
                i = 10
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('SELECT fldData FROM xx_%s' % config.tmp)
        if crsr.rowcount == -1:
            pass
        else:
            self.assertEqual(crsr.rowcount, 9)
        self.helpRollbackTblTemp()

    def testRowCountNoRecordset(self):
        if False:
            print('Hello World!')
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('DELETE FROM xx_%s WHERE fldData >= 5' % config.tmp)
        if crsr.rowcount == -1:
            print(self.getEngine() + ' Provider does not support rowcount (on DELETE)')
        else:
            self.assertEqual(crsr.rowcount, 4)
        self.helpRollbackTblTemp()

    def testFetchMany(self):
        if False:
            print('Hello World!')
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('SELECT fldData FROM xx_%s' % config.tmp)
        rs = crsr.fetchmany(3)
        assert len(rs) == 3
        rs = crsr.fetchmany(5)
        assert len(rs) == 5
        rs = crsr.fetchmany(5)
        assert len(rs) == 1
        self.helpRollbackTblTemp()

    def testFetchManyWithArraySize(self):
        if False:
            return 10
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('SELECT fldData FROM xx_%s' % config.tmp)
        rs = crsr.fetchmany()
        assert len(rs) == 1
        crsr.arraysize = 4
        rs = crsr.fetchmany()
        assert len(rs) == 4
        rs = crsr.fetchmany()
        assert len(rs) == 4
        rs = crsr.fetchmany()
        assert len(rs) == 0
        self.helpRollbackTblTemp()

    def testErrorConnect(self):
        if False:
            while True:
                i = 10
        conn = self.getConnection()
        kw = {}
        if 'proxy_host' in conn.kwargs:
            kw['proxy_host'] = conn.kwargs['proxy_host']
        conn.close()
        self.assertRaises(api.DatabaseError, self.db, 'not a valid connect string', kw)

    def testRowIterator(self):
        if False:
            print('Hello World!')
        self.helpForceDropOnTblTemp()
        conn = self.getConnection()
        crsr = conn.cursor()
        tabdef = '\n            CREATE TABLE xx_%s (\n                fldId integer NOT NULL,\n                fldTwo integer,\n                fldThree integer,\n                fldFour integer)\n                ' % config.tmp
        crsr.execute(tabdef)
        inputs = [(2, 3, 4), (102, 103, 104)]
        fldId = 1
        for inParam in inputs:
            fldId += 1
            try:
                crsr.execute('INSERT INTO xx_%s (fldId,fldTwo,fldThree,fldFour) VALUES (?,?,?,?)' % config.tmp, (fldId, inParam[0], inParam[1], inParam[2]))
            except:
                if self.remote:
                    for message in crsr.messages:
                        print(message)
                else:
                    conn.printADOerrors()
                raise
            crsr.execute('SELECT fldTwo,fldThree,fldFour FROM xx_%s WHERE ?=fldID' % config.tmp, [fldId])
            rec = crsr.fetchone()
            for j in range(len(inParam)):
                assert rec[j] == inParam[j], 'returned value:"%s" != test value:"%s"' % (rec[j], inParam[j])
            assert tuple(rec) == inParam, 'returned value:"%s" != test value:"%s"' % (repr(rec), repr(inParam))
            slice1 = tuple(rec[:-1])
            slice2 = tuple(inParam[0:2])
            assert slice1 == slice2, 'returned value:"%s" != test value:"%s"' % (repr(slice1), repr(slice2))
            assert rec['fldTwo'] == inParam[0]
            assert rec.fldThree == inParam[1]
            assert rec.fldFour == inParam[2]
        crsr.execute('select fldThree,fldFour,fldTwo from xx_%s' % config.tmp)
        recs = crsr.fetchall()
        assert recs[1][0] == 103
        assert recs[0][1] == 4
        assert recs[1]['fldFour'] == 104
        assert recs[0, 0] == 3
        assert recs[0, 'fldTwo'] == 2
        assert recs[1, 2] == 102
        for i in range(1):
            for j in range(2):
                assert recs[i][j] == recs[i, j]

    def testFormatParamstyle(self):
        if False:
            print('Hello World!')
        self.helpForceDropOnTblTemp()
        conn = self.getConnection()
        conn.paramstyle = 'format'
        crsr = conn.cursor()
        tabdef = '\n            CREATE TABLE xx_%s (\n                fldId integer NOT NULL,\n                fldData varchar(10),\n                fldConst varchar(30))\n                ' % config.tmp
        crsr.execute(tabdef)
        inputs = ['one', 'two', 'three']
        fldId = 2
        for inParam in inputs:
            fldId += 1
            sql = 'INSERT INTO xx_' + config.tmp + " (fldId,fldConst,fldData) VALUES (%s,'thi%s :may cause? trouble', %s)"
            try:
                crsr.execute(sql, (fldId, inParam))
            except:
                if self.remote:
                    for message in crsr.messages:
                        print(message)
                else:
                    conn.printADOerrors()
                raise
            crsr.execute('SELECT fldData, fldConst FROM xx_' + config.tmp + ' WHERE %s=fldID', [fldId])
            rec = crsr.fetchone()
            self.assertEqual(rec[0], inParam, 'returned value:"%s" != test value:"%s"' % (rec[0], inParam))
            self.assertEqual(rec[1], 'thi%s :may cause? trouble')
        sel = 'insert into xx_' + config.tmp + " (fldId,fldData) VALUES (%s,'four%sfive')"
        params = (20,)
        crsr.execute(sel, params)
        assert '(?,' in crsr.query, 'expected:"%s" in "%s"' % ('(?,', crsr.query)
        assert crsr.command == sel, 'expected:"%s" but found "%s"' % (sel, crsr.command)
        if not self.remote:
            self.assertEqual(crsr.parameters, params)
        crsr.execute('SELECT fldData FROM xx_%s WHERE fldID=20' % config.tmp)
        rec = crsr.fetchone()
        self.assertEqual(rec[0], 'four%sfive')

    def testNamedParamstyle(self):
        if False:
            while True:
                i = 10
        self.helpForceDropOnTblTemp()
        conn = self.getConnection()
        crsr = conn.cursor()
        crsr.paramstyle = 'named'
        tabdef = '\n            CREATE TABLE xx_%s (\n                fldId integer NOT NULL,\n                fldData varchar(10))\n                ' % config.tmp
        crsr.execute(tabdef)
        inputs = ['four', 'five', 'six']
        fldId = 10
        for inParam in inputs:
            fldId += 1
            try:
                crsr.execute('INSERT INTO xx_%s (fldId,fldData) VALUES (:Id,:f_Val)' % config.tmp, {'f_Val': inParam, 'Id': fldId})
            except:
                if self.remote:
                    for message in crsr.messages:
                        print(message)
                else:
                    conn.printADOerrors()
                raise
            crsr.execute('SELECT fldData FROM xx_%s WHERE fldID=:Id' % config.tmp, {'Id': fldId})
            rec = crsr.fetchone()
            self.assertEqual(rec[0], inParam, 'returned value:"%s" != test value:"%s"' % (rec[0], inParam))
        crsr.execute("insert into xx_%s (fldId,fldData) VALUES (:xyz,'six:five')" % config.tmp, {'xyz': 30})
        crsr.execute('SELECT fldData FROM xx_%s WHERE fldID=30' % config.tmp)
        rec = crsr.fetchone()
        self.assertEqual(rec[0], 'six:five')

    def testPyformatParamstyle(self):
        if False:
            while True:
                i = 10
        self.helpForceDropOnTblTemp()
        conn = self.getConnection()
        crsr = conn.cursor()
        crsr.paramstyle = 'pyformat'
        tabdef = '\n            CREATE TABLE xx_%s (\n                fldId integer NOT NULL,\n                fldData varchar(10))\n                ' % config.tmp
        crsr.execute(tabdef)
        inputs = ['four', 'five', 'six']
        fldId = 10
        for inParam in inputs:
            fldId += 1
            try:
                crsr.execute('INSERT INTO xx_%s (fldId,fldData) VALUES (%%(Id)s,%%(f_Val)s)' % config.tmp, {'f_Val': inParam, 'Id': fldId})
            except:
                if self.remote:
                    for message in crsr.messages:
                        print(message)
                else:
                    conn.printADOerrors()
                raise
            crsr.execute('SELECT fldData FROM xx_%s WHERE fldID=%%(Id)s' % config.tmp, {'Id': fldId})
            rec = crsr.fetchone()
            self.assertEqual(rec[0], inParam, 'returned value:"%s" != test value:"%s"' % (rec[0], inParam))
        crsr.execute("insert into xx_%s (fldId,fldData) VALUES (%%(xyz)s,'six%%five')" % config.tmp, {'xyz': 30})
        crsr.execute('SELECT fldData FROM xx_%s WHERE fldID=30' % config.tmp)
        rec = crsr.fetchone()
        self.assertEqual(rec[0], 'six%five')

    def testAutomaticParamstyle(self):
        if False:
            while True:
                i = 10
        self.helpForceDropOnTblTemp()
        conn = self.getConnection()
        conn.paramstyle = 'dynamic'
        crsr = conn.cursor()
        tabdef = '\n            CREATE TABLE xx_%s (\n                fldId integer NOT NULL,\n                fldData varchar(10),\n                fldConst varchar(30))\n                ' % config.tmp
        crsr.execute(tabdef)
        inputs = ['one', 'two', 'three']
        fldId = 2
        for inParam in inputs:
            fldId += 1
            try:
                crsr.execute('INSERT INTO xx_' + config.tmp + " (fldId,fldConst,fldData) VALUES (?,'thi%s :may cause? troub:1e', ?)", (fldId, inParam))
            except:
                if self.remote:
                    for message in crsr.messages:
                        print(message)
                else:
                    conn.printADOerrors()
                raise
            trouble = 'thi%s :may cause? troub:1e'
            crsr.execute('SELECT fldData, fldConst FROM xx_' + config.tmp + ' WHERE ?=fldID', [fldId])
            rec = crsr.fetchone()
            self.assertEqual(rec[0], inParam, 'returned value:"%s" != test value:"%s"' % (rec[0], inParam))
            self.assertEqual(rec[1], trouble)
        fldId = 10
        for inParam in inputs:
            fldId += 1
            try:
                crsr.execute('INSERT INTO xx_%s (fldId,fldData) VALUES (:Id,:f_Val)' % config.tmp, {'f_Val': inParam, 'Id': fldId})
            except:
                if self.remote:
                    for message in crsr.messages:
                        print(message)
                else:
                    conn.printADOerrors()
                raise
            crsr.execute('SELECT fldData FROM xx_%s WHERE :Id=fldID' % config.tmp, {'Id': fldId})
            rec = crsr.fetchone()
            self.assertEqual(rec[0], inParam, 'returned value:"%s" != test value:"%s"' % (rec[0], inParam))
        ppdcmd = "insert into xx_%s (fldId,fldData) VALUES (:xyz,'six:five')" % config.tmp
        crsr.prepare(ppdcmd)
        crsr.execute(ppdcmd, {'xyz': 30})
        crsr.execute('SELECT fldData FROM xx_%s WHERE fldID=30' % config.tmp)
        rec = crsr.fetchone()
        self.assertEqual(rec[0], 'six:five')

    def testRollBack(self):
        if False:
            print('Hello World!')
        conn = self.getConnection()
        crsr = conn.cursor()
        assert not crsr.connection.autocommit, 'Unexpected beginning condition'
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.connection.commit()
        crsr.execute('INSERT INTO xx_%s (fldData) VALUES(100)' % config.tmp)
        selectSql = 'SELECT fldData FROM xx_%s WHERE fldData=100' % config.tmp
        crsr.execute(selectSql)
        rs = crsr.fetchall()
        assert len(rs) == 1
        self.conn.rollback()
        crsr.execute(selectSql)
        assert crsr.fetchone() is None, 'cursor.fetchone should return None if a query retrieves no rows'
        crsr.execute('SELECT fldData from xx_%s' % config.tmp)
        rs = crsr.fetchall()
        assert len(rs) == 9, 'the original records should still be present'
        self.helpRollbackTblTemp()

    def testCommit(self):
        if False:
            print('Hello World!')
        try:
            con2 = self.getAnotherConnection()
        except NotImplementedError:
            return
        assert not con2.autocommit, 'default should be manual commit'
        crsr = con2.cursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('INSERT INTO xx_%s (fldData) VALUES(100)' % config.tmp)
        con2.commit()
        selectSql = 'SELECT fldData FROM xx_%s WHERE fldData=100' % config.tmp
        crsr.execute(selectSql)
        rs = crsr.fetchall()
        assert len(rs) == 1
        crsr.close()
        con2.close()
        conn = self.getConnection()
        crsr = self.getCursor()
        with conn.cursor() as crsr:
            crsr.execute(selectSql)
            rs = crsr.fetchall()
            assert len(rs) == 1
            assert rs[0][0] == 100
        self.helpRollbackTblTemp()

    def testAutoRollback(self):
        if False:
            return 10
        try:
            con2 = self.getAnotherConnection()
        except NotImplementedError:
            return
        assert not con2.autocommit, 'unexpected beginning condition'
        crsr = con2.cursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('INSERT INTO xx_%s (fldData) VALUES(100)' % config.tmp)
        selectSql = 'SELECT fldData FROM xx_%s WHERE fldData=100' % config.tmp
        crsr.execute(selectSql)
        rs = crsr.fetchall()
        assert len(rs) == 1
        crsr.close()
        con2.close()
        crsr = self.getCursor()
        try:
            crsr.execute(selectSql)
            row = crsr.fetchone()
        except api.DatabaseError:
            row = None
        assert row is None, 'cursor.fetchone should return None if a query retrieves no rows. Got %s' % repr(row)
        self.helpRollbackTblTemp()

    def testAutoCommit(self):
        if False:
            print('Hello World!')
        try:
            ac_conn = self.getAnotherConnection({'autocommit': True})
        except NotImplementedError:
            return
        crsr = ac_conn.cursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('INSERT INTO xx_%s (fldData) VALUES(100)' % config.tmp)
        crsr.close()
        with self.getCursor() as crsr:
            selectSql = 'SELECT fldData from xx_%s' % config.tmp
            crsr.execute(selectSql)
            rs = crsr.fetchall()
            assert len(rs) == 10, 'all records should still be present'
        ac_conn.close()
        self.helpRollbackTblTemp()

    def testSwitchedAutoCommit(self):
        if False:
            return 10
        try:
            ac_conn = self.getAnotherConnection()
        except NotImplementedError:
            return
        ac_conn.autocommit = True
        crsr = ac_conn.cursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        crsr.execute('INSERT INTO xx_%s (fldData) VALUES(100)' % config.tmp)
        crsr.close()
        conn = self.getConnection()
        ac_conn.close()
        with self.getCursor() as crsr:
            selectSql = 'SELECT fldData from xx_%s' % config.tmp
            crsr.execute(selectSql)
            rs = crsr.fetchall()
            assert len(rs) == 10, 'all records should still be present'
        self.helpRollbackTblTemp()

    def testExtendedTypeHandling(self):
        if False:
            for i in range(10):
                print('nop')

        class XtendString(str):
            pass

        class XtendInt(int):
            pass

        class XtendFloat(float):
            pass
        xs = XtendString(randomstring(30))
        xi = XtendInt(random.randint(-100, 500))
        xf = XtendFloat(random.random())
        self.helpForceDropOnTblTemp()
        conn = self.getConnection()
        crsr = conn.cursor()
        tabdef = '\n            CREATE TABLE xx_%s (\n                s VARCHAR(40) NOT NULL,\n                i INTEGER NOT NULL,\n                f REAL NOT NULL)' % config.tmp
        crsr.execute(tabdef)
        crsr.execute('INSERT INTO xx_%s (s, i, f) VALUES (?, ?, ?)' % config.tmp, (xs, xi, xf))
        crsr.close()
        conn = self.getConnection()
        with self.getCursor() as crsr:
            selectSql = 'SELECT s, i, f from xx_%s' % config.tmp
            crsr.execute(selectSql)
            row = crsr.fetchone()
            self.assertEqual(row.s, xs)
            self.assertEqual(row.i, xi)
            self.assertAlmostEqual(row.f, xf)
        self.helpRollbackTblTemp()

class TestADOwithSQLServer(CommonDBTests):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.conn = config.dbSqlServerconnect(*config.connStrSQLServer[0], **config.connStrSQLServer[1])
        self.conn.timeout = 30
        self.engine = 'MSSQL'
        self.db = config.dbSqlServerconnect
        self.remote = config.connStrSQLServer[2]

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.conn.rollback()
        except:
            pass
        try:
            self.conn.close()
        except:
            pass
        self.conn = None

    def getConnection(self):
        if False:
            print('Hello World!')
        return self.conn

    def getAnotherConnection(self, addkeys=None):
        if False:
            while True:
                i = 10
        keys = dict(config.connStrSQLServer[1])
        if addkeys:
            keys.update(addkeys)
        return config.dbSqlServerconnect(*config.connStrSQLServer[0], **keys)

    def testVariableReturningStoredProcedure(self):
        if False:
            while True:
                i = 10
        crsr = self.conn.cursor()
        spdef = '\n            CREATE PROCEDURE sp_DeleteMeOnlyForTesting\n                @theInput varchar(50),\n                @theOtherInput varchar(50),\n                @theOutput varchar(100) OUTPUT\n            AS\n                SET @theOutput=@theInput+@theOtherInput\n                    '
        try:
            crsr.execute('DROP PROCEDURE sp_DeleteMeOnlyForTesting')
            self.conn.commit()
        except:
            pass
        crsr.execute(spdef)
        retvalues = crsr.callproc('sp_DeleteMeOnlyForTesting', ('Dodsworth', 'Anne', '              '))
        assert retvalues[0] == 'Dodsworth', '%s is not "Dodsworth"' % repr(retvalues[0])
        assert retvalues[1] == 'Anne', '%s is not "Anne"' % repr(retvalues[1])
        assert retvalues[2] == 'DodsworthAnne', '%s is not "DodsworthAnne"' % repr(retvalues[2])
        self.conn.rollback()

    def testMultipleSetReturn(self):
        if False:
            while True:
                i = 10
        crsr = self.getCursor()
        self.helpCreateAndPopulateTableTemp(crsr)
        spdef = '\n            CREATE PROCEDURE sp_DeleteMe_OnlyForTesting\n            AS\n                SELECT fldData FROM xx_%s ORDER BY fldData ASC\n                SELECT fldData From xx_%s where fldData = -9999\n                SELECT fldData FROM xx_%s ORDER BY fldData DESC\n                    ' % (config.tmp, config.tmp, config.tmp)
        try:
            crsr.execute('DROP PROCEDURE sp_DeleteMe_OnlyForTesting')
            self.conn.commit()
        except:
            pass
        crsr.execute(spdef)
        retvalues = crsr.callproc('sp_DeleteMe_OnlyForTesting')
        row = crsr.fetchone()
        self.assertEqual(row[0], 0)
        assert crsr.nextset() == True, 'Operation should succeed'
        assert not crsr.fetchall(), 'Should be an empty second set'
        assert crsr.nextset() == True, 'third set should be present'
        rowdesc = crsr.fetchall()
        self.assertEqual(rowdesc[0][0], 8)
        assert crsr.nextset() is None, 'No more return sets, should return None'
        self.helpRollbackTblTemp()

    def testDatetimeProcedureParameter(self):
        if False:
            while True:
                i = 10
        crsr = self.conn.cursor()
        spdef = '\n            CREATE PROCEDURE sp_DeleteMeOnlyForTesting\n                @theInput DATETIME,\n                @theOtherInput varchar(50),\n                @theOutput varchar(100) OUTPUT\n            AS\n                SET @theOutput = CONVERT(CHARACTER(20), @theInput, 0) + @theOtherInput\n                    '
        try:
            crsr.execute('DROP PROCEDURE sp_DeleteMeOnlyForTesting')
            self.conn.commit()
        except:
            pass
        crsr.execute(spdef)
        result = crsr.callproc('sp_DeleteMeOnlyForTesting', [adodbapi.Timestamp(2014, 12, 25, 0, 1, 0), 'Beep', ' ' * 30])
        assert result[2] == 'Dec 25 2014 12:01AM Beep', 'value was="%s"' % result[2]
        self.conn.rollback()

    def testIncorrectStoredProcedureParameter(self):
        if False:
            while True:
                i = 10
        crsr = self.conn.cursor()
        spdef = '\n            CREATE PROCEDURE sp_DeleteMeOnlyForTesting\n                @theInput DATETIME,\n                @theOtherInput varchar(50),\n                @theOutput varchar(100) OUTPUT\n            AS\n                SET @theOutput = CONVERT(CHARACTER(20), @theInput) + @theOtherInput\n                    '
        try:
            crsr.execute('DROP PROCEDURE sp_DeleteMeOnlyForTesting')
            self.conn.commit()
        except:
            pass
        crsr.execute(spdef)
        result = tryconnection.try_operation_with_expected_exception((api.DataError, api.DatabaseError), crsr.callproc, ['sp_DeleteMeOnlyForTesting'], {'parameters': ['this is wrong', 'Anne', 'not Alice']})
        if result[0]:
            assert '@theInput' in str(result[1]) or 'DatabaseError' in str(result), 'Identifies the wrong erroneous parameter'
        else:
            assert result[0], result[1]
        self.conn.rollback()

class TestADOwithAccessDB(CommonDBTests):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.conn = config.dbAccessconnect(*config.connStrAccess[0], **config.connStrAccess[1])
        self.conn.timeout = 30
        self.engine = 'ACCESS'
        self.db = config.dbAccessconnect
        self.remote = config.connStrAccess[2]

    def tearDown(self):
        if False:
            print('Hello World!')
        try:
            self.conn.rollback()
        except:
            pass
        try:
            self.conn.close()
        except:
            pass
        self.conn = None

    def getConnection(self):
        if False:
            while True:
                i = 10
        return self.conn

    def getAnotherConnection(self, addkeys=None):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Jet cannot use a second connection to the database')

    def testOkConnect(self):
        if False:
            return 10
        c = self.db(*config.connStrAccess[0], **config.connStrAccess[1])
        assert c is not None
        c.close()

class TestADOwithMySql(CommonDBTests):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.conn = config.dbMySqlconnect(*config.connStrMySql[0], **config.connStrMySql[1])
        self.conn.timeout = 30
        self.engine = 'MySQL'
        self.db = config.dbMySqlconnect
        self.remote = config.connStrMySql[2]

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        try:
            self.conn.rollback()
        except:
            pass
        try:
            self.conn.close()
        except:
            pass
        self.conn = None

    def getConnection(self):
        if False:
            for i in range(10):
                print('nop')
        return self.conn

    def getAnotherConnection(self, addkeys=None):
        if False:
            for i in range(10):
                print('nop')
        keys = dict(config.connStrMySql[1])
        if addkeys:
            keys.update(addkeys)
        return config.dbMySqlconnect(*config.connStrMySql[0], **keys)

    def testOkConnect(self):
        if False:
            return 10
        c = self.db(*config.connStrMySql[0], **config.connStrMySql[1])
        assert c is not None

class TestADOwithPostgres(CommonDBTests):

    def setUp(self):
        if False:
            print('Hello World!')
        self.conn = config.dbPostgresConnect(*config.connStrPostgres[0], **config.connStrPostgres[1])
        self.conn.timeout = 30
        self.engine = 'PostgreSQL'
        self.db = config.dbPostgresConnect
        self.remote = config.connStrPostgres[2]

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.conn.rollback()
        except:
            pass
        try:
            self.conn.close()
        except:
            pass
        self.conn = None

    def getConnection(self):
        if False:
            return 10
        return self.conn

    def getAnotherConnection(self, addkeys=None):
        if False:
            for i in range(10):
                print('nop')
        keys = dict(config.connStrPostgres[1])
        if addkeys:
            keys.update(addkeys)
        return config.dbPostgresConnect(*config.connStrPostgres[0], **keys)

    def testOkConnect(self):
        if False:
            return 10
        c = self.db(*config.connStrPostgres[0], **config.connStrPostgres[1])
        assert c is not None

class TimeConverterInterfaceTest(unittest.TestCase):

    def testIDate(self):
        if False:
            while True:
                i = 10
        assert self.tc.Date(1990, 2, 2)

    def testITime(self):
        if False:
            while True:
                i = 10
        assert self.tc.Time(13, 2, 2)

    def testITimestamp(self):
        if False:
            print('Hello World!')
        assert self.tc.Timestamp(1990, 2, 2, 13, 2, 1)

    def testIDateObjectFromCOMDate(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.tc.DateObjectFromCOMDate(37435.7604282)

    def testICOMDate(self):
        if False:
            i = 10
            return i + 15
        assert hasattr(self.tc, 'COMDate')

    def testExactDate(self):
        if False:
            return 10
        d = self.tc.Date(1994, 11, 15)
        comDate = self.tc.COMDate(d)
        correct = 34653.0
        assert comDate == correct, comDate

    def testExactTimestamp(self):
        if False:
            i = 10
            return i + 15
        d = self.tc.Timestamp(1994, 11, 15, 12, 0, 0)
        comDate = self.tc.COMDate(d)
        correct = 34653.5
        self.assertEqual(comDate, correct)
        d = self.tc.Timestamp(2003, 5, 6, 14, 15, 17)
        comDate = self.tc.COMDate(d)
        correct = 37747.59394675926
        self.assertEqual(comDate, correct)

    def testIsoFormat(self):
        if False:
            while True:
                i = 10
        d = self.tc.Timestamp(1994, 11, 15, 12, 3, 10)
        iso = self.tc.DateObjectToIsoFormatString(d)
        self.assertEqual(str(iso[:19]), '1994-11-15 12:03:10')
        dt = self.tc.Date(2003, 5, 2)
        iso = self.tc.DateObjectToIsoFormatString(dt)
        self.assertEqual(str(iso[:10]), '2003-05-02')
if config.doMxDateTimeTest:
    import mx.DateTime

class TestMXDateTimeConverter(TimeConverterInterfaceTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tc = api.mxDateTimeConverter()

    def testCOMDate(self):
        if False:
            i = 10
            return i + 15
        t = mx.DateTime.DateTime(2002, 6, 28, 18, 15, 2)
        cmd = self.tc.COMDate(t)
        assert cmd == t.COMDate()

    def testDateObjectFromCOMDate(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = self.tc.DateObjectFromCOMDate(37435.7604282)
        t = mx.DateTime.DateTime(2002, 6, 28, 18, 15, 0)
        t2 = mx.DateTime.DateTime(2002, 6, 28, 18, 15, 2)
        assert t2 > cmd > t

    def testDate(self):
        if False:
            for i in range(10):
                print('nop')
        assert mx.DateTime.Date(1980, 11, 4) == self.tc.Date(1980, 11, 4)

    def testTime(self):
        if False:
            for i in range(10):
                print('nop')
        assert mx.DateTime.Time(13, 11, 4) == self.tc.Time(13, 11, 4)

    def testTimestamp(self):
        if False:
            i = 10
            return i + 15
        t = mx.DateTime.DateTime(2002, 6, 28, 18, 15, 1)
        obj = self.tc.Timestamp(2002, 6, 28, 18, 15, 1)
        assert t == obj
import time

class TestPythonTimeConverter(TimeConverterInterfaceTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tc = api.pythonTimeConverter()

    def testCOMDate(self):
        if False:
            while True:
                i = 10
        mk = time.mktime((2002, 6, 28, 18, 15, 1, 4, 31 + 28 + 31 + 30 + 31 + 28, -1))
        t = time.localtime(mk)
        cmd = self.tc.COMDate(t)
        assert abs(cmd - 37435.7604282) < 1.0 / 24, '%f more than an hour wrong' % cmd

    def testDateObjectFromCOMDate(self):
        if False:
            print('Hello World!')
        cmd = self.tc.DateObjectFromCOMDate(37435.7604282)
        t1 = time.gmtime(time.mktime((2002, 6, 28, 0, 14, 1, 4, 31 + 28 + 31 + 30 + 31 + 28, -1)))
        t2 = time.gmtime(time.mktime((2002, 6, 29, 12, 14, 2, 4, 31 + 28 + 31 + 30 + 31 + 28, -1)))
        assert t1 < cmd < t2, '"%s" should be about 2002-6-28 12:15:01' % repr(cmd)

    def testDate(self):
        if False:
            return 10
        t1 = time.mktime((2002, 6, 28, 18, 15, 1, 4, 31 + 28 + 31 + 30 + 31 + 30, 0))
        t2 = time.mktime((2002, 6, 30, 18, 15, 1, 4, 31 + 28 + 31 + 30 + 31 + 28, 0))
        obj = self.tc.Date(2002, 6, 29)
        assert t1 < time.mktime(obj) < t2, obj

    def testTime(self):
        if False:
            return 10
        self.assertEqual(self.tc.Time(18, 15, 2), time.gmtime(18 * 60 * 60 + 15 * 60 + 2))

    def testTimestamp(self):
        if False:
            for i in range(10):
                print('nop')
        t1 = time.localtime(time.mktime((2002, 6, 28, 18, 14, 1, 4, 31 + 28 + 31 + 30 + 31 + 28, -1)))
        t2 = time.localtime(time.mktime((2002, 6, 28, 18, 16, 1, 4, 31 + 28 + 31 + 30 + 31 + 28, -1)))
        obj = self.tc.Timestamp(2002, 6, 28, 18, 15, 2)
        assert t1 < obj < t2, obj

class TestPythonDateTimeConverter(TimeConverterInterfaceTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tc = api.pythonDateTimeConverter()

    def testCOMDate(self):
        if False:
            i = 10
            return i + 15
        t = datetime.datetime(2002, 6, 28, 18, 15, 1)
        cmd = self.tc.COMDate(t)
        assert abs(cmd - 37435.7604282) < 1.0 / 24, 'more than an hour wrong'

    def testDateObjectFromCOMDate(self):
        if False:
            i = 10
            return i + 15
        cmd = self.tc.DateObjectFromCOMDate(37435.7604282)
        t1 = datetime.datetime(2002, 6, 28, 18, 14, 1)
        t2 = datetime.datetime(2002, 6, 28, 18, 16, 1)
        assert t1 < cmd < t2, cmd
        tx = datetime.datetime(2002, 6, 28, 18, 14, 1, 900000)
        c1 = self.tc.DateObjectFromCOMDate(self.tc.COMDate(tx))
        assert t1 < c1 < t2, c1

    def testDate(self):
        if False:
            i = 10
            return i + 15
        t1 = datetime.date(2002, 6, 28)
        t2 = datetime.date(2002, 6, 30)
        obj = self.tc.Date(2002, 6, 29)
        assert t1 < obj < t2, obj

    def testTime(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.tc.Time(18, 15, 2).isoformat()[:8], '18:15:02')

    def testTimestamp(self):
        if False:
            return 10
        t1 = datetime.datetime(2002, 6, 28, 18, 14, 1)
        t2 = datetime.datetime(2002, 6, 28, 18, 16, 1)
        obj = self.tc.Timestamp(2002, 6, 28, 18, 15, 2)
        assert t1 < obj < t2, obj
suites = []
suites.append(unittest.makeSuite(TestPythonDateTimeConverter, 'test'))
if config.doMxDateTimeTest:
    suites.append(unittest.makeSuite(TestMXDateTimeConverter, 'test'))
if config.doTimeTest:
    suites.append(unittest.makeSuite(TestPythonTimeConverter, 'test'))
if config.doAccessTest:
    suites.append(unittest.makeSuite(TestADOwithAccessDB, 'test'))
if config.doSqlServerTest:
    suites.append(unittest.makeSuite(TestADOwithSQLServer, 'test'))
if config.doMySqlTest:
    suites.append(unittest.makeSuite(TestADOwithMySql, 'test'))
if config.doPostgresTest:
    suites.append(unittest.makeSuite(TestADOwithPostgres, 'test'))

class cleanup_manager(object):

    def __enter__(self):
        if False:
            print('Hello World!')
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        config.cleanup(config.testfolder, config.mdb_name)
suite = unittest.TestSuite(suites)
if __name__ == '__main__':
    mysuite = copy.deepcopy(suite)
    with cleanup_manager():
        defaultDateConverter = adodbapi.dateconverter
        print(__doc__)
        print('Default Date Converter is %s' % (defaultDateConverter,))
        dateconverter = defaultDateConverter
        tag = 'datetime'
        unittest.TextTestRunner().run(mysuite)
        if config.iterateOverTimeTests:
            for (test, dateconverter, tag) in ((config.doTimeTest, api.pythonTimeConverter, 'pythontime'), (config.doMxDateTimeTest, api.mxDateTimeConverter, 'mx')):
                if test:
                    mysuite = copy.deepcopy(suite)
                    adodbapi.adodbapi.dateconverter = dateconverter()
                    print('Changed dateconverter to ')
                    print(adodbapi.adodbapi.dateconverter)
                    unittest.TextTestRunner().run(mysuite)