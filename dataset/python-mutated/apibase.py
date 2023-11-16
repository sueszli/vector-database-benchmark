"""adodbapi.apibase - A python DB API 2.0 (PEP 249) interface to Microsoft ADO

Copyright (C) 2002 Henrik Ekelund, version 2.1 by Vernon Cole
* http://sourceforge.net/projects/pywin32
* http://sourceforge.net/projects/adodbapi
"""
import datetime
import decimal
import numbers
import sys
import time
from . import ado_consts as adc
verbose = False
onIronPython = sys.platform == 'cli'
if onIronPython:
    from System import DateTime, DBNull
    NullTypes = (type(None), DBNull)
else:
    DateTime = type(NotImplemented)
    NullTypes = type(None)
unicodeType = str
longType = int
StringTypes = str
makeByteBuffer = bytes
memoryViewType = memoryview
_BaseException = Exception
try:
    bytes
except NameError:
    bytes = str

def standardErrorHandler(connection, cursor, errorclass, errorvalue):
    if False:
        print('Hello World!')
    err = (errorclass, errorvalue)
    try:
        connection.messages.append(err)
    except:
        pass
    if cursor is not None:
        try:
            cursor.messages.append(err)
        except:
            pass
    raise errorclass(errorvalue)

class Error(_BaseException):
    pass

class Warning(_BaseException):
    pass

class InterfaceError(Error):
    pass

class DatabaseError(Error):
    pass

class InternalError(DatabaseError):
    pass

class OperationalError(DatabaseError):
    pass

class ProgrammingError(DatabaseError):
    pass

class IntegrityError(DatabaseError):
    pass

class DataError(DatabaseError):
    pass

class NotSupportedError(DatabaseError):
    pass

class FetchFailedError(OperationalError):
    """
    Error is used by RawStoredProcedureQuerySet to determine when a fetch
    failed due to a connection being closed or there is no record set
    returned. (Non-standard, added especially for django)
    """
    pass

class TimeConverter(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._ordinal_1899_12_31 = datetime.date(1899, 12, 31).toordinal() - 1
        self.types = {type(self.Date(2000, 1, 1)), type(self.Time(12, 1, 1)), type(self.Timestamp(2000, 1, 1, 12, 1, 1)), datetime.datetime, datetime.time, datetime.date}

    def COMDate(self, obj):
        if False:
            i = 10
            return i + 15
        'Returns a ComDate from a date-time'
        try:
            tt = obj.timetuple()
            try:
                ms = obj.microsecond
            except:
                ms = 0
            return self.ComDateFromTuple(tt, ms)
        except:
            try:
                return self.ComDateFromTuple(obj)
            except:
                try:
                    return obj.COMDate()
                except:
                    raise ValueError('Cannot convert "%s" to COMdate.' % repr(obj))

    def ComDateFromTuple(self, t, microseconds=0):
        if False:
            print('Hello World!')
        d = datetime.date(t[0], t[1], t[2])
        integerPart = d.toordinal() - self._ordinal_1899_12_31
        ms = (t[3] * 3600 + t[4] * 60 + t[5]) * 1000000 + microseconds
        fractPart = float(ms) / 86400000000.0
        return integerPart + fractPart

    def DateObjectFromCOMDate(self, comDate):
        if False:
            print('Hello World!')
        'Returns an object of the wanted type from a ComDate'
        raise NotImplementedError

    def Date(self, year, month, day):
        if False:
            i = 10
            return i + 15
        'This function constructs an object holding a date value.'
        raise NotImplementedError

    def Time(self, hour, minute, second):
        if False:
            print('Hello World!')
        'This function constructs an object holding a time value.'
        raise NotImplementedError

    def Timestamp(self, year, month, day, hour, minute, second):
        if False:
            while True:
                i = 10
        'This function constructs an object holding a time stamp value.'
        raise NotImplementedError

    def DateObjectToIsoFormatString(self, obj):
        if False:
            while True:
                i = 10
        "This function should return a string in the format 'YYYY-MM-dd HH:MM:SS:ms' (ms optional)"
        try:
            s = obj.isoformat(' ')
        except (TypeError, AttributeError):
            if isinstance(obj, datetime.date):
                s = obj.isoformat() + ' 00:00:00'
            else:
                try:
                    s = obj.strftime('%Y-%m-%d %H:%M:%S')
                except AttributeError:
                    try:
                        s = time.strftime('%Y-%m-%d %H:%M:%S', obj)
                    except:
                        raise ValueError('Cannot convert "%s" to isoformat' % repr(obj))
        return s
try:
    import mx.DateTime
    mxDateTime = True
except:
    mxDateTime = False
if mxDateTime:

    class mxDateTimeConverter(TimeConverter):

        def __init__(self):
            if False:
                print('Hello World!')
            TimeConverter.__init__(self)
            self.types.add(type(mx.DateTime))

        def DateObjectFromCOMDate(self, comDate):
            if False:
                while True:
                    i = 10
            return mx.DateTime.DateTimeFromCOMDate(comDate)

        def Date(self, year, month, day):
            if False:
                while True:
                    i = 10
            return mx.DateTime.Date(year, month, day)

        def Time(self, hour, minute, second):
            if False:
                return 10
            return mx.DateTime.Time(hour, minute, second)

        def Timestamp(self, year, month, day, hour, minute, second):
            if False:
                print('Hello World!')
            return mx.DateTime.Timestamp(year, month, day, hour, minute, second)
else:

    class mxDateTimeConverter(TimeConverter):
        pass

class pythonDateTimeConverter(TimeConverter):

    def __init__(self):
        if False:
            while True:
                i = 10
        TimeConverter.__init__(self)

    def DateObjectFromCOMDate(self, comDate):
        if False:
            print('Hello World!')
        if isinstance(comDate, datetime.datetime):
            odn = comDate.toordinal()
            tim = comDate.time()
            new = datetime.datetime.combine(datetime.datetime.fromordinal(odn), tim)
            return new
        elif isinstance(comDate, DateTime):
            fComDate = comDate.ToOADate()
        else:
            fComDate = float(comDate)
        integerPart = int(fComDate)
        floatpart = fComDate - integerPart
        dte = datetime.datetime.fromordinal(integerPart + self._ordinal_1899_12_31) + datetime.timedelta(milliseconds=floatpart * 86400000)
        return dte

    def Date(self, year, month, day):
        if False:
            i = 10
            return i + 15
        return datetime.date(year, month, day)

    def Time(self, hour, minute, second):
        if False:
            while True:
                i = 10
        return datetime.time(hour, minute, second)

    def Timestamp(self, year, month, day, hour, minute, second):
        if False:
            while True:
                i = 10
        return datetime.datetime(year, month, day, hour, minute, second)

class pythonTimeConverter(TimeConverter):

    def __init__(self):
        if False:
            return 10
        TimeConverter.__init__(self)
        self.types.add(time.struct_time)

    def DateObjectFromCOMDate(self, comDate):
        if False:
            while True:
                i = 10
        'Returns ticks since 1970'
        if isinstance(comDate, datetime.datetime):
            return comDate.timetuple()
        elif isinstance(comDate, DateTime):
            fcomDate = comDate.ToOADate()
        else:
            fcomDate = float(comDate)
        secondsperday = 86400
        t = time.gmtime(secondsperday * (fcomDate - 25569.0))
        return t

    def Date(self, year, month, day):
        if False:
            return 10
        return self.Timestamp(year, month, day, 0, 0, 0)

    def Time(self, hour, minute, second):
        if False:
            print('Hello World!')
        return time.gmtime((hour * 60 + minute) * 60 + second)

    def Timestamp(self, year, month, day, hour, minute, second):
        if False:
            print('Hello World!')
        return time.localtime(time.mktime((year, month, day, hour, minute, second, 0, 0, -1)))
base_dateconverter = pythonDateTimeConverter()
threadsafety = 1
apilevel = '2.0'
paramstyle = 'qmark'
accepted_paramstyles = ('qmark', 'named', 'format', 'pyformat', 'dynamic')
adoIntegerTypes = (adc.adInteger, adc.adSmallInt, adc.adTinyInt, adc.adUnsignedInt, adc.adUnsignedSmallInt, adc.adUnsignedTinyInt, adc.adBoolean, adc.adError)
adoRowIdTypes = (adc.adChapter,)
adoLongTypes = (adc.adBigInt, adc.adFileTime, adc.adUnsignedBigInt)
adoExactNumericTypes = (adc.adDecimal, adc.adNumeric, adc.adVarNumeric, adc.adCurrency)
adoApproximateNumericTypes = (adc.adDouble, adc.adSingle)
adoStringTypes = (adc.adBSTR, adc.adChar, adc.adLongVarChar, adc.adLongVarWChar, adc.adVarChar, adc.adVarWChar, adc.adWChar)
adoBinaryTypes = (adc.adBinary, adc.adLongVarBinary, adc.adVarBinary)
adoDateTimeTypes = (adc.adDBTime, adc.adDBTimeStamp, adc.adDate, adc.adDBDate)
adoRemainingTypes = (adc.adEmpty, adc.adIDispatch, adc.adIUnknown, adc.adPropVariant, adc.adArray, adc.adUserDefined, adc.adVariant, adc.adGUID)

class DBAPITypeObject(object):

    def __init__(self, valuesTuple):
        if False:
            while True:
                i = 10
        self.values = frozenset(valuesTuple)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return other in self.values

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return other not in self.values
'This type object is used to describe columns in a database that are string-based (e.g. CHAR). '
STRING = DBAPITypeObject(adoStringTypes)
'This type object is used to describe (long) binary columns in a database (e.g. LONG, RAW, BLOBs). '
BINARY = DBAPITypeObject(adoBinaryTypes)
'This type object is used to describe numeric columns in a database. '
NUMBER = DBAPITypeObject(adoIntegerTypes + adoLongTypes + adoExactNumericTypes + adoApproximateNumericTypes)
'This type object is used to describe date/time columns in a database. '
DATETIME = DBAPITypeObject(adoDateTimeTypes)
'This type object is used to describe the "Row ID" column in a database. '
ROWID = DBAPITypeObject(adoRowIdTypes)
OTHER = DBAPITypeObject(adoRemainingTypes)
typeMap = {memoryViewType: adc.adVarBinary, float: adc.adDouble, type(None): adc.adEmpty, str: adc.adBSTR, bool: adc.adBoolean, decimal.Decimal: adc.adDecimal, int: adc.adBigInt, bytes: adc.adVarBinary}

def pyTypeToADOType(d):
    if False:
        i = 10
        return i + 15
    tp = type(d)
    try:
        return typeMap[tp]
    except KeyError:
        from . import dateconverter
        if tp in dateconverter.types:
            return adc.adDate
        if isinstance(d, StringTypes):
            return adc.adBSTR
        if isinstance(d, numbers.Integral):
            return adc.adBigInt
        if isinstance(d, numbers.Real):
            return adc.adDouble
        raise DataError('cannot convert "%s" (type=%s) to ADO' % (repr(d), tp))

def variantConvertDate(v):
    if False:
        i = 10
        return i + 15
    from . import dateconverter
    return dateconverter.DateObjectFromCOMDate(v)

def cvtString(variant):
    if False:
        while True:
            i = 10
    if onIronPython:
        try:
            return variant.ToString()
        except:
            pass
    return str(variant)

def cvtDecimal(variant):
    if False:
        i = 10
        return i + 15
    return _convertNumberWithCulture(variant, decimal.Decimal)

def cvtNumeric(variant):
    if False:
        i = 10
        return i + 15
    return cvtDecimal(variant)

def cvtFloat(variant):
    if False:
        for i in range(10):
            print('nop')
    return _convertNumberWithCulture(variant, float)

def _convertNumberWithCulture(variant, f):
    if False:
        while True:
            i = 10
    try:
        return f(variant)
    except (ValueError, TypeError, decimal.InvalidOperation):
        try:
            europeVsUS = str(variant).replace(',', '.')
            return f(europeVsUS)
        except (ValueError, TypeError, decimal.InvalidOperation):
            pass

def cvtInt(variant):
    if False:
        for i in range(10):
            print('nop')
    return int(variant)

def cvtLong(variant):
    if False:
        print('Hello World!')
    return int(variant)

def cvtBuffer(variant):
    if False:
        return 10
    return bytes(variant)

def cvtUnicode(variant):
    if False:
        print('Hello World!')
    return str(variant)

def identity(x):
    if False:
        while True:
            i = 10
    return x

def cvtUnusual(variant):
    if False:
        return 10
    if verbose > 1:
        sys.stderr.write('Conversion called for Unusual data=%s\n' % repr(variant))
    if isinstance(variant, DateTime):
        from .adodbapi import dateconverter
        return dateconverter.DateObjectFromCOMDate(variant)
    return variant

def convert_to_python(variant, func):
    if False:
        return 10
    if isinstance(variant, NullTypes):
        return None
    return func(variant)

class MultiMap(dict):
    """A dictionary of ado.type : function -- but you can set multiple items by passing a sequence of keys"""

    def __init__(self, aDict):
        if False:
            print('Hello World!')
        for (k, v) in list(aDict.items()):
            self[k] = v

    def __setitem__(self, adoType, cvtFn):
        if False:
            for i in range(10):
                print('nop')
        'set a single item, or a whole sequence of items'
        try:
            for type in adoType:
                dict.__setitem__(self, type, cvtFn)
        except TypeError:
            dict.__setitem__(self, adoType, cvtFn)
variantConversions = MultiMap({adoDateTimeTypes: variantConvertDate, adoApproximateNumericTypes: cvtFloat, adoExactNumericTypes: cvtDecimal, adoLongTypes: cvtLong, adoIntegerTypes: cvtInt, adoRowIdTypes: cvtInt, adoStringTypes: identity, adoBinaryTypes: cvtBuffer, adoRemainingTypes: cvtUnusual})
(RS_WIN_32, RS_ARRAY, RS_REMOTE) = list(range(1, 4))

class SQLrow(object):

    def __init__(self, rows, index):
        if False:
            while True:
                i = 10
        self.rows = rows
        self.index = index

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        try:
            return self._getValue(self.rows.columnNames[name.lower()])
        except KeyError:
            raise AttributeError('Unknown column name "{}"'.format(name))

    def _getValue(self, key):
        if False:
            for i in range(10):
                print('nop')
        if self.rows.recordset_format == RS_ARRAY:
            v = self.rows.ado_results[key, self.index]
        elif self.rows.recordset_format == RS_REMOTE:
            v = self.rows.ado_results[self.index][key]
        else:
            v = self.rows.ado_results[key][self.index]
        if self.rows.converters is NotImplemented:
            return v
        return convert_to_python(v, self.rows.converters[key])

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.rows.numberOfColumns

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        if isinstance(key, int):
            try:
                return self._getValue(key)
            except IndexError:
                raise
        if isinstance(key, slice):
            indices = key.indices(self.rows.numberOfColumns)
            vl = [self._getValue(i) for i in range(*indices)]
            return tuple(vl)
        try:
            return self._getValue(self.rows.columnNames[key.lower()])
        except (KeyError, TypeError):
            (er, st, tr) = sys.exc_info()
            raise er('No such key as "%s" in %s' % (repr(key), self.__repr__())).with_traceback(tr)

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.__next__())

    def __next__(self):
        if False:
            print('Hello World!')
        for n in range(self.rows.numberOfColumns):
            yield self._getValue(n)

    def __repr__(self):
        if False:
            print('Hello World!')
        taglist = sorted(list(self.rows.columnNames.items()), key=lambda x: x[1])
        s = '<SQLrow={'
        for (name, i) in taglist:
            s += name + ':' + repr(self._getValue(i)) + ', '
        return s[:-2] + '}>'

    def __str__(self):
        if False:
            return 10
        return str(tuple((str(self._getValue(i)) for i in range(self.rows.numberOfColumns))))

class SQLrows(object):

    def __init__(self, ado_results, numberOfRows, cursor):
        if False:
            i = 10
            return i + 15
        self.ado_results = ado_results
        try:
            self.recordset_format = cursor.recordset_format
            self.numberOfColumns = cursor.numberOfColumns
            self.converters = cursor.converters
            self.columnNames = cursor.columnNames
        except AttributeError:
            self.recordset_format = RS_ARRAY
            self.numberOfColumns = 0
            self.converters = []
            self.columnNames = {}
        self.numberOfRows = numberOfRows

    def __len__(self):
        if False:
            return 10
        return self.numberOfRows

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        if not self.ado_results:
            return []
        if isinstance(item, slice):
            indices = item.indices(self.numberOfRows)
            return [SQLrow(self, k) for k in range(*indices)]
        elif isinstance(item, tuple) and len(item) == 2:
            (i, j) = item
            if not isinstance(j, int):
                try:
                    j = self.columnNames[j.lower()]
                except KeyError:
                    raise KeyError('adodbapi: no such column name as "%s"' % repr(j))
            if self.recordset_format == RS_ARRAY:
                v = self.ado_results[j, i]
            elif self.recordset_format == RS_REMOTE:
                v = self.ado_results[i][j]
            else:
                v = self.ado_results[j][i]
            if self.converters is NotImplemented:
                return v
            return convert_to_python(v, self.converters[j])
        else:
            row = SQLrow(self, item)
            return row

    def __iter__(self):
        if False:
            return 10
        return iter(self.__next__())

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        for n in range(self.numberOfRows):
            row = SQLrow(self, n)
            yield row

def changeNamedToQmark(op):
    if False:
        while True:
            i = 10
    outOp = ''
    outparms = []
    chunks = op.split("'")
    inQuotes = False
    for chunk in chunks:
        if inQuotes:
            if chunk == '':
                outOp = outOp[:-1]
            else:
                outOp += "'" + chunk + "'"
        else:
            while chunk:
                sp = chunk.split(':', 1)
                outOp += sp[0]
                s = ''
                try:
                    chunk = sp[1]
                except IndexError:
                    chunk = None
                if chunk:
                    i = 0
                    c = chunk[0]
                    while c.isalnum() or c == '_':
                        i += 1
                        try:
                            c = chunk[i]
                        except IndexError:
                            break
                    s = chunk[:i]
                    chunk = chunk[i:]
                if s:
                    outparms.append(s)
                    outOp += '?'
        inQuotes = not inQuotes
    return (outOp, outparms)

def changeFormatToQmark(op):
    if False:
        while True:
            i = 10
    outOp = ''
    outparams = []
    chunks = op.split("'")
    inQuotes = False
    for chunk in chunks:
        if inQuotes:
            if outOp != '' and chunk == '':
                outOp = outOp[:-1]
            else:
                outOp += "'" + chunk + "'"
        elif '%(' in chunk:
            while chunk:
                sp = chunk.split('%(', 1)
                outOp += sp[0]
                if len(sp) > 1:
                    try:
                        (s, chunk) = sp[1].split(')s', 1)
                    except ValueError:
                        raise ProgrammingError('Pyformat SQL has incorrect format near "%s"' % chunk)
                    outparams.append(s)
                    outOp += '?'
                else:
                    chunk = None
        else:
            sp = chunk.split('%s')
            outOp += '?'.join(sp)
        inQuotes = not inQuotes
    return (outOp, outparams)