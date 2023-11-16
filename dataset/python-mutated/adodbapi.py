"""adodbapi - A python DB API 2.0 (PEP 249) interface to Microsoft ADO

Copyright (C) 2002 Henrik Ekelund, versions 2.1 and later by Vernon Cole
* http://sourceforge.net/projects/pywin32
* https://github.com/mhammond/pywin32
* http://sourceforge.net/projects/adodbapi

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    django adaptations and refactoring by Adam Vandenberg

DB-API 2.0 specification: http://www.python.org/dev/peps/pep-0249/

This module source should run correctly in CPython versions 2.7 and later,
or IronPython version 2.7 and later,
or, after running through 2to3.py, CPython 3.4 or later.
"""
__version__ = '2.6.2.0'
version = 'adodbapi v' + __version__
import copy
import decimal
import os
import sys
import weakref
from . import ado_consts as adc, apibase as api, process_connect_string
try:
    verbose = int(os.environ['ADODBAPI_VERBOSE'])
except:
    verbose = False
if verbose:
    print(version)
onWin32 = False
if api.onIronPython:
    from clr import Reference
    from System import Activator, Array, Byte, DateTime, DBNull, Decimal as SystemDecimal, Type

    def Dispatch(dispatch):
        if False:
            for i in range(10):
                print('nop')
        type = Type.GetTypeFromProgID(dispatch)
        return Activator.CreateInstance(type)

    def getIndexedValue(obj, index):
        if False:
            return 10
        return obj.Item[index]
else:
    try:
        import pythoncom
        import pywintypes
        import win32com.client
        onWin32 = True

        def Dispatch(dispatch):
            if False:
                return 10
            return win32com.client.Dispatch(dispatch)
    except ImportError:
        import warnings
        warnings.warn('pywin32 package (or IronPython) required for adodbapi.', ImportWarning)

    def getIndexedValue(obj, index):
        if False:
            print('Hello World!')
        return obj(index)
from collections.abc import Mapping
unicodeType = str
longType = int
StringTypes = str
maxint = sys.maxsize

def make_COM_connecter():
    if False:
        while True:
            i = 10
    try:
        if onWin32:
            pythoncom.CoInitialize()
        c = Dispatch('ADODB.Connection')
    except:
        raise api.InterfaceError("Windows COM Error: Dispatch('ADODB.Connection') failed.")
    return c

def connect(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Connect to a database.\n\n    call using:\n    :connection_string -- An ADODB formatted connection string, see:\n         * http://www.connectionstrings.com\n         * http://www.asp101.com/articles/john/connstring/default.asp\n    :timeout -- A command timeout value, in seconds (default 30 seconds)\n    '
    co = Connection()
    kwargs = process_connect_string.process(args, kwargs, True)
    try:
        co.connect(kwargs)
        return co
    except Exception as e:
        message = 'Error opening connection to "%s"' % co.connection_string
        raise api.OperationalError(e, message)
defaultIsolationLevel = adc.adXactReadCommitted
defaultCursorLocation = adc.adUseClient
dateconverter = api.pythonDateTimeConverter()

def format_parameters(ADOparameters, show_value=False):
    if False:
        i = 10
        return i + 15
    'Format a collection of ADO Command Parameters.\n\n    Used by error reporting in _execute_command.\n    '
    try:
        if show_value:
            desc = ['Name: %s, Dir.: %s, Type: %s, Size: %s, Value: "%s", Precision: %s, NumericScale: %s' % (p.Name, adc.directions[p.Direction], adc.adTypeNames.get(p.Type, str(p.Type) + ' (unknown type)'), p.Size, p.Value, p.Precision, p.NumericScale) for p in ADOparameters]
        else:
            desc = ['Name: %s, Dir.: %s, Type: %s, Size: %s, Precision: %s, NumericScale: %s' % (p.Name, adc.directions[p.Direction], adc.adTypeNames.get(p.Type, str(p.Type) + ' (unknown type)'), p.Size, p.Precision, p.NumericScale) for p in ADOparameters]
        return '[' + '\n'.join(desc) + ']'
    except:
        return '[]'

def _configure_parameter(p, value, adotype, settings_known):
    if False:
        return 10
    "Configure the given ADO Parameter 'p' with the Python 'value'."
    if adotype in api.adoBinaryTypes:
        p.Size = len(value)
        p.AppendChunk(value)
    elif isinstance(value, StringTypes):
        L = len(value)
        if adotype in api.adoStringTypes:
            if settings_known:
                L = min(L, p.Size)
            p.Value = value[:L]
        else:
            p.Value = value
        if L > 0:
            p.Size = L
    elif isinstance(value, decimal.Decimal):
        if api.onIronPython:
            s = str(value)
            p.Value = s
            p.Size = len(s)
        else:
            p.Value = value
        exponent = value.as_tuple()[2]
        digit_count = len(value.as_tuple()[1])
        p.Precision = digit_count
        if exponent == 0:
            p.NumericScale = 0
        elif exponent < 0:
            p.NumericScale = -exponent
            if p.Precision < p.NumericScale:
                p.Precision = p.NumericScale
        else:
            p.NumericScale = 0
            p.Precision = digit_count + exponent
    elif type(value) in dateconverter.types:
        if settings_known and adotype in api.adoDateTimeTypes:
            p.Value = dateconverter.COMDate(value)
        else:
            s = dateconverter.DateObjectToIsoFormatString(value)
            p.Value = s
            p.Size = len(s)
    elif api.onIronPython and isinstance(value, longType):
        s = str(value)
        p.Value = s
    elif adotype == adc.adEmpty:
        p.Type = adc.adInteger
        p.Value = None
    else:
        p.Value = value

class Connection(object):
    Warning = api.Warning
    Error = api.Error
    InterfaceError = api.InterfaceError
    DataError = api.DataError
    DatabaseError = api.DatabaseError
    OperationalError = api.OperationalError
    IntegrityError = api.IntegrityError
    InternalError = api.InternalError
    NotSupportedError = api.NotSupportedError
    ProgrammingError = api.ProgrammingError
    FetchFailedError = api.FetchFailedError
    verbose = api.verbose

    @property
    def dbapi(self):
        if False:
            while True:
                i = 10
        'Return a reference to the DBAPI module for this Connection.'
        return api

    def __init__(self):
        if False:
            return 10
        self.connector = None
        self.paramstyle = api.paramstyle
        self.supportsTransactions = False
        self.connection_string = ''
        self.cursors = weakref.WeakValueDictionary()
        self.dbms_name = ''
        self.dbms_version = ''
        self.errorhandler = None
        self.transaction_level = 0
        self._autocommit = False

    def connect(self, kwargs, connection_maker=make_COM_connecter):
        if False:
            print('Hello World!')
        if verbose > 9:
            print('kwargs=', repr(kwargs))
        try:
            self.connection_string = kwargs['connection_string'] % kwargs
        except Exception as e:
            self._raiseConnectionError(KeyError, 'Python string format error in connection string->')
        self.timeout = kwargs.get('timeout', 30)
        self.mode = kwargs.get('mode', adc.adModeUnknown)
        self.kwargs = kwargs
        if verbose:
            print('%s attempting: "%s"' % (version, self.connection_string))
        self.connector = connection_maker()
        self.connector.ConnectionTimeout = self.timeout
        self.connector.ConnectionString = self.connection_string
        self.connector.Mode = self.mode
        try:
            self.connector.Open()
        except api.Error:
            self._raiseConnectionError(api.DatabaseError, 'ADO error trying to Open=%s' % self.connection_string)
        try:
            if getIndexedValue(self.connector.Properties, 'Transaction DDL').Value != 0:
                self.supportsTransactions = True
        except pywintypes.com_error:
            pass
        self.dbms_name = getIndexedValue(self.connector.Properties, 'DBMS Name').Value
        try:
            self.dbms_version = getIndexedValue(self.connector.Properties, 'DBMS Version').Value
        except pywintypes.com_error:
            pass
        self.connector.CursorLocation = defaultCursorLocation
        if self.supportsTransactions:
            self.connector.IsolationLevel = defaultIsolationLevel
            self._autocommit = bool(kwargs.get('autocommit', False))
            if not self._autocommit:
                self.transaction_level = self.connector.BeginTrans()
        else:
            self._autocommit = True
        if 'paramstyle' in kwargs:
            self.paramstyle = kwargs['paramstyle']
        self.messages = []
        if verbose:
            print('adodbapi New connection at %X' % id(self))

    def _raiseConnectionError(self, errorclass, errorvalue):
        if False:
            while True:
                i = 10
        eh = self.errorhandler
        if eh is None:
            eh = api.standardErrorHandler
        eh(self, None, errorclass, errorvalue)

    def _closeAdoConnection(self):
        if False:
            print('Hello World!')
        'close the underlying ADO Connection object,\n        rolling it back first if it supports transactions.'
        if self.connector is None:
            return
        if not self._autocommit:
            if self.transaction_level:
                try:
                    self.connector.RollbackTrans()
                except:
                    pass
        self.connector.Close()
        if verbose:
            print('adodbapi Closed connection at %X' % id(self))

    def close(self):
        if False:
            return 10
        'Close the connection now (rather than whenever __del__ is called).\n\n        The connection will be unusable from this point forward;\n        an Error (or subclass) exception will be raised if any operation is attempted with the connection.\n        The same applies to all cursor objects trying to use the connection.\n        '
        for crsr in list(self.cursors.values())[:]:
            crsr.close(dont_tell_me=True)
        self.messages = []
        try:
            self._closeAdoConnection()
        except Exception as e:
            self._raiseConnectionError(sys.exc_info()[0], sys.exc_info()[1])
        self.connector = None

    def commit(self):
        if False:
            return 10
        'Commit any pending transaction to the database.\n\n        Note that if the database supports an auto-commit feature,\n        this must be initially off. An interface method may be provided to turn it back on.\n        Database modules that do not support transactions should implement this method with void functionality.\n        '
        self.messages = []
        if not self.supportsTransactions:
            return
        try:
            self.transaction_level = self.connector.CommitTrans()
            if verbose > 1:
                print('commit done on connection at %X' % id(self))
            if not (self._autocommit or self.connector.Attributes & adc.adXactAbortRetaining):
                self.transaction_level = self.connector.BeginTrans()
        except Exception as e:
            self._raiseConnectionError(api.ProgrammingError, e)

    def _rollback(self):
        if False:
            return 10
        'In case a database does provide transactions this method causes the the database to roll back to\n        the start of any pending transaction. Closing a connection without committing the changes first will\n        cause an implicit rollback to be performed.\n\n        If the database does not support the functionality required by the method, the interface should\n        throw an exception in case the method is used.\n        The preferred approach is to not implement the method and thus have Python generate\n        an AttributeError in case the method is requested. This allows the programmer to check for database\n        capabilities using the standard hasattr() function.\n\n        For some dynamically configured interfaces it may not be appropriate to require dynamically making\n        the method available. These interfaces should then raise a NotSupportedError to indicate the\n        non-ability to perform the roll back when the method is invoked.\n        '
        self.messages = []
        if self.transaction_level:
            try:
                self.transaction_level = self.connector.RollbackTrans()
                if verbose > 1:
                    print('rollback done on connection at %X' % id(self))
                if not self._autocommit and (not self.connector.Attributes & adc.adXactAbortRetaining):
                    if not self.transaction_level:
                        self.transaction_level = self.connector.BeginTrans()
            except Exception as e:
                self._raiseConnectionError(api.ProgrammingError, e)

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        if name == 'autocommit':
            if self.supportsTransactions:
                object.__setattr__(self, '_autocommit', bool(value))
                try:
                    self._rollback()
                except:
                    pass
            return
        elif name == 'paramstyle':
            if value not in api.accepted_paramstyles:
                self._raiseConnectionError(api.NotSupportedError, 'paramstyle="%s" not in:%s' % (value, repr(api.accepted_paramstyles)))
        elif name == 'variantConversions':
            value = copy.copy(value)
        object.__setattr__(self, name, value)

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        if item == 'rollback':
            if self.supportsTransactions:
                return self._rollback
            else:
                raise AttributeError('this data provider does not support Rollback')
        elif item == 'autocommit':
            return self._autocommit
        else:
            raise AttributeError('no such attribute in ADO connection object as="%s"' % item)

    def cursor(self):
        if False:
            while True:
                i = 10
        'Return a new Cursor Object using the connection.'
        self.messages = []
        c = Cursor(self)
        return c

    def _i_am_here(self, crsr):
        if False:
            while True:
                i = 10
        'message from a new cursor proclaiming its existence'
        oid = id(crsr)
        self.cursors[oid] = crsr

    def _i_am_closing(self, crsr):
        if False:
            while True:
                i = 10
        'message from a cursor giving connection a chance to clean up'
        try:
            del self.cursors[id(crsr)]
        except:
            pass

    def printADOerrors(self):
        if False:
            i = 10
            return i + 15
        j = self.connector.Errors.Count
        if j:
            print('ADO Errors:(%i)' % j)
        for e in self.connector.Errors:
            print('Description: %s' % e.Description)
            print('Error: %s %s ' % (e.Number, adc.adoErrors.get(e.Number, 'unknown')))
            if e.Number == adc.ado_error_TIMEOUT:
                print('Timeout Error: Try using adodbpi.connect(constr,timeout=Nseconds)')
            print('Source: %s' % e.Source)
            print('NativeError: %s' % e.NativeError)
            print('SQL State: %s' % e.SQLState)

    def _suggest_error_class(self):
        if False:
            for i in range(10):
                print('nop')
        'Introspect the current ADO Errors and determine an appropriate error class.\n\n        Error.SQLState is a SQL-defined error condition, per the SQL specification:\n        http://www.contrib.andrew.cmu.edu/~shadow/sql/sql1992.txt\n\n        The 23000 class of errors are integrity errors.\n        Error 40002 is a transactional integrity error.\n        '
        if self.connector is not None:
            for e in self.connector.Errors:
                state = str(e.SQLState)
                if state.startswith('23') or state == '40002':
                    return api.IntegrityError
        return api.DatabaseError

    def __del__(self):
        if False:
            while True:
                i = 10
        try:
            self._closeAdoConnection()
        except:
            pass
        self.connector = None

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        if exc_type:
            self._rollback()
        else:
            self.commit()

    def get_table_names(self):
        if False:
            for i in range(10):
                print('nop')
        schema = self.connector.OpenSchema(20)
        tables = []
        while not schema.EOF:
            name = getIndexedValue(schema.Fields, 'TABLE_NAME').Value
            tables.append(name)
            schema.MoveNext()
        del schema
        return tables

class Cursor(object):

    def __init__(self, connection):
        if False:
            return 10
        self.command = None
        self._ado_prepared = False
        self.messages = []
        self.connection = connection
        self.paramstyle = connection.paramstyle
        self._parameter_names = []
        self.recordset_is_remote = False
        self.rs = None
        self.converters = []
        self.columnNames = {}
        self.numberOfColumns = 0
        self._description = None
        self.rowcount = -1
        self.errorhandler = connection.errorhandler
        self.arraysize = 1
        connection._i_am_here(self)
        if verbose:
            print('%s New cursor at %X on conn %X' % (version, id(self), id(self.connection)))

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.fetchone, None)

    def prepare(self, operation):
        if False:
            print('Hello World!')
        self.command = operation
        self._description = None
        self._ado_prepared = 'setup'

    def __next__(self):
        if False:
            i = 10
            return i + 15
        r = self.fetchone()
        if r:
            return r
        raise StopIteration

    def __enter__(self):
        if False:
            print('Hello World!')
        'Allow database cursors to be used with context managers.'
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        'Allow database cursors to be used with context managers.'
        self.close()

    def _raiseCursorError(self, errorclass, errorvalue):
        if False:
            while True:
                i = 10
        eh = self.errorhandler
        if eh is None:
            eh = api.standardErrorHandler
        eh(self.connection, self, errorclass, errorvalue)

    def build_column_info(self, recordset):
        if False:
            i = 10
            return i + 15
        self.converters = []
        self.columnNames = {}
        self._description = None
        if recordset is None or recordset.State == adc.adStateClosed:
            self.rs = None
            self.numberOfColumns = 0
            return
        self.rs = recordset
        self.recordset_format = api.RS_ARRAY if api.onIronPython else api.RS_WIN_32
        self.numberOfColumns = recordset.Fields.Count
        try:
            varCon = self.connection.variantConversions
        except AttributeError:
            varCon = api.variantConversions
        for i in range(self.numberOfColumns):
            f = getIndexedValue(self.rs.Fields, i)
            try:
                self.converters.append(varCon[f.Type])
            except KeyError:
                self._raiseCursorError(api.InternalError, 'Data column of Unknown ADO type=%s' % f.Type)
            self.columnNames[f.Name.lower()] = i

    def _makeDescriptionFromRS(self):
        if False:
            print('Hello World!')
        if self.rs is None:
            self._description = None
            return
        desc = []
        for i in range(self.numberOfColumns):
            f = getIndexedValue(self.rs.Fields, i)
            if self.rs.EOF or self.rs.BOF:
                display_size = None
            else:
                display_size = f.ActualSize
            null_ok = bool(f.Attributes & adc.adFldMayBeNull)
            desc.append((f.Name, f.Type, display_size, f.DefinedSize, f.Precision, f.NumericScale, null_ok))
        self._description = desc

    def get_description(self):
        if False:
            i = 10
            return i + 15
        if not self._description:
            self._makeDescriptionFromRS()
        return self._description

    def __getattr__(self, item):
        if False:
            print('Hello World!')
        if item == 'description':
            return self.get_description()
        object.__getattribute__(self, item)

    def format_description(self, d):
        if False:
            return 10
        'Format db_api description tuple for printing.'
        if self.description is None:
            self._makeDescriptionFromRS()
        if isinstance(d, int):
            d = self.description[d]
        desc = 'Name= %s, Type= %s, DispSize= %s, IntSize= %s, Precision= %s, Scale= %s NullOK=%s' % (d[0], adc.adTypeNames.get(d[1], str(d[1]) + ' (unknown type)'), d[2], d[3], d[4], d[5], d[6])
        return desc

    def close(self, dont_tell_me=False):
        if False:
            i = 10
            return i + 15
        'Close the cursor now (rather than whenever __del__ is called).\n        The cursor will be unusable from this point forward; an Error (or subclass)\n        exception will be raised if any operation is attempted with the cursor.\n        '
        if self.connection is None:
            return
        self.messages = []
        if self.rs and self.rs.State != adc.adStateClosed:
            self.rs.Close()
            self.rs = None
        if not dont_tell_me:
            self.connection._i_am_closing(self)
        self.connection = None
        if verbose:
            print('adodbapi Closed cursor at %X' % id(self))

    def __del__(self):
        if False:
            i = 10
            return i + 15
        try:
            self.close()
        except:
            pass

    def _new_command(self, command_type=adc.adCmdText):
        if False:
            while True:
                i = 10
        self.cmd = None
        self.messages = []
        if self.connection is None:
            self._raiseCursorError(api.InterfaceError, None)
            return
        try:
            self.cmd = Dispatch('ADODB.Command')
            self.cmd.ActiveConnection = self.connection.connector
            self.cmd.CommandTimeout = self.connection.timeout
            self.cmd.CommandType = command_type
            self.cmd.CommandText = self.commandText
            self.cmd.Prepared = bool(self._ado_prepared)
        except:
            self._raiseCursorError(api.DatabaseError, 'Error creating new ADODB.Command object for "%s"' % repr(self.commandText))

    def _execute_command(self):
        if False:
            print('Hello World!')
        self.return_value = None
        recordset = None
        count = -1
        if verbose:
            print('Executing command="%s"' % self.commandText)
        try:
            if api.onIronPython:
                ra = Reference[int]()
                recordset = self.cmd.Execute(ra)
                count = ra.Value
            else:
                (recordset, count) = self.cmd.Execute()
        except Exception as e:
            _message = ''
            if hasattr(e, 'args'):
                _message += str(e.args) + '\n'
            _message += 'Command:\n%s\nParameters:\n%s' % (self.commandText, format_parameters(self.cmd.Parameters, True))
            klass = self.connection._suggest_error_class()
            self._raiseCursorError(klass, _message)
        try:
            self.rowcount = recordset.RecordCount
        except:
            self.rowcount = count
        self.build_column_info(recordset)

    def get_rowcount(self):
        if False:
            print('Hello World!')
        return self.rowcount

    def get_returned_parameters(self):
        if False:
            print('Hello World!')
        'with some providers, returned parameters and the .return_value are not available until\n        after the last recordset has been read.  In that case, you must coll nextset() until it\n        returns None, then call this method to get your returned information.'
        retLst = []
        for p in tuple(self.cmd.Parameters):
            if verbose > 2:
                print('Returned=Name: %s, Dir.: %s, Type: %s, Size: %s, Value: "%s", Precision: %s, NumericScale: %s' % (p.Name, adc.directions[p.Direction], adc.adTypeNames.get(p.Type, str(p.Type) + ' (unknown type)'), p.Size, p.Value, p.Precision, p.NumericScale))
            pyObject = api.convert_to_python(p.Value, api.variantConversions[p.Type])
            if p.Direction == adc.adParamReturnValue:
                self.returnValue = pyObject
                self.return_value = pyObject
            else:
                retLst.append(pyObject)
        return retLst

    def callproc(self, procname, parameters=None):
        if False:
            return 10
        'Call a stored database procedure with the given name.\n        The sequence of parameters must contain one entry for each\n        argument that the sproc expects. The result of the\n        call is returned as modified copy of the input\n        sequence.  Input parameters are left untouched, output and\n        input/output parameters replaced with possibly new values.\n\n        The sproc may also provide a result set as output,\n        which is available through the standard .fetch*() methods.\n        Extension: A "return_value" property may be set on the\n        cursor if the sproc defines an integer return value.\n        '
        self._parameter_names = []
        self.commandText = procname
        self._new_command(command_type=adc.adCmdStoredProc)
        self._buildADOparameterList(parameters, sproc=True)
        if verbose > 2:
            print('Calling Stored Proc with Params=', format_parameters(self.cmd.Parameters, True))
        self._execute_command()
        return self.get_returned_parameters()

    def _reformat_operation(self, operation, parameters):
        if False:
            i = 10
            return i + 15
        if self.paramstyle in ('format', 'pyformat'):
            (operation, self._parameter_names) = api.changeFormatToQmark(operation)
        elif self.paramstyle == 'named' or (self.paramstyle == 'dynamic' and isinstance(parameters, Mapping)):
            (operation, self._parameter_names) = api.changeNamedToQmark(operation)
        return operation

    def _buildADOparameterList(self, parameters, sproc=False):
        if False:
            print('Hello World!')
        self.parameters = parameters
        if parameters is None:
            parameters = []
        parameters_known = False
        if sproc:
            try:
                self.cmd.Parameters.Refresh()
                if verbose > 2:
                    print('ADO detected Params=', format_parameters(self.cmd.Parameters, True))
                    print('Program Parameters=', repr(parameters))
                parameters_known = True
            except api.Error:
                if verbose:
                    print('ADO Parameter Refresh failed')
                pass
            else:
                if len(parameters) != self.cmd.Parameters.Count - 1:
                    raise api.ProgrammingError('You must supply %d parameters for this stored procedure' % (self.cmd.Parameters.Count - 1))
        if sproc or parameters != []:
            i = 0
            if parameters_known:
                if self._parameter_names:
                    for (i, pm_name) in enumerate(self._parameter_names):
                        p = getIndexedValue(self.cmd.Parameters, i)
                        try:
                            _configure_parameter(p, parameters[pm_name], p.Type, parameters_known)
                        except Exception as e:
                            _message = 'Error Converting Parameter %s: %s, %s <- %s\n' % (p.Name, adc.ado_type_name(p.Type), p.Value, repr(parameters[pm_name]))
                            self._raiseCursorError(api.DataError, _message + '->' + repr(e.args))
                else:
                    for value in parameters:
                        p = getIndexedValue(self.cmd.Parameters, i)
                        if p.Direction == adc.adParamReturnValue:
                            i += 1
                            p = getIndexedValue(self.cmd.Parameters, i)
                        try:
                            _configure_parameter(p, value, p.Type, parameters_known)
                        except Exception as e:
                            _message = 'Error Converting Parameter %s: %s, %s <- %s\n' % (p.Name, adc.ado_type_name(p.Type), p.Value, repr(value))
                            self._raiseCursorError(api.DataError, _message + '->' + repr(e.args))
                        i += 1
            else:
                if self._parameter_names:
                    for parm_name in self._parameter_names:
                        elem = parameters[parm_name]
                        adotype = api.pyTypeToADOType(elem)
                        p = self.cmd.CreateParameter(parm_name, adotype, adc.adParamInput)
                        _configure_parameter(p, elem, adotype, parameters_known)
                        try:
                            self.cmd.Parameters.Append(p)
                        except Exception as e:
                            _message = 'Error Building Parameter %s: %s, %s <- %s\n' % (p.Name, adc.ado_type_name(p.Type), p.Value, repr(elem))
                            self._raiseCursorError(api.DataError, _message + '->' + repr(e.args))
                else:
                    if sproc:
                        p = self.cmd.CreateParameter('@RETURN_VALUE', adc.adInteger, adc.adParamReturnValue)
                        self.cmd.Parameters.Append(p)
                    for elem in parameters:
                        name = 'p%i' % i
                        adotype = api.pyTypeToADOType(elem)
                        p = self.cmd.CreateParameter(name, adotype, adc.adParamInput)
                        _configure_parameter(p, elem, adotype, parameters_known)
                        try:
                            self.cmd.Parameters.Append(p)
                        except Exception as e:
                            _message = 'Error Building Parameter %s: %s, %s <- %s\n' % (p.Name, adc.ado_type_name(p.Type), p.Value, repr(elem))
                            self._raiseCursorError(api.DataError, _message + '->' + repr(e.args))
                        i += 1
                if self._ado_prepared == 'setup':
                    self._ado_prepared = True

    def execute(self, operation, parameters=None):
        if False:
            print('Hello World!')
        'Prepare and execute a database operation (query or command).\n\n        Parameters may be provided as sequence or mapping and will be bound to variables in the operation.\n        Variables are specified in a database-specific notation\n        (see the module\'s paramstyle attribute for details). [5]\n        A reference to the operation will be retained by the cursor.\n        If the same operation object is passed in again, then the cursor\n        can optimize its behavior. This is most effective for algorithms\n        where the same operation is used, but different parameters are bound to it (many times).\n\n        For maximum efficiency when reusing an operation, it is best to use\n        the setinputsizes() method to specify the parameter types and sizes ahead of time.\n        It is legal for a parameter to not match the predefined information;\n        the implementation should compensate, possibly with a loss of efficiency.\n\n        The parameters may also be specified as list of tuples to e.g. insert multiple rows in\n        a single operation, but this kind of usage is depreciated: executemany() should be used instead.\n\n        Return value is not defined.\n\n        [5] The module will use the __getitem__ method of the parameters object to map either positions\n        (integers) or names (strings) to parameter values. This allows for both sequences and mappings\n        to be used as input.\n        The term "bound" refers to the process of binding an input value to a database execution buffer.\n        In practical terms, this means that the input value is directly used as a value in the operation.\n        The client should not be required to "escape" the value so that it can be used -- the value\n        should be equal to the actual database value.'
        if self.command is not operation or self._ado_prepared == 'setup' or (not hasattr(self, 'commandText')):
            if self.command is not operation:
                self._ado_prepared = False
                self.command = operation
            self._parameter_names = []
            self.commandText = operation if self.paramstyle == 'qmark' or not parameters else self._reformat_operation(operation, parameters)
        self._new_command()
        self._buildADOparameterList(parameters)
        if verbose > 3:
            print('Params=', format_parameters(self.cmd.Parameters, True))
        self._execute_command()

    def executemany(self, operation, seq_of_parameters):
        if False:
            return 10
        'Prepare a database operation (query or command)\n        and then execute it against all parameter sequences or mappings found in the sequence seq_of_parameters.\n\n            Return values are not defined.\n        '
        self.messages = list()
        total_recordcount = 0
        self.prepare(operation)
        for params in seq_of_parameters:
            self.execute(self.command, params)
            if self.rowcount == -1:
                total_recordcount = -1
            if total_recordcount != -1:
                total_recordcount += self.rowcount
        self.rowcount = total_recordcount

    def _fetch(self, limit=None):
        if False:
            while True:
                i = 10
        'Fetch rows from the current recordset.\n\n        limit -- Number of rows to fetch, or None (default) to fetch all rows.\n        '
        if self.connection is None or self.rs is None:
            self._raiseCursorError(api.FetchFailedError, 'fetch() on closed connection or empty query set')
            return
        if self.rs.State == adc.adStateClosed or self.rs.BOF or self.rs.EOF:
            return list()
        if limit:
            ado_results = self.rs.GetRows(limit)
        else:
            ado_results = self.rs.GetRows()
        if self.recordset_format == api.RS_ARRAY:
            length = len(ado_results) // self.numberOfColumns
        else:
            length = len(ado_results[0])
        fetchObject = api.SQLrows(ado_results, length, self)
        return fetchObject

    def fetchone(self):
        if False:
            while True:
                i = 10
        'Fetch the next row of a query result set, returning a single sequence,\n        or None when no more data is available.\n\n        An Error (or subclass) exception is raised if the previous call to executeXXX()\n        did not produce any result set or no call was issued yet.\n        '
        self.messages = []
        result = self._fetch(1)
        if result:
            return result[0]
        return None

    def fetchmany(self, size=None):
        if False:
            return 10
        "Fetch the next set of rows of a query result, returning a list of tuples. An empty sequence is returned when no more rows are available.\n\n        The number of rows to fetch per call is specified by the parameter.\n        If it is not given, the cursor's arraysize determines the number of rows to be fetched.\n        The method should try to fetch as many rows as indicated by the size parameter.\n        If this is not possible due to the specified number of rows not being available,\n        fewer rows may be returned.\n\n        An Error (or subclass) exception is raised if the previous call to executeXXX()\n        did not produce any result set or no call was issued yet.\n\n        Note there are performance considerations involved with the size parameter.\n        For optimal performance, it is usually best to use the arraysize attribute.\n        If the size parameter is used, then it is best for it to retain the same value from\n        one fetchmany() call to the next.\n        "
        self.messages = []
        if size is None:
            size = self.arraysize
        return self._fetch(size)

    def fetchall(self):
        if False:
            return 10
        "Fetch all (remaining) rows of a query result, returning them as a sequence of sequences (e.g. a list of tuples).\n\n        Note that the cursor's arraysize attribute\n        can affect the performance of this operation.\n        An Error (or subclass) exception is raised if the previous call to executeXXX()\n        did not produce any result set or no call was issued yet.\n        "
        self.messages = []
        return self._fetch()

    def nextset(self):
        if False:
            while True:
                i = 10
        'Skip to the next available recordset, discarding any remaining rows from the current recordset.\n\n        If there are no more sets, the method returns None. Otherwise, it returns a true\n        value and subsequent calls to the fetch methods will return rows from the next result set.\n\n        An Error (or subclass) exception is raised if the previous call to executeXXX()\n        did not produce any result set or no call was issued yet.\n        '
        self.messages = []
        if self.connection is None or self.rs is None:
            self._raiseCursorError(api.OperationalError, 'nextset() on closed connection or empty query set')
            return None
        if api.onIronPython:
            try:
                recordset = self.rs.NextRecordset()
            except TypeError:
                recordset = None
            except api.Error as exc:
                self._raiseCursorError(api.NotSupportedError, exc.args)
        else:
            try:
                rsTuple = self.rs.NextRecordset()
            except pywintypes.com_error as exc:
                self._raiseCursorError(api.NotSupportedError, exc.args)
            recordset = rsTuple[0]
        if recordset is None:
            return None
        self.build_column_info(recordset)
        return True

    def setinputsizes(self, sizes):
        if False:
            i = 10
            return i + 15
        pass

    def setoutputsize(self, size, column=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _last_query(self):
        if False:
            while True:
                i = 10
        try:
            if self.parameters is None:
                ret = self.commandText
            else:
                ret = '%s,parameters=%s' % (self.commandText, repr(self.parameters))
        except:
            ret = None
        return ret
    query = property(_last_query, None, None, 'returns the last query executed')
if __name__ == '__main__':
    raise api.ProgrammingError(version + ' cannot be run as a main program.')