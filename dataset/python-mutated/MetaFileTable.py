from __future__ import absolute_import
import uuid
import Common.EdkLogger as EdkLogger
import Ecc.EccGlobalData as EccGlobalData
from Ecc.MetaFileWorkspace.MetaDataTable import Table
from Ecc.MetaFileWorkspace.MetaDataTable import ConvertToSqlString
from CommonDataClass.DataClass import MODEL_FILE_DSC, MODEL_FILE_DEC, MODEL_FILE_INF, MODEL_FILE_OTHERS

class MetaFileTable(Table):

    def __init__(self, Cursor, MetaFile, FileType, TableName, Temporary=False):
        if False:
            while True:
                i = 10
        self.MetaFile = MetaFile
        self.TblFile = EccGlobalData.gDb.TblFile
        if FileType == MODEL_FILE_INF:
            TableName = 'Inf'
        if FileType == MODEL_FILE_DSC:
            if Temporary:
                TableName = '_%s_%s' % ('Dsc', uuid.uuid4().hex)
            else:
                TableName = 'Dsc'
        if FileType == MODEL_FILE_DEC:
            TableName = 'Dec'
        Table.__init__(self, Cursor, TableName, 0, Temporary)
        self.Create(False)

class ModuleTable(MetaFileTable):
    _COLUMN_ = '\n        ID REAL PRIMARY KEY,\n        Model INTEGER NOT NULL,\n        Value1 TEXT NOT NULL,\n        Value2 TEXT,\n        Value3 TEXT,\n        Usage  TEXT,\n        Scope1 TEXT,\n        Scope2 TEXT,\n        BelongsToItem REAL NOT NULL,\n        BelongsToFile SINGLE NOT NULL,\n        StartLine INTEGER NOT NULL,\n        StartColumn INTEGER NOT NULL,\n        EndLine INTEGER NOT NULL,\n        EndColumn INTEGER NOT NULL,\n        Enabled INTEGER DEFAULT 0\n        '
    _DUMMY_ = "-1, -1, '====', '====', '====', '====', '====', -1, -1, -1, -1, -1, -1, -1"

    def __init__(self, Cursor):
        if False:
            return 10
        MetaFileTable.__init__(self, Cursor, '', MODEL_FILE_INF, 'Inf', False)

    def Insert(self, Model, Value1, Value2, Value3, Scope1='COMMON', Scope2='COMMON', BelongsToItem=-1, BelongsToFile=-1, StartLine=-1, StartColumn=-1, EndLine=-1, EndColumn=-1, Enabled=0, Usage=''):
        if False:
            return 10
        (Value1, Value2, Value3, Usage, Scope1, Scope2) = ConvertToSqlString((Value1, Value2, Value3, Usage, Scope1, Scope2))
        return Table.Insert(self, Model, Value1, Value2, Value3, Usage, Scope1, Scope2, BelongsToItem, BelongsToFile, StartLine, StartColumn, EndLine, EndColumn, Enabled)

    def Query(self, Model, Arch=None, Platform=None):
        if False:
            print('Hello World!')
        ConditionString = 'Model=%s AND Enabled>=0' % Model
        ValueString = 'Value1,Value2,Value3,Usage,Scope1,Scope2,ID,StartLine'
        if Arch is not None and Arch != 'COMMON':
            ConditionString += " AND (Scope1='%s' OR Scope1='COMMON')" % Arch
        if Platform is not None and Platform != 'COMMON':
            ConditionString += " AND (Scope2='%s' OR Scope2='COMMON' OR Scope2='DEFAULT')" % Platform
        SqlCommand = 'SELECT %s FROM %s WHERE %s' % (ValueString, self.Table, ConditionString)
        return self.Exec(SqlCommand)

class PackageTable(MetaFileTable):
    _COLUMN_ = '\n        ID REAL PRIMARY KEY,\n        Model INTEGER NOT NULL,\n        Value1 TEXT NOT NULL,\n        Value2 TEXT,\n        Value3 TEXT,\n        Scope1 TEXT,\n        Scope2 TEXT,\n        BelongsToItem REAL NOT NULL,\n        BelongsToFile SINGLE NOT NULL,\n        StartLine INTEGER NOT NULL,\n        StartColumn INTEGER NOT NULL,\n        EndLine INTEGER NOT NULL,\n        EndColumn INTEGER NOT NULL,\n        Enabled INTEGER DEFAULT 0\n        '
    _DUMMY_ = "-1, -1, '====', '====', '====', '====', '====', -1, -1, -1, -1, -1, -1, -1"

    def __init__(self, Cursor):
        if False:
            while True:
                i = 10
        MetaFileTable.__init__(self, Cursor, '', MODEL_FILE_DEC, 'Dec', False)

    def Insert(self, Model, Value1, Value2, Value3, Scope1='COMMON', Scope2='COMMON', BelongsToItem=-1, BelongsToFile=-1, StartLine=-1, StartColumn=-1, EndLine=-1, EndColumn=-1, Enabled=0):
        if False:
            return 10
        (Value1, Value2, Value3, Scope1, Scope2) = ConvertToSqlString((Value1, Value2, Value3, Scope1, Scope2))
        return Table.Insert(self, Model, Value1, Value2, Value3, Scope1, Scope2, BelongsToItem, BelongsToFile, StartLine, StartColumn, EndLine, EndColumn, Enabled)

    def Query(self, Model, Arch=None):
        if False:
            print('Hello World!')
        ConditionString = 'Model=%s AND Enabled>=0' % Model
        ValueString = 'Value1,Value2,Value3,Scope1,ID,StartLine'
        if Arch is not None and Arch != 'COMMON':
            ConditionString += " AND (Scope1='%s' OR Scope1='COMMON')" % Arch
        SqlCommand = 'SELECT %s FROM %s WHERE %s' % (ValueString, self.Table, ConditionString)
        return self.Exec(SqlCommand)

class PlatformTable(MetaFileTable):
    _COLUMN_ = '\n        ID REAL PRIMARY KEY,\n        Model INTEGER NOT NULL,\n        Value1 TEXT NOT NULL,\n        Value2 TEXT,\n        Value3 TEXT,\n        Scope1 TEXT,\n        Scope2 TEXT,\n        BelongsToItem REAL NOT NULL,\n        BelongsToFile SINGLE NOT NULL,\n        FromItem REAL NOT NULL,\n        StartLine INTEGER NOT NULL,\n        StartColumn INTEGER NOT NULL,\n        EndLine INTEGER NOT NULL,\n        EndColumn INTEGER NOT NULL,\n        Enabled INTEGER DEFAULT 0\n        '
    _DUMMY_ = "-1, -1, '====', '====', '====', '====', '====', -1, -1, -1, -1, -1, -1, -1, -1"

    def __init__(self, Cursor, MetaFile='', FileType=MODEL_FILE_DSC, Temporary=False):
        if False:
            while True:
                i = 10
        MetaFileTable.__init__(self, Cursor, MetaFile, FileType, 'Dsc', Temporary)

    def Insert(self, Model, Value1, Value2, Value3, Scope1='COMMON', Scope2='COMMON', BelongsToItem=-1, BelongsToFile=-1, FromItem=-1, StartLine=-1, StartColumn=-1, EndLine=-1, EndColumn=-1, Enabled=1):
        if False:
            return 10
        (Value1, Value2, Value3, Scope1, Scope2) = ConvertToSqlString((Value1, Value2, Value3, Scope1, Scope2))
        return Table.Insert(self, Model, Value1, Value2, Value3, Scope1, Scope2, BelongsToItem, BelongsToFile, FromItem, StartLine, StartColumn, EndLine, EndColumn, Enabled)

    def Query(self, Model, Scope1=None, Scope2=None, BelongsToItem=None, FromItem=None):
        if False:
            return 10
        ConditionString = 'Model=%s AND Enabled>0' % Model
        ValueString = 'Value1,Value2,Value3,Scope1,Scope2,ID,StartLine'
        if Scope1 is not None and Scope1 != 'COMMON':
            ConditionString += " AND (Scope1='%s' OR Scope1='COMMON')" % Scope1
        if Scope2 is not None and Scope2 != 'COMMON':
            ConditionString += " AND (Scope2='%s' OR Scope2='COMMON' OR Scope2='DEFAULT')" % Scope2
        if BelongsToItem is not None:
            ConditionString += ' AND BelongsToItem=%s' % BelongsToItem
        else:
            ConditionString += ' AND BelongsToItem<0'
        if FromItem is not None:
            ConditionString += ' AND FromItem=%s' % FromItem
        SqlCommand = 'SELECT %s FROM %s WHERE %s' % (ValueString, self.Table, ConditionString)
        return self.Exec(SqlCommand)

class MetaFileStorage(object):
    _FILE_TABLE_ = {MODEL_FILE_INF: ModuleTable, MODEL_FILE_DEC: PackageTable, MODEL_FILE_DSC: PlatformTable, MODEL_FILE_OTHERS: MetaFileTable}
    _FILE_TYPE_ = {'.inf': MODEL_FILE_INF, '.dec': MODEL_FILE_DEC, '.dsc': MODEL_FILE_DSC}

    def __new__(Class, Cursor, MetaFile, FileType=None, Temporary=False):
        if False:
            while True:
                i = 10
        if not FileType:
            if MetaFile.Type in self._FILE_TYPE_:
                FileType = Class._FILE_TYPE_[MetaFile.Type]
            else:
                FileType = MODEL_FILE_OTHERS
        if FileType == MODEL_FILE_OTHERS:
            Args = (Cursor, MetaFile, FileType, Temporary)
        else:
            Args = (Cursor, MetaFile, FileType, Temporary)
        return Class._FILE_TABLE_[FileType](*Args)