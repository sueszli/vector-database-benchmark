from __future__ import print_function
import Common.LongFilePathOs as os
import Common.EdkLogger as EdkLogger
from CommonDataClass import DataClass
from CommonDataClass.DataClass import FileClass

def ConvertToSqlString(StringList):
    if False:
        i = 10
        return i + 15
    return map(lambda s: "'" + s.replace("'", "''") + "'", StringList)

class Table(object):
    _COLUMN_ = ''
    _ID_STEP_ = 1
    _ID_MAX_ = 2147483648
    _DUMMY_ = 0

    def __init__(self, Cursor, Name='', IdBase=0, Temporary=False):
        if False:
            i = 10
            return i + 15
        self.Cur = Cursor
        self.Table = Name
        self.IdBase = int(IdBase)
        self.ID = int(IdBase)
        self.Temporary = Temporary

    def __str__(self):
        if False:
            return 10
        return self.Table

    def Create(self, NewTable=True):
        if False:
            i = 10
            return i + 15
        if NewTable:
            self.Drop()
        if self.Temporary:
            SqlCommand = 'create temp table IF NOT EXISTS %s (%s)' % (self.Table, self._COLUMN_)
        else:
            SqlCommand = 'create table IF NOT EXISTS %s (%s)' % (self.Table, self._COLUMN_)
        EdkLogger.debug(EdkLogger.DEBUG_8, SqlCommand)
        self.Cur.execute(SqlCommand)
        self.ID = self.GetId()

    def Insert(self, *Args):
        if False:
            for i in range(10):
                print('nop')
        self.ID = self.ID + self._ID_STEP_
        if self.ID >= self.IdBase + self._ID_MAX_:
            self.ID = self.IdBase + self._ID_STEP_
        Values = ', '.join((str(Arg) for Arg in Args))
        SqlCommand = 'insert into %s values(%s, %s)' % (self.Table, self.ID, Values)
        EdkLogger.debug(EdkLogger.DEBUG_5, SqlCommand)
        self.Cur.execute(SqlCommand)
        return self.ID

    def Query(self):
        if False:
            i = 10
            return i + 15
        SqlCommand = 'select * from %s' % self.Table
        self.Cur.execute(SqlCommand)
        for Rs in self.Cur:
            EdkLogger.verbose(str(Rs))
        TotalCount = self.GetId()

    def Drop(self):
        if False:
            while True:
                i = 10
        SqlCommand = 'drop table IF EXISTS %s' % self.Table
        try:
            self.Cur.execute(SqlCommand)
        except Exception as e:
            print('An error occurred when Drop a table:', e.args[0])

    def GetCount(self):
        if False:
            return 10
        SqlCommand = 'select count(ID) from %s' % self.Table
        Record = self.Cur.execute(SqlCommand).fetchall()
        return Record[0][0]

    def GetId(self):
        if False:
            print('Hello World!')
        SqlCommand = 'select max(ID) from %s' % self.Table
        Record = self.Cur.execute(SqlCommand).fetchall()
        Id = Record[0][0]
        if Id is None:
            Id = self.IdBase
        return Id

    def InitID(self):
        if False:
            i = 10
            return i + 15
        self.ID = self.GetId()

    def Exec(self, SqlCommand):
        if False:
            print('Hello World!')
        EdkLogger.debug(EdkLogger.DEBUG_5, SqlCommand)
        self.Cur.execute(SqlCommand)
        RecordSet = self.Cur.fetchall()
        return RecordSet

    def SetEndFlag(self):
        if False:
            print('Hello World!')
        pass

    def IsIntegral(self):
        if False:
            i = 10
            return i + 15
        Result = self.Exec('select min(ID) from %s' % self.Table)
        if Result[0][0] != -1:
            return False
        return True

    def GetAll(self):
        if False:
            while True:
                i = 10
        return self.Exec('select * from %s where ID > 0 order by ID' % self.Table)

class TableDataModel(Table):
    _COLUMN_ = '\n        ID INTEGER PRIMARY KEY,\n        CrossIndex INTEGER NOT NULL,\n        Name VARCHAR NOT NULL,\n        Description VARCHAR\n        '

    def __init__(self, Cursor):
        if False:
            print('Hello World!')
        Table.__init__(self, Cursor, 'DataModel')

    def Insert(self, CrossIndex, Name, Description):
        if False:
            while True:
                i = 10
        (Name, Description) = ConvertToSqlString((Name, Description))
        return Table.Insert(self, CrossIndex, Name, Description)

    def InitTable(self):
        if False:
            return 10
        EdkLogger.verbose('\nInitialize table DataModel started ...')
        Count = self.GetCount()
        if Count is not None and Count != 0:
            return
        for Item in DataClass.MODEL_LIST:
            CrossIndex = Item[1]
            Name = Item[0]
            Description = Item[0]
            self.Insert(CrossIndex, Name, Description)
        EdkLogger.verbose('Initialize table DataModel ... DONE!')

    def GetCrossIndex(self, ModelName):
        if False:
            while True:
                i = 10
        CrossIndex = -1
        SqlCommand = "select CrossIndex from DataModel where name = '" + ModelName + "'"
        self.Cur.execute(SqlCommand)
        for Item in self.Cur:
            CrossIndex = Item[0]
        return CrossIndex