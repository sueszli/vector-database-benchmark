from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
import CommonDataClass.DataClass as DataClass
from Table.Table import Table
from Common.StringUtils import ConvertToSqlString

class TableDataModel(Table):

    def __init__(self, Cursor):
        if False:
            print('Hello World!')
        Table.__init__(self, Cursor)
        self.Table = 'DataModel'

    def Create(self):
        if False:
            i = 10
            return i + 15
        SqlCommand = 'create table IF NOT EXISTS %s (ID INTEGER PRIMARY KEY,\n                                                       CrossIndex INTEGER NOT NULL,\n                                                       Name VARCHAR NOT NULL,\n                                                       Description VARCHAR\n                                                      )' % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, CrossIndex, Name, Description):
        if False:
            i = 10
            return i + 15
        self.ID = self.ID + 1
        (Name, Description) = ConvertToSqlString((Name, Description))
        SqlCommand = "insert into %s values(%s, %s, '%s', '%s')" % (self.Table, self.ID, CrossIndex, Name, Description)
        Table.Insert(self, SqlCommand)
        return self.ID

    def InitTable(self):
        if False:
            print('Hello World!')
        EdkLogger.verbose('\nInitialize table DataModel started ...')
        for Item in DataClass.MODEL_LIST:
            CrossIndex = Item[1]
            Name = Item[0]
            Description = Item[0]
            self.Insert(CrossIndex, Name, Description)
        EdkLogger.verbose('Initialize table DataModel ... DONE!')

    def GetCrossIndex(self, ModelName):
        if False:
            return 10
        CrossIndex = -1
        SqlCommand = "select CrossIndex from DataModel where name = '" + ModelName + "'"
        self.Cur.execute(SqlCommand)
        for Item in self.Cur:
            CrossIndex = Item[0]
        return CrossIndex