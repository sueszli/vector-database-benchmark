from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
import Common.LongFilePathOs as os, time
from Table.Table import Table
from Common.StringUtils import ConvertToSqlString2
import Eot.EotToolError as EotToolError
import Eot.EotGlobalData as EotGlobalData

class TableEotReport(Table):

    def __init__(self, Cursor):
        if False:
            return 10
        Table.__init__(self, Cursor)
        self.Table = 'Report'

    def Create(self):
        if False:
            return 10
        SqlCommand = "create table IF NOT EXISTS %s (ID INTEGER PRIMARY KEY,\n                                                       ModuleID INTEGER DEFAULT -1,\n                                                       ModuleName TEXT DEFAULT '',\n                                                       ModuleGuid TEXT DEFAULT '',\n                                                       SourceFileID INTEGER DEFAULT -1,\n                                                       SourceFileFullPath TEXT DEFAULT '',\n                                                       ItemName TEXT DEFAULT '',\n                                                       ItemType TEXT DEFAULT '',\n                                                       ItemMode TEXT DEFAULT '',\n                                                       GuidName TEXT DEFAULT '',\n                                                       GuidMacro TEXT DEFAULT '',\n                                                       GuidValue TEXT DEFAULT '',\n                                                       BelongsToFunction TEXT DEFAULT '',\n                                                       Enabled INTEGER DEFAULT 0\n                                                      )" % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, ModuleID=-1, ModuleName='', ModuleGuid='', SourceFileID=-1, SourceFileFullPath='', ItemName='', ItemType='', ItemMode='', GuidName='', GuidMacro='', GuidValue='', BelongsToFunction='', Enabled=0):
        if False:
            return 10
        self.ID = self.ID + 1
        SqlCommand = "insert into %s values(%s, %s, '%s', '%s', %s, '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', %s)" % (self.Table, self.ID, ModuleID, ModuleName, ModuleGuid, SourceFileID, SourceFileFullPath, ItemName, ItemType, ItemMode, GuidName, GuidMacro, GuidValue, BelongsToFunction, Enabled)
        Table.Insert(self, SqlCommand)

    def GetMaxID(self):
        if False:
            return 10
        SqlCommand = 'select max(ID) from %s' % self.Table
        self.Cur.execute(SqlCommand)
        for Item in self.Cur:
            return Item[0]