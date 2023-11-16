from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
from Table.Table import Table
from Common.StringUtils import ConvertToSqlString
import Common.LongFilePathOs as os
from CommonDataClass.DataClass import FileClass

class TableFile(Table):

    def __init__(self, Cursor):
        if False:
            while True:
                i = 10
        Table.__init__(self, Cursor)
        self.Table = 'File'

    def Create(self):
        if False:
            for i in range(10):
                print('nop')
        SqlCommand = 'create table IF NOT EXISTS %s (ID INTEGER PRIMARY KEY,\n                                                       Name VARCHAR NOT NULL,\n                                                       ExtName VARCHAR,\n                                                       Path VARCHAR,\n                                                       FullPath VARCHAR NOT NULL,\n                                                       Model INTEGER DEFAULT 0,\n                                                       TimeStamp VARCHAR NOT NULL\n                                                      )' % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, Name, ExtName, Path, FullPath, Model, TimeStamp):
        if False:
            for i in range(10):
                print('nop')
        self.ID = self.ID + 1
        (Name, ExtName, Path, FullPath) = ConvertToSqlString((Name, ExtName, Path, FullPath))
        SqlCommand = "insert into %s values(%s, '%s', '%s', '%s', '%s', %s, '%s')" % (self.Table, self.ID, Name, ExtName, Path, FullPath, Model, TimeStamp)
        Table.Insert(self, SqlCommand)
        return self.ID

    def InsertFile(self, FileFullPath, Model):
        if False:
            for i in range(10):
                print('nop')
        (Filepath, Name) = os.path.split(FileFullPath)
        (Root, Ext) = os.path.splitext(FileFullPath)
        TimeStamp = os.stat(FileFullPath)[8]
        File = FileClass(-1, Name, Ext, Filepath, FileFullPath, Model, '', [], [], [])
        return self.Insert(File.Name, File.ExtName, File.Path, File.FullPath, File.Model, TimeStamp)

    def GetFileId(self, File):
        if False:
            print('Hello World!')
        QueryScript = "select ID from %s where FullPath = '%s'" % (self.Table, str(File))
        RecordList = self.Exec(QueryScript)
        if len(RecordList) == 0:
            return None
        return RecordList[0][0]