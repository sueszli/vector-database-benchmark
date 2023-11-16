from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
from Common.StringUtils import ConvertToSqlString
from Table.Table import Table

class TableQuery(Table):

    def __init__(self, Cursor):
        if False:
            for i in range(10):
                print('nop')
        Table.__init__(self, Cursor)
        self.Table = 'Query'

    def Create(self):
        if False:
            print('Hello World!')
        SqlCommand = "create table IF NOT EXISTS %s(ID INTEGER PRIMARY KEY,\n                                                      Name TEXT DEFAULT '',\n                                                      Modifier TEXT DEFAULT '',\n                                                      Value TEXT DEFAULT '',\n                                                      Model INTEGER DEFAULT 0\n                                                     )" % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, Name, Modifier, Value, Model):
        if False:
            while True:
                i = 10
        self.ID = self.ID + 1
        SqlCommand = "insert into %s values(%s, '%s', '%s', '%s', %s)" % (self.Table, self.ID, Name, Modifier, Value, Model)
        Table.Insert(self, SqlCommand)
        return self.ID