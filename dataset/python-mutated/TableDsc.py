from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
import CommonDataClass.DataClass as DataClass
from Table.Table import Table
from Common.StringUtils import ConvertToSqlString

class TableDsc(Table):

    def __init__(self, Cursor):
        if False:
            print('Hello World!')
        Table.__init__(self, Cursor)
        self.Table = 'Dsc'

    def Create(self):
        if False:
            print('Hello World!')
        SqlCommand = 'create table IF NOT EXISTS %s (ID INTEGER PRIMARY KEY,\n                                                       Model INTEGER NOT NULL,\n                                                       Value1 VARCHAR NOT NULL,\n                                                       Value2 VARCHAR,\n                                                       Value3 VARCHAR,\n                                                       Arch VarCHAR,\n                                                       BelongsToItem SINGLE NOT NULL,\n                                                       BelongsToFile SINGLE NOT NULL,\n                                                       StartLine INTEGER NOT NULL,\n                                                       StartColumn INTEGER NOT NULL,\n                                                       EndLine INTEGER NOT NULL,\n                                                       EndColumn INTEGER NOT NULL,\n                                                       Enabled INTEGER DEFAULT 0\n                                                      )' % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, Model, Value1, Value2, Value3, Arch, BelongsToItem, BelongsToFile, StartLine, StartColumn, EndLine, EndColumn, Enabled):
        if False:
            print('Hello World!')
        self.ID = self.ID + 1
        (Value1, Value2, Value3, Arch) = ConvertToSqlString((Value1, Value2, Value3, Arch))
        SqlCommand = "insert into %s values(%s, %s, '%s', '%s', '%s', '%s', %s, %s, %s, %s, %s, %s, %s)" % (self.Table, self.ID, Model, Value1, Value2, Value3, Arch, BelongsToItem, BelongsToFile, StartLine, StartColumn, EndLine, EndColumn, Enabled)
        Table.Insert(self, SqlCommand)
        return self.ID

    def Query(self, Model):
        if False:
            while True:
                i = 10
        SqlCommand = 'select ID, Value1, Value2, Value3, Arch, BelongsToItem, BelongsToFile, StartLine from %s\n                        where Model = %s\n                        and Enabled > -1' % (self.Table, Model)
        EdkLogger.debug(4, 'SqlCommand: %s' % SqlCommand)
        self.Cur.execute(SqlCommand)
        return self.Cur.fetchall()