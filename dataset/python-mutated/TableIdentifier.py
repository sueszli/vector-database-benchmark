from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
from Common.StringUtils import ConvertToSqlString
from Table.Table import Table

class TableIdentifier(Table):

    def __init__(self, Cursor):
        if False:
            return 10
        Table.__init__(self, Cursor)
        self.Table = 'Identifier'

    def Create(self):
        if False:
            print('Hello World!')
        SqlCommand = 'create table IF NOT EXISTS %s(ID INTEGER PRIMARY KEY,\n                                                      Modifier VARCHAR,\n                                                      Type VARCHAR,\n                                                      Name VARCHAR NOT NULL,\n                                                      Value VARCHAR NOT NULL,\n                                                      Model INTEGER NOT NULL,\n                                                      BelongsToFile SINGLE NOT NULL,\n                                                      BelongsToFunction SINGLE DEFAULT -1,\n                                                      StartLine INTEGER NOT NULL,\n                                                      StartColumn INTEGER NOT NULL,\n                                                      EndLine INTEGER NOT NULL,\n                                                      EndColumn INTEGER NOT NULL\n                                                     )' % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, Modifier, Type, Name, Value, Model, BelongsToFile, BelongsToFunction, StartLine, StartColumn, EndLine, EndColumn):
        if False:
            return 10
        self.ID = self.ID + 1
        (Modifier, Type, Name, Value) = ConvertToSqlString((Modifier, Type, Name, Value))
        SqlCommand = "insert into %s values(%s, '%s', '%s', '%s', '%s', %s, %s, %s, %s, %s, %s, %s)" % (self.Table, self.ID, Modifier, Type, Name, Value, Model, BelongsToFile, BelongsToFunction, StartLine, StartColumn, EndLine, EndColumn)
        Table.Insert(self, SqlCommand)
        return self.ID