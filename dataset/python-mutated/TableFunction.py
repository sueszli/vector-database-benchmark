from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
from Table.Table import Table
from Common.StringUtils import ConvertToSqlString

class TableFunction(Table):

    def __init__(self, Cursor):
        if False:
            while True:
                i = 10
        Table.__init__(self, Cursor)
        self.Table = 'Function'

    def Create(self):
        if False:
            while True:
                i = 10
        SqlCommand = 'create table IF NOT EXISTS %s (ID INTEGER PRIMARY KEY,\n                                                       Header TEXT,\n                                                       Modifier VARCHAR,\n                                                       Name VARCHAR NOT NULL,\n                                                       ReturnStatement VARCHAR,\n                                                       StartLine INTEGER NOT NULL,\n                                                       StartColumn INTEGER NOT NULL,\n                                                       EndLine INTEGER NOT NULL,\n                                                       EndColumn INTEGER NOT NULL,\n                                                       BodyStartLine INTEGER NOT NULL,\n                                                       BodyStartColumn INTEGER NOT NULL,\n                                                       BelongsToFile SINGLE NOT NULL,\n                                                       FunNameStartLine INTEGER NOT NULL,\n                                                       FunNameStartColumn INTEGER NOT NULL\n                                                      )' % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, Header, Modifier, Name, ReturnStatement, StartLine, StartColumn, EndLine, EndColumn, BodyStartLine, BodyStartColumn, BelongsToFile, FunNameStartLine, FunNameStartColumn):
        if False:
            for i in range(10):
                print('nop')
        self.ID = self.ID + 1
        (Header, Modifier, Name, ReturnStatement) = ConvertToSqlString((Header, Modifier, Name, ReturnStatement))
        SqlCommand = "insert into %s values(%s, '%s', '%s', '%s', '%s', %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (self.Table, self.ID, Header, Modifier, Name, ReturnStatement, StartLine, StartColumn, EndLine, EndColumn, BodyStartLine, BodyStartColumn, BelongsToFile, FunNameStartLine, FunNameStartColumn)
        Table.Insert(self, SqlCommand)
        return self.ID