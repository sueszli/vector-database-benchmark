from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
from Table.Table import Table
from Common.StringUtils import ConvertToSqlString

class TablePcd(Table):

    def __init__(self, Cursor):
        if False:
            print('Hello World!')
        Table.__init__(self, Cursor)
        self.Table = 'Pcd'

    def Create(self):
        if False:
            return 10
        SqlCommand = 'create table IF NOT EXISTS %s (ID INTEGER PRIMARY KEY,\n                                                       CName VARCHAR NOT NULL,\n                                                       TokenSpaceGuidCName VARCHAR NOT NULL,\n                                                       Token INTEGER,\n                                                       DatumType VARCHAR,\n                                                       Model INTEGER NOT NULL,\n                                                       BelongsToFile SINGLE NOT NULL,\n                                                       BelongsToFunction SINGLE DEFAULT -1,\n                                                       StartLine INTEGER NOT NULL,\n                                                       StartColumn INTEGER NOT NULL,\n                                                       EndLine INTEGER NOT NULL,\n                                                       EndColumn INTEGER NOT NULL\n                                                      )' % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, CName, TokenSpaceGuidCName, Token, DatumType, Model, BelongsToFile, BelongsToFunction, StartLine, StartColumn, EndLine, EndColumn):
        if False:
            i = 10
            return i + 15
        self.ID = self.ID + 1
        (CName, TokenSpaceGuidCName, DatumType) = ConvertToSqlString((CName, TokenSpaceGuidCName, DatumType))
        SqlCommand = "insert into %s values(%s, '%s', '%s', %s, '%s', %s, %s, %s, %s, %s, %s, %s)" % (self.Table, self.ID, CName, TokenSpaceGuidCName, Token, DatumType, Model, BelongsToFile, BelongsToFunction, StartLine, StartColumn, EndLine, EndColumn)
        Table.Insert(self, SqlCommand)
        return self.ID