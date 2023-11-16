from __future__ import absolute_import
import Common.EdkLogger as EdkLogger
import Common.LongFilePathOs as os, time
from Table.Table import Table
from Common.StringUtils import ConvertToSqlString2
import Ecc.EccToolError as EccToolError
import Ecc.EccGlobalData as EccGlobalData
from Common.LongFilePathSupport import OpenLongFilePath as open

class TableReport(Table):

    def __init__(self, Cursor):
        if False:
            while True:
                i = 10
        Table.__init__(self, Cursor)
        self.Table = 'Report'

    def Create(self):
        if False:
            while True:
                i = 10
        SqlCommand = 'create table IF NOT EXISTS %s (ID INTEGER PRIMARY KEY,\n                                                       ErrorID INTEGER NOT NULL,\n                                                       OtherMsg TEXT,\n                                                       BelongsToTable TEXT NOT NULL,\n                                                       BelongsToItem SINGLE NOT NULL,\n                                                       Enabled INTEGER DEFAULT 0,\n                                                       Corrected INTEGER DEFAULT -1\n                                                      )' % self.Table
        Table.Create(self, SqlCommand)

    def Insert(self, ErrorID, OtherMsg='', BelongsToTable='', BelongsToItem=-1, Enabled=0, Corrected=-1):
        if False:
            return 10
        self.ID = self.ID + 1
        SqlCommand = "insert into %s values(%s, %s, '%s', '%s', %s, %s, %s)" % (self.Table, self.ID, ErrorID, ConvertToSqlString2(OtherMsg), BelongsToTable, BelongsToItem, Enabled, Corrected)
        Table.Insert(self, SqlCommand)
        return self.ID

    def Query(self):
        if False:
            for i in range(10):
                print('nop')
        SqlCommand = 'select ID, ErrorID, OtherMsg, BelongsToTable, BelongsToItem, Corrected from %s\n                        where Enabled > -1 order by ErrorID, BelongsToItem' % self.Table
        return self.Exec(SqlCommand)

    def UpdateBelongsToItemByFile(self, ItemID=-1, File=''):
        if False:
            for i in range(10):
                print('nop')
        SqlCommand = "update Report set BelongsToItem=%s where BelongsToTable='File' and BelongsToItem=-2\n                        and OtherMsg like '%%%s%%'" % (ItemID, File)
        return self.Exec(SqlCommand)

    def ToCSV(self, Filename='Report.csv'):
        if False:
            for i in range(10):
                print('nop')
        try:
            File = open(Filename, 'w+')
            File.write('No, Error Code, Error Message, File, LineNo, Other Error Message\n')
            RecordSet = self.Query()
            Index = 0
            for Record in RecordSet:
                Index = Index + 1
                ErrorID = Record[1]
                OtherMsg = Record[2]
                BelongsToTable = Record[3]
                BelongsToItem = Record[4]
                IsCorrected = Record[5]
                SqlCommand = ''
                if BelongsToTable == 'File':
                    SqlCommand = 'select 1, FullPath from %s where ID = %s\n                             ' % (BelongsToTable, BelongsToItem)
                else:
                    SqlCommand = 'select A.StartLine, B.FullPath from %s as A, File as B\n                                    where A.ID = %s and B.ID = A.BelongsToFile\n                                 ' % (BelongsToTable, BelongsToItem)
                NewRecord = self.Exec(SqlCommand)
                if NewRecord != []:
                    File.write('%s,%s,"%s",%s,%s,"%s"\n' % (Index, ErrorID, EccToolError.gEccErrorMessage[ErrorID], NewRecord[0][1], NewRecord[0][0], OtherMsg))
                    EdkLogger.quiet('%s(%s): [%s]%s %s' % (NewRecord[0][1], NewRecord[0][0], ErrorID, EccToolError.gEccErrorMessage[ErrorID], OtherMsg))
            File.close()
        except IOError:
            NewFilename = 'Report_' + time.strftime('%Y%m%d_%H%M%S.csv', time.localtime())
            EdkLogger.warn('ECC', 'The report file %s is locked by other progress, use %s instead!' % (Filename, NewFilename))
            self.ToCSV(NewFilename)