import Common.EdkLogger as EdkLogger

class Table(object):

    def __init__(self, Cursor):
        if False:
            return 10
        self.Cur = Cursor
        self.Table = ''
        self.ID = 0

    def Create(self, SqlCommand):
        if False:
            i = 10
            return i + 15
        self.Cur.execute(SqlCommand)
        self.ID = 0
        EdkLogger.verbose(SqlCommand + ' ... DONE!')

    def Insert(self, SqlCommand):
        if False:
            for i in range(10):
                print('nop')
        self.Exec(SqlCommand)

    def Query(self):
        if False:
            while True:
                i = 10
        EdkLogger.verbose('\nQuery table %s started ...' % self.Table)
        SqlCommand = 'select * from %s' % self.Table
        self.Cur.execute(SqlCommand)
        for Rs in self.Cur:
            EdkLogger.verbose(str(Rs))
        TotalCount = self.GetCount()
        EdkLogger.verbose('*** Total %s records in table %s ***' % (TotalCount, self.Table))
        EdkLogger.verbose('Query tabel %s DONE!' % self.Table)

    def Drop(self):
        if False:
            for i in range(10):
                print('nop')
        SqlCommand = 'drop table IF EXISTS %s' % self.Table
        self.Cur.execute(SqlCommand)
        EdkLogger.verbose('Drop tabel %s ... DONE!' % self.Table)

    def GetCount(self):
        if False:
            for i in range(10):
                print('nop')
        SqlCommand = 'select count(ID) from %s' % self.Table
        self.Cur.execute(SqlCommand)
        for Item in self.Cur:
            return Item[0]

    def GenerateID(self, ID):
        if False:
            for i in range(10):
                print('nop')
        if ID == -1:
            self.ID = self.ID + 1
        return self.ID

    def InitID(self):
        if False:
            print('Hello World!')
        self.ID = self.GetCount()

    def Exec(self, SqlCommand):
        if False:
            i = 10
            return i + 15
        EdkLogger.debug(4, 'SqlCommand: %s' % SqlCommand)
        self.Cur.execute(SqlCommand)
        RecordSet = self.Cur.fetchall()
        EdkLogger.debug(4, 'RecordSet: %s' % RecordSet)
        return RecordSet