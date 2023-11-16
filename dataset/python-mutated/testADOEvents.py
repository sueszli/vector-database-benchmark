import os
import time
import pythoncom
from win32com.client import Dispatch, DispatchWithEvents, constants
finished = 0

class ADOEvents:

    def OnWillConnect(self, str, user, pw, opt, sts, cn):
        if False:
            i = 10
            return i + 15
        pass

    def OnConnectComplete(self, error, status, connection):
        if False:
            print('Hello World!')
        print('connection is', connection)
        print('Connected to', connection.Properties('Data Source'))
        global finished
        finished = 1

    def OnCommitTransComplete(self, pError, adStatus, pConnection):
        if False:
            for i in range(10):
                print('nop')
        pass

    def OnInfoMessage(self, pError, adStatus, pConnection):
        if False:
            for i in range(10):
                print('nop')
        pass

    def OnDisconnect(self, adStatus, pConnection):
        if False:
            return 10
        pass

    def OnBeginTransComplete(self, TransactionLevel, pError, adStatus, pConnection):
        if False:
            while True:
                i = 10
        pass

    def OnRollbackTransComplete(self, pError, adStatus, pConnection):
        if False:
            i = 10
            return i + 15
        pass

    def OnExecuteComplete(self, RecordsAffected, pError, adStatus, pCommand, pRecordset, pConnection):
        if False:
            i = 10
            return i + 15
        pass

    def OnWillExecute(self, Source, CursorType, LockType, Options, adStatus, pCommand, pRecordset, pConnection):
        if False:
            return 10
        pass

def TestConnection(dbname):
    if False:
        print('Hello World!')
    c = DispatchWithEvents('ADODB.Connection', ADOEvents)
    dsn = 'Driver={Microsoft Access Driver (*.mdb)};Dbq=%s' % dbname
    user = 'system'
    pw = 'manager'
    c.Open(dsn, user, pw, constants.adAsyncConnect)
    end_time = time.clock() + 10
    while time.clock() < end_time:
        pythoncom.PumpWaitingMessages()
    if not finished:
        print('XXX - Failed to connect!')

def Test():
    if False:
        for i in range(10):
            print('nop')
    from . import testAccess
    try:
        testAccess.GenerateSupport()
    except pythoncom.com_error:
        print('*** Can not import the MSAccess type libraries - tests skipped')
        return
    dbname = testAccess.CreateTestAccessDatabase()
    try:
        TestConnection(dbname)
    finally:
        os.unlink(dbname)
if __name__ == '__main__':
    Test()