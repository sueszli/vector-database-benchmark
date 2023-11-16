import msvcrt
import sys
import threading
import time
import types
import pythoncom
from win32com.client import Dispatch, DispatchWithEvents
stopEvent = threading.Event()

def TestExcel():
    if False:
        print('Hello World!')

    class ExcelEvents:

        def OnNewWorkbook(self, wb):
            if False:
                print('Hello World!')
            if not isinstance(wb, types.InstanceType):
                raise RuntimeError('The transformer doesnt appear to have translated this for us!')
            self.seen_events['OnNewWorkbook'] = None

        def OnWindowActivate(self, wb, wn):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(wb, types.InstanceType) or not isinstance(wn, types.InstanceType):
                raise RuntimeError('The transformer doesnt appear to have translated this for us!')
            self.seen_events['OnWindowActivate'] = None

        def OnWindowDeactivate(self, wb, wn):
            if False:
                return 10
            self.seen_events['OnWindowDeactivate'] = None

        def OnSheetDeactivate(self, sh):
            if False:
                i = 10
                return i + 15
            self.seen_events['OnSheetDeactivate'] = None

        def OnSheetBeforeDoubleClick(self, Sh, Target, Cancel):
            if False:
                print('Hello World!')
            if Target.Column % 2 == 0:
                print('You can double-click there...')
            else:
                print('You can not double-click there...')
                return 1

    class WorkbookEvents:

        def OnActivate(self):
            if False:
                return 10
            print('workbook OnActivate')

        def OnBeforeRightClick(self, Target, Cancel):
            if False:
                print('Hello World!')
            print("It's a Worksheet Event")
    e = DispatchWithEvents('Excel.Application', ExcelEvents)
    e.seen_events = {}
    e.Visible = 1
    book = e.Workbooks.Add()
    book = DispatchWithEvents(book, WorkbookEvents)
    print('Have book', book)
    print('Double-click in a few of the Excel cells...')
    print('Press any key when finished with Excel, or wait 10 seconds...')
    if not _WaitForFinish(e, 10):
        e.Quit()
    if not _CheckSeenEvents(e, ['OnNewWorkbook', 'OnWindowActivate']):
        sys.exit(1)

def TestWord():
    if False:
        print('Hello World!')

    class WordEvents:

        def OnDocumentChange(self):
            if False:
                print('Hello World!')
            self.seen_events['OnDocumentChange'] = None

        def OnWindowActivate(self, doc, wn):
            if False:
                while True:
                    i = 10
            self.seen_events['OnWindowActivate'] = None

        def OnQuit(self):
            if False:
                while True:
                    i = 10
            self.seen_events['OnQuit'] = None
            stopEvent.set()
    w = DispatchWithEvents('Word.Application', WordEvents)
    w.seen_events = {}
    w.Visible = 1
    w.Documents.Add()
    print('Press any key when finished with Word, or wait 10 seconds...')
    if not _WaitForFinish(w, 10):
        w.Quit()
    if not _CheckSeenEvents(w, ['OnDocumentChange', 'OnWindowActivate']):
        sys.exit(1)

def _WaitForFinish(ob, timeout):
    if False:
        for i in range(10):
            print('nop')
    end = time.time() + timeout
    while 1:
        if msvcrt.kbhit():
            msvcrt.getch()
            break
        pythoncom.PumpWaitingMessages()
        stopEvent.wait(0.2)
        if stopEvent.isSet():
            stopEvent.clear()
            break
        try:
            if not ob.Visible:
                return 0
        except pythoncom.com_error:
            pass
        if time.time() > end:
            return 0
    return 1

def _CheckSeenEvents(o, events):
    if False:
        print('Hello World!')
    rc = 1
    for e in events:
        if e not in o.seen_events:
            print('ERROR: Expected event did not trigger', e)
            rc = 0
    return rc

def test():
    if False:
        while True:
            i = 10
    import sys
    if 'noword' not in sys.argv[1:]:
        TestWord()
    if 'noexcel' not in sys.argv[1:]:
        TestExcel()
    print('Word and Excel event tests passed.')
if __name__ == '__main__':
    test()