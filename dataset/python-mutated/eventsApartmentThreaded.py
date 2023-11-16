import time
import pythoncom
import win32api
import win32com.client
import win32event

class ExplorerEvents:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.event = win32event.CreateEvent(None, 0, 0, None)

    def OnDocumentComplete(self, pDisp=pythoncom.Empty, URL=pythoncom.Empty):
        if False:
            while True:
                i = 10
        thread = win32api.GetCurrentThreadId()
        print('OnDocumentComplete event processed on thread %d' % thread)
        win32event.SetEvent(self.event)

    def OnQuit(self):
        if False:
            while True:
                i = 10
        thread = win32api.GetCurrentThreadId()
        print('OnQuit event processed on thread %d' % thread)
        win32event.SetEvent(self.event)

def WaitWhileProcessingMessages(event, timeout=2):
    if False:
        for i in range(10):
            print('nop')
    start = time.perf_counter()
    while True:
        rc = win32event.MsgWaitForMultipleObjects((event,), 0, 250, win32event.QS_ALLEVENTS)
        if rc == win32event.WAIT_OBJECT_0:
            return True
        if time.perf_counter() - start > timeout:
            return False
        pythoncom.PumpWaitingMessages()

def TestExplorerEvents():
    if False:
        return 10
    iexplore = win32com.client.DispatchWithEvents('InternetExplorer.Application', ExplorerEvents)
    thread = win32api.GetCurrentThreadId()
    print('TestExplorerEvents created IE object on thread %d' % thread)
    iexplore.Visible = 1
    try:
        iexplore.Navigate(win32api.GetFullPathName('..\\readme.html'))
    except pythoncom.com_error as details:
        print('Warning - could not open the test HTML file', details)
    if not WaitWhileProcessingMessages(iexplore.event):
        print('Document load event FAILED to fire!!!')
    iexplore.Quit()
    if not WaitWhileProcessingMessages(iexplore.event):
        print('OnQuit event FAILED to fire!!!')
    iexplore = None
if __name__ == '__main__':
    TestExplorerEvents()