import sys
sys.coinit_flags = 0
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
            return 10
        thread = win32api.GetCurrentThreadId()
        print('OnDocumentComplete event processed on thread %d' % thread)
        win32event.SetEvent(self.event)

    def OnQuit(self):
        if False:
            print('Hello World!')
        thread = win32api.GetCurrentThreadId()
        print('OnQuit event processed on thread %d' % thread)
        win32event.SetEvent(self.event)

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
    rc = win32event.WaitForSingleObject(iexplore.event, 2000)
    if rc != win32event.WAIT_OBJECT_0:
        print('Document load event FAILED to fire!!!')
    iexplore.Quit()
    rc = win32event.WaitForSingleObject(iexplore.event, 2000)
    if rc != win32event.WAIT_OBJECT_0:
        print('OnQuit event FAILED to fire!!!')
    iexplore = None
    print('Finished the IE event sample!')
if __name__ == '__main__':
    TestExplorerEvents()