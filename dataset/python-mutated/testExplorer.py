import os
import time
import pythoncom
import win32api
import win32com.client.dynamic
import win32con
import win32gui
import winerror
from win32com.client import Dispatch
from win32com.test.util import CheckClean
bVisibleEventFired = 0
HRESULTS_IN_AUTOMATION = [-2125463506, winerror.MK_E_UNAVAILABLE]

class ExplorerEvents:

    def OnVisible(self, visible):
        if False:
            print('Hello World!')
        global bVisibleEventFired
        bVisibleEventFired = 1

def TestExplorerEvents():
    if False:
        i = 10
        return i + 15
    global bVisibleEventFired
    try:
        iexplore = win32com.client.DispatchWithEvents('InternetExplorer.Application', ExplorerEvents)
    except pythoncom.com_error as exc:
        if exc.hresult not in HRESULTS_IN_AUTOMATION:
            raise
        print('IE events appear to not be available, so skipping this test')
        return
    iexplore.Visible = 1
    if not bVisibleEventFired:
        raise RuntimeError('The IE event did not appear to fire!')
    iexplore.Quit()
    iexplore = None
    bVisibleEventFired = 0
    ie = win32com.client.Dispatch('InternetExplorer.Application')
    ie_events = win32com.client.DispatchWithEvents(ie, ExplorerEvents)
    ie.Visible = 1
    if not bVisibleEventFired:
        raise RuntimeError('The IE event did not appear to fire!')
    ie.Quit()
    ie = None
    print('IE Event tests worked.')

def TestObjectFromWindow():
    if False:
        print('Hello World!')
    hwnd = win32gui.FindWindow('IEFrame', None)
    for child_class in ['TabWindowClass', 'Shell DocObject View', 'Internet Explorer_Server']:
        hwnd = win32gui.FindWindowEx(hwnd, 0, child_class, None)
        return
    msg = win32gui.RegisterWindowMessage('WM_HTML_GETOBJECT')
    (rc, result) = win32gui.SendMessageTimeout(hwnd, msg, 0, 0, win32con.SMTO_ABORTIFHUNG, 1000)
    ob = pythoncom.ObjectFromLresult(result, pythoncom.IID_IDispatch, 0)
    doc = Dispatch(ob)
    for color in 'red green blue orange white'.split():
        doc.bgColor = color
        time.sleep(0.2)

def TestExplorer(iexplore):
    if False:
        for i in range(10):
            print('nop')
    if not iexplore.Visible:
        iexplore.Visible = -1
    filename = os.path.join(os.path.dirname(__file__), '..\\readme.html')
    iexplore.Navigate(win32api.GetFullPathName(filename))
    win32api.Sleep(1000)
    TestObjectFromWindow()
    win32api.Sleep(3000)
    try:
        iexplore.Quit()
    except (AttributeError, pythoncom.com_error):
        pass

def TestAll():
    if False:
        i = 10
        return i + 15
    try:
        try:
            try:
                iexplore = win32com.client.dynamic.Dispatch('InternetExplorer.Application')
            except pythoncom.com_error as exc:
                if exc.hresult not in HRESULTS_IN_AUTOMATION:
                    raise
                print('IE appears to not be available, so skipping this test')
                return
            TestExplorer(iexplore)
            win32api.Sleep(1000)
            iexplore = None
            TestExplorerEvents()
            time.sleep(2)
            from win32com.client import gencache
            gencache.EnsureModule('{EAB22AC0-30C1-11CF-A7EB-0000C05BAE0B}', 0, 1, 1)
            iexplore = win32com.client.Dispatch('InternetExplorer.Application')
            TestExplorer(iexplore)
        except pythoncom.com_error as exc:
            if exc.hresult != winerror.RPC_E_DISCONNECTED:
                raise
    finally:
        iexplore = None
if __name__ == '__main__':
    TestAll()
    CheckClean()