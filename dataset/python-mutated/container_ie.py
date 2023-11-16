import sys
import pythoncom
import win32api
import win32con
import win32gui
import winerror
from win32com.axcontrol import axcontrol
from win32com.client import Dispatch
from win32com.server.exception import COMException
from win32com.server.util import wrap
debugging = False
IOleClientSite_methods = 'SaveObject GetMoniker GetContainer ShowObject\n                            OnShowWindow RequestNewObjectLayout'.split()
IOleInPlaceSite_methods = 'GetWindow ContextSensitiveHelp CanInPlaceActivate\n                             OnInPlaceActivate OnUIActivate GetWindowContext\n                             Scroll OnUIDeactivate OnInPlaceDeactivate\n                             DiscardUndoState DeactivateAndUndo\n                             OnPosRectChange'.split()
IOleInPlaceFrame_methods = 'GetWindow ContextSensitiveHelp GetBorder\n                              RequestBorderSpace SetBorderSpace\n                              SetActiveObject InsertMenus SetMenu\n                              RemoveMenus SetStatusText EnableModeless\n                              TranslateAccelerator'.split()

class SimpleSite:
    _com_interfaces_ = [axcontrol.IID_IOleClientSite, axcontrol.IID_IOleInPlaceSite]
    _public_methods_ = IOleClientSite_methods + IOleInPlaceSite_methods

    def __init__(self, host_window):
        if False:
            while True:
                i = 10
        self.hw = host_window

    def SaveObject(self):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetMoniker(self, dwAssign, which):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetContainer(self):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOINTERFACE)

    def ShowObject(self):
        if False:
            i = 10
            return i + 15
        pass

    def OnShowWindow(self, fShow):
        if False:
            i = 10
            return i + 15
        pass

    def RequestNewObjectLayout(self):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetWindow(self):
        if False:
            print('Hello World!')
        return self.hw.hwnd

    def ContextSensitiveHelp(self, fEnter):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def CanInPlaceActivate(self):
        if False:
            i = 10
            return i + 15
        pass

    def OnInPlaceActivate(self):
        if False:
            i = 10
            return i + 15
        pass

    def OnUIActivate(self):
        if False:
            return 10
        pass

    def GetWindowContext(self):
        if False:
            while True:
                i = 10
        return (self.hw.ole_frame, None, (0, 0, 0, 0), (0, 0, 0, 0), (True, self.hw.hwnd, None, 0))

    def Scroll(self, size):
        if False:
            while True:
                i = 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def OnUIDeactivate(self, fUndoable):
        if False:
            return 10
        pass

    def OnInPlaceDeactivate(self):
        if False:
            print('Hello World!')
        pass

    def DiscardUndoState(self):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def DeactivateAndUndo(self):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def OnPosRectChange(self, rect):
        if False:
            i = 10
            return i + 15
        browser_ob = self.hw.browser.QueryInterface(axcontrol.IID_IOleInPlaceObject)
        browser_ob.SetObjectRects(rect, rect)

class SimpleFrame:
    _public_methods_ = IOleInPlaceFrame_methods

    def __init__(self, host_window):
        if False:
            i = 10
            return i + 15
        self.hw = host_window

    def GetWindow(self):
        if False:
            while True:
                i = 10
        return self.hw.hwnd

    def ContextSensitiveHelp(self, fEnterMode):
        if False:
            while True:
                i = 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetBorder(self):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def RequestBorderSpace(self, widths):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetBorderSpace(self, widths):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetActiveObject(self, ob, name):
        if False:
            return 10
        pass

    def InsertMenus(self, hmenuShared, menuWidths):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetMenu(self, hmenuShared, holemenu, hwndActiveObject):
        if False:
            while True:
                i = 10
        pass

    def RemoveMenus(self, hmenuShared):
        if False:
            for i in range(10):
                print('nop')
        pass

    def SetStatusText(self, statusText):
        if False:
            return 10
        pass

    def EnableModeless(self, fEnable):
        if False:
            while True:
                i = 10
        pass

    def TranslateAccelerator(self, msg, wID):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

class IEHost:
    wnd_class_name = 'EmbeddedBrowser'

    def __init__(self):
        if False:
            print('Hello World!')
        self.hwnd = None
        self.ole_frame = None

    def __del__(self):
        if False:
            return 10
        try:
            win32gui.UnregisterClass(self.wnd_class_name, None)
        except win32gui.error:
            pass

    def create_window(self):
        if False:
            return 10
        message_map = {win32con.WM_SIZE: self.OnSize, win32con.WM_DESTROY: self.OnDestroy}
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = self.wnd_class_name
        wc.lpfnWndProc = message_map
        class_atom = win32gui.RegisterClass(wc)
        self.hwnd = win32gui.CreateWindow(wc.lpszClassName, 'Embedded browser', win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE, win32con.CW_USEDEFAULT, win32con.CW_USEDEFAULT, win32con.CW_USEDEFAULT, win32con.CW_USEDEFAULT, 0, 0, 0, None)
        browser = pythoncom.CoCreateInstance('{8856F961-340A-11D0-A96B-00C04FD705A2}', None, pythoncom.CLSCTX_INPROC_SERVER | pythoncom.CLSCTX_INPROC_HANDLER, axcontrol.IID_IOleObject)
        self.browser = browser
        site = wrap(SimpleSite(self), axcontrol.IID_IOleClientSite, useDispatcher=debugging)
        browser.SetClientSite(site)
        browser.SetHostNames('IE demo', 'Hi there')
        axcontrol.OleSetContainedObject(self.browser, True)
        rect = win32gui.GetWindowRect(self.hwnd)
        browser.DoVerb(axcontrol.OLEIVERB_SHOW, None, site, -1, self.hwnd, rect)
        b2 = Dispatch(browser.QueryInterface(pythoncom.IID_IDispatch))
        self.browser2 = b2
        b2.Left = 0
        b2.Top = 0
        b2.Width = rect[2]
        b2.Height = rect[3]

    def OnSize(self, hwnd, msg, wparam, lparam):
        if False:
            i = 10
            return i + 15
        self.browser2.Width = win32api.LOWORD(lparam)
        self.browser2.Height = win32api.HIWORD(lparam)

    def OnDestroy(self, hwnd, msg, wparam, lparam):
        if False:
            i = 10
            return i + 15
        self.browser.Close(axcontrol.OLECLOSE_NOSAVE)
        self.browser = None
        self.browser2 = None
        win32gui.PostQuitMessage(0)
if __name__ == '__main__':
    h = IEHost()
    h.create_window()
    if len(sys.argv) < 2:
        h.browser2.Navigate2('about:blank')
        doc = h.browser2.Document
        doc.write('This is an IE page hosted by <a href="http://www.python.org">python</a>')
        doc.write('<br>(you can also specify a URL on the command-line...)')
    else:
        h.browser2.Navigate2(sys.argv[1])
    win32gui.PumpMessages()