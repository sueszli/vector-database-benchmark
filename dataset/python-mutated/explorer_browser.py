import sys
import pythoncom
import win32api
import win32con
import win32gui
from win32com.server.util import unwrap, wrap
from win32com.shell import shell, shellcon
IExplorerBrowserEvents_Methods = 'OnNavigationComplete OnNavigationFailed \n                                    OnNavigationPending OnViewCreated'.split()

class EventHandler:
    _com_interfaces_ = [shell.IID_IExplorerBrowserEvents]
    _public_methods_ = IExplorerBrowserEvents_Methods

    def OnNavigationComplete(self, pidl):
        if False:
            while True:
                i = 10
        print('OnNavComplete', pidl)

    def OnNavigationFailed(self, pidl):
        if False:
            i = 10
            return i + 15
        print('OnNavigationFailed', pidl)

    def OnNavigationPending(self, pidl):
        if False:
            while True:
                i = 10
        print('OnNavigationPending', pidl)

    def OnViewCreated(self, view):
        if False:
            while True:
                i = 10
        print('OnViewCreated', view)
        try:
            pyview = unwrap(view)
            print('and look - its a Python implemented view!', pyview)
        except ValueError:
            pass

class MainWindow:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        message_map = {win32con.WM_DESTROY: self.OnDestroy, win32con.WM_COMMAND: self.OnCommand, win32con.WM_SIZE: self.OnSize}
        wc = win32gui.WNDCLASS()
        hinst = wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = 'test_explorer_browser'
        wc.lpfnWndProc = message_map
        classAtom = win32gui.RegisterClass(wc)
        style = win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE
        self.hwnd = win32gui.CreateWindow(classAtom, 'Python IExplorerBrowser demo', style, 0, 0, win32con.CW_USEDEFAULT, win32con.CW_USEDEFAULT, 0, 0, hinst, None)
        eb = pythoncom.CoCreateInstance(shellcon.CLSID_ExplorerBrowser, None, pythoncom.CLSCTX_ALL, shell.IID_IExplorerBrowser)
        self.event_cookie = eb.Advise(wrap(EventHandler()))
        eb.SetOptions(shellcon.EBO_SHOWFRAMES)
        rect = win32gui.GetClientRect(self.hwnd)
        flags = (shellcon.FVM_LIST, shellcon.FWF_AUTOARRANGE | shellcon.FWF_NOWEBVIEW)
        eb.Initialize(self.hwnd, rect, (0, shellcon.FVM_DETAILS))
        if len(sys.argv) == 2:
            pidl = shell.SHGetDesktopFolder().ParseDisplayName(0, None, sys.argv[1])[1]
        else:
            pidl = []
        eb.BrowseToIDList(pidl, shellcon.SBSP_ABSOLUTE)
        sp = eb.QueryInterface(pythoncom.IID_IServiceProvider)
        try:
            tree = sp.QueryService(shell.IID_INameSpaceTreeControl, shell.IID_INameSpaceTreeControl)
        except pythoncom.com_error as exc:
            print('Strange - failed to get the tree control even though we asked for a EBO_SHOWFRAMES')
            print(exc)
        else:
            si = shell.SHCreateItemFromIDList(pidl, shell.IID_IShellItem)
            tree.SetItemState(si, shellcon.NSTCIS_SELECTED, shellcon.NSTCIS_SELECTED)
        self.eb = eb

    def OnCommand(self, hwnd, msg, wparam, lparam):
        if False:
            i = 10
            return i + 15
        pass

    def OnDestroy(self, hwnd, msg, wparam, lparam):
        if False:
            i = 10
            return i + 15
        print('tearing down ExplorerBrowser...')
        self.eb.Unadvise(self.event_cookie)
        self.eb.Destroy()
        self.eb = None
        print('shutting down app...')
        win32gui.PostQuitMessage(0)

    def OnSize(self, hwnd, msg, wparam, lparam):
        if False:
            return 10
        x = win32api.LOWORD(lparam)
        y = win32api.HIWORD(lparam)
        self.eb.SetRect(None, (0, 0, x, y))

def main():
    if False:
        i = 10
        return i + 15
    w = MainWindow()
    win32gui.PumpMessages()
if __name__ == '__main__':
    main()