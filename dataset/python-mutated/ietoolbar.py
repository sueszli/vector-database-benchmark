"""
This sample implements a simple IE Toolbar COM server
supporting Windows XP styles and access to
the IWebBrowser2 interface.

It also demonstrates how to hijack the parent window
to catch WM_COMMAND messages.
"""
import sys
import winreg
import pythoncom
import win32com
from win32com import universal
from win32com.axcontrol import axcontrol
from win32com.client import Dispatch, DispatchWithEvents, constants, gencache, getevents
from win32com.shell import shell
from win32com.shell.shellcon import *
try:
    import winxpgui as win32gui
except:
    import win32gui
import array
import struct
import commctrl
import win32con
import win32ui
win32com.client.gencache.EnsureModule('{EAB22AC0-30C1-11CF-A7EB-0000C05BAE0B}', 0, 1, 1)
IDeskBand_methods = ['GetBandInfo']
IDockingWindow_methods = ['ShowDW', 'CloseDW', 'ResizeBorderDW']
IOleWindow_methods = ['GetWindow', 'ContextSensitiveHelp']
IInputObject_methods = ['UIActivateIO', 'HasFocusIO', 'TranslateAcceleratorIO']
IObjectWithSite_methods = ['SetSite', 'GetSite']
IPersistStream_methods = ['GetClassID', 'IsDirty', 'Load', 'Save', 'GetSizeMax']
_ietoolbar_methods_ = IDeskBand_methods + IDockingWindow_methods + IOleWindow_methods + IInputObject_methods + IObjectWithSite_methods + IPersistStream_methods
_ietoolbar_com_interfaces_ = [shell.IID_IDeskBand, axcontrol.IID_IObjectWithSite, pythoncom.IID_IPersistStream, axcontrol.IID_IOleCommandTarget]

class WIN32STRUCT:

    def __init__(self, **kw):
        if False:
            return 10
        full_fmt = ''
        for (name, fmt, default) in self._struct_items_:
            self.__dict__[name] = None
            if fmt == 'z':
                full_fmt += 'pi'
            else:
                full_fmt += fmt
        for (name, val) in kw.items():
            self.__dict__[name] = val

    def __setattr__(self, attr, val):
        if False:
            return 10
        if not attr.startswith('_') and attr not in self.__dict__:
            raise AttributeError(attr)
        self.__dict__[attr] = val

    def toparam(self):
        if False:
            while True:
                i = 10
        self._buffs = []
        full_fmt = ''
        vals = []
        for (name, fmt, default) in self._struct_items_:
            val = self.__dict__[name]
            if fmt == 'z':
                fmt = 'Pi'
                if val is None:
                    vals.append(0)
                    vals.append(0)
                else:
                    str_buf = array.array('c', val + '\x00')
                    vals.append(str_buf.buffer_info()[0])
                    vals.append(len(val))
                    self._buffs.append(str_buf)
            else:
                if val is None:
                    val = default
                vals.append(val)
            full_fmt += fmt
        return struct.pack(*(full_fmt,) + tuple(vals))

class TBBUTTON(WIN32STRUCT):
    _struct_items_ = [('iBitmap', 'i', 0), ('idCommand', 'i', 0), ('fsState', 'B', 0), ('fsStyle', 'B', 0), ('bReserved', 'H', 0), ('dwData', 'I', 0), ('iString', 'z', None)]

class Stub:
    """
    this class serves as a method stub,
    outputting debug info whenever the object
    is being called.
    """

    def __init__(self, name):
        if False:
            return 10
        self.name = name

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        print('STUB: ', self.name, args)

class IEToolbarCtrl:
    """
    a tiny wrapper for our winapi-based
    toolbar control implementation.
    """

    def __init__(self, hwndparent):
        if False:
            while True:
                i = 10
        styles = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_CLIPSIBLINGS | win32con.WS_CLIPCHILDREN | commctrl.TBSTYLE_LIST | commctrl.TBSTYLE_FLAT | commctrl.TBSTYLE_TRANSPARENT | commctrl.CCS_TOP | commctrl.CCS_NODIVIDER | commctrl.CCS_NORESIZE | commctrl.CCS_NOPARENTALIGN
        self.hwnd = win32gui.CreateWindow('ToolbarWindow32', None, styles, 0, 0, 100, 100, hwndparent, 0, win32gui.dllhandle, None)
        win32gui.SendMessage(self.hwnd, commctrl.TB_BUTTONSTRUCTSIZE, 20, 0)

    def ShowWindow(self, mode):
        if False:
            for i in range(10):
                print('nop')
        win32gui.ShowWindow(self.hwnd, mode)

    def AddButtons(self, *buttons):
        if False:
            return 10
        tbbuttons = ''
        for button in buttons:
            tbbuttons += button.toparam()
        return win32gui.SendMessage(self.hwnd, commctrl.TB_ADDBUTTONS, len(buttons), tbbuttons)

    def GetSafeHwnd(self):
        if False:
            while True:
                i = 10
        return self.hwnd

class IEToolbar:
    """
    The actual COM server class
    """
    _com_interfaces_ = _ietoolbar_com_interfaces_
    _public_methods_ = _ietoolbar_methods_
    _reg_clsctx_ = pythoncom.CLSCTX_INPROC_SERVER
    _reg_clsid_ = '{F21202A2-959A-4149-B1C3-68B9013F3335}'
    _reg_progid_ = 'PyWin32.IEToolbar'
    _reg_desc_ = 'PyWin32 IE Toolbar'

    def __init__(self):
        if False:
            print('Hello World!')
        for method in self._public_methods_:
            if not hasattr(self, method):
                print('providing default stub for %s' % method)
                setattr(self, method, Stub(method))

    def GetWindow(self):
        if False:
            for i in range(10):
                print('nop')
        return self.toolbar.GetSafeHwnd()

    def Load(self, stream):
        if False:
            return 10
        pass

    def Save(self, pStream, fClearDirty):
        if False:
            print('Hello World!')
        pass

    def CloseDW(self, dwReserved):
        if False:
            print('Hello World!')
        del self.toolbar

    def ShowDW(self, bShow):
        if False:
            for i in range(10):
                print('nop')
        if bShow:
            self.toolbar.ShowWindow(win32con.SW_SHOW)
        else:
            self.toolbar.ShowWindow(win32con.SW_HIDE)

    def on_first_button(self):
        if False:
            return 10
        print('first!')
        self.webbrowser.Navigate2('http://starship.python.net/crew/mhammond/')

    def on_second_button(self):
        if False:
            return 10
        print('second!')

    def on_third_button(self):
        if False:
            return 10
        print('third!')

    def toolbar_command_handler(self, args):
        if False:
            return 10
        (hwnd, message, wparam, lparam, time, point) = args
        if lparam == self.toolbar.GetSafeHwnd():
            self._command_map[wparam]()

    def SetSite(self, unknown):
        if False:
            for i in range(10):
                print('nop')
        if unknown:
            olewindow = unknown.QueryInterface(pythoncom.IID_IOleWindow)
            hwndparent = olewindow.GetWindow()
            cmdtarget = unknown.QueryInterface(axcontrol.IID_IOleCommandTarget)
            serviceprovider = cmdtarget.QueryInterface(pythoncom.IID_IServiceProvider)
            self.webbrowser = win32com.client.Dispatch(serviceprovider.QueryService('{0002DF05-0000-0000-C000-000000000046}', pythoncom.IID_IDispatch))
            self.toolbar = IEToolbarCtrl(hwndparent)
            buttons = [('Visit PyWin32 Homepage', self.on_first_button), ('Another Button', self.on_second_button), ('Yet Another Button', self.on_third_button)]
            self._command_map = {}
            window = win32ui.CreateWindowFromHandle(hwndparent)
            for i in range(len(buttons)):
                button = TBBUTTON()
                (name, func) = buttons[i]
                id = 17476 + i
                button.iBitmap = -2
                button.idCommand = id
                button.fsState = commctrl.TBSTATE_ENABLED
                button.fsStyle = commctrl.TBSTYLE_BUTTON
                button.iString = name
                self._command_map[17476 + i] = func
                self.toolbar.AddButtons(button)
                window.HookMessage(self.toolbar_command_handler, win32con.WM_COMMAND)
        else:
            self.webbrowser = None

    def GetClassID(self):
        if False:
            i = 10
            return i + 15
        return self._reg_clsid_

    def GetBandInfo(self, dwBandId, dwViewMode, dwMask):
        if False:
            return 10
        ptMinSize = (0, 24)
        ptMaxSize = (2000, 24)
        ptIntegral = (0, 0)
        ptActual = (2000, 24)
        wszTitle = 'PyWin32 IE Toolbar'
        dwModeFlags = DBIMF_VARIABLEHEIGHT
        crBkgnd = 0
        return (ptMinSize, ptMaxSize, ptIntegral, ptActual, wszTitle, dwModeFlags, crBkgnd)

def DllInstall(bInstall, cmdLine):
    if False:
        i = 10
        return i + 15
    comclass = IEToolbar

def DllRegisterServer():
    if False:
        print('Hello World!')
    comclass = IEToolbar
    try:
        print('Trying to register Toolbar.\n')
        hkey = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Internet Explorer\\Toolbar')
        subKey = winreg.SetValueEx(hkey, comclass._reg_clsid_, 0, winreg.REG_BINARY, '\x00')
    except OSError:
        print("Couldn't set registry value.\nhkey: %d\tCLSID: %s\n" % (hkey, comclass._reg_clsid_))
    else:
        print('Set registry value.\nhkey: %d\tCLSID: %s\n' % (hkey, comclass._reg_clsid_))

def DllUnregisterServer():
    if False:
        i = 10
        return i + 15
    comclass = IEToolbar
    try:
        print('Trying to unregister Toolbar.\n')
        hkey = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Internet Explorer\\Toolbar')
        winreg.DeleteValue(hkey, comclass._reg_clsid_)
    except OSError:
        print("Couldn't delete registry value.\nhkey: %d\tCLSID: %s\n" % (hkey, comclass._reg_clsid_))
    else:
        print('Deleting reg key succeeded.\n')
if __name__ == '__main__':
    import win32com.server.register
    win32com.server.register.UseCommandLine(IEToolbar)
    if '--unregister' in sys.argv:
        DllUnregisterServer()
    else:
        DllRegisterServer()
else:
    import win32traceutil