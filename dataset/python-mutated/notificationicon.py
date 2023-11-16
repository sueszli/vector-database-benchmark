import ctypes
import ctypes.wintypes
import os
import uuid
import time
import gevent
import threading
try:
    from queue import Empty as queue_Empty
except ImportError:
    from Queue import Empty as queue_Empty
__all__ = ['NotificationIcon']
CreatePopupMenu = ctypes.windll.user32.CreatePopupMenu
CreatePopupMenu.restype = ctypes.wintypes.HMENU
CreatePopupMenu.argtypes = []
MF_BYCOMMAND = 0
MF_BYPOSITION = 1024
MF_BITMAP = 4
MF_CHECKED = 8
MF_DISABLED = 2
MF_ENABLED = 0
MF_GRAYED = 1
MF_MENUBARBREAK = 32
MF_MENUBREAK = 64
MF_OWNERDRAW = 256
MF_POPUP = 16
MF_SEPARATOR = 2048
MF_STRING = 0
MF_UNCHECKED = 0
InsertMenu = ctypes.windll.user32.InsertMenuW
InsertMenu.restype = ctypes.wintypes.BOOL
InsertMenu.argtypes = [ctypes.wintypes.HMENU, ctypes.wintypes.UINT, ctypes.wintypes.UINT, ctypes.wintypes.UINT, ctypes.wintypes.LPCWSTR]
AppendMenu = ctypes.windll.user32.AppendMenuW
AppendMenu.restype = ctypes.wintypes.BOOL
AppendMenu.argtypes = [ctypes.wintypes.HMENU, ctypes.wintypes.UINT, ctypes.wintypes.UINT, ctypes.wintypes.LPCWSTR]
SetMenuDefaultItem = ctypes.windll.user32.SetMenuDefaultItem
SetMenuDefaultItem.restype = ctypes.wintypes.BOOL
SetMenuDefaultItem.argtypes = [ctypes.wintypes.HMENU, ctypes.wintypes.UINT, ctypes.wintypes.UINT]

class POINT(ctypes.Structure):
    _fields_ = [('x', ctypes.wintypes.LONG), ('y', ctypes.wintypes.LONG)]
GetCursorPos = ctypes.windll.user32.GetCursorPos
GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
SetForegroundWindow = ctypes.windll.user32.SetForegroundWindow
SetForegroundWindow.argtypes = [ctypes.wintypes.HWND]
TPM_LEFTALIGN = 0
TPM_CENTERALIGN = 4
TPM_RIGHTALIGN = 8
TPM_TOPALIGN = 0
TPM_VCENTERALIGN = 16
TPM_BOTTOMALIGN = 32
TPM_NONOTIFY = 128
TPM_RETURNCMD = 256
TPM_LEFTBUTTON = 0
TPM_RIGHTBUTTON = 2
TPM_HORNEGANIMATION = 2048
TPM_HORPOSANIMATION = 1024
TPM_NOANIMATION = 16384
TPM_VERNEGANIMATION = 8192
TPM_VERPOSANIMATION = 4096
TrackPopupMenu = ctypes.windll.user32.TrackPopupMenu
TrackPopupMenu.restype = ctypes.wintypes.BOOL
TrackPopupMenu.argtypes = [ctypes.wintypes.HMENU, ctypes.wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.wintypes.HWND, ctypes.c_void_p]
PostMessage = ctypes.windll.user32.PostMessageW
PostMessage.restype = ctypes.wintypes.BOOL
PostMessage.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.UINT, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM]
DestroyMenu = ctypes.windll.user32.DestroyMenu
DestroyMenu.restype = ctypes.wintypes.BOOL
DestroyMenu.argtypes = [ctypes.wintypes.HMENU]
GUID = ctypes.c_ubyte * 16

class TimeoutVersionUnion(ctypes.Union):
    _fields_ = [('uTimeout', ctypes.wintypes.UINT), ('uVersion', ctypes.wintypes.UINT)]
NIS_HIDDEN = 1
NIS_SHAREDICON = 2

class NOTIFYICONDATA(ctypes.Structure):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(NOTIFYICONDATA, self).__init__(*args, **kwargs)
        self.cbSize = ctypes.sizeof(self)
    _fields_ = [('cbSize', ctypes.wintypes.DWORD), ('hWnd', ctypes.wintypes.HWND), ('uID', ctypes.wintypes.UINT), ('uFlags', ctypes.wintypes.UINT), ('uCallbackMessage', ctypes.wintypes.UINT), ('hIcon', ctypes.wintypes.HICON), ('szTip', ctypes.wintypes.WCHAR * 64), ('dwState', ctypes.wintypes.DWORD), ('dwStateMask', ctypes.wintypes.DWORD), ('szInfo', ctypes.wintypes.WCHAR * 256), ('union', TimeoutVersionUnion), ('szInfoTitle', ctypes.wintypes.WCHAR * 64), ('dwInfoFlags', ctypes.wintypes.DWORD), ('guidItem', GUID), ('hBalloonIcon', ctypes.wintypes.HICON)]
NIM_ADD = 0
NIM_MODIFY = 1
NIM_DELETE = 2
NIM_SETFOCUS = 3
NIM_SETVERSION = 4
NIF_MESSAGE = 1
NIF_ICON = 2
NIF_TIP = 4
NIF_STATE = 8
NIF_INFO = 16
NIF_GUID = 32
NIF_REALTIME = 64
NIF_SHOWTIP = 128
NIIF_NONE = 0
NIIF_INFO = 1
NIIF_WARNING = 2
NIIF_ERROR = 3
NIIF_USER = 4
NOTIFYICON_VERSION = 3
NOTIFYICON_VERSION_4 = 4
Shell_NotifyIcon = ctypes.windll.shell32.Shell_NotifyIconW
Shell_NotifyIcon.restype = ctypes.wintypes.BOOL
Shell_NotifyIcon.argtypes = [ctypes.wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATA)]
IMAGE_BITMAP = 0
IMAGE_ICON = 1
IMAGE_CURSOR = 2
LR_CREATEDIBSECTION = 8192
LR_DEFAULTCOLOR = 0
LR_DEFAULTSIZE = 64
LR_LOADFROMFILE = 16
LR_LOADMAP3DCOLORS = 4096
LR_LOADTRANSPARENT = 32
LR_MONOCHROME = 1
LR_SHARED = 32768
LR_VGACOLOR = 128
OIC_SAMPLE = 32512
OIC_HAND = 32513
OIC_QUES = 32514
OIC_BANG = 32515
OIC_NOTE = 32516
OIC_WINLOGO = 32517
OIC_WARNING = OIC_BANG
OIC_ERROR = OIC_HAND
OIC_INFORMATION = OIC_NOTE
LoadImage = ctypes.windll.user32.LoadImageW
LoadImage.restype = ctypes.wintypes.HANDLE
LoadImage.argtypes = [ctypes.wintypes.HINSTANCE, ctypes.wintypes.LPCWSTR, ctypes.wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.wintypes.UINT]
WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.wintypes.HWND, ctypes.c_uint, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM)
DefWindowProc = ctypes.windll.user32.DefWindowProcW
DefWindowProc.restype = ctypes.c_int
DefWindowProc.argtypes = [ctypes.wintypes.HWND, ctypes.c_uint, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM]
WS_OVERLAPPED = 0
WS_POPUP = 2147483648
WS_CHILD = 1073741824
WS_MINIMIZE = 536870912
WS_VISIBLE = 268435456
WS_DISABLED = 134217728
WS_CLIPSIBLINGS = 67108864
WS_CLIPCHILDREN = 33554432
WS_MAXIMIZE = 16777216
WS_CAPTION = 12582912
WS_BORDER = 8388608
WS_DLGFRAME = 4194304
WS_VSCROLL = 2097152
WS_HSCROLL = 1048576
WS_SYSMENU = 524288
WS_THICKFRAME = 262144
WS_GROUP = 131072
WS_TABSTOP = 65536
WS_MINIMIZEBOX = 131072
WS_MAXIMIZEBOX = 65536
WS_OVERLAPPEDWINDOW = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
SM_CMONITORS = 80
SM_SAMEDISPLAYFORMAT = 81
WM_NULL = 0
WM_CREATE = 1
WM_DESTROY = 2
WM_MOVE = 3
WM_SIZE = 5
WM_ACTIVATE = 6
WM_SETFOCUS = 7
WM_KILLFOCUS = 8
WM_ENABLE = 10
WM_SETREDRAW = 11
WM_SETTEXT = 12
WM_GETTEXT = 13
WM_GETTEXTLENGTH = 14
WM_PAINT = 15
WM_CLOSE = 16
WM_QUERYENDSESSION = 17
WM_QUIT = 18
WM_QUERYOPEN = 19
WM_ERASEBKGND = 20
WM_SYSCOLORCHANGE = 21
WM_ENDSESSION = 22
WM_SHOWWINDOW = 24
WM_CTLCOLOR = 25
WM_WININICHANGE = 26
WM_SETTINGCHANGE = 26
WM_DEVMODECHANGE = 27
WM_ACTIVATEAPP = 28
WM_FONTCHANGE = 29
WM_TIMECHANGE = 30
WM_CANCELMODE = 31
WM_SETCURSOR = 32
WM_MOUSEACTIVATE = 33
WM_CHILDACTIVATE = 34
WM_QUEUESYNC = 35
WM_GETMINMAXINFO = 36
WM_PAINTICON = 38
WM_ICONERASEBKGND = 39
WM_NEXTDLGCTL = 40
WM_SPOOLERSTATUS = 42
WM_DRAWITEM = 43
WM_MEASUREITEM = 44
WM_DELETEITEM = 45
WM_VKEYTOITEM = 46
WM_CHARTOITEM = 47
WM_SETFONT = 48
WM_GETFONT = 49
WM_SETHOTKEY = 50
WM_GETHOTKEY = 51
WM_QUERYDRAGICON = 55
WM_COMPAREITEM = 57
WM_GETOBJECT = 61
WM_COMPACTING = 65
WM_COMMNOTIFY = 68
WM_WINDOWPOSCHANGING = 70
WM_WINDOWPOSCHANGED = 71
WM_POWER = 72
WM_COPYDATA = 74
WM_CANCELJOURNAL = 75
WM_NOTIFY = 78
WM_INPUTLANGCHANGEREQUEST = 80
WM_INPUTLANGCHANGE = 81
WM_TCARD = 82
WM_HELP = 83
WM_USERCHANGED = 84
WM_NOTIFYFORMAT = 85
WM_CONTEXTMENU = 123
WM_STYLECHANGING = 124
WM_STYLECHANGED = 125
WM_DISPLAYCHANGE = 126
WM_GETICON = 127
WM_SETICON = 128
WM_NCCREATE = 129
WM_NCDESTROY = 130
WM_NCCALCSIZE = 131
WM_NCHITTEST = 132
WM_NCPAINT = 133
WM_NCACTIVATE = 134
WM_GETDLGCODE = 135
WM_SYNCPAINT = 136
WM_NCMOUSEMOVE = 160
WM_NCLBUTTONDOWN = 161
WM_NCLBUTTONUP = 162
WM_NCLBUTTONDBLCLK = 163
WM_NCRBUTTONDOWN = 164
WM_NCRBUTTONUP = 165
WM_NCRBUTTONDBLCLK = 166
WM_NCMBUTTONDOWN = 167
WM_NCMBUTTONUP = 168
WM_NCMBUTTONDBLCLK = 169
WM_KEYDOWN = 256
WM_KEYUP = 257
WM_CHAR = 258
WM_DEADCHAR = 259
WM_SYSKEYDOWN = 260
WM_SYSKEYUP = 261
WM_SYSCHAR = 262
WM_SYSDEADCHAR = 263
WM_KEYLAST = 264
WM_IME_STARTCOMPOSITION = 269
WM_IME_ENDCOMPOSITION = 270
WM_IME_COMPOSITION = 271
WM_IME_KEYLAST = 271
WM_INITDIALOG = 272
WM_COMMAND = 273
WM_SYSCOMMAND = 274
WM_TIMER = 275
WM_HSCROLL = 276
WM_VSCROLL = 277
WM_INITMENU = 278
WM_INITMENUPOPUP = 279
WM_MENUSELECT = 287
WM_MENUCHAR = 288
WM_ENTERIDLE = 289
WM_MENURBUTTONUP = 290
WM_MENUDRAG = 291
WM_MENUGETOBJECT = 292
WM_UNINITMENUPOPUP = 293
WM_MENUCOMMAND = 294
WM_CTLCOLORMSGBOX = 306
WM_CTLCOLOREDIT = 307
WM_CTLCOLORLISTBOX = 308
WM_CTLCOLORBTN = 309
WM_CTLCOLORDLG = 310
WM_CTLCOLORSCROLLBAR = 311
WM_CTLCOLORSTATIC = 312
WM_MOUSEMOVE = 512
WM_LBUTTONDOWN = 513
WM_LBUTTONUP = 514
WM_LBUTTONDBLCLK = 515
WM_RBUTTONDOWN = 516
WM_RBUTTONUP = 517
WM_RBUTTONDBLCLK = 518
WM_MBUTTONDOWN = 519
WM_MBUTTONUP = 520
WM_MBUTTONDBLCLK = 521
WM_MOUSEWHEEL = 522
WM_PARENTNOTIFY = 528
WM_ENTERMENULOOP = 529
WM_EXITMENULOOP = 530
WM_NEXTMENU = 531
WM_SIZING = 532
WM_CAPTURECHANGED = 533
WM_MOVING = 534
WM_DEVICECHANGE = 537
WM_MDICREATE = 544
WM_MDIDESTROY = 545
WM_MDIACTIVATE = 546
WM_MDIRESTORE = 547
WM_MDINEXT = 548
WM_MDIMAXIMIZE = 549
WM_MDITILE = 550
WM_MDICASCADE = 551
WM_MDIICONARRANGE = 552
WM_MDIGETACTIVE = 553
WM_MDISETMENU = 560
WM_ENTERSIZEMOVE = 561
WM_EXITSIZEMOVE = 562
WM_DROPFILES = 563
WM_MDIREFRESHMENU = 564
WM_IME_SETCONTEXT = 641
WM_IME_NOTIFY = 642
WM_IME_CONTROL = 643
WM_IME_COMPOSITIONFULL = 644
WM_IME_SELECT = 645
WM_IME_CHAR = 646
WM_IME_REQUEST = 648
WM_IME_KEYDOWN = 656
WM_IME_KEYUP = 657
WM_MOUSEHOVER = 673
WM_MOUSELEAVE = 675
WM_CUT = 768
WM_COPY = 769
WM_PASTE = 770
WM_CLEAR = 771
WM_UNDO = 772
WM_RENDERFORMAT = 773
WM_RENDERALLFORMATS = 774
WM_DESTROYCLIPBOARD = 775
WM_DRAWCLIPBOARD = 776
WM_PAINTCLIPBOARD = 777
WM_VSCROLLCLIPBOARD = 778
WM_SIZECLIPBOARD = 779
WM_ASKCBFORMATNAME = 780
WM_CHANGECBCHAIN = 781
WM_HSCROLLCLIPBOARD = 782
WM_QUERYNEWPALETTE = 783
WM_PALETTEISCHANGING = 784
WM_PALETTECHANGED = 785
WM_HOTKEY = 786
WM_PRINT = 791
WM_PRINTCLIENT = 792
WM_HANDHELDFIRST = 856
WM_HANDHELDLAST = 863
WM_AFXFIRST = 864
WM_AFXLAST = 895
WM_PENWINFIRST = 896
WM_PENWINLAST = 911
WM_APP = 32768
WM_USER = 1024
WM_REFLECT = WM_USER + 7168

class WNDCLASSEX(ctypes.Structure):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(WNDCLASSEX, self).__init__(*args, **kwargs)
        self.cbSize = ctypes.sizeof(self)
    _fields_ = [('cbSize', ctypes.c_uint), ('style', ctypes.c_uint), ('lpfnWndProc', WNDPROC), ('cbClsExtra', ctypes.c_int), ('cbWndExtra', ctypes.c_int), ('hInstance', ctypes.wintypes.HANDLE), ('hIcon', ctypes.wintypes.HANDLE), ('hCursor', ctypes.wintypes.HANDLE), ('hBrush', ctypes.wintypes.HANDLE), ('lpszMenuName', ctypes.wintypes.LPCWSTR), ('lpszClassName', ctypes.wintypes.LPCWSTR), ('hIconSm', ctypes.wintypes.HANDLE)]
ShowWindow = ctypes.windll.user32.ShowWindow
ShowWindow.argtypes = [ctypes.wintypes.HWND, ctypes.c_int]

def GenerateDummyWindow(callback, uid):
    if False:
        print('Hello World!')
    newclass = WNDCLASSEX()
    newclass.lpfnWndProc = callback
    newclass.lpszClassName = uid.replace('-', '')
    ATOM = ctypes.windll.user32.RegisterClassExW(ctypes.byref(newclass))
    hwnd = ctypes.windll.user32.CreateWindowExW(0, newclass.lpszClassName, None, WS_POPUP, 0, 0, 0, 0, 0, 0, 0, 0)
    return hwnd
TIMERCALLBACK = ctypes.WINFUNCTYPE(None, ctypes.wintypes.HWND, ctypes.wintypes.UINT, ctypes.POINTER(ctypes.wintypes.UINT), ctypes.wintypes.DWORD)
SetTimer = ctypes.windll.user32.SetTimer
SetTimer.restype = ctypes.POINTER(ctypes.wintypes.UINT)
SetTimer.argtypes = [ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.UINT), ctypes.wintypes.UINT, TIMERCALLBACK]
KillTimer = ctypes.windll.user32.KillTimer
KillTimer.restype = ctypes.wintypes.BOOL
KillTimer.argtypes = [ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.UINT)]

class MSG(ctypes.Structure):
    _fields_ = [('HWND', ctypes.wintypes.HWND), ('message', ctypes.wintypes.UINT), ('wParam', ctypes.wintypes.WPARAM), ('lParam', ctypes.wintypes.LPARAM), ('time', ctypes.wintypes.DWORD), ('pt', POINT)]
GetMessage = ctypes.windll.user32.GetMessageW
GetMessage.restype = ctypes.wintypes.BOOL
GetMessage.argtypes = [ctypes.POINTER(MSG), ctypes.wintypes.HWND, ctypes.wintypes.UINT, ctypes.wintypes.UINT]
TranslateMessage = ctypes.windll.user32.TranslateMessage
TranslateMessage.restype = ctypes.wintypes.ULONG
TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
DispatchMessage = ctypes.windll.user32.DispatchMessageW
DispatchMessage.restype = ctypes.wintypes.ULONG
DispatchMessage.argtypes = [ctypes.POINTER(MSG)]

def LoadIcon(iconfilename, small=False):
    if False:
        return 10
    return LoadImage(0, str(iconfilename), IMAGE_ICON, 16 if small else 0, 16 if small else 0, LR_LOADFROMFILE)

class NotificationIcon(object):

    def __init__(self, iconfilename, tooltip=None):
        if False:
            while True:
                i = 10
        assert os.path.isfile(str(iconfilename)), "{} doesn't exist".format(iconfilename)
        self._iconfile = str(iconfilename)
        self._hicon = LoadIcon(self._iconfile, True)
        assert self._hicon, 'Failed to load {}'.format(iconfilename)
        self._die = False
        self._timerid = None
        self._uid = uuid.uuid4()
        self._tooltip = str(tooltip) if tooltip else ''
        self._info_bubble = None
        self.items = []

    def _bubble(self, iconinfo):
        if False:
            while True:
                i = 10
        if self._info_bubble:
            info_bubble = self._info_bubble
            self._info_bubble = None
            message = str(self._info_bubble)
            iconinfo.uFlags |= NIF_INFO
            iconinfo.szInfo = message
            iconinfo.szInfoTitle = message
            iconinfo.dwInfoFlags = NIIF_INFO
            iconinfo.union.uTimeout = 10000
            Shell_NotifyIcon(NIM_MODIFY, ctypes.pointer(iconinfo))

    def _run(self):
        if False:
            i = 10
            return i + 15
        self.WM_TASKBARCREATED = ctypes.windll.user32.RegisterWindowMessageW('TaskbarCreated')
        self._windowproc = WNDPROC(self._callback)
        self._hwnd = GenerateDummyWindow(self._windowproc, str(self._uid))
        iconinfo = NOTIFYICONDATA()
        iconinfo.hWnd = self._hwnd
        iconinfo.uID = 100
        iconinfo.uFlags = NIF_ICON | NIF_SHOWTIP | NIF_MESSAGE | (NIF_TIP if self._tooltip else 0)
        iconinfo.uCallbackMessage = WM_MENUCOMMAND
        iconinfo.hIcon = self._hicon
        iconinfo.szTip = self._tooltip
        Shell_NotifyIcon(NIM_ADD, ctypes.pointer(iconinfo))
        self.iconinfo = iconinfo
        PostMessage(self._hwnd, WM_NULL, 0, 0)
        message = MSG()
        last_time = -1
        ret = None
        while not self._die:
            try:
                ret = GetMessage(ctypes.pointer(message), 0, 0, 0)
                TranslateMessage(ctypes.pointer(message))
                DispatchMessage(ctypes.pointer(message))
            except Exception as err:
                message = MSG()
            time.sleep(0.125)
        print('Icon thread stopped, removing icon (hicon: %s, hwnd: %s)...' % (self._hicon, self._hwnd))
        Shell_NotifyIcon(NIM_DELETE, ctypes.cast(ctypes.pointer(iconinfo), ctypes.POINTER(NOTIFYICONDATA)))
        ctypes.windll.user32.DestroyWindow(self._hwnd)
        ctypes.windll.user32.DestroyIcon.argtypes = [ctypes.wintypes.HICON]
        ctypes.windll.user32.DestroyIcon(self._hicon)

    def _menu(self):
        if False:
            return 10
        if not hasattr(self, 'items'):
            return
        menu = CreatePopupMenu()
        func = None
        try:
            iidx = 1000
            defaultitem = -1
            item_map = {}
            for fs in self.items:
                iidx += 1
                if isinstance(fs, str):
                    if fs and (not fs.strip('-_=')):
                        AppendMenu(menu, MF_SEPARATOR, iidx, fs)
                    else:
                        AppendMenu(menu, MF_STRING | MF_GRAYED, iidx, fs)
                elif isinstance(fs, tuple):
                    if callable(fs[0]):
                        itemstring = fs[0]()
                    else:
                        itemstring = str(fs[0])
                    flags = MF_STRING
                    if itemstring.startswith('!'):
                        itemstring = itemstring[1:]
                        defaultitem = iidx
                    if itemstring.startswith('+'):
                        itemstring = itemstring[1:]
                        flags = flags | MF_CHECKED
                    itemcallable = fs[1]
                    item_map[iidx] = itemcallable
                    if itemcallable is False:
                        flags = flags | MF_DISABLED
                    elif not callable(itemcallable):
                        flags = flags | MF_GRAYED
                    AppendMenu(menu, flags, iidx, itemstring)
            if defaultitem != -1:
                SetMenuDefaultItem(menu, defaultitem, 0)
            pos = POINT()
            GetCursorPos(ctypes.pointer(pos))
            PostMessage(self._hwnd, WM_NULL, 0, 0)
            SetForegroundWindow(self._hwnd)
            ti = TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY, pos.x, pos.y, 0, self._hwnd, None)
            if ti in item_map:
                func = item_map[ti]
            PostMessage(self._hwnd, WM_NULL, 0, 0)
        finally:
            DestroyMenu(menu)
        if func:
            func()

    def clicked(self):
        if False:
            return 10
        self._menu()

    def _callback(self, hWnd, msg, wParam, lParam):
        if False:
            while True:
                i = 10
        if msg == WM_TIMER:
            if not any((thread.getName() == 'MainThread' and thread.isAlive() for thread in threading.enumerate())):
                self._die = True
        elif msg == WM_MENUCOMMAND and lParam == WM_LBUTTONUP:
            self.clicked()
        elif msg == WM_MENUCOMMAND and lParam == WM_RBUTTONUP:
            self._menu()
        elif msg == self.WM_TASKBARCREATED:
            Shell_NotifyIcon(NIM_ADD, ctypes.pointer(self.iconinfo))
        else:
            return DefWindowProc(hWnd, msg, wParam, lParam)
        return 1

    def die(self):
        if False:
            return 10
        self._die = True
        PostMessage(self._hwnd, WM_NULL, 0, 0)
        time.sleep(0.2)
        try:
            Shell_NotifyIcon(NIM_DELETE, self.iconinfo)
        except Exception as err:
            print('Icon remove error', err)
        ctypes.windll.user32.DestroyWindow(self._hwnd)
        ctypes.windll.user32.DestroyIcon(self._hicon)

    def pump(self):
        if False:
            print('Hello World!')
        try:
            while not self._pumpqueue.empty():
                callable = self._pumpqueue.get(False)
                callable()
        except queue_Empty:
            pass

    def announce(self, text):
        if False:
            return 10
        self._info_bubble = text

def hideConsole():
    if False:
        while True:
            i = 10
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

def showConsole():
    if False:
        print('Hello World!')
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 1)

def hasConsole():
    if False:
        i = 10
        return i + 15
    return ctypes.windll.kernel32.GetConsoleWindow() != 0
if __name__ == '__main__':
    import time

    def greet():
        if False:
            while True:
                i = 10
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        print('Hello')

    def quit():
        if False:
            i = 10
            return i + 15
        ni._die = True

    def announce():
        if False:
            while True:
                i = 10
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 1)
        ni.announce('Hello there')

    def clicked():
        if False:
            while True:
                i = 10
        ni.announce('Hello')

    def dynamicTitle():
        if False:
            while True:
                i = 10
        return '!The time is: %s' % time.time()
    ni = NotificationIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../trayicon.ico'), 'ZeroNet 0.2.9')
    ni.items = [(dynamicTitle, False), ('Hello', greet), ('Title', False), ('!Default', greet), ('+Popup bubble', announce), 'Nothing', '--', ('Quit', quit)]
    ni.clicked = clicked
    import atexit

    @atexit.register
    def goodbye():
        if False:
            print('Hello World!')
        print('You are now leaving the Python sector.')
    ni._run()