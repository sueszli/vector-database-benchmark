"""Functions to retrieve properties from a window handle

These are implemented in a procedural way so as to to be
useful to other modules with the least conceptual overhead
"""
import warnings
from ctypes import wintypes
from ctypes import WINFUNCTYPE
from ctypes import c_int
from ctypes import byref
from ctypes import sizeof
from ctypes import create_unicode_buffer
import win32process
import win32api
import win32con
import win32gui
from .windows import win32functions
from .windows import win32defines
from .windows import win32structures
from .actionlogger import ActionLogger

def text(handle):
    if False:
        i = 10
        return i + 15
    'Return the text of the window'
    class_name = classname(handle)
    if class_name == 'IME':
        return 'Default IME'
    if class_name == 'MSCTFIME UI':
        return 'M'
    if class_name is None:
        return ''
    c_length = win32structures.DWORD_PTR(0)
    result = win32functions.SendMessageTimeout(handle, win32defines.WM_GETTEXTLENGTH, 0, 0, win32defines.SMTO_ABORTIFHUNG, 500, byref(c_length))
    if result == 0:
        if isvisible(handle):
            ActionLogger().log('WARNING! Cannot retrieve text length for handle = ' + hex(handle))
        return ''
    else:
        length = c_length.value
    textval = ''
    if length > 0:
        length += 1
        buffer_ = create_unicode_buffer(length)
        ret = win32functions.SendMessage(handle, win32defines.WM_GETTEXT, length, byref(buffer_))
        if ret:
            textval = buffer_.value
    return textval

def classname(handle):
    if False:
        return 10
    'Return the class name of the window'
    if handle is None:
        return None
    class_name = create_unicode_buffer(u'', 257)
    win32functions.GetClassName(handle, class_name, 256)
    return class_name.value

def parent(handle):
    if False:
        while True:
            i = 10
    'Return the handle of the parent of the window'
    return win32functions.GetParent(handle)

def style(handle):
    if False:
        print('Hello World!')
    'Return the style of the window'
    return win32functions.GetWindowLong(handle, win32defines.GWL_STYLE)

def exstyle(handle):
    if False:
        return 10
    'Return the extended style of the window'
    return win32functions.GetWindowLong(handle, win32defines.GWL_EXSTYLE)

def controlid(handle):
    if False:
        for i in range(10):
            print('nop')
    'Return the ID of the control'
    return win32functions.GetWindowLong(handle, win32defines.GWL_ID)

def userdata(handle):
    if False:
        for i in range(10):
            print('nop')
    'Return the value of any user data associated with the window'
    return win32functions.GetWindowLong(handle, win32defines.GWL_USERDATA)

def contexthelpid(handle):
    if False:
        while True:
            i = 10
    'Return the context help id of the window'
    return win32functions.GetWindowContextHelpId(handle)

def iswindow(handle):
    if False:
        return 10
    'Return True if the handle is a window'
    return False if handle is None else bool(win32functions.IsWindow(handle))

def isvisible(handle):
    if False:
        i = 10
        return i + 15
    'Return True if the window is visible'
    return False if handle is None else bool(win32functions.IsWindowVisible(handle))

def isunicode(handle):
    if False:
        i = 10
        return i + 15
    'Return True if the window is a Unicode window'
    return False if handle is None else bool(win32functions.IsWindowUnicode(handle))

def isenabled(handle):
    if False:
        return 10
    'Return True if the window is enabled'
    return False if handle is None else bool(win32functions.IsWindowEnabled(handle))

def is64bitprocess(process_id):
    if False:
        print('Hello World!')
    'Return True if the specified process is a 64-bit process on x64\n\n    Return False if it is only a 32-bit process running under Wow64.\n    Always return False for x86.\n    '
    from .base_application import ProcessNotFoundError
    from .sysinfo import is_x64_OS
    is32 = True
    if is_x64_OS():
        try:
            phndl = win32api.OpenProcess(win32con.MAXIMUM_ALLOWED, 0, process_id)
            if phndl:
                is32 = win32process.IsWow64Process(phndl)
        except win32gui.error as e:
            if e.winerror == win32defines.ERROR_INVALID_PARAMETER:
                raise ProcessNotFoundError
            else:
                raise e
    return not is32

def is64bitbinary(filename):
    if False:
        print('Hello World!')
    'Check if the file is 64-bit binary'
    import win32file
    try:
        binary_type = win32file.GetBinaryType(filename)
        return binary_type != win32file.SCS_32BIT_BINARY
    except Exception as exc:
        warnings.warn('Cannot get binary type for file "{}". Error: {}'.format(filename, exc), RuntimeWarning, stacklevel=2)
        return None

def clientrect(handle):
    if False:
        for i in range(10):
            print('nop')
    'Return the client rectangle of the control'
    client_rect = win32structures.RECT()
    win32functions.GetClientRect(handle, byref(client_rect))
    return client_rect

def rectangle(handle):
    if False:
        i = 10
        return i + 15
    'Return the rectangle of the window'
    rect = win32structures.RECT()
    win32functions.GetWindowRect(handle, byref(rect))
    return rect

def font(handle):
    if False:
        while True:
            i = 10
    'Return the font as a LOGFONTW of the window'
    if handle is None:
        handle = 0
    font_handle = win32functions.SendMessage(handle, win32defines.WM_GETFONT, 0, 0)
    if not font_handle:
        font_handle = win32functions.GetStockObject(win32defines.DEFAULT_GUI_FONT)
        if not font_handle:
            if win32functions.GetSystemMetrics(win32defines.SM_DBCSENABLED):
                font_handle = win32functions.GetStockObject(win32defines.SYSTEM_FONT)
            else:
                font_handle = win32functions.GetStockObject(win32defines.ANSI_VAR_FONT)
    fontval = win32structures.LOGFONTW()
    ret = win32functions.GetObject(font_handle, sizeof(fontval), byref(fontval))
    if not ret:
        fontval = win32structures.LOGFONTW()
    if is_toplevel_window(handle):
        if 'MS Shell Dlg' in fontval.lfFaceName or fontval.lfFaceName == 'System':
            ncms = win32structures.NONCLIENTMETRICSW()
            ncms.cbSize = sizeof(ncms)
            win32functions.SystemParametersInfo(win32defines.SPI_GETNONCLIENTMETRICS, sizeof(ncms), byref(ncms), 0)
            if has_style(handle, win32defines.WS_EX_TOOLWINDOW) or has_style(handle, win32defines.WS_EX_PALETTEWINDOW):
                fontval = ncms.lfSmCaptionFont
            else:
                fontval = ncms.lfCaptionFont
    return fontval

def processid(handle):
    if False:
        i = 10
        return i + 15
    'Return the ID of process that controls this window'
    pid = wintypes.DWORD()
    win32functions.GetWindowThreadProcessId(handle, byref(pid))
    return pid.value

def has_enough_privileges(process_id):
    if False:
        for i in range(10):
            print('nop')
    'Check if target process has enough rights to query GUI actions'
    try:
        access_level = win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ
        process_handle = win32api.OpenProcess(access_level, 0, process_id)
        if process_handle:
            win32api.CloseHandle(process_handle)
            return True
        return False
    except win32gui.error:
        return False

def children(handle):
    if False:
        return 10
    'Return a list of handles to the children of this window'
    child_windows = []

    def enum_child_proc(hwnd, lparam):
        if False:
            for i in range(10):
                print('nop')
        'Called for each child - adds child hwnd to list'
        child_windows.append(hwnd)
        return True
    enum_child_proc_t = WINFUNCTYPE(c_int, wintypes.HWND, wintypes.LPARAM)
    proc = enum_child_proc_t(enum_child_proc)
    win32functions.EnumChildWindows(handle, proc, 0)
    return child_windows

def has_style(handle, tocheck):
    if False:
        for i in range(10):
            print('nop')
    'Return True if the control has style tocheck'
    hwnd_style = style(handle)
    return tocheck & hwnd_style == tocheck

def has_exstyle(handle, tocheck):
    if False:
        while True:
            i = 10
    'Return True if the control has extended style tocheck'
    hwnd_exstyle = exstyle(handle)
    return tocheck & hwnd_exstyle == tocheck

def is_toplevel_window(handle):
    if False:
        return 10
    'Return whether the window is a top level window or not'
    style_ = style(handle)
    if (style_ & win32defines.WS_OVERLAPPED == win32defines.WS_OVERLAPPED or style_ & win32defines.WS_CAPTION == win32defines.WS_CAPTION) and (not style_ & win32defines.WS_CHILD == win32defines.WS_CHILD):
        return True
    else:
        return False

def dumpwindow(handle):
    if False:
        while True:
            i = 10
    'Dump a window to a set of properties'
    props = {}
    for func in (text, classname, rectangle, clientrect, style, exstyle, contexthelpid, controlid, userdata, font, parent, processid, isenabled, isunicode, isvisible, children):
        props[func.__name__] = func(handle)
    return props