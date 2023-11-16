"""Definition of Windows structures"""
from ctypes import c_int, c_long, c_void_p, c_char, POINTER, sizeof, alignment, Union, c_longlong, c_size_t, wintypes
from ..base_types import Structure
from ..base_types import StructureMixIn
from ..base_types import PointIteratorMixin
from ..base_types import RectExtMixin
from ..base_types import _reduce
from .win32defines import LF_FACESIZE
from pywinauto import sysinfo
BOOL = wintypes.BOOL
BYTE = wintypes.BYTE
CHAR = c_char
DWORD = wintypes.DWORD
HANDLE = wintypes.HANDLE
HBITMAP = HANDLE
LONG = wintypes.LONG
LPVOID = wintypes.LPVOID
PVOID = c_void_p
UINT = wintypes.UINT
WCHAR = wintypes.WCHAR
WORD = wintypes.WORD
LRESULT = wintypes.LPARAM
COLORREF = wintypes.COLORREF
LPBYTE = POINTER(BYTE)
LPWSTR = c_size_t
DWORD_PTR = UINT_PTR = ULONG_PTR = c_size_t
PDWORD_PTR = POINTER(DWORD_PTR)
if sysinfo.is_x64_Python():
    INT_PTR = LONG_PTR = c_longlong
else:
    INT_PTR = LONG_PTR = c_long
HINSTANCE = LONG_PTR
HMENU = LONG_PTR
HBRUSH = wintypes.HBRUSH
HTREEITEM = LONG_PTR
HWND = wintypes.HWND
LPARAM = wintypes.LPARAM
WPARAM = wintypes.WPARAM

class POINT(wintypes.POINT, PointIteratorMixin, StructureMixIn):
    pass
assert sizeof(POINT) == 8, sizeof(POINT)
assert alignment(POINT) == 4, alignment(POINT)
RectExtMixin._POINT = POINT

class RECT(wintypes.RECT, RectExtMixin, StructureMixIn):
    """Wrap the wintypes.RECT structure and add extra functionality"""

    def __init__(self, other=0, top=0, right=0, bottom=0):
        if False:
            while True:
                i = 10
        '\n        Try to construct RECT from wintypes.RECT otherwise pass it down to RecExtMixin\n        '
        if isinstance(other, wintypes.RECT):
            self.left = other.left
            self.right = other.right
            self.top = other.top
            self.bottom = other.bottom
        else:
            RectExtMixin.__init__(self, other, top, right, bottom)
assert sizeof(RECT) == 16, sizeof(RECT)
assert alignment(RECT) == 4, alignment(RECT)
RectExtMixin._RECT = RECT

class SETTEXTEX(Structure):
    _pack_ = 1
    _fields_ = [('flags', DWORD), ('codepage', UINT)]
assert sizeof(SETTEXTEX) == 8, sizeof(SETTEXTEX)

class LVCOLUMNW(Structure):
    """The main layout for LVCOLUMN on x86 and x64 archs"""
    _fields_ = [('mask', UINT), ('fmt', c_int), ('cx', c_int), ('pszText', c_void_p), ('cchTextMax', c_int), ('iSubItem', c_int), ('iImage', c_int), ('iOrder', c_int), ('cxMin', c_int), ('cxDefault', c_int), ('cxIdeal', c_int)]

class LVCOLUMNW32(Structure):
    """A special layout for LVCOLUMN for a 32-bit process running on x64"""
    _fields_ = [('mask', UINT), ('fmt', c_int), ('cx', c_int), ('pszText', UINT), ('cchTextMax', c_int), ('iSubItem', c_int), ('iImage', c_int), ('iOrder', c_int), ('cxMin', c_int), ('cxDefault', c_int), ('cxIdeal', c_int)]

class LVITEMW(Structure):
    """The main layout for LVITEM, naturally fits for x86 and x64 archs"""
    _fields_ = [('mask', UINT), ('iItem', c_int), ('iSubItem', c_int), ('state', UINT), ('stateMask', UINT), ('pszText', c_void_p), ('cchTextMax', c_int), ('iImage', c_int), ('lParam', LPARAM), ('iIndent', c_int), ('iGroupId', c_int), ('cColumns', UINT), ('puColumns', POINTER(UINT)), ('piColFmt', POINTER(c_int)), ('iGroup', c_int)]
if sysinfo.is_x64_Python():
    assert sizeof(LVITEMW) == 88, sizeof(LVITEMW)
    assert alignment(LVITEMW) == 8, alignment(LVITEMW)
else:
    assert sizeof(LVITEMW) == 60, sizeof(LVITEMW)
    assert alignment(LVITEMW) == 4, alignment(LVITEMW)

class LVITEMW32(Structure):
    """A special layout for LVITEM for a 32-bit process running on x64"""
    _pack_ = 4
    _fields_ = [('mask', UINT), ('iItem', c_int), ('iSubItem', c_int), ('state', UINT), ('stateMask', UINT), ('pszText', UINT), ('cchTextMax', c_int), ('iImage', c_int), ('lParam', LPARAM), ('iIndent', c_int), ('iGroupId', c_int), ('cColumns', UINT), ('puColumns', UINT), ('piColFmt', c_int), ('iGroup', c_int)]
assert alignment(LVITEMW32) == 4, alignment(LVITEMW32)

class TVITEMW(Structure):
    """The main layout for TVITEM, naturally fits for x86 and x64 archs"""
    _fields_ = [('mask', UINT), ('hItem', HTREEITEM), ('state', UINT), ('stateMask', UINT), ('pszText', LPWSTR), ('cchTextMax', c_int), ('iImage', c_int), ('iSelectedImage', c_int), ('cChildren', c_int), ('lParam', LPARAM)]
if sysinfo.is_x64_Python():
    assert sizeof(TVITEMW) == 56, sizeof(TVITEMW)
    assert alignment(TVITEMW) == 8, alignment(TVITEMW)
else:
    assert sizeof(TVITEMW) == 40, sizeof(TVITEMW)
    assert alignment(TVITEMW) == 4, alignment(TVITEMW)

class TVITEMW32(Structure):
    """An additional layout for TVITEM, used in a combination of 64-bit python and 32-bit app"""
    _fields_ = [('mask', UINT), ('hItem', UINT), ('state', UINT), ('stateMask', UINT), ('pszText', UINT), ('cchTextMax', c_int), ('iImage', c_int), ('iSelectedImage', c_int), ('cChildren', c_int), ('lParam', UINT)]
assert sizeof(TVITEMW32) == 40, sizeof(TVITEMW32)
assert alignment(TVITEMW32) == 4, alignment(TVITEMW32)

class NMHDR(Structure):
    _fields_ = [('hwndFrom', HWND), ('idFrom', UINT_PTR), ('code', UINT)]
if sysinfo.is_x64_Python():
    assert sizeof(NMHDR) == 24, sizeof(NMHDR)
    assert alignment(NMHDR) == 8, alignment(NMHDR)
else:
    assert sizeof(NMHDR) == 12, sizeof(NMHDR)
    assert alignment(NMHDR) == 4, alignment(NMHDR)

class NMTVDISPINFOW(Structure):
    _pack_ = 1
    _fields_ = [('hdr', NMHDR), ('item', TVITEMW)]
assert alignment(NMTVDISPINFOW) == 1, alignment(NMTVDISPINFOW)

class LOGFONTW(Structure):
    _fields_ = [('lfHeight', LONG), ('lfWidth', LONG), ('lfEscapement', LONG), ('lfOrientation', LONG), ('lfWeight', LONG), ('lfItalic', BYTE), ('lfUnderline', BYTE), ('lfStrikeOut', BYTE), ('lfCharSet', BYTE), ('lfOutPrecision', BYTE), ('lfClipPrecision', BYTE), ('lfQuality', BYTE), ('lfPitchAndFamily', BYTE), ('lfFaceName', WCHAR * LF_FACESIZE)]

    def __str__(self):
        if False:
            return 10
        return "('%s' %d)" % (self.lfFaceName, self.lfHeight)

    def __repr__(self):
        if False:
            return 10
        return "<LOGFONTW '%s' %d>" % (self.lfFaceName, self.lfHeight)
LOGFONTW.__reduce__ = _reduce
assert sizeof(LOGFONTW) == 92, sizeof(LOGFONTW)
assert alignment(LOGFONTW) == 4, alignment(LOGFONTW)

class TEXTMETRICW(Structure):
    _pack_ = 2
    _fields_ = [('tmHeight', LONG), ('tmAscent', LONG), ('tmDescent', LONG), ('tmInternalLeading', LONG), ('tmExternalLeading', LONG), ('tmAveCharWidth', LONG), ('tmMaxCharWidth', LONG), ('tmWeight', LONG), ('tmOverhang', LONG), ('tmDigitizedAspectX', LONG), ('tmDigitizedAspectY', LONG), ('tmFirstChar', WCHAR), ('tmLastChar', WCHAR), ('tmDefaultChar', WCHAR), ('tmBreakChar', WCHAR), ('tmItalic', BYTE), ('tmUnderlined', BYTE), ('tmStruckOut', BYTE), ('tmPitchAndFamily', BYTE), ('tmCharSet', BYTE)]
assert sizeof(TEXTMETRICW) == 58, sizeof(TEXTMETRICW)
assert alignment(TEXTMETRICW) == 2, alignment(TEXTMETRICW)

class NONCLIENTMETRICSW(Structure):
    _pack_ = 2
    _fields_ = [('cbSize', UINT), ('iBorderWidth', c_int), ('iScrollWidth', c_int), ('iScrollHeight', c_int), ('iCaptionWidth', c_int), ('iCaptionHeight', c_int), ('lfCaptionFont', LOGFONTW), ('iSmCaptionWidth', c_int), ('iSmCaptionHeight', c_int), ('lfSmCaptionFont', LOGFONTW), ('iMenuWidth', c_int), ('iMenuHeight', c_int), ('lfMenuFont', LOGFONTW), ('lfStatusFont', LOGFONTW), ('lfMessageFont', LOGFONTW)]
assert sizeof(NONCLIENTMETRICSW) == 500, sizeof(NONCLIENTMETRICSW)
assert alignment(NONCLIENTMETRICSW) == 2, alignment(NONCLIENTMETRICSW)

class LOGBRUSH(Structure):
    _fields_ = [('lbStyle', UINT), ('lbColor', COLORREF), ('lbHatch', LONG)]
assert sizeof(LOGBRUSH) == 12, sizeof(LOGBRUSH)
assert alignment(LOGBRUSH) == 4, alignment(LOGBRUSH)

class MENUITEMINFOW(Structure):
    _fields_ = [('cbSize', UINT), ('fMask', UINT), ('fType', UINT), ('fState', UINT), ('wID', UINT), ('hSubMenu', HMENU), ('hbmpChecked', HBITMAP), ('hbmpUnchecked', HBITMAP), ('dwItemData', ULONG_PTR), ('dwTypeData', LPWSTR), ('cch', UINT), ('hbmpItem', HBITMAP)]
if sysinfo.is_x64_Python():
    assert sizeof(MENUITEMINFOW) == 80, sizeof(MENUITEMINFOW)
    assert alignment(MENUITEMINFOW) == 8, alignment(MENUITEMINFOW)
else:
    assert sizeof(MENUITEMINFOW) == 48, sizeof(MENUITEMINFOW)
    assert alignment(MENUITEMINFOW) == 4, alignment(MENUITEMINFOW)

class MENUBARINFO(Structure):
    _fields_ = [('cbSize', DWORD), ('rcBar', RECT), ('hMenu', HMENU), ('hwndMenu', HWND), ('fBarFocused', BOOL, 1), ('fFocused', BOOL, 1)]

class MSG(Structure):
    _fields_ = [('hwnd', HWND), ('message', UINT), ('wParam', WPARAM), ('lParam', LPARAM), ('time', DWORD), ('pt', POINT)]
if sysinfo.is_x64_Python():
    assert sizeof(MSG) == 48, sizeof(MSG)
    assert alignment(MSG) == 8, alignment(MSG)
else:
    assert sizeof(MSG) == 28, sizeof(MSG)
    assert alignment(MSG) == 4, alignment(MSG)

class TOOLINFOW(Structure):
    _fields_ = [('cbSize', UINT), ('uFlags', UINT), ('hwnd', HWND), ('uId', UINT_PTR), ('rect', RECT), ('hinst', HINSTANCE), ('lpszText', LPWSTR), ('lParam', LPARAM), ('lpReserved', LPVOID)]
if sysinfo.is_x64_Python():
    assert sizeof(TOOLINFOW) == 72, sizeof(TOOLINFOW)
    assert alignment(TOOLINFOW) == 8, alignment(TOOLINFOW)
else:
    assert sizeof(TOOLINFOW) == 48, sizeof(TOOLINFOW)
    assert alignment(TOOLINFOW) == 4, alignment(TOOLINFOW)

class HDITEMW(Structure):
    _fields_ = [('mask', UINT), ('cxy', c_int), ('pszText', LPWSTR), ('hbm', HBITMAP), ('cchTextMax', c_int), ('fmt', c_int), ('lParam', LPARAM), ('iImage', c_int), ('iOrder', c_int), ('type', UINT), ('pvFilter', LPVOID), ('state', UINT)]
if sysinfo.is_x64_Python():
    assert sizeof(HDITEMW) == 72, sizeof(HDITEMW)
    assert alignment(HDITEMW) == 8, alignment(HDITEMW)
else:
    assert sizeof(HDITEMW) == 48, sizeof(HDITEMW)
    assert alignment(HDITEMW) == 4, alignment(HDITEMW)

class COMBOBOXEXITEMW(Structure):
    _fields_ = [('mask', UINT), ('iItem', INT_PTR), ('pszText', LPWSTR), ('cchTextMax', c_int), ('iImage', c_int), ('iSelectedImage', c_int), ('iOverlay', c_int), ('iIndent', c_int), ('lParam', LPARAM)]
if sysinfo.is_x64_Python():
    assert sizeof(COMBOBOXEXITEMW) == 56, sizeof(COMBOBOXEXITEMW)
    assert alignment(COMBOBOXEXITEMW) == 8, alignment(COMBOBOXEXITEMW)
else:
    assert sizeof(COMBOBOXEXITEMW) == 36, sizeof(COMBOBOXEXITEMW)
    assert alignment(COMBOBOXEXITEMW) == 4, alignment(COMBOBOXEXITEMW)

class TCITEMHEADERW(Structure):
    _fields_ = [('mask', UINT), ('lpReserved1', UINT), ('lpReserved2', UINT), ('pszText', LPWSTR), ('cchTextMax', c_int), ('iImage', c_int)]
if sysinfo.is_x64_Python():
    assert sizeof(TCITEMHEADERW) == 32, sizeof(TCITEMHEADERW)
    assert alignment(TCITEMHEADERW) == 8, alignment(TCITEMHEADERW)
else:
    assert sizeof(TCITEMHEADERW) == 24, sizeof(TCITEMHEADERW)
    assert alignment(TCITEMHEADERW) == 4, alignment(TCITEMHEADERW)

class TCITEMW(Structure):
    _fields_ = [('mask', UINT), ('dwState', DWORD), ('dwStateMask', DWORD), ('pszText', LPWSTR), ('cchTextMax', c_int), ('iImage', c_int), ('lParam', LPARAM)]
if sysinfo.is_x64_Python():
    assert sizeof(TCITEMW) == 40, sizeof(TCITEMW)
    assert alignment(TCITEMW) == 8, alignment(TCITEMW)
else:
    assert sizeof(TCITEMW) == 28, sizeof(TCITEMW)
    assert alignment(TCITEMW) == 4, alignment(TCITEMW)

class TBBUTTONINFOW(Structure):
    _fields_ = [('cbSize', UINT), ('dwMask', DWORD), ('idCommand', c_int), ('iImage', c_int), ('fsState', BYTE), ('fsStyle', BYTE), ('cx', WORD), ('lParam', POINTER(DWORD)), ('pszText', LPWSTR), ('cchText', c_int)]
if sysinfo.is_x64_Python():
    assert sizeof(TBBUTTONINFOW) == 48, sizeof(TBBUTTONINFOW)
    assert alignment(TBBUTTONINFOW) == 8, alignment(TBBUTTONINFOW)
else:
    assert sizeof(TBBUTTONINFOW) == 32, sizeof(TBBUTTONINFOW)
    assert alignment(TBBUTTONINFOW) == 4, alignment(TBBUTTONINFOW)

class TBBUTTONINFOW32(Structure):
    _fields_ = [('cbSize', UINT), ('dwMask', DWORD), ('idCommand', c_int), ('iImage', c_int), ('fsState', BYTE), ('fsStyle', BYTE), ('cx', WORD), ('lParam', UINT), ('pszText', UINT), ('cchText', c_int)]
assert sizeof(TBBUTTONINFOW32) == 32, sizeof(TBBUTTONINFOW32)
assert alignment(TBBUTTONINFOW32) == 4, alignment(TBBUTTONINFOW32)
if sysinfo.is_x64_Python():

    class TBBUTTON(Structure):
        _fields_ = [('iBitmap', c_int), ('idCommand', c_int), ('fsState', BYTE), ('fsStyle', BYTE), ('bReserved', BYTE * 6), ('dwData', DWORD_PTR), ('iString', INT_PTR)]
else:

    class TBBUTTON(Structure):
        _fields_ = [('iBitmap', c_int), ('idCommand', c_int), ('fsState', BYTE), ('fsStyle', BYTE), ('bReserved', BYTE * 2), ('dwData', DWORD_PTR), ('iString', INT_PTR)]
if sysinfo.is_x64_Python():
    assert sizeof(TBBUTTON) == 32, sizeof(TBBUTTON)
    assert alignment(TBBUTTON) == 8, alignment(TBBUTTON)
else:
    assert sizeof(TBBUTTON) == 20, sizeof(TBBUTTON)
    assert alignment(TBBUTTON) == 4, alignment(TBBUTTON)

class TBBUTTON32(Structure):
    _fields_ = [('iBitmap', c_int), ('idCommand', c_int), ('fsState', BYTE), ('fsStyle', BYTE), ('bReserved', BYTE * 2), ('dwData', UINT), ('iString', UINT)]
assert sizeof(TBBUTTON32) == 20, sizeof(TBBUTTON32)
assert alignment(TBBUTTON32) == 4, alignment(TBBUTTON32)

class REBARBANDINFOW(Structure):
    _fields_ = [('cbSize', UINT), ('fMask', UINT), ('fStyle', UINT), ('clrFore', COLORREF), ('clrBack', COLORREF), ('lpText', LPWSTR), ('cch', UINT), ('iImage', c_int), ('hwndChild', HWND), ('cxMinChild', UINT), ('cyMinChild', UINT), ('cx', UINT), ('hbmBack', HBITMAP), ('wID', UINT), ('cyChild', UINT), ('cyMaxChild', UINT), ('cyIntegral', UINT), ('cxIdeal', UINT), ('lParam', LPARAM), ('cxHeader', UINT), ('rcChevronLocation', RECT), ('uChevronState', UINT)]
if sysinfo.is_x64_Python():
    assert sizeof(REBARBANDINFOW) == 128, sizeof(REBARBANDINFOW)
    assert alignment(REBARBANDINFOW) == 8, alignment(REBARBANDINFOW)
else:
    assert sizeof(REBARBANDINFOW) == 100, sizeof(REBARBANDINFOW)
    assert alignment(REBARBANDINFOW) == 4, alignment(REBARBANDINFOW)

class SECURITY_ATTRIBUTES(Structure):
    _fields_ = [('nLength', DWORD), ('lpSecurityDescriptor', LPVOID), ('bInheritHandle', BOOL)]
assert sizeof(SECURITY_ATTRIBUTES) == 12 or sizeof(SECURITY_ATTRIBUTES) == 24, sizeof(SECURITY_ATTRIBUTES)
assert alignment(SECURITY_ATTRIBUTES) == 4 or alignment(SECURITY_ATTRIBUTES) == 8, alignment(SECURITY_ATTRIBUTES)

class STARTUPINFOW(Structure):
    _fields_ = [('cb', DWORD), ('lpReserved', LPWSTR), ('lpDesktop', LPWSTR), ('lpTitle', LPWSTR), ('dwX', DWORD), ('dwY', DWORD), ('dwXSize', DWORD), ('dwYSize', DWORD), ('dwXCountChars', DWORD), ('dwYCountChars', DWORD), ('dwFillAttribute', DWORD), ('dwFlags', DWORD), ('wShowWindow', WORD), ('cbReserved2', WORD), ('lpReserved2', LPBYTE), ('hStdInput', HANDLE), ('hStdOutput', HANDLE), ('hStdError', HANDLE)]
assert sizeof(STARTUPINFOW) == 68 or sizeof(STARTUPINFOW) == 104, sizeof(STARTUPINFOW)
assert alignment(STARTUPINFOW) == 4 or alignment(STARTUPINFOW) == 8, alignment(STARTUPINFOW)

class PROCESS_INFORMATION(Structure):
    _fields_ = [('hProcess', HANDLE), ('hThread', HANDLE), ('dwProcessId', DWORD), ('dwThreadId', DWORD)]
assert sizeof(PROCESS_INFORMATION) == 16 or sizeof(PROCESS_INFORMATION) == 24, sizeof(PROCESS_INFORMATION)
assert alignment(PROCESS_INFORMATION) == 4 or alignment(PROCESS_INFORMATION) == 8, alignment(PROCESS_INFORMATION)

class NMLISTVIEW(Structure):
    _fields_ = [('hdr', NMHDR), ('iItem', c_int), ('iSubItem', c_int), ('uNewState', UINT), ('uOldState', UINT), ('uChanged', UINT), ('ptAction', POINT), ('lParam', LPARAM)]
if sysinfo.is_x64_Python():
    assert sizeof(NMLISTVIEW) == 64, sizeof(NMLISTVIEW)
    assert alignment(NMLISTVIEW) == 8, alignment(NMLISTVIEW)
else:
    assert sizeof(NMLISTVIEW) == 44, sizeof(NMLISTVIEW)
    assert alignment(NMLISTVIEW) == 4, alignment(NMLISTVIEW)

class NMMOUSE(Structure):
    _fields_ = [('hdr', NMHDR), ('dwItemSpec', DWORD_PTR), ('dwItemData', DWORD_PTR), ('pt', POINT), ('dwHitInfo', LPARAM)]
if sysinfo.is_x64_Python():
    assert sizeof(NMMOUSE) == 56, sizeof(NMMOUSE)
    assert alignment(NMMOUSE) == 8, alignment(NMMOUSE)
else:
    assert sizeof(NMMOUSE) == 32, sizeof(NMMOUSE)
    assert alignment(NMMOUSE) == 4, alignment(NMMOUSE)

class MOUSEINPUT(Structure):
    if sysinfo.is_x64_Python():
        _pack_ = 8
    else:
        _pack_ = 2
    _fields_ = [('dx', LONG), ('dy', LONG), ('mouseData', DWORD), ('dwFlags', DWORD), ('time', DWORD), ('dwExtraInfo', ULONG_PTR)]
if sysinfo.is_x64_Python():
    assert sizeof(MOUSEINPUT) == 32, sizeof(MOUSEINPUT)
    assert alignment(MOUSEINPUT) == 8, alignment(MOUSEINPUT)
else:
    assert sizeof(MOUSEINPUT) == 24, sizeof(MOUSEINPUT)
    assert alignment(MOUSEINPUT) == 2, alignment(MOUSEINPUT)

class KEYBDINPUT(Structure):
    if sysinfo.is_x64_Python():
        _pack_ = 8
    else:
        _pack_ = 2
    _fields_ = [('wVk', WORD), ('wScan', WORD), ('dwFlags', DWORD), ('time', DWORD), ('dwExtraInfo', ULONG_PTR)]
if sysinfo.is_x64_Python():
    assert sizeof(KEYBDINPUT) == 24, sizeof(KEYBDINPUT)
    assert alignment(KEYBDINPUT) == 8, alignment(KEYBDINPUT)
else:
    assert sizeof(KEYBDINPUT) == 16, sizeof(KEYBDINPUT)
    assert alignment(KEYBDINPUT) == 2, alignment(KEYBDINPUT)

class HARDWAREINPUT(Structure):
    if sysinfo.is_x64_Python():
        _pack_ = 8
    else:
        _pack_ = 2
    _fields_ = [('uMsg', DWORD), ('wParamL', WORD), ('wParamH', WORD)]
assert sizeof(HARDWAREINPUT) == 8, sizeof(HARDWAREINPUT)
if sysinfo.is_x64_Python():
    assert alignment(HARDWAREINPUT) == 4, alignment(HARDWAREINPUT)
else:
    assert alignment(HARDWAREINPUT) == 2, alignment(HARDWAREINPUT)

class UNION_INPUT_STRUCTS(Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]
if sysinfo.is_x64_Python():
    assert sizeof(UNION_INPUT_STRUCTS) == 32, sizeof(UNION_INPUT_STRUCTS)
    assert alignment(UNION_INPUT_STRUCTS) == 8, alignment(UNION_INPUT_STRUCTS)
else:
    assert sizeof(UNION_INPUT_STRUCTS) == 24, sizeof(UNION_INPUT_STRUCTS)
    assert alignment(UNION_INPUT_STRUCTS) == 2, alignment(UNION_INPUT_STRUCTS)

class INPUT(Structure):
    if sysinfo.is_x64_Python():
        _pack_ = 8
    else:
        _pack_ = 2
    _anonymous_ = ('_',)
    _fields_ = [('type', c_int), ('_', UNION_INPUT_STRUCTS)]
if sysinfo.is_x64_Python():
    assert sizeof(INPUT) == 40, sizeof(INPUT)
    assert alignment(INPUT) == 8, alignment(INPUT)
else:
    assert sizeof(INPUT) == 28, sizeof(INPUT)
    assert alignment(INPUT) == 2, alignment(INPUT)

class NMUPDOWN(Structure):
    _pack_ = 1
    _fields_ = [('hdr', NMHDR), ('iPos', c_int), ('iDelta', c_int)]
if sysinfo.is_x64_Python():
    assert sizeof(NMUPDOWN) == 32, sizeof(NMUPDOWN)
    assert alignment(NMUPDOWN) == 1, alignment(NMUPDOWN)
else:
    assert sizeof(NMUPDOWN) == 20, sizeof(NMUPDOWN)
    assert alignment(NMUPDOWN) == 1, alignment(NMUPDOWN)

class GUITHREADINFO(Structure):
    _pack_ = 2
    _fields_ = [('cbSize', DWORD), ('flags', DWORD), ('hwndActive', HWND), ('hwndFocus', HWND), ('hwndCapture', HWND), ('hwndMenuOwner', HWND), ('hwndMoveSize', HWND), ('hwndCaret', HWND), ('rcCaret', RECT)]
if sysinfo.is_x64_Python():
    assert sizeof(GUITHREADINFO) == 72, sizeof(GUITHREADINFO)
    assert alignment(GUITHREADINFO) == 2, alignment(GUITHREADINFO)
else:
    assert sizeof(GUITHREADINFO) == 48, sizeof(GUITHREADINFO)
    assert alignment(GUITHREADINFO) == 2, alignment(GUITHREADINFO)

class MENUINFO(Structure):
    _fields_ = [('cbSize', DWORD), ('fMask', DWORD), ('dwStyle', DWORD), ('cyMax', UINT), ('hbrBack', HBRUSH), ('dwContextHelpID', DWORD), ('dwMenuData', ULONG_PTR)]
if sysinfo.is_x64_Python():
    assert sizeof(MENUINFO) == 40, sizeof(MENUINFO)
    assert alignment(MENUINFO) == 8, alignment(MENUINFO)
else:
    assert sizeof(MENUINFO) == 28, sizeof(MENUINFO)
    assert alignment(MENUINFO) == 4, alignment(MENUINFO)

class NMTTDISPINFOW(Structure):
    _fields_ = [('hdr', NMHDR), ('lpszText', LPWSTR), ('szText', WCHAR * 80), ('hinst', HINSTANCE), ('uFlags', UINT), ('lParam', LPARAM)]
if sysinfo.is_x64_Python():
    assert sizeof(NMTTDISPINFOW) == 216, sizeof(NMTTDISPINFOW)
    assert alignment(NMTTDISPINFOW) == 8, alignment(NMTTDISPINFOW)
else:
    assert sizeof(NMTTDISPINFOW) == 188, sizeof(NMTTDISPINFOW)
    assert alignment(NMTTDISPINFOW) == 4, alignment(NMTTDISPINFOW)

class WINDOWPLACEMENT(Structure):
    _fields_ = [('length', UINT), ('flags', UINT), ('showCmd', UINT), ('ptMinPosition', POINT), ('ptMaxPosition', POINT), ('rcNormalPosition', RECT)]
assert sizeof(WINDOWPLACEMENT) == 44, sizeof(WINDOWPLACEMENT)
assert alignment(WINDOWPLACEMENT) == 4, alignment(WINDOWPLACEMENT)

class LVHITTESTINFO(Structure):
    _fields_ = [('pt', POINT), ('flags', UINT), ('iItem', c_int), ('iSubItem', c_int), ('iGroup', c_int)]
assert sizeof(LVHITTESTINFO) == 24, sizeof(LVHITTESTINFO)
assert alignment(LVHITTESTINFO) == 4, alignment(LVHITTESTINFO)

class TVHITTESTINFO(Structure):
    _fields_ = [('pt', POINT), ('flags', UINT), ('hItem', HTREEITEM)]
if sysinfo.is_x64_Python():
    assert sizeof(TVHITTESTINFO) == 24, sizeof(TVHITTESTINFO)
    assert alignment(TVHITTESTINFO) == 8, alignment(TVHITTESTINFO)
else:
    assert sizeof(TVHITTESTINFO) == 16, sizeof(TVHITTESTINFO)
    assert alignment(TVHITTESTINFO) == 4, alignment(TVHITTESTINFO)

class LOGFONTA(Structure):
    _fields_ = [('lfHeight', LONG), ('lfHeight', LONG), ('lfHeight', LONG), ('lfHeight', LONG), ('lfHeight', LONG), ('lfHeight', LONG), ('lfHeight', LONG), ('lfHeight', LONG), ('lfHeight', LONG)]

class GV_ITEM(Structure):
    _pack_ = 1
    _fields_ = [('row', c_int), ('col', c_int), ('mask', UINT), ('state', UINT), ('nFormat', UINT)]

class SYSTEMTIME(Structure):
    """Wrap the SYSTEMTIME structure"""
    _fields_ = [('wYear', WORD), ('wMonth', WORD), ('wDayOfWeek', WORD), ('wDay', WORD), ('wHour', WORD), ('wMinute', WORD), ('wSecond', WORD), ('wMilliseconds', WORD)]

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<wYear=' + str(self.wYear) + ', wMonth=' + str(self.wMonth) + ', wDayOfWeek=' + str(self.wDayOfWeek) + ', wDay=' + str(self.wDay) + ', wHour=' + str(self.wHour) + ', wMinute=' + str(self.wMinute) + ', wSecond=' + str(self.wSecond) + ', wMilliseconds=' + str(self.wMilliseconds) + '>'

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.__repr__()
assert sizeof(SYSTEMTIME) == 16, sizeof(SYSTEMTIME)

class MCHITTESTINFO(Structure):
    _fields_ = [('cbSize', UINT), ('pt', POINT), ('uHit', UINT), ('st', SYSTEMTIME), ('rc', RECT), ('iOffset', c_int), ('iRow', c_int), ('iCol', c_int)]

class KBDLLHOOKSTRUCT(Structure):
    """Wrap KBDLLHOOKSTRUCT structure"""
    _fields_ = [('vkCode', DWORD), ('scanCode', DWORD), ('flags', DWORD), ('time', DWORD), ('dwExtraInfo', DWORD)]
assert sizeof(KBDLLHOOKSTRUCT) == 20, sizeof(KBDLLHOOKSTRUCT)

class MSLLHOOKSTRUCT(Structure):
    """Wrap MSLLHOOKSTRUCT structure"""
    _fields_ = [('pt', POINT), ('mouseData', DWORD), ('flags', DWORD), ('time', DWORD), ('dwExtraInfo', DWORD)]
assert sizeof(MSLLHOOKSTRUCT) == 24, sizeof(MSLLHOOKSTRUCT)