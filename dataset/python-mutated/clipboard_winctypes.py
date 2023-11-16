"""
Clipboard windows: an implementation of the Clipboard using ctypes.
"""
__all__ = ('ClipboardWindows',)
from kivy.utils import platform
from kivy.core.clipboard import ClipboardBase
if platform != 'win':
    raise SystemError('unsupported platform for Windows clipboard')
import ctypes
from ctypes import wintypes
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
msvcrt = ctypes.cdll.msvcrt
c_char_p = ctypes.c_char_p
c_wchar_p = ctypes.c_wchar_p

class ClipboardWindows(ClipboardBase):

    def get(self, mimetype='text/plain'):
        if False:
            print('Hello World!')
        GetClipboardData = user32.GetClipboardData
        GetClipboardData.argtypes = [wintypes.UINT]
        GetClipboardData.restype = wintypes.HANDLE
        user32.OpenClipboard(user32.GetActiveWindow())
        pcontents = GetClipboardData(13)
        if not pcontents:
            user32.CloseClipboard()
            return ''
        data = c_wchar_p(pcontents).value.encode(self._encoding)
        user32.CloseClipboard()
        return data

    def put(self, text, mimetype='text/plain'):
        if False:
            i = 10
            return i + 15
        SetClipboardData = user32.SetClipboardData
        SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        SetClipboardData.restype = wintypes.HANDLE
        GlobalAlloc = kernel32.GlobalAlloc
        GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        GlobalAlloc.restype = wintypes.HGLOBAL
        user32.OpenClipboard(user32.GetActiveWindow())
        user32.EmptyClipboard()
        GMEM_FIXED = 0
        hCd = GlobalAlloc(GMEM_FIXED, len(text) + 2)
        msvcrt.wcscpy(c_wchar_p(hCd), text)
        CF_UNICODETEXT = 13
        SetClipboardData(CF_UNICODETEXT, hCd)
        user32.CloseClipboard()

    def get_types(self):
        if False:
            i = 10
            return i + 15
        return ['text/plain']