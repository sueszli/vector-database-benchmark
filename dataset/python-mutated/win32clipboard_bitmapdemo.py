import win32api
import win32clipboard
import win32con
import win32gui

class ViewerWindow:

    def __init__(self):
        if False:
            return 10
        self.hwndNextViewer = None

    def OnPaint(self, hwnd, msg, wp, lp):
        if False:
            for i in range(10):
                print('nop')
        (dc, ps) = win32gui.BeginPaint(hwnd)
        wndrect = win32gui.GetClientRect(hwnd)
        wndwidth = wndrect[2] - wndrect[0]
        wndheight = wndrect[3] - wndrect[1]
        win32clipboard.OpenClipboard()
        try:
            try:
                hbitmap = win32clipboard.GetClipboardData(win32clipboard.CF_BITMAP)
            except TypeError:
                font = win32gui.LOGFONT()
                font.lfHeight = 15
                font.lfWidth = 15
                hf = win32gui.CreateFontIndirect(font)
                win32gui.SelectObject(dc, hf)
                win32gui.SetBkMode(dc, win32con.TRANSPARENT)
                win32gui.SetTextColor(dc, win32api.RGB(0, 0, 0))
                win32gui.DrawText(dc, 'No bitmaps are in the clipboard\n(try pressing the PrtScn button)', -1, (0, 0, wndwidth, wndheight), win32con.DT_CENTER)
            else:
                bminfo = win32gui.GetObject(hbitmap)
                dcDC = win32gui.CreateCompatibleDC(None)
                win32gui.SelectObject(dcDC, hbitmap)
                win32gui.StretchBlt(dc, 0, 0, wndwidth, wndheight, dcDC, 0, 0, bminfo.bmWidth, bminfo.bmHeight, win32con.SRCCOPY)
                win32gui.DeleteDC(dcDC)
                win32gui.EndPaint(hwnd, ps)
        finally:
            win32clipboard.CloseClipboard()
        return 0

    def OnDrawClipboard(self, hwnd, msg, wp, lp):
        if False:
            for i in range(10):
                print('nop')
        win32gui.InvalidateRect(hwnd, None, True)

    def OnChangeCBChain(self, hwnd, msg, wp, lp):
        if False:
            for i in range(10):
                print('nop')
        if wp == self.hwndNextViewer:
            self.hwndNextViewer = lp
        elif self.hwndNextViewer:
            win32gui.SendMessage(self.hwndNextViewer, msg, wp, lp)

    def OnCreate(self, hwnd, msg, wp, lp):
        if False:
            return 10
        self.hwndNextViewer = win32gui.SetClipboardViewer(hwnd)

    def OnClose(self, hwnd, msg, wp, lp):
        if False:
            i = 10
            return i + 15
        win32clipboard.ChangeClipboardChain(hwnd, self.hwndNextViewer)
        win32gui.DestroyWindow(hwnd)
        win32gui.PostQuitMessage(0)

    def go(self):
        if False:
            i = 10
            return i + 15
        wndproc = {win32con.WM_PAINT: self.OnPaint, win32con.WM_CLOSE: self.OnClose, win32con.WM_CREATE: self.OnCreate, win32con.WM_DRAWCLIPBOARD: self.OnDrawClipboard, win32con.WM_CHANGECBCHAIN: self.OnChangeCBChain}
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = 'test_win32clipboard_bmp'
        wc.style = win32con.CS_GLOBALCLASS | win32con.CS_VREDRAW | win32con.CS_HREDRAW
        wc.hbrBackground = win32con.COLOR_WINDOW + 1
        wc.lpfnWndProc = wndproc
        class_atom = win32gui.RegisterClass(wc)
        hwnd = win32gui.CreateWindowEx(0, class_atom, 'ClipboardViewer', win32con.WS_CAPTION | win32con.WS_VISIBLE | win32con.WS_THICKFRAME | win32con.WS_SYSMENU, 100, 100, 900, 900, 0, 0, 0, None)
        win32clipboard.SetClipboardViewer(hwnd)
        win32gui.PumpMessages()
        win32gui.UnregisterClass(class_atom, None)
if __name__ == '__main__':
    w = ViewerWindow()
    w.go()