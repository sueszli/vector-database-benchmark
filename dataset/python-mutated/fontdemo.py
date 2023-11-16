import win32api
import win32con
import win32ui
from pywin.mfc import docview

class FontView(docview.ScrollView):

    def __init__(self, doc, text='Python Rules!', font_spec={'name': 'Arial', 'height': 42}):
        if False:
            for i in range(10):
                print('nop')
        docview.ScrollView.__init__(self, doc)
        self.font = win32ui.CreateFont(font_spec)
        self.text = text
        self.width = self.height = 0
        self.HookMessage(self.OnSize, win32con.WM_SIZE)

    def OnAttachedObjectDeath(self):
        if False:
            i = 10
            return i + 15
        docview.ScrollView.OnAttachedObjectDeath(self)
        del self.font

    def SetFont(self, new_font):
        if False:
            print('Hello World!')
        self.font = win32ui.CreateFont(new_font)
        selfInvalidateRect(None)

    def OnSize(self, params):
        if False:
            for i in range(10):
                print('nop')
        lParam = params[3]
        self.width = win32api.LOWORD(lParam)
        self.height = win32api.HIWORD(lParam)

    def OnPrepareDC(self, dc, printinfo):
        if False:
            for i in range(10):
                print('nop')
        self.SetScrollSizes(win32con.MM_TEXT, (100, 100))
        dc.SetTextColor(win32api.RGB(0, 0, 255))
        dc.SetBkColor(win32api.GetSysColor(win32con.COLOR_WINDOW))
        dc.SelectObject(self.font)
        dc.SetTextAlign(win32con.TA_CENTER | win32con.TA_BASELINE)

    def OnDraw(self, dc):
        if False:
            print('Hello World!')
        if self.width == 0 and self.height == 0:
            (left, top, right, bottom) = self.GetClientRect()
            self.width = right - left
            self.height = bottom - top
        (x, y) = (self.width // 2, self.height // 2)
        dc.TextOut(x, y, self.text)

def FontDemo():
    if False:
        return 10
    template = docview.DocTemplate(win32ui.IDR_PYTHONTYPE, None, None, FontView)
    doc = template.OpenDocumentFile(None)
    doc.SetTitle('Font Demo')
    template.close()
if __name__ == '__main__':
    import demoutils
    if demoutils.NeedGoodGUI():
        FontDemo()