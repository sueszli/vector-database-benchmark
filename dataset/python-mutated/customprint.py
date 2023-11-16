import win32api
import win32con
import win32ui
from pywin.framework import app
from pywin.mfc import afxres, dialog, docview
PRINTDLGORD = 1538
IDC_PRINT_MAG_EDIT = 1010

class PrintDemoTemplate(docview.DocTemplate):

    def _SetupSharedMenu_(self):
        if False:
            while True:
                i = 10
        pass

class PrintDemoView(docview.ScrollView):

    def OnInitialUpdate(self):
        if False:
            for i in range(10):
                print('nop')
        ret = self._obj_.OnInitialUpdate()
        self.colors = {'Black': (0 << 0) + (0 << 8) + (0 << 16), 'Red': (255 << 0) + (0 << 8) + (0 << 16), 'Green': (0 << 0) + (255 << 8) + (0 << 16), 'Blue': (0 << 0) + (0 << 8) + (255 << 16), 'Cyan': (0 << 0) + (255 << 8) + (255 << 16), 'Magenta': (255 << 0) + (0 << 8) + (255 << 16), 'Yellow': (255 << 0) + (255 << 8) + (0 << 16)}
        self.pens = {}
        for (name, color) in self.colors.items():
            self.pens[name] = win32ui.CreatePen(win32con.PS_SOLID, 5, color)
        self.pen = None
        self.size = (128, 128)
        self.SetScaleToFitSize(self.size)
        self.HookCommand(self.OnFilePrint, afxres.ID_FILE_PRINT)
        self.HookCommand(self.OnFilePrintPreview, win32ui.ID_FILE_PRINT_PREVIEW)
        return ret

    def OnDraw(self, dc):
        if False:
            i = 10
            return i + 15
        oldPen = None
        (x, y) = self.size
        delta = 2
        colors = list(self.colors.keys())
        colors.sort()
        colors = colors * 2
        for color in colors:
            if oldPen is None:
                oldPen = dc.SelectObject(self.pens[color])
            else:
                dc.SelectObject(self.pens[color])
            dc.MoveTo((delta, delta))
            dc.LineTo((x - delta, delta))
            dc.LineTo((x - delta, y - delta))
            dc.LineTo((delta, y - delta))
            dc.LineTo((delta, delta))
            delta = delta + 4
            if x - delta <= 0 or y - delta <= 0:
                break
        dc.SelectObject(oldPen)

    def OnPrepareDC(self, dc, pInfo):
        if False:
            while True:
                i = 10
        if dc.IsPrinting():
            mag = self.prtDlg['mag']
            dc.SetMapMode(win32con.MM_ANISOTROPIC)
            dc.SetWindowOrg((0, 0))
            dc.SetWindowExt((1, 1))
            dc.SetViewportOrg((0, 0))
            dc.SetViewportExt((mag, mag))

    def OnPreparePrinting(self, pInfo):
        if False:
            print('Hello World!')
        flags = win32ui.PD_USEDEVMODECOPIES | win32ui.PD_PAGENUMS | win32ui.PD_NOPAGENUMS | win32ui.PD_NOSELECTION
        self.prtDlg = ImagePrintDialog(pInfo, PRINTDLGORD, flags)
        pInfo.SetPrintDialog(self.prtDlg)
        pInfo.SetMinPage(1)
        pInfo.SetMaxPage(1)
        pInfo.SetFromPage(1)
        pInfo.SetToPage(1)
        ret = self.DoPreparePrinting(pInfo)
        return ret

    def OnBeginPrinting(self, dc, pInfo):
        if False:
            print('Hello World!')
        return self._obj_.OnBeginPrinting(dc, pInfo)

    def OnEndPrinting(self, dc, pInfo):
        if False:
            return 10
        del self.prtDlg
        return self._obj_.OnEndPrinting(dc, pInfo)

    def OnFilePrintPreview(self, *arg):
        if False:
            return 10
        self._obj_.OnFilePrintPreview()

    def OnFilePrint(self, *arg):
        if False:
            i = 10
            return i + 15
        self._obj_.OnFilePrint()

    def OnPrint(self, dc, pInfo):
        if False:
            while True:
                i = 10
        doc = self.GetDocument()
        metrics = dc.GetTextMetrics()
        cxChar = metrics['tmAveCharWidth']
        cyChar = metrics['tmHeight']
        (left, top, right, bottom) = pInfo.GetDraw()
        dc.TextOut(0, 2 * cyChar, doc.GetTitle())
        top = top + 7 * cyChar / 2
        dc.MoveTo(left, top)
        dc.LineTo(right, top)
        top = top + cyChar
        pInfo.SetDraw((left, top, right, bottom))
        dc.SetWindowOrg((0, -top))
        self.OnDraw(dc)
        dc.SetTextAlign(win32con.TA_LEFT | win32con.TA_BOTTOM)
        rect = self.GetWindowRect()
        rect = self.ScreenToClient(rect)
        height = rect[3] - rect[1]
        dc.SetWindowOrg((0, -(top + height + cyChar)))
        dc.MoveTo(left, 0)
        dc.LineTo(right, 0)
        x = 0
        y = 3 * cyChar / 2
        dc.TextOut(x, y, doc.GetTitle())
        y = y + cyChar

class PrintDemoApp(app.CApp):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        app.CApp.__init__(self)

    def InitInstance(self):
        if False:
            while True:
                i = 10
        template = PrintDemoTemplate(None, None, None, PrintDemoView)
        self.AddDocTemplate(template)
        self._obj_.InitMDIInstance()
        self.LoadMainFrame()
        doc = template.OpenDocumentFile(None)
        doc.SetTitle('Custom Print Document')

class ImagePrintDialog(dialog.PrintDialog):
    sectionPos = 'Image Print Demo'

    def __init__(self, pInfo, dlgID, flags=win32ui.PD_USEDEVMODECOPIES):
        if False:
            print('Hello World!')
        dialog.PrintDialog.__init__(self, pInfo, dlgID, flags=flags)
        mag = win32ui.GetProfileVal(self.sectionPos, 'Document Magnification', 0)
        if mag <= 0:
            mag = 2
            win32ui.WriteProfileVal(self.sectionPos, 'Document Magnification', mag)
        self['mag'] = mag

    def OnInitDialog(self):
        if False:
            while True:
                i = 10
        self.magCtl = self.GetDlgItem(IDC_PRINT_MAG_EDIT)
        self.magCtl.SetWindowText(repr(self['mag']))
        return dialog.PrintDialog.OnInitDialog(self)

    def OnOK(self):
        if False:
            for i in range(10):
                print('nop')
        dialog.PrintDialog.OnOK(self)
        strMag = self.magCtl.GetWindowText()
        try:
            self['mag'] = int(strMag)
        except:
            pass
        win32ui.WriteProfileVal(self.sectionPos, 'Document Magnification', self['mag'])
if __name__ == '__main__':

    def test():
        if False:
            return 10
        template = PrintDemoTemplate(None, None, None, PrintDemoView)
        template.OpenDocumentFile(None)
    test()
else:
    app = PrintDemoApp()