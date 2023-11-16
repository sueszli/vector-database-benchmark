import os
import win32api
import win32con
import win32ui
from pywin.mfc import docview, window
from . import app
bStretch = 1

class BitmapDocument(docview.Document):
    """A bitmap document.  Holds the bitmap data itself."""

    def __init__(self, template):
        if False:
            i = 10
            return i + 15
        docview.Document.__init__(self, template)
        self.bitmap = None

    def OnNewDocument(self):
        if False:
            print('Hello World!')
        win32ui.MessageBox('Bitmaps can not be created.')

    def OnOpenDocument(self, filename):
        if False:
            i = 10
            return i + 15
        self.bitmap = win32ui.CreateBitmap()
        f = open(filename, 'rb')
        try:
            try:
                self.bitmap.LoadBitmapFile(f)
            except OSError:
                win32ui.MessageBox('Could not load the bitmap from %s' % filename)
                return 0
        finally:
            f.close()
        self.size = self.bitmap.GetSize()
        return 1

    def DeleteContents(self):
        if False:
            return 10
        self.bitmap = None

class BitmapView(docview.ScrollView):
    """A view of a bitmap.  Obtains data from document."""

    def __init__(self, doc):
        if False:
            while True:
                i = 10
        docview.ScrollView.__init__(self, doc)
        self.width = self.height = 0
        self.HookMessage(self.OnSize, win32con.WM_SIZE)

    def OnInitialUpdate(self):
        if False:
            i = 10
            return i + 15
        doc = self.GetDocument()
        if doc.bitmap:
            bitmapSize = doc.bitmap.GetSize()
            self.SetScrollSizes(win32con.MM_TEXT, bitmapSize)

    def OnSize(self, params):
        if False:
            for i in range(10):
                print('nop')
        lParam = params[3]
        self.width = win32api.LOWORD(lParam)
        self.height = win32api.HIWORD(lParam)

    def OnDraw(self, dc):
        if False:
            print('Hello World!')
        doc = self.GetDocument()
        if doc.bitmap is None:
            return
        bitmapSize = doc.bitmap.GetSize()
        if bStretch:
            viewRect = (0, 0, self.width, self.height)
            bitmapRect = (0, 0, bitmapSize[0], bitmapSize[1])
            doc.bitmap.Paint(dc, viewRect, bitmapRect)
        else:
            doc.bitmap.Paint(dc)

class BitmapFrame(window.MDIChildWnd):

    def OnCreateClient(self, createparams, context):
        if False:
            while True:
                i = 10
        borderX = win32api.GetSystemMetrics(win32con.SM_CXFRAME)
        borderY = win32api.GetSystemMetrics(win32con.SM_CYFRAME)
        titleY = win32api.GetSystemMetrics(win32con.SM_CYCAPTION)
        mdiClient = win32ui.GetMainFrame().GetWindow(win32con.GW_CHILD)
        clientWindowRect = mdiClient.ScreenToClient(mdiClient.GetWindowRect())
        clientWindowSize = (clientWindowRect[2] - clientWindowRect[0], clientWindowRect[3] - clientWindowRect[1])
        (left, top, right, bottom) = mdiClient.ScreenToClient(self.GetWindowRect())
        window.MDIChildWnd.OnCreateClient(self, createparams, context)
        return 1

class BitmapTemplate(docview.DocTemplate):

    def __init__(self):
        if False:
            print('Hello World!')
        docview.DocTemplate.__init__(self, win32ui.IDR_PYTHONTYPE, BitmapDocument, BitmapFrame, BitmapView)

    def MatchDocType(self, fileName, fileType):
        if False:
            return 10
        doc = self.FindOpenDocument(fileName)
        if doc:
            return doc
        ext = os.path.splitext(fileName)[1].lower()
        if ext == '.bmp':
            return win32ui.CDocTemplate_Confidence_yesAttemptNative
        return win32ui.CDocTemplate_Confidence_maybeAttemptForeign
try:
    win32ui.GetApp().RemoveDocTemplate(bitmapTemplate)
except NameError:
    pass
bitmapTemplate = BitmapTemplate()
bitmapTemplate.SetDocStrings('\nBitmap\nBitmap\nBitmap (*.bmp)\n.bmp\nPythonBitmapFileType\nPython Bitmap File')
win32ui.GetApp().AddDocTemplate(bitmapTemplate)

def t():
    if False:
        return 10
    bitmapTemplate.OpenDocumentFile('d:\\winnt\\arcade.bmp')

def demo():
    if False:
        while True:
            i = 10
    import glob
    winDir = win32api.GetWindowsDirectory()
    for fileName in glob.glob1(winDir, '*.bmp')[:2]:
        bitmapTemplate.OpenDocumentFile(os.path.join(winDir, fileName))