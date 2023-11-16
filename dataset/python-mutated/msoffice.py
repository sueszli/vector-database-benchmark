import win32con
import win32ui
import win32uiole
from pywin.mfc import docview, object, window
from win32com.client import gencache

class OleClientItem(object.CmdTarget):

    def __init__(self, doc):
        if False:
            while True:
                i = 10
        object.CmdTarget.__init__(self, win32uiole.CreateOleClientItem(doc))

    def OnGetItemPosition(self):
        if False:
            print('Hello World!')
        return (10, 10, 210, 210)

    def OnActivate(self):
        if False:
            print('Hello World!')
        view = self.GetActiveView()
        item = self.GetDocument().GetInPlaceActiveItem(view)
        if item is not None and item._obj_ != self._obj_:
            item.Close()
        self._obj_.OnActivate()

    def OnChange(self, oleNotification, dwParam):
        if False:
            return 10
        self._obj_.OnChange(oleNotification, dwParam)
        self.GetDocument().UpdateAllViews(None)

    def OnChangeItemPosition(self, rect):
        if False:
            for i in range(10):
                print('nop')
        if not self._obj_.OnChangeItemPosition(self, rect):
            return 0
        return 1

class OleDocument(object.CmdTarget):

    def __init__(self, template):
        if False:
            print('Hello World!')
        object.CmdTarget.__init__(self, win32uiole.CreateOleDocument(template))
        self.EnableCompoundFile()

class ExcelView(docview.ScrollView):

    def OnInitialUpdate(self):
        if False:
            i = 10
            return i + 15
        self.HookMessage(self.OnSetFocus, win32con.WM_SETFOCUS)
        self.HookMessage(self.OnSize, win32con.WM_SIZE)
        self.SetScrollSizes(win32con.MM_TEXT, (100, 100))
        rc = self._obj_.OnInitialUpdate()
        self.EmbedExcel()
        return rc

    def EmbedExcel(self):
        if False:
            for i in range(10):
                print('nop')
        doc = self.GetDocument()
        self.clientItem = OleClientItem(doc)
        self.clientItem.CreateNewItem('Excel.Sheet')
        self.clientItem.DoVerb(-1, self)
        doc.UpdateAllViews(None)

    def OnDraw(self, dc):
        if False:
            print('Hello World!')
        doc = self.GetDocument()
        pos = doc.GetStartPosition()
        (clientItem, pos) = doc.GetNextItem(pos)
        clientItem.Draw(dc, (10, 10, 210, 210))

    def OnSetFocus(self, msg):
        if False:
            i = 10
            return i + 15
        item = self.GetDocument().GetInPlaceActiveItem(self)
        if item is not None and item.GetItemState() == win32uiole.COleClientItem_activeUIState:
            wnd = item.GetInPlaceWindow()
            if wnd is not None:
                wnd.SetFocus()
            return 0
        return 1

    def OnSize(self, params):
        if False:
            print('Hello World!')
        item = self.GetDocument().GetInPlaceActiveItem(self)
        if item is not None:
            item.SetItemRects()
        return 1

class OleTemplate(docview.DocTemplate):

    def __init__(self, resourceId=None, MakeDocument=None, MakeFrame=None, MakeView=None):
        if False:
            return 10
        if MakeDocument is None:
            MakeDocument = OleDocument
        if MakeView is None:
            MakeView = ExcelView
        docview.DocTemplate.__init__(self, resourceId, MakeDocument, MakeFrame, MakeView)

class WordFrame(window.MDIChildWnd):

    def __init__(self, doc=None):
        if False:
            print('Hello World!')
        self._obj_ = win32ui.CreateMDIChild()
        self._obj_.AttachObject(self)

    def Create(self, title, rect=None, parent=None):
        if False:
            while True:
                i = 10
        style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_OVERLAPPEDWINDOW
        self._obj_.CreateWindow(None, title, style, rect, parent)
        rect = self.GetClientRect()
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
        self.ocx = MyWordControl()
        self.ocx.CreateControl('Microsoft Word', win32con.WS_VISIBLE | win32con.WS_CHILD, rect, self, 20000)

def Demo():
    if False:
        print('Hello World!')
    import sys
    import win32api
    docName = None
    if len(sys.argv) > 1:
        docName = win32api.GetFullPathName(sys.argv[1])
    OleTemplate().OpenDocumentFile(None)
if __name__ == '__main__':
    Demo()