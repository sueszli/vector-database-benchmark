import commctrl
import fontdemo
import win32ui
from pywin.mfc import docview, window

class SplitterFrame(window.MDIChildWnd):

    def __init__(self):
        if False:
            print('Hello World!')
        self.images = None
        window.MDIChildWnd.__init__(self)

    def OnCreateClient(self, cp, context):
        if False:
            return 10
        splitter = win32ui.CreateSplitter()
        doc = context.doc
        frame_rect = self.GetWindowRect()
        size = (frame_rect[2] - frame_rect[0], (frame_rect[3] - frame_rect[1]) // 2)
        sub_size = (size[0] // 2, size[1])
        splitter.CreateStatic(self, 2, 1)
        self.v1 = win32ui.CreateEditView(doc)
        self.v2 = fontdemo.FontView(doc)
        self.v3 = win32ui.CreateListView(doc)
        sub_splitter = win32ui.CreateSplitter()
        sub_splitter.CreateStatic(splitter, 1, 2)
        sub_splitter.CreateView(self.v1, 0, 0, sub_size)
        sub_splitter.CreateView(self.v2, 0, 1, (0, 0))
        splitter.SetRowInfo(0, size[1], 0)
        splitter.CreateView(self.v3, 1, 0, (0, 0))
        self.images = win32ui.CreateImageList(32, 32, 1, 5, 5)
        self.images.Add(win32ui.GetApp().LoadIcon(win32ui.IDR_MAINFRAME))
        self.images.Add(win32ui.GetApp().LoadIcon(win32ui.IDR_PYTHONCONTYPE))
        self.images.Add(win32ui.GetApp().LoadIcon(win32ui.IDR_TEXTTYPE))
        self.v3.SetImageList(self.images, commctrl.LVSIL_NORMAL)
        self.v3.InsertItem(0, 'Icon 1', 0)
        self.v3.InsertItem(0, 'Icon 2', 1)
        self.v3.InsertItem(0, 'Icon 3', 2)
        return 1

    def OnDestroy(self, msg):
        if False:
            print('Hello World!')
        window.MDIChildWnd.OnDestroy(self, msg)
        if self.images:
            self.images.DeleteImageList()
            self.images = None

    def InitialUpdateFrame(self, doc, makeVisible):
        if False:
            while True:
                i = 10
        self.v1.ReplaceSel('Hello from Edit Window 1')
        self.v1.SetModifiedFlag(0)

class SampleTemplate(docview.DocTemplate):

    def __init__(self):
        if False:
            print('Hello World!')
        docview.DocTemplate.__init__(self, win32ui.IDR_PYTHONTYPE, None, SplitterFrame, None)

    def InitialUpdateFrame(self, frame, doc, makeVisible):
        if False:
            return 10
        self._obj_.InitialUpdateFrame(frame, doc, makeVisible)
        frame.InitialUpdateFrame(doc, makeVisible)

def demo():
    if False:
        print('Hello World!')
    template = SampleTemplate()
    doc = template.OpenDocumentFile(None)
    doc.SetTitle('Splitter Demo')
if __name__ == '__main__':
    import demoutils
    if demoutils.NeedGoodGUI():
        demo()