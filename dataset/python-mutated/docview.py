import win32ui
from . import object, window

class View(window.Wnd):

    def __init__(self, initobj):
        if False:
            print('Hello World!')
        window.Wnd.__init__(self, initobj)

    def OnInitialUpdate(self):
        if False:
            print('Hello World!')
        pass

class CtrlView(View):

    def __init__(self, doc, wndclass, style=0):
        if False:
            print('Hello World!')
        View.__init__(self, win32ui.CreateCtrlView(doc, wndclass, style))

class EditView(CtrlView):

    def __init__(self, doc):
        if False:
            return 10
        View.__init__(self, win32ui.CreateEditView(doc))

class RichEditView(CtrlView):

    def __init__(self, doc):
        if False:
            return 10
        View.__init__(self, win32ui.CreateRichEditView(doc))

class ListView(CtrlView):

    def __init__(self, doc):
        if False:
            print('Hello World!')
        View.__init__(self, win32ui.CreateListView(doc))

class TreeView(CtrlView):

    def __init__(self, doc):
        if False:
            while True:
                i = 10
        View.__init__(self, win32ui.CreateTreeView(doc))

class ScrollView(View):

    def __init__(self, doc):
        if False:
            i = 10
            return i + 15
        View.__init__(self, win32ui.CreateView(doc))

class FormView(View):

    def __init__(self, doc, id):
        if False:
            for i in range(10):
                print('nop')
        View.__init__(self, win32ui.CreateFormView(doc, id))

class Document(object.CmdTarget):

    def __init__(self, template, docobj=None):
        if False:
            i = 10
            return i + 15
        if docobj is None:
            docobj = template.DoCreateDoc()
        object.CmdTarget.__init__(self, docobj)

class RichEditDoc(object.CmdTarget):

    def __init__(self, template):
        if False:
            while True:
                i = 10
        object.CmdTarget.__init__(self, template.DoCreateRichEditDoc())

class CreateContext:
    """A transient base class used as a CreateContext"""

    def __init__(self, template, doc=None):
        if False:
            print('Hello World!')
        self.template = template
        self.doc = doc

    def __del__(self):
        if False:
            while True:
                i = 10
        self.close()

    def close(self):
        if False:
            print('Hello World!')
        self.doc = None
        self.template = None

class DocTemplate(object.CmdTarget):

    def __init__(self, resourceId=None, MakeDocument=None, MakeFrame=None, MakeView=None):
        if False:
            i = 10
            return i + 15
        if resourceId is None:
            resourceId = win32ui.IDR_PYTHONTYPE
        object.CmdTarget.__init__(self, self._CreateDocTemplate(resourceId))
        self.MakeDocument = MakeDocument
        self.MakeFrame = MakeFrame
        self.MakeView = MakeView
        self._SetupSharedMenu_()

    def _SetupSharedMenu_(self):
        if False:
            while True:
                i = 10
        pass

    def _CreateDocTemplate(self, resourceId):
        if False:
            i = 10
            return i + 15
        return win32ui.CreateDocTemplate(resourceId)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        object.CmdTarget.__del__(self)

    def CreateCreateContext(self, doc=None):
        if False:
            i = 10
            return i + 15
        return CreateContext(self, doc)

    def CreateNewFrame(self, doc):
        if False:
            for i in range(10):
                print('nop')
        makeFrame = self.MakeFrame
        if makeFrame is None:
            makeFrame = window.MDIChildWnd
        wnd = makeFrame()
        context = self.CreateCreateContext(doc)
        wnd.LoadFrame(self.GetResourceID(), -1, None, context)
        return wnd

    def CreateNewDocument(self):
        if False:
            return 10
        makeDocument = self.MakeDocument
        if makeDocument is None:
            makeDocument = Document
        return makeDocument(self)

    def CreateView(self, frame, context):
        if False:
            for i in range(10):
                print('nop')
        makeView = self.MakeView
        if makeView is None:
            makeView = EditView
        view = makeView(context.doc)
        view.CreateWindow(frame)

class RichEditDocTemplate(DocTemplate):

    def __init__(self, resourceId=None, MakeDocument=None, MakeFrame=None, MakeView=None):
        if False:
            print('Hello World!')
        if MakeView is None:
            MakeView = RichEditView
        if MakeDocument is None:
            MakeDocument = RichEditDoc
        DocTemplate.__init__(self, resourceId, MakeDocument, MakeFrame, MakeView)

    def _CreateDocTemplate(self, resourceId):
        if False:
            print('Hello World!')
        return win32ui.CreateRichEditDocTemplate(resourceId)

def t():
    if False:
        while True:
            i = 10

    class FormTemplate(DocTemplate):

        def CreateView(self, frame, context):
            if False:
                return 10
            makeView = self.MakeView
            view = ListView(context.doc)
            view.CreateWindow(frame)
    t = FormTemplate()
    return t.OpenDocumentFile(None)