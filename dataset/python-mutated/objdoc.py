import win32ui
from pywin.mfc import docview

class object_template(docview.DocTemplate):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        docview.DocTemplate.__init__(self, None, None, None, object_view)

    def OpenObject(self, object):
        if False:
            print('Hello World!')
        for doc in self.GetDocumentList():
            print('document is ', doc)
            if doc.object is object:
                doc.GetFirstView().ActivateFrame()
                return doc
        doc = object_document(self, object)
        frame = self.CreateNewFrame(doc)
        doc.OnNewDocument()
        doc.SetTitle(str(object))
        self.InitialUpdateFrame(frame, doc)
        return doc

class object_document(docview.Document):

    def __init__(self, template, object):
        if False:
            return 10
        docview.Document.__init__(self, template)
        self.object = object

    def OnOpenDocument(self, name):
        if False:
            return 10
        raise RuntimeError('Should not be called if template strings set up correctly')
        return 0

class object_view(docview.EditView):

    def OnInitialUpdate(self):
        if False:
            return 10
        self.ReplaceSel('Object is %s' % repr(self.GetDocument().object))

def demo():
    if False:
        return 10
    t = object_template()
    d = t.OpenObject(win32ui)
    return (t, d)
if __name__ == '__main__':
    import demoutils
    if demoutils.NeedGoodGUI():
        demo()