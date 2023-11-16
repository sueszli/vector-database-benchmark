import os
import commctrl
import win32ui
from pywin.mfc import docview, window
from pywin.tools import hierlist

class DirHierList(hierlist.HierList):

    def __init__(self, root, listBoxID=win32ui.IDC_LIST1):
        if False:
            i = 10
            return i + 15
        hierlist.HierList.__init__(self, root, win32ui.IDB_HIERFOLDERS, listBoxID)

    def GetText(self, item):
        if False:
            return 10
        return os.path.basename(item)

    def GetSubList(self, item):
        if False:
            for i in range(10):
                print('nop')
        if os.path.isdir(item):
            ret = [os.path.join(item, fname) for fname in os.listdir(item)]
        else:
            ret = None
        return ret

    def IsExpandable(self, item):
        if False:
            i = 10
            return i + 15
        return os.path.isdir(item)

    def GetSelectedBitmapColumn(self, item):
        if False:
            print('Hello World!')
        return self.GetBitmapColumn(item) + 6

class TestDocument(docview.Document):

    def __init__(self, template):
        if False:
            print('Hello World!')
        docview.Document.__init__(self, template)
        self.hierlist = hierlist.HierListWithItems(HLIFileDir('\\'), win32ui.IDB_HIERFOLDERS, win32ui.AFX_IDW_PANE_FIRST)

class HierListView(docview.TreeView):

    def OnInitialUpdate(self):
        if False:
            while True:
                i = 10
        rc = self._obj_.OnInitialUpdate()
        self.hierList = self.GetDocument().hierlist
        self.hierList.HierInit(self.GetParent())
        self.hierList.SetStyle(commctrl.TVS_HASLINES | commctrl.TVS_LINESATROOT | commctrl.TVS_HASBUTTONS)
        return rc

class HierListFrame(window.MDIChildWnd):
    pass

def GetTestRoot():
    if False:
        return 10
    tree1 = ('Tree 1', [('Item 1', 'Item 1 data'), 'Item 2', 3])
    tree2 = ('Tree 2', [('Item 2.1', 'Item 2 data'), 'Item 2.2', 2.3])
    return ('Root', [tree1, tree2, 'Item 3'])

def demoboth():
    if False:
        return 10
    template = docview.DocTemplate(win32ui.IDR_PYTHONTYPE, TestDocument, HierListFrame, HierListView)
    template.OpenDocumentFile(None).SetTitle('Hierlist demo')
    demomodeless()

def demomodeless():
    if False:
        i = 10
        return i + 15
    testList2 = DirHierList('\\')
    dlg = hierlist.HierDialog('hier list test', testList2)
    dlg.CreateWindow()

def demodlg():
    if False:
        while True:
            i = 10
    testList2 = DirHierList('\\')
    dlg = hierlist.HierDialog('hier list test', testList2)
    dlg.DoModal()

def demo():
    if False:
        return 10
    template = docview.DocTemplate(win32ui.IDR_PYTHONTYPE, TestDocument, HierListFrame, HierListView)
    template.OpenDocumentFile(None).SetTitle('Hierlist demo')

class HLIFileDir(hierlist.HierListItem):

    def __init__(self, filename):
        if False:
            return 10
        self.filename = filename
        hierlist.HierListItem.__init__(self)

    def GetText(self):
        if False:
            print('Hello World!')
        try:
            return '%-20s %d bytes' % (os.path.basename(self.filename), os.stat(self.filename)[6])
        except OSError as details:
            return '%-20s - %s' % (self.filename, details[1])

    def IsExpandable(self):
        if False:
            i = 10
            return i + 15
        return os.path.isdir(self.filename)

    def GetSubList(self):
        if False:
            print('Hello World!')
        ret = []
        for newname in os.listdir(self.filename):
            if newname not in ('.', '..'):
                ret.append(HLIFileDir(os.path.join(self.filename, newname)))
        return ret

def demohli():
    if False:
        return 10
    template = docview.DocTemplate(win32ui.IDR_PYTHONTYPE, TestDocument, hierlist.HierListFrame, hierlist.HierListView)
    template.OpenDocumentFile(None).SetTitle('Hierlist demo')
if __name__ == '__main__':
    import demoutils
    if demoutils.HaveGoodGUI():
        demoboth()
    else:
        demodlg()