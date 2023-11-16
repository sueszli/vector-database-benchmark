import cefpython3.wx.chromectrl as chrome
import wx
import wx.lib.agw.flatnotebook as fnb
import platform
import sys
ROOT_NAME = 'My Locations'
URLS = ['http://gmail.com', 'http://maps.google.com', 'http://youtube.com', 'http://yahoo.com', 'http://wikipedia.com', 'http://cyaninc.com', 'http://tavmjong.free.fr/INKSCAPE/MANUAL/web/svg_tests.php']

class MainFrame(wx.Frame):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        wx.Frame.__init__(self, parent=None, id=wx.ID_ANY, title='cefwx example2', size=(800, 600))
        self.initComponents()
        self.layoutComponents()
        self.initEventHandlers()
        if len(sys.argv) == 2 and sys.argv[1] == 'test-launch':
            wx.CallLater(500, self.testLaunch)

    def testLaunch(self):
        if False:
            print('Hello World!')
        print('b8ba7d9945c22425328df2e21fbb64cd')
        self.Close()

    def initComponents(self):
        if False:
            i = 10
            return i + 15
        self.tree = wx.TreeCtrl(self, id=-1, size=(200, -1))
        self.root = self.tree.AddRoot(ROOT_NAME)
        for url in URLS:
            self.tree.AppendItem(self.root, url)
        self.tree.Expand(self.root)
        self.tabs = fnb.FlatNotebook(self, wx.ID_ANY, agwStyle=fnb.FNB_NODRAG | fnb.FNB_X_ON_TAB)
        self.tabs.SetWindowStyleFlag(wx.WANTS_CHARS)

    def layoutComponents(self):
        if False:
            for i in range(10):
                print('nop')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.tree, 0, wx.EXPAND)
        sizer.Add(self.tabs, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def initEventHandlers(self):
        if False:
            return 10
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnSelChanged, self.tree)
        self.Bind(fnb.EVT_FLATNOTEBOOK_PAGE_CLOSING, self.OnPageClosing)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnSelChanged(self, event):
        if False:
            return 10
        self.item = event.GetItem()
        url = self.tree.GetItemText(self.item)
        if url and url != ROOT_NAME:
            cefPanel = chrome.ChromeCtrl(self.tabs, useTimer=True, url=str(url))
            self.tabs.AddPage(cefPanel, url)
            self.tabs.SetSelection(self.tabs.GetPageCount() - 1)
        event.Skip()

    def OnPageClosing(self, event):
        if False:
            return 10
        print('sample2.py: One could place some extra closing stuff here')
        event.Skip()

    def OnClose(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.Destroy()
        if platform.system() == 'Darwin':
            chrome.Shutdown()
            wx.GetApp().Exit()

class MyApp(wx.App):

    def OnInit(self):
        if False:
            for i in range(10):
                print('nop')
        frame = MainFrame()
        self.SetTopWindow(frame)
        frame.Show()
        return True
if __name__ == '__main__':
    chrome.Initialize()
    if platform.system() == 'Linux':
        import time
        time.sleep(0.5)
    print('sample2.py: wx.version=%s' % wx.version())
    app = MyApp(False)
    app.MainLoop()
    del app
    if platform.system() in ['Linux', 'Windows']:
        chrome.Shutdown()