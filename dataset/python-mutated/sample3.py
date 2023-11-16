import cefpython3.wx.chromectrl as chrome
import os
import wx
import wx.lib.agw.flatnotebook as fnb
import platform

class MainFrame(wx.Frame):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        wx.Frame.__init__(self, parent=None, id=wx.ID_ANY, title='cefwx example3', size=(800, 600))
        self._InitComponents()
        self._LayoutComponents()
        self._InitEventHandlers()

    def _InitComponents(self):
        if False:
            for i in range(10):
                print('nop')
        self.tabs = fnb.FlatNotebook(self, wx.ID_ANY, agwStyle=fnb.FNB_NODRAG | fnb.FNB_X_ON_TAB)
        self.tabs.SetWindowStyleFlag(wx.WANTS_CHARS)
        ctrl1 = chrome.ChromeCtrl(self.tabs, useTimer=True, url='wikipedia.org')
        ctrl1.GetNavigationBar().GetUrlCtrl().SetEditable(False)
        ctrl1.GetNavigationBar().GetBackButton().SetBitmapLabel(wx.Bitmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'back.png'), wx.BITMAP_TYPE_PNG))
        ctrl1.GetNavigationBar().GetForwardButton().SetBitmapLabel(wx.Bitmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forward.png'), wx.BITMAP_TYPE_PNG))
        ctrl1.GetNavigationBar().GetReloadButton().SetBitmapLabel(wx.Bitmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reload_page.png'), wx.BITMAP_TYPE_PNG))
        self.tabs.AddPage(ctrl1, 'Wikipedia')
        ctrl2 = chrome.ChromeCtrl(self.tabs, useTimer=True, url='google.com', hasNavBar=False)
        self.tabs.AddPage(ctrl2, 'Google')
        ctrl3 = chrome.ChromeCtrl(self.tabs, useTimer=True, url='greenpeace.org')
        ctrl3.SetNavigationBar(CustomNavigationBar(ctrl3))
        self.tabs.AddPage(ctrl3, 'Greenpeace')

    def _LayoutComponents(self):
        if False:
            return 10
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.tabs, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def _InitEventHandlers(self):
        if False:
            for i in range(10):
                print('nop')
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, event):
        if False:
            while True:
                i = 10
        self.Destroy()
        if platform.system() == 'Darwin':
            chrome.Shutdown()
            wx.GetApp().Exit()

class CustomNavigationBar(chrome.NavigationBar):

    def _LayoutComponents(self):
        if False:
            return 10
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.url, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 12)
        sizer.Add(self.GetBackButton(), 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        sizer.Add(self.GetForwardButton(), 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        self.GetReloadButton().Hide()
        self.SetSizer(sizer)
        self.Fit()

class MyApp(wx.App):

    def OnInit(self):
        if False:
            print('Hello World!')
        frame = MainFrame()
        self.SetTopWindow(frame)
        frame.Show()
        return True
if __name__ == '__main__':
    chrome.Initialize()
    print('sample3.py: wx.version=%s' % wx.version())
    app = MyApp()
    app.MainLoop()
    del app
    if platform.system() in ['Linux', 'Windows']:
        chrome.Shutdown()