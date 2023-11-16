import cefpython3.wx.chromectrl as chrome
import os
import wx
import platform

class MainFrame(wx.Frame):

    def __init__(self):
        if False:
            while True:
                i = 10
        wx.Frame.__init__(self, parent=None, id=wx.ID_ANY, title='cefwx example1', size=(800, 600))
        self.cefWindow = chrome.ChromeWindow(self, url=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample1.html'))
        sizer = wx.BoxSizer()
        sizer.Add(self.cefWindow, 1, wx.EXPAND, 0)
        self.SetSizer(sizer)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, event):
        if False:
            i = 10
            return i + 15
        self.Destroy()
        if platform.system() == 'Darwin':
            chrome.Shutdown()
            wx.GetApp().Exit()

class MyApp(wx.App):

    def OnInit(self):
        if False:
            print('Hello World!')
        frame = MainFrame()
        self.SetTopWindow(frame)
        frame.Show()
        return True
if __name__ == '__main__':
    chrome.Initialize({'debug': True, 'log_file': 'debug.log', 'log_severity': chrome.cefpython.LOGSEVERITY_INFO})
    print('[sample1.py] wx.version=%s' % wx.version())
    app = MyApp(False)
    app.MainLoop()
    del app
    if platform.system() in ['Linux', 'Windows']:
        chrome.Shutdown()