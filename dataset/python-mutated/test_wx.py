import wx
ID_NEW = 1
ID_RENAME = 2
ID_CLEAR = 3
ID_DELETE = 4

class Example(wx.Frame):

    def __init__(self, parent, title):
        if False:
            while True:
                i = 10
        super(Example, self).__init__(parent, title=title, size=(260, 180))
        self.InitUI()
        self.Centre()
        self.Show()

    def InitUI(self):
        if False:
            for i in range(10):
                print('nop')
        panel = wx.Panel(self)
        panel.SetBackgroundColour('#4f5049')
        lbox = wx.BoxSizer(wx.VERTICAL)
        listbox = wx.ListBox(panel, -1, size=(100, 50))
        btnPanel = wx.Panel(panel, -1, size=(30, 30))
        bbox = wx.BoxSizer(wx.HORIZONTAL)
        new = wx.Button(btnPanel, ID_NEW, '+', size=(24, 24))
        ren = wx.Button(btnPanel, ID_RENAME, '-', size=(24, 24))
        bbox.Add(new, flag=wx.LEFT, border=2)
        bbox.Add(ren, flag=wx.LEFT, border=2)
        btnPanel.SetSizer(bbox)
        lbox.Add(listbox, 1, wx.EXPAND | wx.ALL, 1)
        lbox.Add(btnPanel, 0, wx.EXPAND | wx.ALL, 1)
        panel.SetSizer(lbox)
if __name__ == '__main__':
    app = wx.App()
    Example(None, title='Gmvault-test')
    app.MainLoop()