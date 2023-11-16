"""
WxAppShell provides a GUI application framework using wxPython.
This is an wxPython version of AppShell.py
"""
import wx
import sys

class WxAppShell(wx.Frame):
    appversion = '1.0'
    appname = 'Generic Application Frame'
    copyright = 'Copyright 2008 Walt Disney Internet Group.' + '\nAll Rights Reserved.'
    contactname = 'Gyedo Jeon'
    contactemail = 'Gyedo.Jeon@disney.com'
    frameWidth = 450
    frameHeight = 320
    padx = 5
    pady = 5
    usecommandarea = 0
    usestatusarea = 0
    balloonState = 'none'
    panelCount = 0

    def __init__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        if not kw.get(''):
            kw['title'] = self.appname
        if not kw.get('size'):
            kw['size'] = wx.Size(self.frameWidth, self.frameHeight)
        wx.Frame.__init__(self, None, -1, *args, **kw)
        self._logWin = None
        self.appInit()
        self.__createInterface()
        self.Show()

    def __createInterface(self):
        if False:
            print('Hello World!')
        self.__createLogWin()
        self.__createMenuBar()
        self.__createAboutBox()
        self.Bind(wx.EVT_CLOSE, self.quit)
        self.createMenuBar()
        self.createInterface()

    def __createLogWin(self, evt=None):
        if False:
            for i in range(10):
                print('nop')
        if self._logWin:
            self._logWin.Destroy()
        self._logWin = wx.Frame(None)
        self._logWin.Bind(wx.EVT_CLOSE, self.__createLogWin)
        wx.Log.SetActiveTarget(wx.LogTextCtrl(wx.TextCtrl(self._logWin, style=wx.TE_MULTILINE)))

    def __createMenuBar(self):
        if False:
            for i in range(10):
                print('nop')
        self.menuBar = wx.MenuBar()
        self.SetMenuBar(self.menuBar)

    def __createAboutBox(self):
        if False:
            i = 10
            return i + 15
        self.about = wx.MessageDialog(None, self.appname + '\n\n' + 'Version %s' % self.appversion + '\n\n' + self.copyright + '\n\n' + 'For more information, contact:\n%s\nEmail: %s' % (self.contactname, self.contactemail), 'About %s' % self.appname, wx.OK | wx.ICON_INFORMATION)

    def showAbout(self, event):
        if False:
            print('Hello World!')
        self.about.ShowModal()

    def quit(self, event=None):
        if False:
            i = 10
            return i + 15
        self.onDestroy(event)
        from direct.showbase import ShowBaseGlobal
        if hasattr(ShowBaseGlobal, 'base'):
            ShowBaseGlobal.base.userExit()
        else:
            sys.exit()

    def appInit(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def createInterface(self):
        if False:
            print('Hello World!')
        pass

    def onDestroy(self, event):
        if False:
            i = 10
            return i + 15
        pass

    def createMenuBar(self):
        if False:
            i = 10
            return i + 15
        self.menuFile = wx.Menu()
        self.menuBar.Append(self.menuFile, '&File')
        self.menuHelp = wx.Menu()
        self.menuBar.Append(self.menuHelp, '&Help')
        menuItem = self.menuFile.Append(wx.ID_EXIT, '&Quit')
        self.Bind(wx.EVT_MENU, self.quit, menuItem)
        menuItem = self.menuHelp.Append(wx.ID_ABOUT, '&About...')
        self.Bind(wx.EVT_MENU, self.showAbout, menuItem)