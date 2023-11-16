from cefpython3 import cefpython
import os, sys, platform
import wx
import wx.lib.buttons as buttons
g_settings = None

def Debug(msg):
    if False:
        return 10
    if g_settings and 'debug' in g_settings and g_settings['debug']:
        print('[chromectrl.py] ' + msg)
DEFAULT_TIMER_MILLIS = 10
g_messageLoopTimer = None

def CreateMessageLoopTimer(timerMillis):
    if False:
        for i in range(10):
            print('nop')
    global g_messageLoopTimer
    Debug('CreateMesageLoopTimer')
    if g_messageLoopTimer:
        return
    g_messageLoopTimer = wx.Timer()
    g_messageLoopTimer.Start(timerMillis)
    Debug('g_messageLoopTimer.GetId() = ' + str(g_messageLoopTimer.GetId()))
    wx.EVT_TIMER(g_messageLoopTimer, g_messageLoopTimer.GetId(), MessageLoopTimer)

def MessageLoopTimer(event):
    if False:
        print('Hello World!')
    cefpython.MessageLoopWork()

def DestroyMessageLoopTimer():
    if False:
        i = 10
        return i + 15
    global g_messageLoopTimer
    Debug('DestroyMessageLoopTimer')
    if g_messageLoopTimer:
        g_messageLoopTimer.Stop()
        g_messageLoopTimer = None
    else:
        Debug('DestroyMessageLoopTimer: timer not started')

class NavigationBar(wx.Panel):

    def __init__(self, parent, *args, **kwargs):
        if False:
            return 10
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.bitmapDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
        self._InitComponents()
        self._LayoutComponents()
        self._InitEventHandlers()

    def _InitComponents(self):
        if False:
            print('Hello World!')
        self.backBtn = buttons.GenBitmapButton(self, -1, wx.Bitmap(os.path.join(self.bitmapDir, 'back.png'), wx.BITMAP_TYPE_PNG), style=wx.BORDER_NONE)
        self.forwardBtn = buttons.GenBitmapButton(self, -1, wx.Bitmap(os.path.join(self.bitmapDir, 'forward.png'), wx.BITMAP_TYPE_PNG), style=wx.BORDER_NONE)
        self.reloadBtn = buttons.GenBitmapButton(self, -1, wx.Bitmap(os.path.join(self.bitmapDir, 'reload_page.png'), wx.BITMAP_TYPE_PNG), style=wx.BORDER_NONE)
        self.url = wx.TextCtrl(self, id=-1, style=0)
        self.historyPopup = wx.Menu()

    def _LayoutComponents(self):
        if False:
            for i in range(10):
                print('nop')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.backBtn, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        sizer.Add(self.forwardBtn, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        sizer.Add(self.reloadBtn, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        sizer.Add(self.url, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 12)
        self.SetSizer(sizer)
        self.Fit()

    def _InitEventHandlers(self):
        if False:
            while True:
                i = 10
        self.backBtn.Bind(wx.EVT_CONTEXT_MENU, self.OnButtonContext)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self.historyPopup.Destroy()

    def GetBackButton(self):
        if False:
            return 10
        return self.backBtn

    def GetForwardButton(self):
        if False:
            while True:
                i = 10
        return self.forwardBtn

    def GetReloadButton(self):
        if False:
            while True:
                i = 10
        return self.reloadBtn

    def GetUrlCtrl(self):
        if False:
            return 10
        return self.url

    def InitHistoryPopup(self):
        if False:
            for i in range(10):
                print('nop')
        self.historyPopup = wx.Menu()

    def AddToHistory(self, url):
        if False:
            print('Hello World!')
        self.historyPopup.Append(-1, url)

    def OnButtonContext(self, event):
        if False:
            i = 10
            return i + 15
        self.PopupMenu(self.historyPopup)

class ChromeWindow(wx.Window):
    """
    Standalone CEF component. The class provides facilites for interacting
    with wx message loop
    """

    def __init__(self, parent, url='', useTimer=True, timerMillis=DEFAULT_TIMER_MILLIS, browserSettings=None, size=(-1, -1), *args, **kwargs):
        if False:
            print('Hello World!')
        wx.Window.__init__(self, parent, *args, id=wx.ID_ANY, size=size, **kwargs)
        self.timer = wx.Timer()
        if platform.system() in ['Linux', 'Darwin']:
            if url.startswith('/'):
                url = 'file://' + url
        self.url = url
        windowInfo = cefpython.WindowInfo()
        if platform.system() == 'Windows':
            windowInfo.SetAsChild(self.GetHandle())
        elif platform.system() == 'Linux':
            windowInfo.SetAsChild(self.GetGtkWidget())
        elif platform.system() == 'Darwin':
            (width, height) = self.GetClientSizeTuple()
            windowInfo.SetAsChild(self.GetHandle(), [0, 0, width, height])
        else:
            raise Exception('Unsupported OS')
        if not browserSettings:
            browserSettings = {}
        self.browser = cefpython.CreateBrowserSync(windowInfo, browserSettings=browserSettings, navigateUrl=url)
        if platform.system() == 'Windows':
            self.Bind(wx.EVT_SET_FOCUS, self.OnSetFocus)
            self.Bind(wx.EVT_SIZE, self.OnSize)
        self._useTimer = useTimer
        if useTimer:
            CreateMessageLoopTimer(timerMillis)
        else:
            Debug('WARNING: Using EVT_IDLE for CEF message  loop processing is not recommended')
            self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, event):
        if False:
            return 10
        if not self._useTimer:
            try:
                self.Unbind(wx.EVT_IDLE)
            except:
                pass
        self.browser.ParentWindowWillClose()

    def OnIdle(self, event):
        if False:
            print('Hello World!')
        'Service CEF message loop when useTimer is False'
        cefpython.MessageLoopWork()
        event.Skip()

    def OnSetFocus(self, event):
        if False:
            print('Hello World!')
        'OS_WIN only.'
        cefpython.WindowUtils.OnSetFocus(self.GetHandle(), 0, 0, 0)
        event.Skip()

    def OnSize(self, event):
        if False:
            while True:
                i = 10
        'OS_WIN only. Handle the the size event'
        cefpython.WindowUtils.OnSize(self.GetHandle(), 0, 0, 0)
        event.Skip()

    def GetBrowser(self):
        if False:
            return 10
        "Returns the CEF's browser object"
        return self.browser

    def LoadUrl(self, url, onLoadStart=None, onLoadEnd=None):
        if False:
            return 10
        if onLoadStart or onLoadEnd:
            self.GetBrowser().SetClientHandler(CallbackClientHandler(onLoadStart, onLoadEnd))
        browser = self.GetBrowser()
        if cefpython.g_debug:
            Debug('LoadUrl() self: %s' % self)
            Debug('browser: %s' % browser)
            Debug('browser id: %s' % browser.GetIdentifier())
            Debug('mainframe: %s' % browser.GetMainFrame())
            Debug('mainframe id: %s' % browser.GetMainFrame().GetIdentifier())
        self.GetBrowser().GetMainFrame().LoadUrl(url)

class ChromeCtrl(wx.Panel):

    def __init__(self, parent, url='', useTimer=True, timerMillis=DEFAULT_TIMER_MILLIS, browserSettings=None, hasNavBar=True, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        wx.Panel.__init__(self, parent, *args, style=wx.WANTS_CHARS, **kwargs)
        self.chromeWindow = ChromeWindow(self, url=str(url), useTimer=useTimer, browserSettings=browserSettings)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.navigationBar = None
        if hasNavBar:
            self.navigationBar = self.CreateNavigationBar()
            sizer.Add(self.navigationBar, 0, wx.EXPAND | wx.ALL, 0)
            self._InitEventHandlers()
        sizer.Add(self.chromeWindow, 1, wx.EXPAND, 0)
        self.SetSizer(sizer)
        self.Fit()
        ch = DefaultClientHandler(self)
        self.SetClientHandler(ch)
        if self.navigationBar:
            self.UpdateButtonsState()

    def _InitEventHandlers(self):
        if False:
            return 10
        self.navigationBar.backBtn.Bind(wx.EVT_BUTTON, self.OnLeft)
        self.navigationBar.forwardBtn.Bind(wx.EVT_BUTTON, self.OnRight)
        self.navigationBar.reloadBtn.Bind(wx.EVT_BUTTON, self.OnReload)

    def GetNavigationBar(self):
        if False:
            for i in range(10):
                print('nop')
        return self.navigationBar

    def SetNavigationBar(self, navigationBar):
        if False:
            i = 10
            return i + 15
        sizer = self.GetSizer()
        if self.navigationBar:
            sizer.Replace(self.navigationBar, navigationBar)
            self.navigationBar.Hide()
            del self.navigationBar
        else:
            sizer.Insert(0, navigationBar, 0, wx.EXPAND)
        self.navigationBar = navigationBar
        sizer.Fit(self)

    def CreateNavigationBar(self):
        if False:
            i = 10
            return i + 15
        np = NavigationBar(self)
        return np

    def SetClientHandler(self, handler):
        if False:
            i = 10
            return i + 15
        self.chromeWindow.GetBrowser().SetClientHandler(handler)

    def OnLeft(self, event):
        if False:
            print('Hello World!')
        if self.chromeWindow.GetBrowser().CanGoBack():
            self.chromeWindow.GetBrowser().GoBack()
        self.UpdateButtonsState()
        self.chromeWindow.GetBrowser().SetFocus(True)

    def OnRight(self, event):
        if False:
            return 10
        if self.chromeWindow.GetBrowser().CanGoForward():
            self.chromeWindow.GetBrowser().GoForward()
        self.UpdateButtonsState()
        self.chromeWindow.GetBrowser().SetFocus(True)

    def OnReload(self, event):
        if False:
            while True:
                i = 10
        self.chromeWindow.GetBrowser().Reload()
        self.UpdateButtonsState()
        self.chromeWindow.GetBrowser().SetFocus(True)

    def UpdateButtonsState(self):
        if False:
            i = 10
            return i + 15
        self.navigationBar.backBtn.Enable(self.chromeWindow.GetBrowser().CanGoBack())
        self.navigationBar.forwardBtn.Enable(self.chromeWindow.GetBrowser().CanGoForward())

    def OnLoadStart(self, browser, frame):
        if False:
            for i in range(10):
                print('nop')
        if self.navigationBar:
            self.UpdateButtonsState()
            self.navigationBar.GetUrlCtrl().SetValue(browser.GetMainFrame().GetUrl())
            self.navigationBar.AddToHistory(browser.GetMainFrame().GetUrl())

    def OnLoadEnd(self, browser, frame, httpStatusCode):
        if False:
            return 10
        if self.navigationBar:
            self.UpdateButtonsState()

class DefaultClientHandler(object):

    def __init__(self, parentCtrl):
        if False:
            i = 10
            return i + 15
        self.parentCtrl = parentCtrl

    def OnLoadStart(self, browser, frame):
        if False:
            for i in range(10):
                print('nop')
        self.parentCtrl.OnLoadStart(browser, frame)

    def OnLoadEnd(self, browser, frame, httpStatusCode):
        if False:
            print('Hello World!')
        self.parentCtrl.OnLoadEnd(browser, frame, httpStatusCode)

    def OnLoadError(self, browser, frame, errorCode, errorText, failedUrl):
        if False:
            return 10
        Debug('ERROR LOADING URL : %s' % failedUrl)

class CallbackClientHandler(object):

    def __init__(self, onLoadStart=None, onLoadEnd=None):
        if False:
            for i in range(10):
                print('nop')
        self._onLoadStart = onLoadStart
        self._onLoadEnd = onLoadEnd

    def OnLoadStart(self, browser, frame):
        if False:
            print('Hello World!')
        if self._onLoadStart and frame.GetUrl() != 'about:blank':
            self._onLoadStart(browser, frame)

    def OnLoadEnd(self, browser, frame, httpStatusCode):
        if False:
            return 10
        if self._onLoadEnd and frame.GetUrl() != 'about:blank':
            self._onLoadEnd(browser, frame, httpStatusCode)

    def OnLoadError(self, browser, frame, errorCode, errorText, failedUrl):
        if False:
            print('Hello World!')
        Debug('ERROR LOADING URL : %s, %s' % (failedUrl, frame.GetUrl()))

def Initialize(settings=None, debug=False):
    if False:
        i = 10
        return i + 15
    'Initializes CEF, We should do it before initializing wx\n       If no settings passed a default is used\n    '
    switches = {}
    global g_settings
    if not settings:
        settings = {}
    if not 'log_severity' in settings:
        settings['log_severity'] = cefpython.LOGSEVERITY_INFO
    if not 'log_file' in settings:
        settings['log_file'] = ''
    if platform.system() == 'Linux':
        if not 'locales_dir_path' in settings:
            settings['locales_dir_path'] = cefpython.GetModuleDirectory() + '/locales'
        if not 'resources_dir_path' in settings:
            settings['resources_dir_path'] = cefpython.GetModuleDirectory()
    elif platform.system() == 'Darwin':
        if not 'resources_dir_path' in settings:
            settings['resources_dir_path'] = cefpython.GetModuleDirectory() + '/Resources'
        locale_pak = cefpython.GetModuleDirectory() + '/Resources/en.lproj/locale.pak'
        if 'locale_pak' in settings:
            locale_pak = settings['locale_pak']
            del settings['locale_pak']
        switches['locale_pak'] = locale_pak
    if not 'browser_subprocess_path' in settings:
        settings['browser_subprocess_path'] = '%s/%s' % (cefpython.GetModuleDirectory(), 'subprocess')
    if debug:
        settings['debug'] = True
        settings['log_severity'] = cefpython.LOGSEVERITY_VERBOSE
        settings['log_file'] = 'debug.log'
    g_settings = settings
    cefpython.Initialize(settings, switches)

def Shutdown():
    if False:
        while True:
            i = 10
    'Shuts down CEF, should be called by app exiting code'
    DestroyMessageLoopTimer()
    cefpython.Shutdown()