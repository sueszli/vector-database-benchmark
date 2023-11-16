import wx
from cefpython3 import cefpython as cef
import platform
import sys
import os
WINDOWS = platform.system() == 'Windows'
LINUX = platform.system() == 'Linux'
MAC = platform.system() == 'Darwin'
if MAC:
    try:
        from AppKit import NSApp
    except ImportError:
        print('[wxpython.py] Error: PyObjC package is missing, cannot fix Issue #371')
        print('[wxpython.py] To install PyObjC type: pip install -U pyobjc')
        sys.exit(1)
WIDTH = 900
HEIGHT = 640
g_count_windows = 0

def main():
    if False:
        print('Hello World!')
    check_versions()
    sys.excepthook = cef.ExceptHook
    settings = {}
    if MAC:
        settings['external_message_pump'] = True
    if WINDOWS:
        cef.DpiAware.EnableHighDpiSupport()
    cef.Initialize(settings=settings)
    app = CefApp(False)
    app.MainLoop()
    del app
    if not MAC:
        cef.Shutdown()

def check_versions():
    if False:
        return 10
    print('[wxpython.py] CEF Python {ver}'.format(ver=cef.__version__))
    print('[wxpython.py] Python {ver} {arch}'.format(ver=platform.python_version(), arch=platform.architecture()[0]))
    print('[wxpython.py] wxPython {ver}'.format(ver=wx.version()))
    assert cef.__version__ >= '66.0', 'CEF Python v66.0+ required to run this'

def scale_window_size_for_high_dpi(width, height):
    if False:
        print('Hello World!')
    'Scale window size for high DPI devices. This func can be\n    called on all operating systems, but scales only for Windows.\n    If scaled value is bigger than the work area on the display\n    then it will be reduced.'
    if not WINDOWS:
        return (width, height)
    (_, _, max_width, max_height) = wx.GetClientDisplayRect().Get()
    (width, height) = cef.DpiAware.Scale((width, height))
    if width > max_width:
        width = max_width
    if height > max_height:
        height = max_height
    return (width, height)

class MainFrame(wx.Frame):

    def __init__(self):
        if False:
            print('Hello World!')
        self.browser = None
        if LINUX:
            cef.WindowUtils.InstallX11ErrorHandlers()
        global g_count_windows
        g_count_windows += 1
        if WINDOWS:
            print('[wxpython.py] System DPI settings: %s' % str(cef.DpiAware.GetSystemDpi()))
        if hasattr(wx, 'GetDisplayPPI'):
            print('[wxpython.py] wx.GetDisplayPPI = %s' % wx.GetDisplayPPI())
        print('[wxpython.py] wx.GetDisplaySize = %s' % wx.GetDisplaySize())
        print('[wxpython.py] MainFrame declared size: %s' % str((WIDTH, HEIGHT)))
        size = scale_window_size_for_high_dpi(WIDTH, HEIGHT)
        print('[wxpython.py] MainFrame DPI scaled size: %s' % str(size))
        wx.Frame.__init__(self, parent=None, id=wx.ID_ANY, title='wxPython example', size=size)
        print('[wxpython.py] MainFrame actual size: %s' % self.GetSize())
        self.setup_icon()
        self.create_menu()
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.browser_panel = wx.Panel(self, style=wx.WANTS_CHARS)
        self.browser_panel.Bind(wx.EVT_SET_FOCUS, self.OnSetFocus)
        self.browser_panel.Bind(wx.EVT_SIZE, self.OnSize)
        if MAC:
            NSApp.windows()[0].contentView().setWantsLayer_(True)
        if LINUX:
            self.Show()
            if wx.version().startswith('3.') or wx.version().startswith('4.'):
                wx.CallLater(100, self.embed_browser)
            else:
                self.embed_browser()
        else:
            self.embed_browser()
            self.Show()

    def setup_icon(self):
        if False:
            i = 10
            return i + 15
        icon_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resources', 'wxpython.png')
        if os.path.exists(icon_file) and hasattr(wx, 'IconFromBitmap'):
            icon = wx.IconFromBitmap(wx.Bitmap(icon_file, wx.BITMAP_TYPE_PNG))
            self.SetIcon(icon)

    def create_menu(self):
        if False:
            print('Hello World!')
        filemenu = wx.Menu()
        filemenu.Append(1, 'Some option')
        filemenu.Append(2, 'Another option')
        menubar = wx.MenuBar()
        menubar.Append(filemenu, '&File')
        self.SetMenuBar(menubar)

    def embed_browser(self):
        if False:
            for i in range(10):
                print('nop')
        window_info = cef.WindowInfo()
        (width, height) = self.browser_panel.GetClientSize().Get()
        assert self.browser_panel.GetHandle(), 'Window handle not available'
        window_info.SetAsChild(self.browser_panel.GetHandle(), [0, 0, width, height])
        self.browser = cef.CreateBrowserSync(window_info, url='https://www.google.com/')
        self.browser.SetClientHandler(FocusHandler())

    def OnSetFocus(self, _):
        if False:
            return 10
        if not self.browser:
            return
        if WINDOWS:
            cef.WindowUtils.OnSetFocus(self.browser_panel.GetHandle(), 0, 0, 0)
        self.browser.SetFocus(True)

    def OnSize(self, _):
        if False:
            while True:
                i = 10
        if not self.browser:
            return
        if WINDOWS:
            cef.WindowUtils.OnSize(self.browser_panel.GetHandle(), 0, 0, 0)
        elif LINUX:
            (x, y) = (0, 0)
            (width, height) = self.browser_panel.GetSize().Get()
            self.browser.SetBounds(x, y, width, height)
        self.browser.NotifyMoveOrResizeStarted()

    def OnClose(self, event):
        if False:
            return 10
        print('[wxpython.py] OnClose called')
        if not self.browser:
            return
        if MAC:
            self.browser.CloseBrowser()
            self.clear_browser_references()
            self.Destroy()
            global g_count_windows
            g_count_windows -= 1
            if g_count_windows == 0:
                cef.Shutdown()
                wx.GetApp().ExitMainLoop()
                os._exit(0)
        else:
            self.browser.ParentWindowWillClose()
            event.Skip()
            self.clear_browser_references()

    def clear_browser_references(self):
        if False:
            print('Hello World!')
        self.browser = None

class FocusHandler(object):

    def OnGotFocus(self, browser, **_):
        if False:
            print('Hello World!')
        if LINUX:
            print('[wxpython.py] FocusHandler.OnGotFocus: keyboard focus fix (Issue #284)')
            browser.SetFocus(True)

class CefApp(wx.App):

    def __init__(self, redirect):
        if False:
            i = 10
            return i + 15
        self.timer = None
        self.timer_id = 1
        self.is_initialized = False
        super(CefApp, self).__init__(redirect=redirect)

    def OnPreInit(self):
        if False:
            i = 10
            return i + 15
        super(CefApp, self).OnPreInit()
        if MAC and wx.version().startswith('4.'):
            print('[wxpython.py] OnPreInit: initialize here (wxPython 4.0 fix)')
            self.initialize()

    def OnInit(self):
        if False:
            for i in range(10):
                print('nop')
        self.initialize()
        return True

    def initialize(self):
        if False:
            i = 10
            return i + 15
        if self.is_initialized:
            return
        self.is_initialized = True
        self.create_timer()
        frame = MainFrame()
        self.SetTopWindow(frame)
        frame.Show()

    def create_timer(self):
        if False:
            print('Hello World!')
        self.timer = wx.Timer(self, self.timer_id)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(10)

    def on_timer(self, _):
        if False:
            while True:
                i = 10
        cef.MessageLoopWork()

    def OnExit(self):
        if False:
            while True:
                i = 10
        self.timer.Stop()
        return 0
if __name__ == '__main__':
    main()