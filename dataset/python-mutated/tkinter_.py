from cefpython3 import cefpython as cef
import ctypes
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk
import sys
import os
import platform
import logging as _logging
WindowUtils = cef.WindowUtils()
WINDOWS = platform.system() == 'Windows'
LINUX = platform.system() == 'Linux'
MAC = platform.system() == 'Darwin'
logger = _logging.getLogger('tkinter_.py')
IMAGE_EXT = '.png' if tk.TkVersion > 8.5 else '.gif'

def main():
    if False:
        return 10
    logger.setLevel(_logging.DEBUG)
    stream_handler = _logging.StreamHandler()
    formatter = _logging.Formatter('[%(filename)s] %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info('CEF Python {ver}'.format(ver=cef.__version__))
    logger.info('Python {ver} {arch}'.format(ver=platform.python_version(), arch=platform.architecture()[0]))
    logger.info('Tk {ver}'.format(ver=tk.Tcl().eval('info patchlevel')))
    assert cef.__version__ >= '55.3', 'CEF Python v55.3+ required to run this'
    sys.excepthook = cef.ExceptHook
    root = tk.Tk()
    app = MainFrame(root)
    settings = {}
    if MAC:
        settings['external_message_pump'] = True
    cef.Initialize(settings=settings)
    app.mainloop()
    logger.debug('Main loop exited')
    cef.Shutdown()

class MainFrame(tk.Frame):

    def __init__(self, root):
        if False:
            i = 10
            return i + 15
        self.browser_frame = None
        self.navigation_bar = None
        self.root = root
        root.geometry('900x640')
        tk.Grid.rowconfigure(root, 0, weight=1)
        tk.Grid.columnconfigure(root, 0, weight=1)
        tk.Frame.__init__(self, root)
        self.master.title('Tkinter example')
        self.master.protocol('WM_DELETE_WINDOW', self.on_close)
        self.master.bind('<Configure>', self.on_root_configure)
        self.setup_icon()
        self.bind('<Configure>', self.on_configure)
        self.bind('<FocusIn>', self.on_focus_in)
        self.bind('<FocusOut>', self.on_focus_out)
        self.navigation_bar = NavigationBar(self)
        self.navigation_bar.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        tk.Grid.rowconfigure(self, 0, weight=0)
        tk.Grid.columnconfigure(self, 0, weight=0)
        self.browser_frame = BrowserFrame(self, self.navigation_bar)
        self.browser_frame.grid(row=1, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        tk.Grid.rowconfigure(self, 1, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)
        self.pack(fill=tk.BOTH, expand=tk.YES)

    def on_root_configure(self, _):
        if False:
            i = 10
            return i + 15
        logger.debug('MainFrame.on_root_configure')
        if self.browser_frame:
            self.browser_frame.on_root_configure()

    def on_configure(self, event):
        if False:
            return 10
        logger.debug('MainFrame.on_configure')
        if self.browser_frame:
            width = event.width
            height = event.height
            if self.navigation_bar:
                height = height - self.navigation_bar.winfo_height()
            self.browser_frame.on_mainframe_configure(width, height)

    def on_focus_in(self, _):
        if False:
            print('Hello World!')
        logger.debug('MainFrame.on_focus_in')

    def on_focus_out(self, _):
        if False:
            print('Hello World!')
        logger.debug('MainFrame.on_focus_out')

    def on_close(self):
        if False:
            print('Hello World!')
        if self.browser_frame:
            self.browser_frame.on_root_close()
            self.browser_frame = None
        else:
            self.master.destroy()

    def get_browser(self):
        if False:
            return 10
        if self.browser_frame:
            return self.browser_frame.browser
        return None

    def get_browser_frame(self):
        if False:
            while True:
                i = 10
        if self.browser_frame:
            return self.browser_frame
        return None

    def setup_icon(self):
        if False:
            i = 10
            return i + 15
        resources = os.path.join(os.path.dirname(__file__), 'resources')
        icon_path = os.path.join(resources, 'tkinter' + IMAGE_EXT)
        if os.path.exists(icon_path):
            self.icon = tk.PhotoImage(file=icon_path)
            self.master.call('wm', 'iconphoto', self.master._w, self.icon)

class BrowserFrame(tk.Frame):

    def __init__(self, mainframe, navigation_bar=None):
        if False:
            print('Hello World!')
        self.navigation_bar = navigation_bar
        self.closing = False
        self.browser = None
        tk.Frame.__init__(self, mainframe)
        self.mainframe = mainframe
        self.bind('<FocusIn>', self.on_focus_in)
        self.bind('<FocusOut>', self.on_focus_out)
        self.bind('<Configure>', self.on_configure)
        'For focus problems see Issue #255 and Issue #535. '
        self.focus_set()

    def embed_browser(self):
        if False:
            return 10
        window_info = cef.WindowInfo()
        rect = [0, 0, self.winfo_width(), self.winfo_height()]
        window_info.SetAsChild(self.get_window_handle(), rect)
        self.browser = cef.CreateBrowserSync(window_info, url='https://www.google.com/')
        assert self.browser
        self.browser.SetClientHandler(LifespanHandler(self))
        self.browser.SetClientHandler(LoadHandler(self))
        self.browser.SetClientHandler(FocusHandler(self))
        self.message_loop_work()

    def get_window_handle(self):
        if False:
            print('Hello World!')
        if MAC:
            from AppKit import NSApp
            import objc
            logger.info('winfo_id={}'.format(self.winfo_id()))
            content_view = objc.pyobjc_id(NSApp.windows()[-1].contentView())
            logger.info('content_view={}'.format(content_view))
            return content_view
        elif self.winfo_id() > 0:
            return self.winfo_id()
        else:
            raise Exception("Couldn't obtain window handle")

    def message_loop_work(self):
        if False:
            return 10
        cef.MessageLoopWork()
        self.after(10, self.message_loop_work)

    def on_configure(self, _):
        if False:
            i = 10
            return i + 15
        if not self.browser:
            self.embed_browser()

    def on_root_configure(self):
        if False:
            while True:
                i = 10
        if self.browser:
            self.browser.NotifyMoveOrResizeStarted()

    def on_mainframe_configure(self, width, height):
        if False:
            return 10
        if self.browser:
            if WINDOWS:
                ctypes.windll.user32.SetWindowPos(self.browser.GetWindowHandle(), 0, 0, 0, width, height, 2)
            elif LINUX:
                self.browser.SetBounds(0, 0, width, height)
            self.browser.NotifyMoveOrResizeStarted()

    def on_focus_in(self, _):
        if False:
            while True:
                i = 10
        logger.debug('BrowserFrame.on_focus_in')
        if self.browser:
            self.browser.SetFocus(True)

    def on_focus_out(self, _):
        if False:
            print('Hello World!')
        logger.debug('BrowserFrame.on_focus_out')
        'For focus problems see Issue #255 and Issue #535. '
        if LINUX and self.browser:
            self.browser.SetFocus(False)

    def on_root_close(self):
        if False:
            for i in range(10):
                print('nop')
        logger.info('BrowserFrame.on_root_close')
        if self.browser:
            logger.debug('CloseBrowser')
            self.browser.CloseBrowser(True)
            self.clear_browser_references()
        else:
            logger.debug('tk.Frame.destroy')
            self.destroy()

    def clear_browser_references(self):
        if False:
            for i in range(10):
                print('nop')
        self.browser = None

class LifespanHandler(object):

    def __init__(self, tkFrame):
        if False:
            return 10
        self.tkFrame = tkFrame

    def OnBeforeClose(self, browser, **_):
        if False:
            while True:
                i = 10
        logger.debug('LifespanHandler.OnBeforeClose')
        self.tkFrame.quit()

class LoadHandler(object):

    def __init__(self, browser_frame):
        if False:
            return 10
        self.browser_frame = browser_frame

    def OnLoadStart(self, browser, **_):
        if False:
            i = 10
            return i + 15
        if self.browser_frame.master.navigation_bar:
            self.browser_frame.master.navigation_bar.set_url(browser.GetUrl())

class FocusHandler(object):
    """For focus problems see Issue #255 and Issue #535. """

    def __init__(self, browser_frame):
        if False:
            while True:
                i = 10
        self.browser_frame = browser_frame

    def OnTakeFocus(self, next_component, **_):
        if False:
            return 10
        logger.debug('FocusHandler.OnTakeFocus, next={next}'.format(next=next_component))

    def OnSetFocus(self, source, **_):
        if False:
            while True:
                i = 10
        logger.debug('FocusHandler.OnSetFocus, source={source}'.format(source=source))
        if LINUX:
            return False
        else:
            return True

    def OnGotFocus(self, **_):
        if False:
            while True:
                i = 10
        logger.debug('FocusHandler.OnGotFocus')
        if LINUX:
            self.browser_frame.focus_set()

class NavigationBar(tk.Frame):

    def __init__(self, master):
        if False:
            print('Hello World!')
        self.back_state = tk.NONE
        self.forward_state = tk.NONE
        self.back_image = None
        self.forward_image = None
        self.reload_image = None
        tk.Frame.__init__(self, master)
        resources = os.path.join(os.path.dirname(__file__), 'resources')
        back_png = os.path.join(resources, 'back' + IMAGE_EXT)
        if os.path.exists(back_png):
            self.back_image = tk.PhotoImage(file=back_png)
        self.back_button = tk.Button(self, image=self.back_image, command=self.go_back)
        self.back_button.grid(row=0, column=0)
        forward_png = os.path.join(resources, 'forward' + IMAGE_EXT)
        if os.path.exists(forward_png):
            self.forward_image = tk.PhotoImage(file=forward_png)
        self.forward_button = tk.Button(self, image=self.forward_image, command=self.go_forward)
        self.forward_button.grid(row=0, column=1)
        reload_png = os.path.join(resources, 'reload' + IMAGE_EXT)
        if os.path.exists(reload_png):
            self.reload_image = tk.PhotoImage(file=reload_png)
        self.reload_button = tk.Button(self, image=self.reload_image, command=self.reload)
        self.reload_button.grid(row=0, column=2)
        self.url_entry = tk.Entry(self)
        self.url_entry.bind('<FocusIn>', self.on_url_focus_in)
        self.url_entry.bind('<FocusOut>', self.on_url_focus_out)
        self.url_entry.bind('<Return>', self.on_load_url)
        self.url_entry.bind('<Button-1>', self.on_button1)
        self.url_entry.grid(row=0, column=3, sticky=tk.N + tk.S + tk.E + tk.W)
        tk.Grid.rowconfigure(self, 0, weight=100)
        tk.Grid.columnconfigure(self, 3, weight=100)
        self.update_state()

    def go_back(self):
        if False:
            i = 10
            return i + 15
        if self.master.get_browser():
            self.master.get_browser().GoBack()

    def go_forward(self):
        if False:
            return 10
        if self.master.get_browser():
            self.master.get_browser().GoForward()

    def reload(self):
        if False:
            print('Hello World!')
        if self.master.get_browser():
            self.master.get_browser().Reload()

    def set_url(self, url):
        if False:
            print('Hello World!')
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, url)

    def on_url_focus_in(self, _):
        if False:
            i = 10
            return i + 15
        logger.debug('NavigationBar.on_url_focus_in')

    def on_url_focus_out(self, _):
        if False:
            while True:
                i = 10
        logger.debug('NavigationBar.on_url_focus_out')

    def on_load_url(self, _):
        if False:
            while True:
                i = 10
        if self.master.get_browser():
            self.master.get_browser().StopLoad()
            self.master.get_browser().LoadUrl(self.url_entry.get())

    def on_button1(self, _):
        if False:
            return 10
        'For focus problems see Issue #255 and Issue #535. '
        logger.debug('NavigationBar.on_button1')
        self.master.master.focus_force()

    def update_state(self):
        if False:
            for i in range(10):
                print('nop')
        browser = self.master.get_browser()
        if not browser:
            if self.back_state != tk.DISABLED:
                self.back_button.config(state=tk.DISABLED)
                self.back_state = tk.DISABLED
            if self.forward_state != tk.DISABLED:
                self.forward_button.config(state=tk.DISABLED)
                self.forward_state = tk.DISABLED
            self.after(100, self.update_state)
            return
        if browser.CanGoBack():
            if self.back_state != tk.NORMAL:
                self.back_button.config(state=tk.NORMAL)
                self.back_state = tk.NORMAL
        elif self.back_state != tk.DISABLED:
            self.back_button.config(state=tk.DISABLED)
            self.back_state = tk.DISABLED
        if browser.CanGoForward():
            if self.forward_state != tk.NORMAL:
                self.forward_button.config(state=tk.NORMAL)
                self.forward_state = tk.NORMAL
        elif self.forward_state != tk.DISABLED:
            self.forward_button.config(state=tk.DISABLED)
            self.forward_state = tk.DISABLED
        self.after(100, self.update_state)

class Tabs(tk.Frame):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        tk.Frame.__init__(self)
if __name__ == '__main__':
    main()