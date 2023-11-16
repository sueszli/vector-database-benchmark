"""
Acceptance tests for wxreactor.

Please test on Linux, Win32 and macOS:
1. Startup event is called at startup.
2. Scheduled event is called after 2 seconds.
3. Shutdown takes 3 seconds, both when quiting from menu and when closing
   window (e.g. Alt-F4 in metacity). This tests reactor.stop() and
   wxApp.ExitEventLoop().
4. 'hello, world' continues to be printed even when modal dialog is open
   (use dialog menu item), when menus are held down, when window is being
   dragged.
"""
import sys
import time
try:
    from wx import EVT_MENU, App as wxApp, DefaultPosition as wxDefaultPosition, Frame as wxFrame, Menu as wxMenu, MenuBar as wxMenuBar, MessageDialog as wxMessageDialog, Size as wxSize
except ImportError:
    from wxPython.wx import *
from twisted.internet import wxreactor
from twisted.python import log
wxreactor.install()
from twisted.internet import defer, reactor
dc = None

def helloWorld():
    if False:
        i = 10
        return i + 15
    global dc
    print('hello, world', time.time())
    dc = reactor.callLater(0.1, helloWorld)
dc = reactor.callLater(0.1, helloWorld)

def twoSecondsPassed():
    if False:
        print('Hello World!')
    print('two seconds passed')

def printer(s):
    if False:
        return 10
    print(s)

def shutdown():
    if False:
        return 10
    print('shutting down in 3 seconds')
    if dc.active():
        dc.cancel()
    reactor.callLater(1, printer, '2...')
    reactor.callLater(2, printer, '1...')
    reactor.callLater(3, printer, '0...')
    d = defer.Deferred()
    reactor.callLater(3, d.callback, 1)
    return d

def startup():
    if False:
        for i in range(10):
            print('nop')
    print('Start up event!')
reactor.callLater(2, twoSecondsPassed)
reactor.addSystemEventTrigger('after', 'startup', startup)
reactor.addSystemEventTrigger('before', 'shutdown', shutdown)
ID_EXIT = 101
ID_DIALOG = 102

class MyFrame(wxFrame):

    def __init__(self, parent, ID, title):
        if False:
            for i in range(10):
                print('nop')
        wxFrame.__init__(self, parent, ID, title, wxDefaultPosition, wxSize(300, 200))
        menu = wxMenu()
        menu.Append(ID_DIALOG, 'D&ialog', 'Show dialog')
        menu.Append(ID_EXIT, 'E&xit', 'Terminate the program')
        menuBar = wxMenuBar()
        menuBar.Append(menu, '&File')
        self.SetMenuBar(menuBar)
        EVT_MENU(self, ID_EXIT, self.DoExit)
        EVT_MENU(self, ID_DIALOG, self.DoDialog)

    def DoDialog(self, event):
        if False:
            for i in range(10):
                print('nop')
        dl = wxMessageDialog(self, 'Check terminal to see if messages are still being printed by Twisted.')
        dl.ShowModal()
        dl.Destroy()

    def DoExit(self, event):
        if False:
            for i in range(10):
                print('nop')
        reactor.stop()

class MyApp(wxApp):

    def OnInit(self):
        if False:
            print('Hello World!')
        frame = MyFrame(None, -1, 'Hello, world')
        frame.Show(True)
        self.SetTopWindow(frame)
        return True

def demo():
    if False:
        return 10
    log.startLogging(sys.stdout)
    app = MyApp(0)
    reactor.registerWxApp(app)
    reactor.run()
if __name__ == '__main__':
    demo()