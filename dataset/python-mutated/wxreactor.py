"""
This module provides wxPython event loop support for Twisted.

In order to use this support, simply do the following::

    |  from twisted.internet import wxreactor
    |  wxreactor.install()

Then, when your root wxApp has been created::

    | from twisted.internet import reactor
    | reactor.registerWxApp(yourApp)
    | reactor.run()

Then use twisted.internet APIs as usual. Stop the event loop using
reactor.stop(), not yourApp.ExitMainLoop().

IMPORTANT: tests will fail when run under this reactor. This is
expected and probably does not reflect on the reactor's ability to run
real applications.
"""
from queue import Empty, Queue
try:
    from wx import CallAfter as wxCallAfter, PySimpleApp as wxPySimpleApp, Timer as wxTimer
except ImportError:
    from wxPython.wx import wxPySimpleApp, wxCallAfter, wxTimer
from twisted.internet import _threadedselect
from twisted.python import log, runtime

class ProcessEventsTimer(wxTimer):
    """
    Timer that tells wx to process pending events.

    This is necessary on macOS, probably due to a bug in wx, if we want
    wxCallAfters to be handled when modal dialogs, menus, etc.  are open.
    """

    def __init__(self, wxapp):
        if False:
            print('Hello World!')
        wxTimer.__init__(self)
        self.wxapp = wxapp

    def Notify(self):
        if False:
            return 10
        '\n        Called repeatedly by wx event loop.\n        '
        self.wxapp.ProcessPendingEvents()

class WxReactor(_threadedselect.ThreadedSelectReactor):
    """
    wxPython reactor.

    wxPython drives the event loop, select() runs in a thread.
    """
    _stopping = False

    def registerWxApp(self, wxapp):
        if False:
            print('Hello World!')
        '\n        Register wxApp instance with the reactor.\n        '
        self.wxapp = wxapp

    def _installSignalHandlersAgain(self):
        if False:
            print('Hello World!')
        '\n        wx sometimes removes our own signal handlers, so re-add them.\n        '
        try:
            import signal
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except ImportError:
            return
        self._signals.install()

    def stop(self):
        if False:
            i = 10
            return i + 15
        '\n        Stop the reactor.\n        '
        if self._stopping:
            return
        self._stopping = True
        _threadedselect.ThreadedSelectReactor.stop(self)

    def _runInMainThread(self, f):
        if False:
            while True:
                i = 10
        '\n        Schedule function to run in main wx/Twisted thread.\n\n        Called by the select() thread.\n        '
        if hasattr(self, 'wxapp'):
            wxCallAfter(f)
        else:
            self._postQueue.put(f)

    def _stopWx(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Stop the wx event loop if it hasn't already been stopped.\n\n        Called during Twisted event loop shutdown.\n        "
        if hasattr(self, 'wxapp'):
            self.wxapp.ExitMainLoop()

    def run(self, installSignalHandlers=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start the reactor.\n        '
        self._postQueue = Queue()
        if not hasattr(self, 'wxapp'):
            log.msg('registerWxApp() was not called on reactor, registering my own wxApp instance.')
            self.registerWxApp(wxPySimpleApp())
        self.interleave(self._runInMainThread, installSignalHandlers=installSignalHandlers)
        if installSignalHandlers:
            self.callLater(0, self._installSignalHandlersAgain)
        self.addSystemEventTrigger('after', 'shutdown', self._stopWx)
        self.addSystemEventTrigger('after', 'shutdown', lambda : self._postQueue.put(None))
        if runtime.platform.isMacOSX():
            t = ProcessEventsTimer(self.wxapp)
            t.Start(2)
        self.wxapp.MainLoop()
        wxapp = self.wxapp
        del self.wxapp
        if not self._stopping:
            self.stop()
            wxapp.ProcessPendingEvents()
            while 1:
                try:
                    f = self._postQueue.get(timeout=0.01)
                except Empty:
                    continue
                else:
                    if f is None:
                        break
                    try:
                        f()
                    except BaseException:
                        log.err()

def install():
    if False:
        for i in range(10):
            print('nop')
    '\n    Configure the twisted mainloop to be run inside the wxPython mainloop.\n    '
    reactor = WxReactor()
    from twisted.internet.main import installReactor
    installReactor(reactor)
    return reactor
__all__ = ['install']