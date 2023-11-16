"""Old method of wxPython support for Twisted.

twisted.internet.wxreactor is probably a better choice.

To use::

    | # given a wxApp instance called myWxAppInstance:
    | from twisted.internet import wxsupport
    | wxsupport.install(myWxAppInstance)

Use Twisted's APIs for running and stopping the event loop, don't use
wxPython's methods.

On Windows the Twisted event loop might block when dialogs are open
or menus are selected.

Maintainer: Itamar Shtull-Trauring
"""
import warnings
warnings.warn('wxsupport is not fully functional on Windows, wxreactor is better.')
from twisted.internet import reactor

class wxRunner:
    """Make sure GUI events are handled."""

    def __init__(self, app):
        if False:
            return 10
        self.app = app

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute pending WX events followed by WX idle events and\n        reschedule.\n        '
        while self.app.Pending():
            self.app.Dispatch()
        self.app.ProcessIdle()
        reactor.callLater(0.02, self.run)

def install(app):
    if False:
        while True:
            i = 10
    'Install the wxPython support, given a wxApp instance'
    runner = wxRunner(app)
    reactor.callLater(0.02, runner.run)
__all__ = ['install']