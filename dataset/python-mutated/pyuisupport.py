"""
This module integrates PyUI with twisted.internet's mainloop.

Maintainer: Jp Calderone

See doc/examples/pyuidemo.py for example usage.
"""
import pyui

def _guiUpdate(reactor, delay):
    if False:
        while True:
            i = 10
    pyui.draw()
    if pyui.update() == 0:
        pyui.quit()
        reactor.stop()
    else:
        reactor.callLater(delay, _guiUpdate, reactor, delay)

def install(ms=10, reactor=None, args=(), kw={}):
    if False:
        for i in range(10):
            print('nop')
    "\n    Schedule PyUI's display to be updated approximately every C{ms}\n    milliseconds, and initialize PyUI with the specified arguments.\n    "
    d = pyui.init(*args, **kw)
    if reactor is None:
        from twisted.internet import reactor
    _guiUpdate(reactor, ms / 1000.0)
    return d
__all__ = ['install']