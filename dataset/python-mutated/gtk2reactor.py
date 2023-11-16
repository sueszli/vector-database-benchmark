"""
This module provides support for Twisted to interact with the glib/gtk2
mainloop.

In order to use this support, simply do the following::

    from twisted.internet import gtk2reactor
    gtk2reactor.install()

Then use twisted.internet APIs as usual.  The other methods here are not
intended to be called directly.
"""
from incremental import Version
from ._deprecate import deprecatedGnomeReactor
deprecatedGnomeReactor('gtk2reactor', Version('Twisted', 23, 8, 0))
import sys
from twisted.internet import _glibbase
from twisted.python import runtime
try:
    if not hasattr(sys, 'frozen'):
        import pygtk
        pygtk.require('2.0')
except (ImportError, AttributeError):
    pass
import gobject
if not hasattr(gobject, 'IO_HUP'):
    raise ImportError('pygobject 2.x is not installed. Use the `gi` reactor.')
if hasattr(gobject, 'threads_init'):
    gobject.threads_init()

class Gtk2Reactor(_glibbase.GlibReactorBase):
    """
    PyGTK+ 2 event loop reactor.
    """

    def __init__(self, useGtk=True):
        if False:
            return 10
        _gtk = None
        if useGtk is True:
            import gtk as _gtk
        _glibbase.GlibReactorBase.__init__(self, gobject, _gtk, useGtk=useGtk)
PortableGtkReactor = Gtk2Reactor

def install(useGtk=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Configure the twisted mainloop to be run inside the gtk mainloop.\n\n    @param useGtk: should glib rather than GTK+ event loop be\n        used (this will be slightly faster but does not support GUI).\n    '
    reactor = Gtk2Reactor(useGtk)
    from twisted.internet.main import installReactor
    installReactor(reactor)
    return reactor

def portableInstall(useGtk=True):
    if False:
        i = 10
        return i + 15
    '\n    Configure the twisted mainloop to be run inside the gtk mainloop.\n    '
    reactor = PortableGtkReactor()
    from twisted.internet.main import installReactor
    installReactor(reactor)
    return reactor
if runtime.platform.getType() == 'posix':
    install = install
else:
    install = portableInstall
__all__ = ['install']