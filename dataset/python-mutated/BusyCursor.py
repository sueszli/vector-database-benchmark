from contextlib import contextmanager
from ..Qt import QtCore, QtGui, QtWidgets
__all__ = ['BusyCursor']

@contextmanager
def BusyCursor():
    if False:
        while True:
            i = 10
    '\n    Display a busy mouse cursor during long operations.\n    Usage::\n\n        with BusyCursor():\n            doLongOperation()\n\n    May be nested. If called from a non-gui thread, then the cursor will not be affected.\n    '
    app = QtCore.QCoreApplication.instance()
    in_gui_thread = app is not None and QtCore.QThread.currentThread() == app.thread()
    try:
        if in_gui_thread:
            guard = QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        yield
    finally:
        if in_gui_thread:
            if hasattr(guard, 'restoreOverrideCursor'):
                guard.restoreOverrideCursor()
            else:
                QtWidgets.QApplication.restoreOverrideCursor()