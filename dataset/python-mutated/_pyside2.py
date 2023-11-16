"""PySide2 proxy backend for the qt backend."""
import sys
from .. import backends
from ...util import logger
from ... import config
USE_EGL = config['gl_backend'].lower().startswith('es')
try:
    if not USE_EGL:
        from PySide2 import QtOpenGL
    from PySide2 import QtGui, QtCore
except Exception as exp:
    (available, testable, why_not, which) = (False, False, str(exp), None)
else:
    (available, testable, why_not) = (True, True, None)
    has_uic = False
    import PySide2
    from PySide2 import QtTest
    if not hasattr(QtTest.QTest, 'qWait'):

        @staticmethod
        def qWait(msec):
            if False:
                i = 10
                return i + 15
            import time
            start = time.time()
            PySide2.QtWidgets.QApplication.processEvents()
            while time.time() < start + msec * 0.001:
                PySide2.QtWidgets.QApplication.processEvents()
        QtTest.QTest.qWait = qWait
    which = ('PySide2', PySide2.__version__, QtCore.__version__)
    sys.modules.pop(__name__.replace('_pyside2', '_qt'), None)
    if backends.qt_lib is None:
        backends.qt_lib = 'pyside2'
        from . import _qt
        from ._qt import *
    else:
        logger.warning('%s already imported, cannot switch to %s' % (backends.qt_lib, 'pyside2'))