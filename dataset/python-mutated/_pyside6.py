"""PySide6 proxy backend for the qt backend.
Based on PySide2 backend.
"""
import sys
from .. import backends
from ...util import logger
from ... import config
USE_EGL = config['gl_backend'].lower().startswith('es')
try:
    if not USE_EGL:
        from PySide6 import QtOpenGL
    from PySide6 import QtGui, QtCore
except Exception as exp:
    (available, testable, why_not, which) = (False, False, str(exp), None)
else:
    (available, testable, why_not) = (True, True, None)
    has_uic = False
    import PySide6
    from PySide6 import QtTest

    @staticmethod
    def qWait(msec):
        if False:
            for i in range(10):
                print('nop')
        import time
        start = time.time()
        PySide6.QtWidgets.QApplication.processEvents()
        while time.time() < start + msec * 0.001:
            PySide6.QtWidgets.QApplication.processEvents()
    QtTest.QTest.qWait = qWait
    which = ('PySide6', PySide6.__version__, QtCore.__version__)
    sys.modules.pop(__name__.replace('_pyside6', '_qt'), None)
    if backends.qt_lib is None:
        backends.qt_lib = 'pyside6'
        from . import _qt
        from ._qt import *
    else:
        logger.warning('%s already imported, cannot switch to %s' % (backends.qt_lib, 'pyside6'))