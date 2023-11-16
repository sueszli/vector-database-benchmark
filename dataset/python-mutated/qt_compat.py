"""
Qt binding and backend selector.

The selection logic is as follows:
- if any of PyQt6, PySide6, PyQt5, or PySide2 have already been
  imported (checked in that order), use it;
- otherwise, if the QT_API environment variable (used by Enthought) is set, use
  it to determine which binding to use;
- otherwise, use whatever the rcParams indicate.
"""
import operator
import os
import platform
import sys
from packaging.version import parse as parse_version
import matplotlib as mpl
from . import _QT_FORCE_QT5_BINDING
QT_API_PYQT6 = 'PyQt6'
QT_API_PYSIDE6 = 'PySide6'
QT_API_PYQT5 = 'PyQt5'
QT_API_PYSIDE2 = 'PySide2'
QT_API_ENV = os.environ.get('QT_API')
if QT_API_ENV is not None:
    QT_API_ENV = QT_API_ENV.lower()
_ETS = {'pyqt6': QT_API_PYQT6, 'pyside6': QT_API_PYSIDE6, 'pyqt5': QT_API_PYQT5, 'pyside2': QT_API_PYSIDE2}
if sys.modules.get('PyQt6.QtCore'):
    QT_API = QT_API_PYQT6
elif sys.modules.get('PySide6.QtCore'):
    QT_API = QT_API_PYSIDE6
elif sys.modules.get('PyQt5.QtCore'):
    QT_API = QT_API_PYQT5
elif sys.modules.get('PySide2.QtCore'):
    QT_API = QT_API_PYSIDE2
elif (mpl.rcParams._get_backend_or_none() or '').lower().startswith('qt5'):
    if QT_API_ENV in ['pyqt5', 'pyside2']:
        QT_API = _ETS[QT_API_ENV]
    else:
        _QT_FORCE_QT5_BINDING = True
        QT_API = None
elif QT_API_ENV is None:
    QT_API = None
elif QT_API_ENV in _ETS:
    QT_API = _ETS[QT_API_ENV]
else:
    raise RuntimeError('The environment variable QT_API has the unrecognized value {!r}; valid values are {}'.format(QT_API_ENV, ', '.join(_ETS)))

def _setup_pyqt5plus():
    if False:
        while True:
            i = 10
    global QtCore, QtGui, QtWidgets, __version__
    global _isdeleted, _to_int
    if QT_API == QT_API_PYQT6:
        from PyQt6 import QtCore, QtGui, QtWidgets, sip
        __version__ = QtCore.PYQT_VERSION_STR
        QtCore.Signal = QtCore.pyqtSignal
        QtCore.Slot = QtCore.pyqtSlot
        QtCore.Property = QtCore.pyqtProperty
        _isdeleted = sip.isdeleted
        _to_int = operator.attrgetter('value')
    elif QT_API == QT_API_PYSIDE6:
        from PySide6 import QtCore, QtGui, QtWidgets, __version__
        import shiboken6

        def _isdeleted(obj):
            if False:
                while True:
                    i = 10
            return not shiboken6.isValid(obj)
        if parse_version(__version__) >= parse_version('6.4'):
            _to_int = operator.attrgetter('value')
        else:
            _to_int = int
    elif QT_API == QT_API_PYQT5:
        from PyQt5 import QtCore, QtGui, QtWidgets
        import sip
        __version__ = QtCore.PYQT_VERSION_STR
        QtCore.Signal = QtCore.pyqtSignal
        QtCore.Slot = QtCore.pyqtSlot
        QtCore.Property = QtCore.pyqtProperty
        _isdeleted = sip.isdeleted
        _to_int = int
    elif QT_API == QT_API_PYSIDE2:
        from PySide2 import QtCore, QtGui, QtWidgets, __version__
        try:
            from PySide2 import shiboken2
        except ImportError:
            import shiboken2

        def _isdeleted(obj):
            if False:
                while True:
                    i = 10
            return not shiboken2.isValid(obj)
        _to_int = int
    else:
        raise AssertionError(f'Unexpected QT_API: {QT_API}')
if QT_API in [QT_API_PYQT6, QT_API_PYQT5, QT_API_PYSIDE6, QT_API_PYSIDE2]:
    _setup_pyqt5plus()
elif QT_API is None:
    if _QT_FORCE_QT5_BINDING:
        _candidates = [(_setup_pyqt5plus, QT_API_PYQT5), (_setup_pyqt5plus, QT_API_PYSIDE2)]
    else:
        _candidates = [(_setup_pyqt5plus, QT_API_PYQT6), (_setup_pyqt5plus, QT_API_PYSIDE6), (_setup_pyqt5plus, QT_API_PYQT5), (_setup_pyqt5plus, QT_API_PYSIDE2)]
    for (_setup, QT_API) in _candidates:
        try:
            _setup()
        except ImportError:
            continue
        break
    else:
        raise ImportError('Failed to import any of the following Qt binding modules: {}'.format(', '.join([QT_API for (_, QT_API) in _candidates])))
else:
    raise AssertionError(f'Unexpected QT_API: {QT_API}')
_version_info = tuple(QtCore.QLibraryInfo.version().segments())
if _version_info < (5, 12):
    raise ImportError(f'The Qt version imported is {QtCore.QLibraryInfo.version().toString()} but Matplotlib requires Qt>=5.12')
if sys.platform == 'darwin' and parse_version(platform.mac_ver()[0]) >= parse_version('10.16') and (_version_info < (5, 15, 2)):
    os.environ.setdefault('QT_MAC_WANTS_LAYER', '1')

def _exec(obj):
    if False:
        i = 10
        return i + 15
    obj.exec() if hasattr(obj, 'exec') else obj.exec_()