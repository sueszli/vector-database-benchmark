"""
This module contains factory functions that attempt
to return Qt submodules from the various python Qt bindings.

It also protects against double-importing Qt with different
bindings, which is unstable and likely to crash

This is used primarily by qt and qt_for_kernel, and shouldn't
be accessed directly from the outside
"""
import importlib.abc
import sys
import os
import types
from functools import partial, lru_cache
import operator
QT_API_PYQT6 = 'pyqt6'
QT_API_PYSIDE6 = 'pyside6'
QT_API_PYQT5 = 'pyqt5'
QT_API_PYSIDE2 = 'pyside2'
QT_API_PYQT = 'pyqt'
QT_API_PYQTv1 = 'pyqtv1'
QT_API_PYSIDE = 'pyside'
QT_API_PYQT_DEFAULT = 'pyqtdefault'
api_to_module = {QT_API_PYQT6: 'PyQt6', QT_API_PYSIDE6: 'PySide6', QT_API_PYQT5: 'PyQt5', QT_API_PYSIDE2: 'PySide2', QT_API_PYSIDE: 'PySide', QT_API_PYQT: 'PyQt4', QT_API_PYQTv1: 'PyQt4', QT_API_PYQT_DEFAULT: 'PyQt6'}

class ImportDenier(importlib.abc.MetaPathFinder):
    """Import Hook that will guard against bad Qt imports
    once IPython commits to a specific binding
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__forbidden = set()

    def forbid(self, module_name):
        if False:
            return 10
        sys.modules.pop(module_name, None)
        self.__forbidden.add(module_name)

    def find_spec(self, fullname, path, target=None):
        if False:
            for i in range(10):
                print('nop')
        if path:
            return
        if fullname in self.__forbidden:
            raise ImportError('\n    Importing %s disabled by IPython, which has\n    already imported an Incompatible QT Binding: %s\n    ' % (fullname, loaded_api()))
ID = ImportDenier()
sys.meta_path.insert(0, ID)

def commit_api(api):
    if False:
        while True:
            i = 10
    'Commit to a particular API, and trigger ImportErrors on subsequent\n    dangerous imports'
    modules = set(api_to_module.values())
    modules.remove(api_to_module[api])
    for mod in modules:
        ID.forbid(mod)

def loaded_api():
    if False:
        while True:
            i = 10
    "Return which API is loaded, if any\n\n    If this returns anything besides None,\n    importing any other Qt binding is unsafe.\n\n    Returns\n    -------\n    None, 'pyside6', 'pyqt6', 'pyside2', 'pyside', 'pyqt', 'pyqt5', 'pyqtv1'\n    "
    if sys.modules.get('PyQt6.QtCore'):
        return QT_API_PYQT6
    elif sys.modules.get('PySide6.QtCore'):
        return QT_API_PYSIDE6
    elif sys.modules.get('PyQt5.QtCore'):
        return QT_API_PYQT5
    elif sys.modules.get('PySide2.QtCore'):
        return QT_API_PYSIDE2
    elif sys.modules.get('PyQt4.QtCore'):
        if qtapi_version() == 2:
            return QT_API_PYQT
        else:
            return QT_API_PYQTv1
    elif sys.modules.get('PySide.QtCore'):
        return QT_API_PYSIDE
    return None

def has_binding(api):
    if False:
        for i in range(10):
            print('nop')
    "Safely check for PyQt4/5, PySide or PySide2, without importing submodules\n\n    Parameters\n    ----------\n    api : str [ 'pyqtv1' | 'pyqt' | 'pyqt5' | 'pyside' | 'pyside2' | 'pyqtdefault']\n        Which module to check for\n\n    Returns\n    -------\n    True if the relevant module appears to be importable\n    "
    module_name = api_to_module[api]
    from importlib.util import find_spec
    required = ['QtCore', 'QtGui', 'QtSvg']
    if api in (QT_API_PYQT5, QT_API_PYSIDE2, QT_API_PYQT6, QT_API_PYSIDE6):
        required.append('QtWidgets')
    for submod in required:
        try:
            spec = find_spec('%s.%s' % (module_name, submod))
        except ImportError:
            return False
        else:
            if spec is None:
                return False
    if api == QT_API_PYSIDE:
        import PySide
        return PySide.__version_info__ >= (1, 0, 3)
    return True

def qtapi_version():
    if False:
        for i in range(10):
            print('nop')
    'Return which QString API has been set, if any\n\n    Returns\n    -------\n    The QString API version (1 or 2), or None if not set\n    '
    try:
        import sip
    except ImportError:
        try:
            from PyQt5 import sip
        except ImportError:
            return
    try:
        return sip.getapi('QString')
    except ValueError:
        return

def can_import(api):
    if False:
        for i in range(10):
            print('nop')
    'Safely query whether an API is importable, without importing it'
    if not has_binding(api):
        return False
    current = loaded_api()
    if api == QT_API_PYQT_DEFAULT:
        return current in [QT_API_PYQT6, None]
    else:
        return current in [api, None]

def import_pyqt4(version=2):
    if False:
        print('Hello World!')
    '\n    Import PyQt4\n\n    Parameters\n    ----------\n    version : 1, 2, or None\n        Which QString/QVariant API to use. Set to None to use the system\n        default\n    ImportErrors raised within this function are non-recoverable\n    '
    import sip
    if version is not None:
        sip.setapi('QString', version)
        sip.setapi('QVariant', version)
    from PyQt4 import QtGui, QtCore, QtSvg
    if QtCore.PYQT_VERSION < 263936:
        raise ImportError('IPython requires PyQt4 >= 4.7, found %s' % QtCore.PYQT_VERSION_STR)
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    version = sip.getapi('QString')
    api = QT_API_PYQTv1 if version == 1 else QT_API_PYQT
    return (QtCore, QtGui, QtSvg, api)

def import_pyqt5():
    if False:
        while True:
            i = 10
    '\n    Import PyQt5\n\n    ImportErrors raised within this function are non-recoverable\n    '
    from PyQt5 import QtCore, QtSvg, QtWidgets, QtGui
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtGuiCompat = types.ModuleType('QtGuiCompat')
    QtGuiCompat.__dict__.update(QtGui.__dict__)
    QtGuiCompat.__dict__.update(QtWidgets.__dict__)
    api = QT_API_PYQT5
    return (QtCore, QtGuiCompat, QtSvg, api)

def import_pyqt6():
    if False:
        for i in range(10):
            print('nop')
    '\n    Import PyQt6\n\n    ImportErrors raised within this function are non-recoverable\n    '
    from PyQt6 import QtCore, QtSvg, QtWidgets, QtGui
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtGuiCompat = types.ModuleType('QtGuiCompat')
    QtGuiCompat.__dict__.update(QtGui.__dict__)
    QtGuiCompat.__dict__.update(QtWidgets.__dict__)
    api = QT_API_PYQT6
    return (QtCore, QtGuiCompat, QtSvg, api)

def import_pyside():
    if False:
        i = 10
        return i + 15
    '\n    Import PySide\n\n    ImportErrors raised within this function are non-recoverable\n    '
    from PySide import QtGui, QtCore, QtSvg
    return (QtCore, QtGui, QtSvg, QT_API_PYSIDE)

def import_pyside2():
    if False:
        for i in range(10):
            print('nop')
    '\n    Import PySide2\n\n    ImportErrors raised within this function are non-recoverable\n    '
    from PySide2 import QtGui, QtCore, QtSvg, QtWidgets, QtPrintSupport
    QtGuiCompat = types.ModuleType('QtGuiCompat')
    QtGuiCompat.__dict__.update(QtGui.__dict__)
    QtGuiCompat.__dict__.update(QtWidgets.__dict__)
    QtGuiCompat.__dict__.update(QtPrintSupport.__dict__)
    return (QtCore, QtGuiCompat, QtSvg, QT_API_PYSIDE2)

def import_pyside6():
    if False:
        for i in range(10):
            print('nop')
    '\n    Import PySide6\n\n    ImportErrors raised within this function are non-recoverable\n    '
    from PySide6 import QtGui, QtCore, QtSvg, QtWidgets, QtPrintSupport
    QtGuiCompat = types.ModuleType('QtGuiCompat')
    QtGuiCompat.__dict__.update(QtGui.__dict__)
    QtGuiCompat.__dict__.update(QtWidgets.__dict__)
    QtGuiCompat.__dict__.update(QtPrintSupport.__dict__)
    return (QtCore, QtGuiCompat, QtSvg, QT_API_PYSIDE6)

def load_qt(api_options):
    if False:
        print('Hello World!')
    "\n    Attempt to import Qt, given a preference list\n    of permissible bindings\n\n    It is safe to call this function multiple times.\n\n    Parameters\n    ----------\n    api_options : List of strings\n        The order of APIs to try. Valid items are 'pyside', 'pyside2',\n        'pyqt', 'pyqt5', 'pyqtv1' and 'pyqtdefault'\n\n    Returns\n    -------\n    A tuple of QtCore, QtGui, QtSvg, QT_API\n    The first three are the Qt modules. The last is the\n    string indicating which module was loaded.\n\n    Raises\n    ------\n    ImportError, if it isn't possible to import any requested\n    bindings (either because they aren't installed, or because\n    an incompatible library has already been installed)\n    "
    loaders = {QT_API_PYQT6: import_pyqt6, QT_API_PYSIDE6: import_pyside6, QT_API_PYQT5: import_pyqt5, QT_API_PYSIDE2: import_pyside2, QT_API_PYSIDE: import_pyside, QT_API_PYQT: import_pyqt4, QT_API_PYQTv1: partial(import_pyqt4, version=1), QT_API_PYQT_DEFAULT: import_pyqt6}
    for api in api_options:
        if api not in loaders:
            raise RuntimeError('Invalid Qt API %r, valid values are: %s' % (api, ', '.join(['%r' % k for k in loaders.keys()])))
        if not can_import(api):
            continue
        result = loaders[api]()
        api = result[-1]
        commit_api(api)
        return result
    else:
        if 'QT_API' in os.environ:
            del os.environ['QT_API']
        raise ImportError('\n    Could not load requested Qt binding. Please ensure that\n    PyQt4 >= 4.7, PyQt5, PyQt6, PySide >= 1.0.3, PySide2, or\n    PySide6 is available, and only one is imported per session.\n\n    Currently-imported Qt library:                              %r\n    PyQt5 available (requires QtCore, QtGui, QtSvg, QtWidgets): %s\n    PyQt6 available (requires QtCore, QtGui, QtSvg, QtWidgets): %s\n    PySide2 installed:                                          %s\n    PySide6 installed:                                          %s\n    Tried to load:                                              %r\n    ' % (loaded_api(), has_binding(QT_API_PYQT5), has_binding(QT_API_PYQT6), has_binding(QT_API_PYSIDE2), has_binding(QT_API_PYSIDE6), api_options))

def enum_factory(QT_API, QtCore):
    if False:
        for i in range(10):
            print('nop')
    'Construct an enum helper to account for PyQt5 <-> PyQt6 changes.'

    @lru_cache(None)
    def _enum(name):
        if False:
            while True:
                i = 10
        return operator.attrgetter(name if QT_API == QT_API_PYQT6 else name.rpartition('.')[0])(sys.modules[QtCore.__package__])
    return _enum