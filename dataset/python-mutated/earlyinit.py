"""Things which need to be done really early (e.g. before importing Qt).

At this point we can be sure we have all python 3.8 features available.
"""
try:
    import hunter
except ImportError:
    hunter = None
import sys
import faulthandler
import traceback
import signal
import importlib
import datetime
from typing import NoReturn
try:
    import tkinter
except ImportError:
    tkinter = None
from qutebrowser.qt import machinery
START_TIME = datetime.datetime.now()

def _missing_str(name, *, webengine=False):
    if False:
        i = 10
        return i + 15
    'Get an error string for missing packages.\n\n    Args:\n        name: The name of the package.\n        webengine: Whether this is checking the QtWebEngine package\n    '
    blocks = ["Fatal error: <b>{}</b> is required to run qutebrowser but could not be imported! Maybe it's not installed?".format(name), '<b>The error encountered was:</b><br />%ERROR%']
    lines = ['Please search for the python3 version of {} in your distributions packages, or see https://github.com/qutebrowser/qutebrowser/blob/main/doc/install.asciidoc'.format(name)]
    blocks.append('<br />'.join(lines))
    if not webengine:
        lines = ['<b>If you installed a qutebrowser package for your distribution, please report this as a bug.</b>']
        blocks.append('<br />'.join(lines))
    return '<br /><br />'.join(blocks)

def _die(message, exception=None):
    if False:
        i = 10
        return i + 15
    "Display an error message using Qt and quit.\n\n    We import the imports here as we want to do other stuff before the imports.\n\n    Args:\n        message: The message to display.\n        exception: The exception object if we're handling an exception.\n    "
    from qutebrowser.qt.widgets import QApplication, QMessageBox
    from qutebrowser.qt.core import Qt
    if ('--debug' in sys.argv or '--no-err-windows' in sys.argv) and exception is not None:
        print(file=sys.stderr)
        traceback.print_exc()
    app = QApplication(sys.argv)
    if '--no-err-windows' in sys.argv:
        print(message, file=sys.stderr)
        print('Exiting because of --no-err-windows.', file=sys.stderr)
    else:
        if exception is not None:
            message = message.replace('%ERROR%', str(exception))
        msgbox = QMessageBox(QMessageBox.Icon.Critical, 'qutebrowser: Fatal error!', message)
        msgbox.setTextFormat(Qt.TextFormat.RichText)
        msgbox.resize(msgbox.sizeHint())
        msgbox.exec()
    app.quit()
    sys.exit(1)

def init_faulthandler(fileobj=sys.__stderr__):
    if False:
        i = 10
        return i + 15
    'Enable faulthandler module if available.\n\n    This print a nice traceback on segfaults.\n\n    We use sys.__stderr__ instead of sys.stderr here so this will still work\n    when sys.stderr got replaced, e.g. by "Python Tools for Visual Studio".\n\n    Args:\n        fileobj: An opened file object to write the traceback to.\n    '
    try:
        faulthandler.enable(fileobj)
    except (RuntimeError, AttributeError):
        return
    if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1') and (sys.stderr is not None):
        faulthandler.register(signal.SIGUSR1)

def _fatal_qt_error(text: str) -> NoReturn:
    if False:
        print('Hello World!')
    'Show a fatal error about Qt being missing.'
    if tkinter and '--no-err-windows' not in sys.argv:
        root = tkinter.Tk()
        root.withdraw()
        tkinter.messagebox.showerror('qutebrowser: Fatal error!', text)
    else:
        print(text, file=sys.stderr)
    if '--debug' in sys.argv or '--no-err-windows' in sys.argv:
        print(file=sys.stderr)
        traceback.print_exc()
    sys.exit(1)

def check_qt_available(info: machinery.SelectionInfo) -> None:
    if False:
        i = 10
        return i + 15
    'Check if Qt core modules (QtCore/QtWidgets) are installed.'
    if info.wrapper is None:
        _fatal_qt_error(f'No Qt wrapper was importable.\n\n{info}')
    packages = [f'{info.wrapper}.QtCore', f'{info.wrapper}.QtWidgets']
    for name in packages:
        try:
            importlib.import_module(name)
        except ImportError as e:
            text = _missing_str(name)
            text = text.replace('<b>', '')
            text = text.replace('</b>', '')
            text = text.replace('<br />', '\n')
            text = text.replace('%ERROR%', str(e))
            text += '\n\n' + str(info)
            _fatal_qt_error(text)

def qt_version(qversion=None, qt_version_str=None):
    if False:
        for i in range(10):
            print('nop')
    'Get a Qt version string based on the runtime/compiled versions.'
    if qversion is None:
        from qutebrowser.qt.core import qVersion
        qversion = qVersion()
    if qt_version_str is None:
        from qutebrowser.qt.core import QT_VERSION_STR
        qt_version_str = QT_VERSION_STR
    if qversion != qt_version_str:
        return '{} (compiled {})'.format(qversion, qt_version_str)
    else:
        return qversion

def get_qt_version():
    if False:
        i = 10
        return i + 15
    'Get the Qt version, or None if too old for QLibaryInfo.version().'
    try:
        from qutebrowser.qt.core import QLibraryInfo
        return QLibraryInfo.version().normalized()
    except (ImportError, AttributeError):
        return None

def check_qt_version():
    if False:
        for i in range(10):
            print('nop')
    'Check if the Qt version is recent enough.'
    from qutebrowser.qt.core import QT_VERSION, PYQT_VERSION, PYQT_VERSION_STR
    from qutebrowser.qt.core import QVersionNumber
    qt_ver = get_qt_version()
    recent_qt_runtime = qt_ver is not None and qt_ver >= QVersionNumber(5, 15)
    if QT_VERSION < 331520 or PYQT_VERSION < 331520 or (not recent_qt_runtime):
        text = 'Fatal error: Qt >= 5.15.0 and PyQt >= 5.15.0 are required, but Qt {} / PyQt {} is installed.'.format(qt_version(), PYQT_VERSION_STR)
        _die(text)
    if 393216 <= PYQT_VERSION < 393730:
        text = 'Fatal error: With Qt 6, PyQt >= 6.2.2 is required, but {} is installed.'.format(PYQT_VERSION_STR)
        _die(text)

def check_ssl_support():
    if False:
        i = 10
        return i + 15
    'Check if SSL support is available.'
    try:
        from qutebrowser.qt.network import QSslSocket
    except ImportError:
        _die('Fatal error: Your Qt is built without SSL support.')

def _check_modules(modules):
    if False:
        while True:
            i = 10
    'Make sure the given modules are available.'
    from qutebrowser.utils import log
    for (name, text) in modules.items():
        try:
            with log.py_warning_filter(category=DeprecationWarning, message='invalid escape sequence'), log.py_warning_filter(category=ImportWarning, message='Not importing directory .*: missing __init__'), log.py_warning_filter(category=DeprecationWarning, message='the imp module is deprecated'), log.py_warning_filter(category=DeprecationWarning, message='Creating a LegacyVersion has been deprecated'):
                importlib.import_module(name)
        except ImportError as e:
            _die(text, e)

def check_libraries():
    if False:
        i = 10
        return i + 15
    'Check if all needed Python libraries are installed.'
    modules = {'jinja2': _missing_str('jinja2'), 'yaml': _missing_str('PyYAML')}
    for subpkg in ['QtQml', 'QtOpenGL', 'QtDBus']:
        package = f'{machinery.INFO.wrapper}.{subpkg}'
        modules[package] = _missing_str(package)
    if sys.version_info < (3, 9):
        modules['importlib_resources'] = _missing_str('importlib_resources')
    if sys.platform.startswith('darwin'):
        from qutebrowser.qt.core import QVersionNumber
        qt_ver = get_qt_version()
        if qt_ver is not None and qt_ver < QVersionNumber(6, 3):
            modules['objc'] = _missing_str('pyobjc-core')
            modules['AppKit'] = _missing_str('pyobjc-framework-Cocoa')
    _check_modules(modules)

def configure_pyqt():
    if False:
        print('Hello World!')
    "Remove the PyQt input hook and enable overflow checking.\n\n    Doing this means we can't use the interactive shell anymore (which we don't\n    anyways), but we can use pdb instead.\n    "
    from qutebrowser.qt.core import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    from qutebrowser.qt import sip
    if machinery.IS_QT5:
        sip.enableoverflowchecking(True)

def init_log(args):
    if False:
        while True:
            i = 10
    'Initialize logging.\n\n    Args:\n        args: The argparse namespace.\n    '
    from qutebrowser.utils import log
    log.init_log(args)
    log.init.debug('Log initialized.')

def init_qtlog(args):
    if False:
        print('Hello World!')
    'Initialize Qt logging.\n\n    Args:\n        args: The argparse namespace.\n    '
    from qutebrowser.utils import log, qtlog
    qtlog.init(args)
    log.init.debug('Qt log initialized.')

def check_optimize_flag():
    if False:
        i = 10
        return i + 15
    'Check whether qutebrowser is running with -OO.'
    from qutebrowser.utils import log
    if sys.flags.optimize >= 2:
        log.init.warning('Running on optimize level higher than 1, unexpected behavior may occur.')

def webengine_early_import():
    if False:
        i = 10
        return i + 15
    'If QtWebEngine is available, import it early.\n\n    We need to ensure that QtWebEngine is imported before a QApplication is created for\n    everything to work properly.\n\n    This needs to be done even when using the QtWebKit backend, to ensure that e.g.\n    error messages in backendproblem.py are accurate.\n    '
    try:
        from qutebrowser.qt import webenginewidgets
    except ImportError:
        pass

def early_init(args):
    if False:
        print('Hello World!')
    "Do all needed early initialization.\n\n    Note that it's vital the other earlyinit functions get called in the right\n    order!\n\n    Args:\n        args: The argparse namespace.\n    "
    init_log(args)
    init_faulthandler()
    info = machinery.init(args)
    init_qtlog(args)
    check_qt_available(info)
    check_libraries()
    check_qt_version()
    configure_pyqt()
    check_ssl_support()
    check_optimize_flag()
    webengine_early_import()