"""Handlers for crashes and OS signals."""
import os
import os.path
import sys
import bdb
import pdb
import signal
import argparse
import functools
import threading
import faulthandler
import dataclasses
from typing import TYPE_CHECKING, Optional, MutableMapping, cast, List
from qutebrowser.qt.core import pyqtSlot, qInstallMessageHandler, QObject, QSocketNotifier, QTimer, QUrl
from qutebrowser.qt.widgets import QApplication
from qutebrowser.api import cmdutils
from qutebrowser.misc import earlyinit, crashdialog, ipc, objects
from qutebrowser.utils import usertypes, standarddir, log, objreg, debug, utils
from qutebrowser.qt import sip
if TYPE_CHECKING:
    from qutebrowser.misc import quitter

@dataclasses.dataclass
class ExceptionInfo:
    """Information stored when there was an exception."""
    pages: List[List[str]]
    cmd_history: List[str]
    objects: str
crash_handler = cast('CrashHandler', None)

class CrashHandler(QObject):
    """Handler for crashes, reports and exceptions.

    Attributes:
        _app: The QApplication instance.
        _quitter: The Quitter instance.
        _args: The argparse namespace.
        _crash_dialog: The CrashDialog currently being shown.
        _crash_log_file: The file handle for the faulthandler crash log.
        _crash_log_data: Crash data read from the previous crash log.
        is_crashing: Used by mainwindow.py to skip confirm questions on
                     crashes.
    """

    def __init__(self, *, app, quitter, args, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._app = app
        self._quitter = quitter
        self._args = args
        self._crash_log_file = None
        self._crash_log_data = None
        self._crash_dialog = None
        self.is_crashing = False

    def activate(self):
        if False:
            return 10
        'Activate the exception hook.'
        sys.excepthook = self.exception_hook

    def init_faulthandler(self):
        if False:
            while True:
                i = 10
        'Handle a segfault from a previous run and set up faulthandler.'
        logname = os.path.join(standarddir.data(), 'crash.log')
        try:
            if os.path.exists(logname):
                with open(logname, 'r', encoding='ascii') as f:
                    self._crash_log_data = f.read()
                os.remove(logname)
                self._init_crashlogfile()
            else:
                self._init_crashlogfile()
        except (OSError, UnicodeDecodeError):
            log.init.exception('Error while handling crash log file!')
            self._init_crashlogfile()

    def display_faulthandler(self):
        if False:
            for i in range(10):
                print('nop')
        'If there was data in the crash log file, display a dialog.'
        assert not self._args.no_err_windows
        if self._crash_log_data:
            self._crash_dialog = crashdialog.FatalCrashDialog(self._args.debug, self._crash_log_data)
            self._crash_dialog.show()
        self._crash_log_data = None

    def _recover_pages(self, forgiving=False):
        if False:
            i = 10
            return i + 15
        'Try to recover all open pages.\n\n        Called from exception_hook, so as forgiving as possible.\n\n        Args:\n            forgiving: Whether to ignore exceptions.\n\n        Return:\n            A list containing a list for each window, which in turn contain the\n            opened URLs.\n        '
        pages = []
        for win_id in objreg.window_registry:
            win_pages = []
            tabbed_browser = objreg.get('tabbed-browser', scope='window', window=win_id)
            for tab in tabbed_browser.widgets():
                try:
                    urlstr = tab.url().toString(QUrl.UrlFormattingOption.RemovePassword | QUrl.ComponentFormattingOption.FullyEncoded)
                    if urlstr:
                        win_pages.append(urlstr)
                except Exception:
                    if forgiving:
                        log.destroy.exception('Error while recovering tab')
                    else:
                        raise
            pages.append(win_pages)
        return pages

    def _init_crashlogfile(self):
        if False:
            i = 10
            return i + 15
        'Start a new logfile and redirect faulthandler to it.'
        logname = os.path.join(standarddir.data(), 'crash.log')
        try:
            self._crash_log_file = open(logname, 'w', encoding='ascii')
        except OSError:
            log.init.exception('Error while opening crash log file!')
        else:
            earlyinit.init_faulthandler(self._crash_log_file)

    @cmdutils.register(instance='crash-handler')
    def report(self, info=None, contact=None):
        if False:
            while True:
                i = 10
        'Report a bug in qutebrowser.\n\n        Args:\n            info: Information about the bug report. If given, no report dialog\n                  shows up.\n            contact: Contact information for the report.\n        '
        pages = self._recover_pages()
        cmd_history = objreg.get('command-history')[-5:]
        all_objects = debug.get_all_objects()
        self._crash_dialog = crashdialog.ReportDialog(pages, cmd_history, all_objects)
        if info is None:
            self._crash_dialog.show()
        else:
            self._crash_dialog.report(info=info, contact=contact)

    @pyqtSlot()
    def shutdown(self):
        if False:
            print('Hello World!')
        self.destroy_crashlogfile()

    def destroy_crashlogfile(self):
        if False:
            while True:
                i = 10
        'Clean up the crash log file and delete it.'
        if self._crash_log_file is None:
            return
        if sys.__stderr__ is not None:
            faulthandler.enable(sys.__stderr__)
        else:
            faulthandler.disable()
        try:
            self._crash_log_file.close()
            os.remove(self._crash_log_file.name)
        except OSError:
            log.destroy.exception('Could not remove crash log!')

    def _get_exception_info(self):
        if False:
            print('Hello World!')
        'Get info needed for the exception hook/dialog.\n\n        Return:\n            An ExceptionInfo object.\n        '
        try:
            pages = self._recover_pages(forgiving=True)
        except Exception as e:
            log.destroy.exception('Error while recovering pages: {}'.format(e))
            pages = []
        try:
            cmd_history = objreg.get('command-history')[-5:]
        except Exception as e:
            log.destroy.exception('Error while getting history: {}'.format(e))
            cmd_history = []
        try:
            all_objects = debug.get_all_objects()
        except Exception:
            log.destroy.exception('Error while getting objects')
            all_objects = ''
        return ExceptionInfo(pages, cmd_history, all_objects)

    def _handle_early_exits(self, exc):
        if False:
            print('Hello World!')
        'Handle some special cases for the exception hook.\n\n        Return value:\n            True: Exception hook should be aborted.\n            False: Continue handling exception.\n        '
        (exctype, _excvalue, tb) = exc
        if not self._quitter.quit_status['crash']:
            log.misc.error('ARGH, there was an exception while the crash dialog is already shown:', exc_info=exc)
            return True
        log.misc.error('Uncaught exception', exc_info=exc)
        is_ignored_exception = exctype is bdb.BdbQuit or not issubclass(exctype, Exception)
        if 'pdb-postmortem' in objects.debug_flags:
            if tb is None:
                pdb.set_trace()
            else:
                pdb.post_mortem(tb)
        if is_ignored_exception or 'pdb-postmortem' in objects.debug_flags:
            sys.exit(usertypes.Exit.exception)
        if threading.current_thread() != threading.main_thread():
            log.misc.error('Ignoring exception outside of main thread... Please report this as a bug.')
            return True
        return False

    def exception_hook(self, exctype, excvalue, tb):
        if False:
            print('Hello World!')
        "Handle uncaught python exceptions.\n\n        It'll try very hard to write all open tabs to a file, and then exit\n        gracefully.\n        "
        exc = (exctype, excvalue, tb)
        if self._handle_early_exits(exc):
            return
        self._quitter.quit_status['crash'] = False
        info = self._get_exception_info()
        if ipc.server is not None:
            try:
                ipc.server.ignored = True
            except Exception:
                log.destroy.exception('Error while ignoring ipc')
        try:
            self._app.lastWindowClosed.disconnect(self._quitter.on_last_window_closed)
        except TypeError:
            log.destroy.exception('Error while preventing shutdown')
        self.is_crashing = True
        self._app.closeAllWindows()
        if self._args.no_err_windows:
            crashdialog.dump_exception_info(exc, info.pages, info.cmd_history, info.objects)
        else:
            self._crash_dialog = crashdialog.ExceptionCrashDialog(self._args.debug, info.pages, info.cmd_history, exc, info.objects)
            ret = self._crash_dialog.exec()
            if ret == crashdialog.Result.restore:
                self._quitter.restart(info.pages)
        qInstallMessageHandler(None)
        self.destroy_crashlogfile()
        sys.exit(usertypes.Exit.exception)

    def raise_crashdlg(self):
        if False:
            while True:
                i = 10
        'Raise the crash dialog if one exists.'
        if self._crash_dialog is not None:
            self._crash_dialog.raise_()

class SignalHandler(QObject):
    """Handler responsible for handling OS signals (SIGINT, SIGTERM, etc.).

    Attributes:
        _app: The QApplication instance.
        _quitter: The Quitter instance.
        _activated: Whether activate() was called.
        _notifier: A QSocketNotifier used for signals on Unix.
        _timer: A QTimer used to poll for signals on Windows.
        _orig_handlers: A {signal: handler} dict of original signal handlers.
        _orig_wakeup_fd: The original wakeup filedescriptor.
    """

    def __init__(self, *, app, quitter, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self._app = app
        self._quitter = quitter
        self._notifier = None
        self._timer = usertypes.Timer(self, 'python_hacks')
        self._orig_handlers: MutableMapping[int, 'signal._HANDLER'] = {}
        self._activated = False
        self._orig_wakeup_fd: Optional[int] = None

    def activate(self):
        if False:
            print('Hello World!')
        'Set up signal handlers.\n\n        On Windows this uses a QTimer to periodically hand control over to\n        Python so it can handle signals.\n\n        On Unix, it uses a QSocketNotifier with os.set_wakeup_fd to get\n        notified.\n        '
        self._orig_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self.interrupt)
        self._orig_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self.interrupt)
        if utils.is_posix and hasattr(signal, 'set_wakeup_fd'):
            import fcntl
            (read_fd, write_fd) = os.pipe()
            for fd in [read_fd, write_fd]:
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self._notifier = QSocketNotifier(cast(sip.voidptr, read_fd), QSocketNotifier.Type.Read, self)
            self._notifier.activated.connect(self.handle_signal_wakeup)
            self._orig_wakeup_fd = signal.set_wakeup_fd(write_fd)
        else:
            self._timer.start(1000)
            self._timer.timeout.connect(lambda : None)
        self._activated = True

    def deactivate(self):
        if False:
            for i in range(10):
                print('nop')
        'Deactivate all signal handlers.'
        if not self._activated:
            return
        if self._notifier is not None:
            assert self._orig_wakeup_fd is not None
            self._notifier.setEnabled(False)
            rfd = self._notifier.socket()
            wfd = signal.set_wakeup_fd(self._orig_wakeup_fd)
            os.close(int(rfd))
            os.close(wfd)
        for (sig, handler) in self._orig_handlers.items():
            signal.signal(sig, handler)
        self._timer.stop()
        self._activated = False

    @pyqtSlot()
    def handle_signal_wakeup(self):
        if False:
            for i in range(10):
                print('nop')
        "Handle a newly arrived signal.\n\n        This gets called via self._notifier when there's a signal.\n\n        Python will get control here, so the signal will get handled.\n        "
        assert self._notifier is not None
        log.destroy.debug('Handling signal wakeup!')
        self._notifier.setEnabled(False)
        read_fd = self._notifier.socket()
        try:
            os.read(int(read_fd), 1)
        except OSError:
            log.destroy.exception('Failed to read wakeup fd.')
        self._notifier.setEnabled(True)

    def _log_later(self, *lines):
        if False:
            return 10
        'Log the given text line-wise with a QTimer.'
        for line in lines:
            QTimer.singleShot(0, functools.partial(log.destroy.info, line))

    def interrupt(self, signum, _frame):
        if False:
            return 10
        'Handler for signals to gracefully shutdown (SIGINT/SIGTERM).\n\n        This calls shutdown and remaps the signal to call\n        interrupt_forcefully the next time.\n        '
        signal.signal(signal.SIGINT, self.interrupt_forcefully)
        signal.signal(signal.SIGTERM, self.interrupt_forcefully)
        self._log_later('SIGINT/SIGTERM received, shutting down!', 'Do the same again to forcefully quit.')
        QTimer.singleShot(0, functools.partial(self._quitter.shutdown, 128 + signum))

    def interrupt_forcefully(self, signum, _frame):
        if False:
            while True:
                i = 10
        'Interrupt forcefully on the second SIGINT/SIGTERM request.\n\n        This skips our shutdown routine and calls QApplication:exit instead.\n        It then remaps the signals to call self.interrupt_really_forcefully the\n        next time.\n        '
        signal.signal(signal.SIGINT, self.interrupt_really_forcefully)
        signal.signal(signal.SIGTERM, self.interrupt_really_forcefully)
        self._log_later('Forceful quit requested, goodbye cruel world!', 'Do the same again to quit with even more force.')
        QTimer.singleShot(0, functools.partial(self._app.exit, 128 + signum))

    def interrupt_really_forcefully(self, signum, _frame):
        if False:
            print('Hello World!')
        "Interrupt with even more force on the third SIGINT/SIGTERM request.\n\n        This doesn't run *any* Qt cleanup and simply exits via Python.\n        It will most likely lead to a segfault.\n        "
        print('WHY ARE YOU DOING THIS TO ME? :(')
        sys.exit(128 + signum)

def init(q_app: QApplication, args: argparse.Namespace, quitter: 'quitter.Quitter') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Initialize crash/signal handlers.'
    global crash_handler
    crash_handler = CrashHandler(app=q_app, quitter=quitter, args=args, parent=q_app)
    objreg.register('crash-handler', crash_handler, command_only=True)
    crash_handler.activate()
    quitter.shutting_down.connect(crash_handler.shutdown)
    signal_handler = SignalHandler(app=q_app, quitter=quitter, parent=q_app)
    signal_handler.activate()
    quitter.shutting_down.connect(signal_handler.deactivate)