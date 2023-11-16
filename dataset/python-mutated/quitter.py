"""Helpers related to quitting qutebrowser cleanly."""
import os
import os.path
import sys
import json
import atexit
import shutil
import argparse
import tokenize
import functools
import warnings
import subprocess
from typing import Iterable, Mapping, MutableSequence, Sequence, cast
from qutebrowser.qt.core import QObject, pyqtSignal, QTimer
try:
    import hunter
except ImportError:
    hunter = None
import qutebrowser
from qutebrowser.api import cmdutils
from qutebrowser.utils import log, qtlog
from qutebrowser.misc import sessions, ipc, objects
from qutebrowser.mainwindow import prompt
from qutebrowser.completion.models import miscmodels
instance = cast('Quitter', None)

class Quitter(QObject):
    """Utility class to quit/restart the QApplication.

    Attributes:
        quit_status: The current quitting status.
        is_shutting_down: Whether we're currently shutting down.
        _args: The argparse namespace.
    """
    shutting_down = pyqtSignal()

    def __init__(self, *, args: argparse.Namespace, parent: QObject=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.quit_status = {'crash': True, 'tabs': False, 'main': False}
        self.is_shutting_down = False
        self._args = args

    def on_last_window_closed(self) -> None:
        if False:
            i = 10
            return i + 15
        'Slot which gets invoked when the last window was closed.'
        self.shutdown(last_window=True)

    def _compile_modules(self) -> None:
        if False:
            while True:
                i = 10
        'Compile all modules to catch SyntaxErrors.'
        if os.path.basename(sys.argv[0]) == 'qutebrowser':
            return
        elif hasattr(sys, 'frozen'):
            return
        else:
            path = os.path.abspath(os.path.dirname(qutebrowser.__file__))
            if not os.path.isdir(path):
                return
        for (dirpath, _dirnames, filenames) in os.walk(path):
            for fn in filenames:
                if os.path.splitext(fn)[1] == '.py' and os.path.isfile(fn):
                    with tokenize.open(os.path.join(dirpath, fn)) as f:
                        compile(f.read(), fn, 'exec')

    def _get_restart_args(self, pages: Iterable[str]=(), session: str=None, override_args: Mapping[str, str]=None) -> Sequence[str]:
        if False:
            return 10
        'Get args to relaunch qutebrowser.\n\n        Args:\n            pages: The pages to re-open.\n            session: The session to load, or None.\n            override_args: Argument overrides as a dict.\n\n        Return:\n            The commandline as a list of strings.\n        '
        if os.path.basename(sys.argv[0]) == 'qutebrowser':
            args = [sys.argv[0]]
        elif hasattr(sys, 'frozen'):
            args = [sys.executable]
        else:
            args = [sys.executable, '-m', 'qutebrowser']
        page_args: MutableSequence[str] = []
        for win in pages:
            page_args.extend(win)
            page_args.append('')
        argdict = vars(self._args)
        argdict['session'] = None
        argdict['url'] = []
        argdict['command'] = page_args[:-1]
        argdict['json_args'] = None
        if session is None:
            argdict['session'] = None
            argdict['override_restore'] = True
        else:
            argdict['session'] = session
            argdict['override_restore'] = False
        if self._args.temp_basedir:
            argdict['temp_basedir'] = False
            argdict['temp_basedir_restarted'] = True
        if override_args is not None:
            argdict.update(override_args)
        data = json.dumps(argdict)
        args += ['--json-args', data]
        log.destroy.debug('args: {}'.format(args))
        return args

    def restart(self, pages: Sequence[str]=(), session: str=None, override_args: Mapping[str, str]=None) -> bool:
        if False:
            print('Hello World!')
        'Inner logic to restart qutebrowser.\n\n        The "better" way to restart is to pass a session (_restart usually) as\n        that\'ll save the complete state.\n\n        However we don\'t do that (and pass a list of pages instead) when we\n        restart because of an exception, as that\'s a lot simpler and we don\'t\n        want to risk anything going wrong.\n\n        Args:\n            pages: A list of URLs to open.\n            session: The session to load, or None.\n            override_args: Argument overrides as a dict.\n\n        Return:\n            True if the restart succeeded, False otherwise.\n        '
        self._compile_modules()
        log.destroy.debug('sys.executable: {}'.format(sys.executable))
        log.destroy.debug('sys.path: {}'.format(sys.path))
        log.destroy.debug('sys.argv: {}'.format(sys.argv))
        log.destroy.debug('frozen: {}'.format(hasattr(sys, 'frozen')))
        if session is not None:
            sessions.session_manager.save(session, with_private=True)
        assert ipc.server is not None
        ipc.server.shutdown()
        try:
            args = self._get_restart_args(pages, session, override_args)
            proc = subprocess.Popen(args)
        except OSError:
            log.destroy.exception('Failed to restart')
            return False
        else:
            log.destroy.debug(f'New process PID: {proc.pid}')
            warnings.filterwarnings('ignore', category=ResourceWarning, message=f'subprocess {proc.pid} is still running')
            return True

    def shutdown(self, status: int=0, session: sessions.ArgType=None, last_window: bool=False, is_restart: bool=False) -> None:
        if False:
            return 10
        "Quit qutebrowser.\n\n        Args:\n            status: The status code to exit with.\n            session: A session name if saving should be forced.\n            last_window: If the shutdown was triggered due to the last window\n                            closing.\n            is_restart: If we're planning to restart.\n        "
        if self.is_shutting_down:
            return
        self.is_shutting_down = True
        log.destroy.debug('Shutting down with status {}, session {}...'.format(status, session))
        sessions.shutdown(session, last_window=last_window)
        if prompt.prompt_queue is not None:
            prompt.prompt_queue.shutdown()
        log.destroy.debug('Deferring shutdown stage 2')
        QTimer.singleShot(0, functools.partial(self._shutdown_2, status, is_restart=is_restart))

    def _shutdown_2(self, status: int, is_restart: bool) -> None:
        if False:
            print('Hello World!')
        'Second stage of shutdown.'
        log.destroy.debug('Stage 2 of shutting down...')
        self.shutting_down.emit()
        if (self._args.temp_basedir or self._args.temp_basedir_restarted) and (not is_restart):
            atexit.register(shutil.rmtree, self._args.basedir, ignore_errors=True)
        log.destroy.debug('Deferring QApplication::exit...')
        QTimer.singleShot(0, functools.partial(self._shutdown_3, status))

    def _shutdown_3(self, status: int) -> None:
        if False:
            i = 10
            return i + 15
        'Finally shut down the QApplication.'
        log.destroy.debug('Now calling QApplication::exit.')
        if 'debug-exit' in objects.debug_flags:
            if hunter is None:
                print('Not logging late shutdown because hunter could not be imported!', file=sys.stderr)
            else:
                print('Now logging late shutdown.', file=sys.stderr)
                hunter.trace()
        objects.qapp.exit(status)

@cmdutils.register(name='quit')
@cmdutils.argument('session', completion=miscmodels.session)
def quit_(save: bool=False, session: sessions.ArgType=None) -> None:
    if False:
        print('Hello World!')
    'Quit qutebrowser.\n\n    Args:\n        save: When given, save the open windows even if auto_save.session\n                is turned off.\n        session: The name of the session to save.\n    '
    if session is not None and (not save):
        raise cmdutils.CommandError('Session name given without --save!')
    if save and session is None:
        session = sessions.default
    instance.shutdown(session=session)

@cmdutils.register()
def restart() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Restart qutebrowser while keeping existing tabs open.'
    try:
        ok = instance.restart(session='_restart')
    except sessions.SessionError as e:
        log.destroy.exception('Failed to save session!')
        raise cmdutils.CommandError('Failed to save session: {}!'.format(e))
    except SyntaxError as e:
        log.destroy.exception('Got SyntaxError')
        raise cmdutils.CommandError('SyntaxError in {}:{}: {}'.format(e.filename, e.lineno, e))
    if ok:
        instance.shutdown(is_restart=True)

def init(args: argparse.Namespace) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Initialize the global Quitter instance.'
    global instance
    instance = Quitter(args=args, parent=objects.qapp)
    instance.shutting_down.connect(qtlog.shutdown_log)
    objects.qapp.lastWindowClosed.connect(instance.on_last_window_closed)