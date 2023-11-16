"""A QProcess which shows notifications in the GUI."""
import dataclasses
import locale
import shlex
import shutil
import signal
from typing import Mapping, Sequence, Dict, Optional
from qutebrowser.qt.core import pyqtSlot, pyqtSignal, QObject, QProcess, QProcessEnvironment, QByteArray, QUrl, Qt
from qutebrowser.utils import message, log, utils, usertypes, version, qtutils
from qutebrowser.api import cmdutils, apitypes
from qutebrowser.completion.models import miscmodels
all_processes: Dict[int, Optional['GUIProcess']] = {}
last_pid: Optional[int] = None

@cmdutils.register()
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('pid', completion=miscmodels.process)
@cmdutils.argument('action', choices=['show', 'terminate', 'kill'])
def process(tab: apitypes.Tab, pid: int=None, action: str='show') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Manage processes spawned by qutebrowser.\n\n    Note that processes with a successful exit get cleaned up after 1h.\n\n    Args:\n        pid: The process ID of the process to manage.\n        action: What to do with the given process:\n\n            - show: Show information about the process.\n            - terminate: Try to gracefully terminate the process (SIGTERM).\n            - kill: Kill the process forcefully (SIGKILL).\n    '
    if pid is None:
        if last_pid is None:
            raise cmdutils.CommandError('No process executed yet!')
        pid = last_pid
    try:
        proc = all_processes[pid]
    except KeyError:
        raise cmdutils.CommandError(f'No process found with pid {pid}')
    if proc is None:
        raise cmdutils.CommandError(f'Data for process {pid} got cleaned up')
    if action == 'show':
        tab.load_url(QUrl(f'qute://process/{pid}'))
    elif action == 'terminate':
        proc.terminate()
    elif action == 'kill':
        proc.terminate(kill=True)
    else:
        raise utils.Unreachable(action)

@dataclasses.dataclass
class ProcessOutcome:
    """The outcome of a finished process."""
    what: str
    running: bool = False
    status: Optional[QProcess.ExitStatus] = None
    code: Optional[int] = None

    def was_successful(self) -> bool:
        if False:
            i = 10
            return i + 15
        "Whether the process exited successfully.\n\n        This must not be called if the process didn't exit yet.\n        "
        assert self.status is not None, "Process didn't finish yet"
        assert self.code is not None
        return self.status == QProcess.ExitStatus.NormalExit and self.code == 0

    def was_sigterm(self) -> bool:
        if False:
            while True:
                i = 10
        "Whether the process was terminated by a SIGTERM.\n\n        This must not be called if the process didn't exit yet.\n        "
        assert self.status is not None, "Process didn't finish yet"
        assert self.code is not None
        return self.status == QProcess.ExitStatus.CrashExit and self.code == signal.SIGTERM

    def _crash_signal(self) -> Optional[signal.Signals]:
        if False:
            print('Hello World!')
        'Get a Python signal (e.g. signal.SIGTERM) from a crashed process.'
        assert self.status == QProcess.ExitStatus.CrashExit
        if self.code is None:
            return None
        try:
            return signal.Signals(self.code)
        except ValueError:
            return None

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        if self.running:
            return f'{self.what.capitalize()} is running.'
        elif self.status is None:
            return f'{self.what.capitalize()} did not start.'
        assert self.status is not None
        assert self.code is not None
        if self.status == QProcess.ExitStatus.CrashExit:
            msg = f'{self.what.capitalize()} {self.state_str()} with status {self.code}'
            sig = self._crash_signal()
            if sig is None:
                return f'{msg}.'
            return f'{msg} ({sig.name}).'
        elif self.was_successful():
            return f'{self.what.capitalize()} exited successfully.'
        assert self.status == QProcess.ExitStatus.NormalExit
        return f'{self.what.capitalize()} exited with status {self.code}.'

    def state_str(self) -> str:
        if False:
            while True:
                i = 10
        'Get a short string describing the state of the process.\n\n        This is used in the :process completion.\n        '
        if self.running:
            return 'running'
        elif self.status is None:
            return 'not started'
        elif self.was_sigterm():
            return 'terminated'
        elif self.status == QProcess.ExitStatus.CrashExit:
            return 'crashed'
        elif self.was_successful():
            return 'successful'
        else:
            return 'unsuccessful'

class GUIProcess(QObject):
    """An external process which shows notifications in the GUI.

    Args:
        cmd: The command which was started.
        args: A list of arguments which gets passed.
        verbose: Whether to show more messages.
        running: Whether the underlying process is started.
        what: What kind of thing is spawned (process/editor/userscript/...).
              Used in messages.
        _output_messages: Show output as messages.
        _proc: The underlying QProcess.

    Signals:
        error/finished/started signals proxied from QProcess.
    """
    error = pyqtSignal(QProcess.ProcessError)
    finished = pyqtSignal(int, QProcess.ExitStatus)
    started = pyqtSignal()

    def __init__(self, what: str, *, verbose: bool=False, additional_env: Mapping[str, str]=None, output_messages: bool=False, parent: QObject=None):
        if False:
            return 10
        super().__init__(parent)
        self.what = what
        self.verbose = verbose
        self._output_messages = output_messages
        self.outcome = ProcessOutcome(what=what)
        self.cmd: Optional[str] = None
        self.resolved_cmd: Optional[str] = None
        self.args: Optional[Sequence[str]] = None
        self.pid: Optional[int] = None
        self.stdout: str = ''
        self.stderr: str = ''
        self._cleanup_timer = usertypes.Timer(self, 'process-cleanup')
        self._cleanup_timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
        self._cleanup_timer.setInterval(3600 * 1000)
        self._cleanup_timer.timeout.connect(self._on_cleanup_timer)
        self._cleanup_timer.setSingleShot(True)
        self._proc = QProcess(self)
        self._proc.errorOccurred.connect(self._on_error)
        self._proc.errorOccurred.connect(self.error)
        self._proc.finished.connect(self._on_finished)
        self._proc.finished.connect(self.finished)
        self._proc.started.connect(self._on_started)
        self._proc.started.connect(self.started)
        self._proc.readyReadStandardOutput.connect(self._on_ready_read_stdout)
        self._proc.readyReadStandardError.connect(self._on_ready_read_stderr)
        if additional_env is not None:
            procenv = QProcessEnvironment.systemEnvironment()
            for (k, v) in additional_env.items():
                procenv.insert(k, v)
            self._proc.setProcessEnvironment(procenv)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        if self.cmd is None or self.args is None:
            return f'<unknown {self.what} command>'
        return ' '.join((shlex.quote(e) for e in [self.cmd] + list(self.args)))

    def _decode_data(self, qba: QByteArray) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Decode data coming from a process.'
        encoding = locale.getpreferredencoding(do_setlocale=False)
        return qba.data().decode(encoding, 'replace')

    def _process_text(self, data: QByteArray, attr: str) -> None:
        if False:
            return 10
        "Process new stdout/stderr text.\n\n        Arguments:\n            data: The new process data.\n            attr: Either 'stdout' or 'stderr'.\n        "
        text = self._decode_data(data)
        if '\r' in text and (not utils.is_windows):
            text = text.rsplit('\r', maxsplit=1)[-1]
            existing = getattr(self, attr)
            if '\n' in existing:
                new = existing.rsplit('\n', maxsplit=1)[0] + '\n'
            else:
                new = ''
            setattr(self, attr, new)
        if attr == 'stdout':
            self.stdout += text
        elif attr == 'stderr':
            self.stderr += text
        else:
            raise utils.Unreachable(attr)

    @pyqtSlot()
    def _on_ready_read_stdout(self) -> None:
        if False:
            return 10
        if not self._output_messages:
            return
        self._process_text(self._proc.readAllStandardOutput(), 'stdout')
        message.info(self._elide_output(self.stdout), replace=f'stdout-{self.pid}')

    @pyqtSlot()
    def _on_ready_read_stderr(self) -> None:
        if False:
            return 10
        if not self._output_messages:
            return
        self._process_text(self._proc.readAllStandardError(), 'stderr')
        message.error(self._elide_output(self.stderr), replace=f'stderr-{self.pid}')

    @pyqtSlot(QProcess.ProcessError)
    def _on_error(self, error: QProcess.ProcessError) -> None:
        if False:
            i = 10
            return i + 15
        'Show a message if there was an error while spawning.'
        if error == QProcess.ProcessError.Crashed and (not utils.is_windows):
            return
        what = f'{self.what} {self.cmd!r}'
        error_descriptions = {QProcess.ProcessError.FailedToStart: f'{what.capitalize()} failed to start', QProcess.ProcessError.Crashed: f'{what.capitalize()} crashed', QProcess.ProcessError.Timedout: f'{what.capitalize()} timed out', QProcess.ProcessError.WriteError: f'Write error for {what}', QProcess.ProcessError.ReadError: f'Read error for {what}'}
        if self.resolved_cmd is None:
            error_string = f"{self.cmd!r} doesn't exist or isn't executable"
            if version.is_flatpak():
                error_string += ' inside the Flatpak container'
        else:
            error_string = self._proc.errorString()
        msg = ': '.join([error_descriptions[error], error_string])
        message.error(msg)

    def _elide_output(self, output: str) -> str:
        if False:
            print('Hello World!')
        'Shorten long output before showing it.'
        output = output.strip()
        lines = output.splitlines()
        count = len(lines)
        threshold = 20
        if count > threshold:
            lines = [f'[{count - threshold} lines hidden, see :process for the full output]'] + lines[-threshold:]
            output = '\n'.join(lines)
        return output

    @pyqtSlot(int, QProcess.ExitStatus)
    def _on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if False:
            print('Hello World!')
        'Show a message when the process finished.'
        log.procs.debug('Process finished with code {}, status {}.'.format(code, status))
        self.outcome.running = False
        self.outcome.code = code
        self.outcome.status = status
        self.stderr += self._decode_data(self._proc.readAllStandardError())
        self.stdout += self._decode_data(self._proc.readAllStandardOutput())
        if self._output_messages:
            if self.stdout:
                message.info(self._elide_output(self.stdout), replace=f'stdout-{self.pid}')
            if self.stderr:
                message.error(self._elide_output(self.stderr), replace=f'stderr-{self.pid}')
        msg = f'{self.outcome} See :process {self.pid} for details.'
        if self.outcome.was_successful() or self.outcome.was_sigterm():
            if self.verbose:
                message.info(msg)
            self._cleanup_timer.start()
        else:
            if self.stdout:
                log.procs.error('Process stdout:\n' + self.stdout.strip())
            if self.stderr:
                log.procs.error('Process stderr:\n' + self.stderr.strip())
            message.error(msg)

    @pyqtSlot()
    def _on_started(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Called when the process started successfully.'
        log.procs.debug('Process started.')
        assert not self.outcome.running
        self.outcome.running = True

    def _pre_start(self, cmd: str, args: Sequence[str]) -> None:
        if False:
            while True:
                i = 10
        'Resolve the given command and prepare starting of a QProcess.\n\n        Doing the resolving in Python here instead of letting Qt do it serves\n        two purposes:\n\n        - Being able to show a nicer error message without having to parse the\n          string we get from Qt: https://bugreports.qt.io/browse/QTBUG-44769\n        - Not running the file from the current directory on Unix with\n          Qt < 5.15.? and 6.2.4, as a WORKAROUND for CVE-2022-25255:\n          https://invent.kde.org/qt/qt/qtbase/-/merge_requests/139\n          https://www.qt.io/blog/security-advisory-qprocess\n          https://lists.qt-project.org/pipermail/announce/2022-February/000333.html\n        '
        if self.outcome.running:
            raise ValueError('Trying to start a running QProcess!')
        self.cmd = cmd
        self.resolved_cmd = shutil.which(cmd)
        self.args = args
        log.procs.debug(f'Executing: {self}')
        if self.verbose:
            message.info(f'Executing: {self}')

    def start(self, cmd: str, args: Sequence[str]) -> None:
        if False:
            return 10
        'Convenience wrapper around QProcess::start.'
        log.procs.debug('Starting process.')
        self._pre_start(cmd, args)
        self._proc.start(qtutils.remove_optional(self.resolved_cmd), args)
        self._post_start()
        self._proc.closeWriteChannel()

    def start_detached(self, cmd: str, args: Sequence[str]) -> bool:
        if False:
            while True:
                i = 10
        'Convenience wrapper around QProcess::startDetached.'
        log.procs.debug('Starting detached.')
        self._pre_start(cmd, args)
        (ok, self.pid) = self._proc.startDetached(self.resolved_cmd, args, None)
        if not ok:
            message.error('Error while spawning {}'.format(self.what))
            return False
        log.procs.debug('Process started.')
        self.outcome.running = True
        self._post_start()
        return True

    def _post_start(self) -> None:
        if False:
            i = 10
            return i + 15
        'Register this process and remember the process ID after starting.'
        self.pid = self._proc.processId()
        all_processes[self.pid] = self
        global last_pid
        last_pid = self.pid

    @pyqtSlot()
    def _on_cleanup_timer(self) -> None:
        if False:
            while True:
                i = 10
        'Remove the process from all registered processes.'
        log.procs.debug(f'Cleaning up data for {self.pid}')
        assert self.pid in all_processes
        all_processes[self.pid] = None
        self.deleteLater()

    def terminate(self, kill: bool=False) -> None:
        if False:
            print('Hello World!')
        'Terminate or kill the process.'
        if kill:
            self._proc.kill()
        else:
            self._proc.terminate()