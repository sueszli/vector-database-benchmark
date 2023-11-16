import sys
import json
import textwrap
import os
import signal
import time
import pytest
import pytest_bdd as bdd
from qutebrowser.qt.core import pyqtSignal, pyqtSlot, QObject, QFileSystemWatcher
bdd.scenarios('editor.feature')
from qutebrowser.utils import utils

@bdd.when(bdd.parsers.parse('I setup a fake editor replacing "{text}" by "{replacement}"'))
def set_up_editor_replacement(quteproc, server, tmpdir, text, replacement):
    if False:
        for i in range(10):
            print('nop')
    'Set up editor.command to a small python script doing a replacement.'
    text = text.replace('(port)', str(server.port))
    script = tmpdir / 'script.py'
    script.write(textwrap.dedent('\n        import sys\n\n        with open(sys.argv[1], encoding=\'utf-8\') as f:\n            data = f.read()\n\n        data = data.replace("{text}", "{replacement}")\n\n        with open(sys.argv[1], \'w\', encoding=\'utf-8\') as f:\n            f.write(data)\n    '.format(text=text, replacement=replacement)))
    editor = json.dumps([sys.executable, str(script), '{}'])
    quteproc.set_setting('editor.command', editor)

@bdd.when(bdd.parsers.parse('I setup a fake editor returning "{text}"'))
def set_up_editor(quteproc, tmpdir, text):
    if False:
        for i in range(10):
            print('nop')
    'Set up editor.command to a small python script inserting a text.'
    script = tmpdir / 'script.py'
    script.write(textwrap.dedent("\n        import sys\n\n        with open(sys.argv[1], 'w', encoding='utf-8') as f:\n            f.write({text!r})\n    ".format(text=text)))
    editor = json.dumps([sys.executable, str(script), '{}'])
    quteproc.set_setting('editor.command', editor)

@bdd.when(bdd.parsers.parse('I setup a fake editor returning empty text'))
def set_up_editor_empty(quteproc, tmpdir):
    if False:
        print('Hello World!')
    'Set up editor.command to a small python script inserting empty text.'
    set_up_editor(quteproc, tmpdir, '')

class EditorPidWatcher(QObject):
    appeared = pyqtSignal()

    def __init__(self, directory, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self._pidfile = directory / 'editor_pid'
        self._watcher = QFileSystemWatcher(self)
        self._watcher.addPath(str(directory))
        self._watcher.directoryChanged.connect(self._check_update)
        self.has_pidfile = False
        self._check_update()

    @pyqtSlot()
    def _check_update(self):
        if False:
            print('Hello World!')
        if self.has_pidfile:
            return
        if self._pidfile.check():
            if self._pidfile.read():
                self.has_pidfile = True
                self.appeared.emit()
            else:
                self._watcher.addPath(str(self._pidfile))

    def manual_check(self):
        if False:
            return 10
        return self._pidfile.check()

@pytest.fixture
def editor_pid_watcher(tmpdir):
    if False:
        i = 10
        return i + 15
    return EditorPidWatcher(tmpdir)

@bdd.when(bdd.parsers.parse('I setup a fake editor that writes "{text}" on save'))
def set_up_editor_wait(quteproc, tmpdir, text, editor_pid_watcher):
    if False:
        return 10
    'Set up editor.command to a small python script inserting a text.'
    assert not utils.is_windows
    pidfile = tmpdir / 'editor_pid'
    script = tmpdir / 'script.py'
    script.write(textwrap.dedent("\n        import os\n        import sys\n        import time\n        import signal\n\n        def handle(sig, _frame):\n            filename = sys.argv[1]\n            old_mtime = new_mtime = os.stat(filename).st_mtime\n            while old_mtime == new_mtime:\n                time.sleep(0.1)\n                with open(filename, 'w', encoding='utf-8') as f:\n                    f.write({text!r})\n                new_mtime = os.stat(filename).st_mtime\n            if sig == signal.SIGUSR1:\n                sys.exit(0)\n\n        signal.signal(signal.SIGUSR1, handle)\n        signal.signal(signal.SIGUSR2, handle)\n\n        with open(r'{pidfile}', 'w') as f:\n            f.write(str(os.getpid()))\n\n        time.sleep(100)\n    ".format(pidfile=pidfile, text=text)))
    editor = json.dumps([sys.executable, str(script), '{}'])
    quteproc.set_setting('editor.command', editor)

@bdd.when('I wait until the editor has started')
def wait_editor(qtbot, editor_pid_watcher):
    if False:
        print('Hello World!')
    if not editor_pid_watcher.has_pidfile:
        with qtbot.wait_signal(editor_pid_watcher.appeared, raising=False):
            pass
    if not editor_pid_watcher.manual_check():
        pytest.fail('Editor pidfile failed to appear!')

@bdd.when(bdd.parsers.parse('I kill the waiting editor'))
def kill_editor_wait(tmpdir):
    if False:
        while True:
            i = 10
    'Kill the waiting editor.'
    pidfile = tmpdir / 'editor_pid'
    pid = int(pidfile.read())
    os.kill(pid, signal.SIGUSR1)

@bdd.when(bdd.parsers.parse('I save without exiting the editor'))
def save_editor_wait(tmpdir):
    if False:
        return 10
    'Trigger the waiting editor to write without exiting.'
    pidfile = tmpdir / 'editor_pid'
    for _ in range(10):
        if pidfile.check():
            break
        time.sleep(1)
    pid = int(pidfile.read())
    os.kill(pid, signal.SIGUSR2)