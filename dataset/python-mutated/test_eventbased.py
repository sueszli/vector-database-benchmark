import threading
from subprocess import Popen
import mock
import pytest
try:
    from watchdog.events import FileSystemEvent, DirModifiedEvent
    from chalice.cli.filewatch.eventbased import WatchdogRestarter
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
import chalice.local
from chalice.cli import reloader
from chalice.local import LocalDevServer
pytestmark = pytest.mark.skipif(not HAS_WATCHDOG, reason='Tests require watchdog package.')

class RecordingPopen(object):

    def __init__(self, process, return_codes=None):
        if False:
            for i in range(10):
                print('nop')
        self.process = process
        self.recorded_args = []
        if return_codes is None:
            return_codes = []
        self.return_codes = return_codes

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.recorded_args.append((args, kwargs))
        if self.return_codes:
            rc = self.return_codes.pop(0)
            self.process.returncode = rc
        return self.process

def test_restarter_triggers_event():
    if False:
        i = 10
        return i + 15
    restart_event = threading.Event()
    restarter = WatchdogRestarter(restart_event)
    app_modified = FileSystemEvent(src_path='./app.py')
    restarter.on_any_event(app_modified)
    assert restart_event.is_set()

def test_directory_events_ignored():
    if False:
        for i in range(10):
            print('nop')
    restart_event = threading.Event()
    restarter = WatchdogRestarter(restart_event)
    app_modified = DirModifiedEvent(src_path='./')
    restarter.on_any_event(app_modified)
    assert not restart_event.is_set()

def test_http_server_thread_starts_server_and_shutsdown():
    if False:
        while True:
            i = 10
    server = mock.Mock(spec=LocalDevServer)
    thread = chalice.local.HTTPServerThread(lambda : server)
    thread.run()
    thread.shutdown()
    server.serve_forever.assert_called_with()
    server.shutdown.assert_called_with()

def test_shutdown_noop_if_server_not_started():
    if False:
        print('Hello World!')
    server = mock.Mock(spec=LocalDevServer)
    thread = chalice.local.HTTPServerThread(lambda : server)
    thread.shutdown()
    assert not server.shutdown.called

def test_parent_process_starts_child_with_worker_env_var():
    if False:
        while True:
            i = 10
    process = mock.Mock(spec=Popen)
    process.returncode = 0
    popen = RecordingPopen(process)
    env = {'original-env': 'foo'}
    parent = reloader.ParentProcess(env, popen)
    parent.main()
    assert len(popen.recorded_args) == 1
    kwargs = popen.recorded_args[-1][1]
    assert kwargs == {'env': {'original-env': 'foo', 'CHALICE_WORKER': 'true'}}

def test_assert_child_restarted_until_not_restart_rc():
    if False:
        i = 10
        return i + 15
    process = mock.Mock(spec=Popen)
    popen = RecordingPopen(process, return_codes=[chalice.cli.filewatch.RESTART_REQUEST_RC, 0])
    parent = reloader.ParentProcess({}, popen)
    parent.main()
    assert len(popen.recorded_args) == 2

def test_ctrl_c_kill_child_process():
    if False:
        print('Hello World!')
    process = mock.Mock(spec=Popen)
    process.communicate.side_effect = KeyboardInterrupt
    popen = RecordingPopen(process)
    parent = reloader.ParentProcess({}, popen)
    with pytest.raises(KeyboardInterrupt):
        parent.main()
    assert process.terminate.called