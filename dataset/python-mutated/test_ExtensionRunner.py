import signal
from unittest import mock
import pytest
from ulauncher.modes.extensions.ExtensionRunner import ExtensionRunner, ExtensionRuntimeError

class TestExtensionRunner:

    @pytest.fixture
    def runner(self):
        if False:
            print('Hello World!')
        return ExtensionRunner()

    @pytest.fixture(autouse=True)
    def find_extensions(self, mocker):
        if False:
            return 10
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.find_extensions')

    @pytest.fixture(autouse=True)
    def timer(self, mocker):
        if False:
            while True:
                i = 10
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.timer')

    @pytest.fixture(autouse=True)
    def get_options(self, mocker):
        if False:
            i = 10
            return i + 15
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.get_options')

    @pytest.fixture(autouse=True)
    def ExtensionManifest(self, mocker):
        if False:
            return 10
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.ExtensionManifest')

    @pytest.fixture(autouse=True)
    def json_dumps(self, mocker):
        if False:
            print('Hello World!')
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.json.dumps', return_value='{}')

    @pytest.fixture(autouse=True)
    def SubprocessLauncher(self, mocker):
        if False:
            while True:
                i = 10
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.Gio.SubprocessLauncher')

    @pytest.fixture(autouse=True)
    def DataInputStream(self, mocker):
        if False:
            return 10
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.Gio.DataInputStream')

    @pytest.fixture
    def ProcessErrorExtractor(self, mocker):
        if False:
            return 10
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.ProcessErrorExtractor')

    @pytest.fixture
    def time(self, mocker):
        if False:
            for i in range(10):
                print('nop')
        return mocker.patch('ulauncher.modes.extensions.ExtensionRunner.time')

    def test_run__basic_execution__is_called(self, runner, SubprocessLauncher):
        if False:
            while True:
                i = 10
        extid = 'id'
        runner.run(extid)
        SubprocessLauncher.new.assert_called_once()
        extproc = runner.extension_procs[extid]
        extproc.subprocess.wait_async.assert_called_once()
        extproc.error_stream.read_line_async.assert_called_once()

    def test_run_all__run__called_with_extension_ids(self, runner, mocker, find_extensions):
        if False:
            i = 10
            return i + 15
        mocker.patch.object(runner, 'run')
        find_extensions.return_value = [('id_1', 'path_1'), ('id_2', 'path_2'), ('id_3', 'path_3')]
        runner.run_all()
        runner.run.assert_any_call('id_1')
        runner.run.assert_any_call('id_2')
        runner.run.assert_any_call('id_3')

    def test_set_extension_error(self, runner):
        if False:
            while True:
                i = 10
        runner.set_extension_error('id_1', ExtensionRuntimeError.Terminated, 'message')
        error = runner.get_extension_error('id_1')
        assert error['name'] == ExtensionRuntimeError.Terminated.value
        assert error['message'] == 'message'

    def test_read_stderr_line(self, runner):
        if False:
            i = 10
            return i + 15
        test_output1 = 'Test Output 1'
        test_output2 = 'Test Output 2'
        extid = 'id'
        read_line_finish_utf8 = mock.Mock()
        runner.run(extid)
        extproc = runner.extension_procs[extid]
        extproc.error_stream.read_line_finish_utf8 = read_line_finish_utf8
        read_line_finish_utf8.return_value = (test_output1, len(test_output1))
        runner.handle_stderr(extproc.error_stream, mock.Mock(), extid)
        assert extproc.recent_errors[0] == test_output1
        assert extproc.error_stream.read_line_async.call_count == 2
        read_line_finish_utf8.return_value = (test_output2, len(test_output2))
        runner.handle_stderr(extproc.error_stream, mock.Mock(), extid)
        assert extproc.recent_errors[0] == test_output2
        assert extproc.error_stream.read_line_async.call_count == 3

    def test_handle_wait__signaled(self, runner):
        if False:
            return 10
        extid = 'id'
        runner.run(extid)
        extproc = runner.extension_procs[extid]
        extproc.subprocess.get_if_signaled.return_value = True
        extproc.subprocess.get_term_sig.return_value = 9
        assert extid not in runner.extension_errors
        runner.handle_wait(extproc.subprocess, mock.Mock(), extid)
        assert extid in runner.extension_errors
        assert extid not in runner.extension_procs

    def test_handle_wait__rapid_exit(self, runner, time, ProcessErrorExtractor):
        if False:
            for i in range(10):
                print('nop')
        extid = 'id'
        curtime = 100.0
        starttime = curtime - 0.5
        time.return_value = starttime
        runner.run(extid)
        extproc = runner.extension_procs[extid]
        extproc.subprocess.get_if_signaled.return_value = False
        extproc.subprocess.get_exit_status.return_value = 9
        time.return_value = curtime
        ProcessErrorExtractor.is_import_error.return_value = True
        ProcessErrorExtractor.get_missing_package_name.return_value = 'TestPackage'
        runner.handle_wait(extproc.subprocess, mock.Mock(), extid)
        assert extid in runner.extension_errors
        assert extid not in runner.extension_procs

    def test_handle_wait__restart(self, runner, time):
        if False:
            while True:
                i = 10
        extid = 'id'
        curtime = 100.0
        starttime = curtime - 5
        time.return_value = starttime
        runner.run(extid)
        extproc = runner.extension_procs[extid]
        extproc.subprocess.get_if_signaled.return_value = False
        extproc.subprocess.get_exit_status.return_value = 9
        time.return_value = curtime
        runner.run = mock.Mock()
        runner.handle_wait(extproc.subprocess, mock.Mock(), extid)
        assert extid in runner.extension_errors
        assert extid not in runner.extension_procs
        runner.run.assert_called_once()

    def test_stop(self, runner, timer):
        if False:
            while True:
                i = 10
        extid = 'id'
        runner.run(extid)
        extproc = runner.extension_procs[extid]
        runner.stop(extid)
        assert extid not in runner.extension_procs
        extproc.subprocess.send_signal.assert_called_with(signal.SIGTERM)
        timer.assert_called_once()

    def test_confirm_termination(self, runner):
        if False:
            while True:
                i = 10
        extproc = mock.Mock()
        extproc.subprocess.get_identifier.return_value = None
        runner.confirm_termination(extproc)
        extproc.subprocess.send_signal.assert_not_called()
        extproc.subprocess.get_identifier.return_value = 1
        runner.confirm_termination(extproc)
        extproc.subprocess.send_signal.assert_called_with(signal.SIGKILL)