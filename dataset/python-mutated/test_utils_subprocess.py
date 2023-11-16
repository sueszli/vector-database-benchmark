import locale
import sys
from logging import DEBUG, ERROR, INFO, WARNING
from typing import List, Optional, Tuple, Type
import pytest
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import InstallationSubprocessError
from pip._internal.utils.logging import VERBOSE
from pip._internal.utils.misc import hide_value
from pip._internal.utils.subprocess import CommandArgs, call_subprocess, format_command_args, make_command, subprocess_logger

@pytest.mark.parametrize('args, expected', [(['pip', 'list'], 'pip list'), (['foo', 'space space', 'new\nline', 'double"quote', "single'quote"], 'foo \'space space\' \'new\nline\' \'double"quote\' \'single\'"\'"\'quote\''), (make_command(hide_value('secret1'), 'foo', hide_value('secret2')), "'****' foo '****'")])
def test_format_command_args(args: CommandArgs, expected: str) -> None:
    if False:
        print('Hello World!')
    actual = format_command_args(args)
    assert actual == expected

@pytest.mark.parametrize(('stdout_only', 'expected'), [(True, ('out\n', 'out\r\n')), (False, ('out\nerr\n', 'out\r\nerr\r\n', 'err\nout\n', 'err\r\nout\r\n'))])
def test_call_subprocess_stdout_only(capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, stdout_only: bool, expected: Tuple[str, ...]) -> None:
    if False:
        return 10
    log = []
    monkeypatch.setattr(subprocess_logger, 'log', lambda level, *args: log.append(args[0]))
    out = call_subprocess([sys.executable, '-c', "import sys; sys.stdout.write('out\\n'); sys.stderr.write('err\\n')"], command_desc='test stdout_only', stdout_only=stdout_only)
    assert out in expected
    captured = capfd.readouterr()
    assert captured.err == ''
    assert log == ['Running command %s', 'out', 'err'] or log == ['Running command %s', 'err', 'out']

class FakeSpinner(SpinnerInterface):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.spin_count = 0
        self.final_status: Optional[str] = None

    def spin(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.spin_count += 1

    def finish(self, final_status: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.final_status = final_status

class TestCallSubprocess:
    """
    Test call_subprocess().
    """

    def check_result(self, capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture, log_level: int, spinner: FakeSpinner, result: Optional[str], expected: Tuple[Optional[List[str]], List[Tuple[str, int, str]]], expected_spinner: Tuple[int, Optional[str]]) -> None:
        if False:
            print('Hello World!')
        "\n        Check the result of calling call_subprocess().\n\n        :param log_level: the logging level that caplog was set to.\n        :param spinner: the FakeSpinner object passed to call_subprocess()\n            to be checked.\n        :param result: the call_subprocess() return value to be checked.\n        :param expected: a pair (expected_proc, expected_records), where\n            1) `expected_proc` is the expected return value of\n              call_subprocess() as a list of lines, or None if the return\n              value is expected to be None;\n            2) `expected_records` is the expected value of\n              caplog.record_tuples.\n        :param expected_spinner: a 2-tuple of the spinner's expected\n            (spin_count, final_status).\n        "
        (expected_proc, expected_records) = expected
        if expected_proc is None:
            assert result is None
        else:
            assert result is not None
            assert result.splitlines() == expected_proc
        captured = capfd.readouterr()
        assert (captured.out, captured.err) == ('', '')
        records = caplog.record_tuples
        if len(records) != len(expected_records):
            raise RuntimeError(f'{records} != {expected_records}')
        for (record, expected_record) in zip(records, expected_records):
            assert record[:2] == expected_record[:2]
            assert expected_record[2] in record[2]
        assert (spinner.spin_count, spinner.final_status) == expected_spinner

    def prepare_call(self, caplog: pytest.LogCaptureFixture, log_level: int, command: Optional[str]=None) -> Tuple[List[str], FakeSpinner]:
        if False:
            i = 10
            return i + 15
        if command is None:
            command = 'print("Hello"); print("world")'
        caplog.set_level(log_level)
        spinner = FakeSpinner()
        args = [sys.executable, '-c', command]
        return (args, spinner)

    def test_debug_logging(self, capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
        if False:
            return 10
        '\n        Test DEBUG logging (and without passing show_stdout=True).\n        '
        log_level = DEBUG
        (args, spinner) = self.prepare_call(caplog, log_level)
        result = call_subprocess(args, command_desc='test debug logging', spinner=spinner)
        expected = (['Hello', 'world'], [('pip.subprocessor', VERBOSE, 'Running '), ('pip.subprocessor', VERBOSE, 'Hello'), ('pip.subprocessor', VERBOSE, 'world')])
        self.check_result(capfd, caplog, log_level, spinner, result, expected, expected_spinner=(0, None))

    def test_info_logging(self, capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test INFO logging (and without passing show_stdout=True).\n        '
        log_level = INFO
        (args, spinner) = self.prepare_call(caplog, log_level)
        result = call_subprocess(args, command_desc='test info logging', spinner=spinner)
        expected: Tuple[List[str], List[Tuple[str, int, str]]] = (['Hello', 'world'], [])
        self.check_result(capfd, caplog, log_level, spinner, result, expected, expected_spinner=(2, 'done'))

    def test_info_logging__subprocess_error(self, capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
        if False:
            return 10
        '\n        Test INFO logging of a subprocess with an error (and without passing\n        show_stdout=True).\n        '
        log_level = INFO
        command = 'print("Hello"); print("world"); exit("fail")'
        (args, spinner) = self.prepare_call(caplog, log_level, command=command)
        with pytest.raises(InstallationSubprocessError) as exc:
            call_subprocess(args, command_desc='test info logging with subprocess error', spinner=spinner)
        result = None
        exception = exc.value
        assert exception.reference == 'subprocess-exited-with-error'
        assert 'exit code: 1' in exception.message
        assert exception.note_stmt
        assert 'not a problem with pip' in exception.note_stmt
        assert exception.context
        assert 'Hello\n' in exception.context
        assert 'fail\n' in exception.context
        assert 'world\n' in exception.context
        expected = (None, [('pip.subprocessor', ERROR, 'subprocess error exited with 1')])
        self.check_result(capfd, caplog, log_level, spinner, result, expected, expected_spinner=(3, 'error'))

    def test_info_logging_with_show_stdout_true(self, capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> None:
        if False:
            while True:
                i = 10
        '\n        Test INFO logging with show_stdout=True.\n        '
        log_level = INFO
        (args, spinner) = self.prepare_call(caplog, log_level)
        result = call_subprocess(args, command_desc='test info logging with show_stdout', spinner=spinner, show_stdout=True)
        expected = (['Hello', 'world'], [('pip.subprocessor', INFO, 'Running '), ('pip.subprocessor', INFO, 'Hello'), ('pip.subprocessor', INFO, 'world')])
        self.check_result(capfd, caplog, log_level, spinner, result, expected, expected_spinner=(0, None))

    @pytest.mark.parametrize(('exit_status', 'show_stdout', 'extra_ok_returncodes', 'log_level', 'expected'), [(0, False, None, INFO, (None, 'done', 2)), (0, False, None, DEBUG, (None, None, 0)), (0, True, None, DEBUG, (None, None, 0)), (0, True, None, INFO, (None, None, 0)), (0, True, None, WARNING, (None, 'done', 2)), (3, False, None, INFO, (InstallationSubprocessError, 'error', 2)), (3, False, (3,), INFO, (None, 'done', 2))])
    def test_spinner_finish(self, exit_status: int, show_stdout: bool, extra_ok_returncodes: Optional[Tuple[int, ...]], log_level: int, caplog: pytest.LogCaptureFixture, expected: Tuple[Optional[Type[Exception]], Optional[str], int]) -> None:
        if False:
            print('Hello World!')
        '\n        Test that the spinner finishes correctly.\n        '
        expected_exc_type = expected[0]
        expected_final_status = expected[1]
        expected_spin_count = expected[2]
        command = f'print("Hello"); print("world"); exit({exit_status})'
        (args, spinner) = self.prepare_call(caplog, log_level, command=command)
        exc_type: Optional[Type[Exception]]
        try:
            call_subprocess(args, command_desc='spinner go spinny', show_stdout=show_stdout, extra_ok_returncodes=extra_ok_returncodes, spinner=spinner)
        except Exception as exc:
            exc_type = type(exc)
        else:
            exc_type = None
        assert exc_type == expected_exc_type
        assert spinner.final_status == expected_final_status
        assert spinner.spin_count == expected_spin_count

    def test_closes_stdin(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.raises(InstallationSubprocessError):
            call_subprocess([sys.executable, '-c', 'input()'], show_stdout=True, command_desc='stdin reader')

def test_unicode_decode_error(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    if locale.getpreferredencoding() != 'UTF-8':
        pytest.skip('locale.getpreferredencoding() is not UTF-8')
    caplog.set_level(INFO)
    call_subprocess([sys.executable, '-c', "import sys; sys.stdout.buffer.write(b'\\xff')"], command_desc='invalid decode output', show_stdout=True)
    assert len(caplog.records) == 2
    assert caplog.record_tuples[1] == ('pip.subprocessor', INFO, '\\xff')