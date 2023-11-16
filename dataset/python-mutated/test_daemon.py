import unittest
from unittest import mock
import octoprint.daemon

class ExpectedExit(BaseException):
    pass

class DaemonTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        run_method = mock.MagicMock()
        echo_method = mock.MagicMock()
        error_method = mock.MagicMock()

        class TestDaemon(octoprint.daemon.Daemon):

            def run(self):
                if False:
                    return 10
                run_method()

            def echo(self, line):
                if False:
                    for i in range(10):
                        print('nop')
                echo_method(line)

            def error(self, line):
                if False:
                    for i in range(10):
                        print('nop')
                error_method(line)
        self.pidfile = '/my/pid/file'
        self.daemon = TestDaemon(self.pidfile)
        self.run_method = run_method
        self.echo_method = echo_method
        self.error_method = error_method

    @mock.patch('os.fork', create=True)
    @mock.patch('os.chdir')
    @mock.patch('os.setsid', create=True)
    @mock.patch('os.umask')
    @mock.patch('sys.exit')
    def test_double_fork(self, mock_exit, mock_umask, mock_setsid, mock_chdir, mock_fork):
        if False:
            return 10
        pid1 = 1234
        pid2 = 2345
        mock_fork.side_effect = [pid1, pid2]
        self.daemon._double_fork()
        self.assertListEqual(mock_fork.mock_calls, [mock.call(), mock.call()])
        self.assertListEqual(mock_exit.mock_calls, [mock.call(0), mock.call(0)])
        mock_chdir.assert_called_once_with('/')
        mock_setsid.assert_called_once_with()
        mock_umask.assert_called_once_with(2)

    @mock.patch('os.fork', create=True)
    @mock.patch('sys.exit')
    def test_double_fork_failed_first(self, mock_exit, mock_fork):
        if False:
            for i in range(10):
                print('nop')
        mock_fork.side_effect = OSError()
        mock_exit.side_effect = ExpectedExit()
        try:
            self.daemon._double_fork()
            self.fail('Expected an exit')
        except ExpectedExit:
            pass
        self.assertListEqual(mock_fork.mock_calls, [mock.call()])
        self.assertListEqual(mock_exit.mock_calls, [mock.call(1)])
        self.assertEqual(len(self.error_method.mock_calls), 1)

    @mock.patch('os.fork', create=True)
    @mock.patch('os.chdir')
    @mock.patch('os.setsid', create=True)
    @mock.patch('os.umask')
    @mock.patch('sys.exit')
    def test_double_fork_failed_second(self, mock_exit, mock_umask, mock_setsid, mock_chdir, mock_fork):
        if False:
            i = 10
            return i + 15
        mock_fork.side_effect = [1234, OSError()]
        mock_exit.side_effect = [None, ExpectedExit()]
        try:
            self.daemon._double_fork()
            self.fail('Expected an exit')
        except ExpectedExit:
            pass
        self.assertEqual(mock_fork.call_count, 2)
        self.assertListEqual(mock_exit.mock_calls, [mock.call(0), mock.call(1)])
        self.assertEqual(self.error_method.call_count, 1)
        mock_chdir.assert_called_once_with('/')
        mock_setsid.assert_called_once_with()
        mock_umask.assert_called_once_with(2)

    @mock.patch('sys.stdin')
    @mock.patch('sys.stdout')
    @mock.patch('sys.stderr')
    @mock.patch('os.devnull')
    @mock.patch('builtins.open')
    @mock.patch('os.dup2')
    def test_redirect_io(self, mock_dup2, mock_open, mock_devnull, mock_stderr, mock_stdout, mock_stdin):
        if False:
            while True:
                i = 10
        mock_stdin.fileno.return_value = 'stdin'
        mock_stdout.fileno.return_value = 'stdout'
        mock_stderr.fileno.return_value = 'stderr'
        new_stdin = mock.MagicMock()
        new_stdout = mock.MagicMock()
        new_stderr = mock.MagicMock()
        new_stdin.fileno.return_value = 'new_stdin'
        new_stdout.fileno.return_value = 'new_stdout'
        new_stderr.fileno.return_value = 'new_stderr'
        mock_open.side_effect = [new_stdin, new_stdout, new_stderr]
        self.daemon._redirect_io()
        mock_stdout.flush.assert_called_once_with()
        mock_stderr.flush.assert_called_once_with()
        self.assertListEqual(mock_open.mock_calls, [mock.call(mock_devnull, encoding='utf-8'), mock.call(mock_devnull, 'a+', encoding='utf-8'), mock.call(mock_devnull, 'a+', encoding='utf-8')])
        self.assertListEqual(mock_dup2.mock_calls, [mock.call('new_stdin', 'stdin'), mock.call('new_stdout', 'stdout'), mock.call('new_stderr', 'stderr')])

    @mock.patch('os.getpid')
    @mock.patch('signal.signal')
    def test_daemonize(self, mock_signal, mock_getpid):
        if False:
            while True:
                i = 10
        self.daemon._double_fork = mock.MagicMock()
        self.daemon._redirect_io = mock.MagicMock()
        self.daemon.set_pid = mock.MagicMock()
        pid = 1234
        mock_getpid.return_value = pid
        self.daemon.start()
        self.daemon._double_fork.assert_called_once_with()
        self.daemon._redirect_io.assert_called_once_with()
        self.daemon.set_pid.assert_called_once_with(str(pid))

    def test_terminated(self):
        if False:
            print('Hello World!')
        self.daemon.remove_pidfile = mock.MagicMock()
        self.daemon.terminated()
        self.daemon.remove_pidfile.assert_called_once_with()

    def test_start(self):
        if False:
            print('Hello World!')
        self.daemon._daemonize = mock.MagicMock()
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = None
        self.daemon.start()
        self.daemon._daemonize.assert_called_once_with()
        self.daemon.get_pid.assert_called_once_with()
        self.echo_method.assert_called_once_with('Starting daemon...')
        self.assertTrue(self.run_method.called)

    @mock.patch('sys.exit')
    def test_start_running(self, mock_exit):
        if False:
            print('Hello World!')
        pid = '1234'
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = pid
        mock_exit.side_effect = ExpectedExit()
        try:
            self.daemon.start()
            self.fail('Expected an exit')
        except ExpectedExit:
            pass
        self.daemon.get_pid.assert_called_once_with()
        self.assertTrue(self.error_method.called)
        mock_exit.assert_called_once_with(1)

    @mock.patch('os.kill')
    @mock.patch('time.sleep')
    def test_stop(self, mock_sleep, mock_kill):
        if False:
            return 10
        import signal
        pid = '1234'
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = pid
        self.daemon.remove_pidfile = mock.MagicMock()
        mock_kill.side_effect = [None, OSError('No such process')]
        self.daemon.stop()
        self.daemon.get_pid.assert_called_once_with()
        self.assertListEqual(mock_kill.mock_calls, [mock.call(pid, signal.SIGTERM), mock.call(pid, signal.SIGTERM)])
        mock_sleep.assert_called_once_with(0.1)
        self.daemon.remove_pidfile.assert_called_once_with()

    @mock.patch('sys.exit')
    def test_stop_not_running(self, mock_exit):
        if False:
            print('Hello World!')
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = None
        mock_exit.side_effect = ExpectedExit()
        try:
            self.daemon.stop()
            self.fail('Expected an exit')
        except ExpectedExit:
            pass
        self.daemon.get_pid.assert_called_once_with()
        self.assertEqual(self.error_method.call_count, 1)
        mock_exit.assert_called_once_with(1)

    @mock.patch('sys.exit')
    def test_stop_not_running_no_error(self, mock_exit):
        if False:
            while True:
                i = 10
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = None
        self.daemon.stop(check_running=False)
        self.daemon.get_pid.assert_called_once_with()
        self.assertFalse(mock_exit.called)

    @mock.patch('os.kill')
    @mock.patch('sys.exit')
    def test_stop_unknown_error(self, mock_exit, mock_kill):
        if False:
            while True:
                i = 10
        pid = '1234'
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = pid
        mock_exit.side_effect = ExpectedExit()
        mock_kill.side_effect = OSError('Unknown')
        try:
            self.daemon.stop()
            self.fail('Expected an exit')
        except ExpectedExit:
            pass
        self.assertTrue(self.error_method.called)
        mock_exit.assert_called_once_with(1)

    def test_restart(self):
        if False:
            while True:
                i = 10
        self.daemon.start = mock.MagicMock()
        self.daemon.stop = mock.MagicMock()
        self.daemon.restart()
        self.daemon.stop.assert_called_once_with(check_running=False)
        self.daemon.start.assert_called_once_with()

    def test_status_running(self):
        if False:
            for i in range(10):
                print('nop')
        self.daemon.is_running = mock.MagicMock()
        self.daemon.is_running.return_value = True
        self.daemon.status()
        self.echo_method.assert_called_once_with('Daemon is running')

    def test_status_not_running(self):
        if False:
            print('Hello World!')
        self.daemon.is_running = mock.MagicMock()
        self.daemon.is_running.return_value = False
        self.daemon.status()
        self.echo_method.assert_called_once_with('Daemon is not running')

    @mock.patch('os.kill')
    def test_is_running_true(self, mock_kill):
        if False:
            print('Hello World!')
        pid = '1234'
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = pid
        self.daemon.remove_pidfile = mock.MagicMock()
        result = self.daemon.is_running()
        self.assertTrue(result)
        mock_kill.assert_called_once_with(pid, 0)
        self.assertFalse(self.daemon.remove_pidfile.called)
        self.assertFalse(self.error_method.called)

    def test_is_running_false_no_pid(self):
        if False:
            i = 10
            return i + 15
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = None
        result = self.daemon.is_running()
        self.assertFalse(result)

    @mock.patch('os.kill')
    def test_is_running_false_pidfile_removed(self, mock_kill):
        if False:
            return 10
        pid = '1234'
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = pid
        mock_kill.side_effect = OSError()
        self.daemon.remove_pidfile = mock.MagicMock()
        result = self.daemon.is_running()
        self.assertFalse(result)
        mock_kill.assert_called_once_with(pid, 0)
        self.daemon.remove_pidfile.assert_called_once_with()
        self.assertFalse(self.error_method.called)

    @mock.patch('os.kill')
    def test_is_running_false_pidfile_error(self, mock_kill):
        if False:
            return 10
        pid = '1234'
        self.daemon.get_pid = mock.MagicMock()
        self.daemon.get_pid.return_value = pid
        mock_kill.side_effect = OSError()
        self.daemon.remove_pidfile = mock.MagicMock()
        self.daemon.remove_pidfile.side_effect = IOError()
        result = self.daemon.is_running()
        self.assertFalse(result)
        mock_kill.assert_called_once_with(pid, 0)
        self.daemon.remove_pidfile.assert_called_once_with()
        self.assertTrue(self.error_method.called)

    def test_get_pid(self):
        if False:
            for i in range(10):
                print('nop')
        pid = 1234
        with mock.patch('builtins.open', mock.mock_open(read_data=f'{pid}\n'), create=True) as m:
            result = self.daemon.get_pid()
        self.assertEqual(result, pid)
        m.assert_called_once_with(self.pidfile, encoding='utf-8')

    def test_get_pid_ioerror(self):
        if False:
            for i in range(10):
                print('nop')
        handle = mock.MagicMock()
        handle.__enter__.side_effect = IOError()
        with mock.patch('builtins.open', mock.mock_open(), create=True) as m:
            result = self.daemon.get_pid()
        self.assertIsNone(result)
        m.assert_called_once_with(self.pidfile, encoding='utf-8')

    def test_get_pid_valueerror(self):
        if False:
            print('Hello World!')
        pid = 'not an integer'
        with mock.patch('builtins.open', mock.mock_open(read_data=f'{pid}\n'), create=True) as m:
            result = self.daemon.get_pid()
        self.assertIsNone(result)
        m.assert_called_once_with(self.pidfile, encoding='utf-8')

    def test_set_pid(self):
        if False:
            return 10
        pid = '1234'
        with mock.patch('builtins.open', mock.mock_open(), create=True) as m:
            self.daemon.set_pid(pid)
        m.assert_called_once_with(self.pidfile, 'w+', encoding='utf-8')
        handle = m()
        handle.write.assert_called_once_with(f'{pid}\n')

    def test_set_pid_int(self):
        if False:
            i = 10
            return i + 15
        pid = 1234
        with mock.patch('builtins.open', mock.mock_open(), create=True) as m:
            self.daemon.set_pid(pid)
        m.assert_called_once_with(self.pidfile, 'w+', encoding='utf-8')
        handle = m()
        handle.write.assert_called_once_with(f'{pid}\n')

    @mock.patch('os.path.isfile')
    @mock.patch('os.remove')
    def test_remove_pidfile_exists(self, mock_remove, mock_isfile):
        if False:
            print('Hello World!')
        mock_isfile.return_value = True
        self.daemon.remove_pidfile()
        mock_isfile.assert_called_once_with(self.pidfile)
        mock_remove.assert_called_once_with(self.pidfile)

    @mock.patch('os.path.isfile')
    @mock.patch('os.remove')
    def test_remove_pidfile_doesnt_exist(self, mock_remove, mock_isfile):
        if False:
            i = 10
            return i + 15
        mock_isfile.return_value = False
        self.daemon.remove_pidfile()
        mock_isfile.assert_called_once_with(self.pidfile)
        self.assertFalse(mock_remove.called)