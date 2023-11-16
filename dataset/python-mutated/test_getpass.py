import getpass
import os
import unittest
from io import BytesIO, StringIO, TextIOWrapper
from unittest import mock
from test import support
try:
    import termios
except ImportError:
    termios = None
try:
    import pwd
except ImportError:
    pwd = None

@mock.patch('os.environ')
class GetpassGetuserTest(unittest.TestCase):

    def test_username_takes_username_from_env(self, environ):
        if False:
            print('Hello World!')
        expected_name = 'some_name'
        environ.get.return_value = expected_name
        self.assertEqual(expected_name, getpass.getuser())

    def test_username_priorities_of_env_values(self, environ):
        if False:
            i = 10
            return i + 15
        environ.get.return_value = None
        try:
            getpass.getuser()
        except ImportError:
            pass
        self.assertEqual(environ.get.call_args_list, [mock.call(x) for x in ('LOGNAME', 'USER', 'LNAME', 'USERNAME')])

    def test_username_falls_back_to_pwd(self, environ):
        if False:
            return 10
        expected_name = 'some_name'
        environ.get.return_value = None
        if pwd:
            with mock.patch('os.getuid') as uid, mock.patch('pwd.getpwuid') as getpw:
                uid.return_value = 42
                getpw.return_value = [expected_name]
                self.assertEqual(expected_name, getpass.getuser())
                getpw.assert_called_once_with(42)
        else:
            self.assertRaises(ImportError, getpass.getuser)

class GetpassRawinputTest(unittest.TestCase):

    def test_flushes_stream_after_prompt(self):
        if False:
            while True:
                i = 10
        stream = mock.Mock(spec=StringIO)
        input = StringIO('input_string')
        getpass._raw_input('some_prompt', stream, input=input)
        stream.flush.assert_called_once_with()

    def test_uses_stderr_as_default(self):
        if False:
            return 10
        input = StringIO('input_string')
        prompt = 'some_prompt'
        with mock.patch('sys.stderr') as stderr:
            getpass._raw_input(prompt, input=input)
            stderr.write.assert_called_once_with(prompt)

    @mock.patch('sys.stdin')
    def test_uses_stdin_as_default_input(self, mock_input):
        if False:
            i = 10
            return i + 15
        mock_input.readline.return_value = 'input_string'
        getpass._raw_input(stream=StringIO())
        mock_input.readline.assert_called_once_with()

    @mock.patch('sys.stdin')
    def test_uses_stdin_as_different_locale(self, mock_input):
        if False:
            print('Hello World!')
        stream = TextIOWrapper(BytesIO(), encoding='ascii')
        mock_input.readline.return_value = 'HasÅ‚o: '
        getpass._raw_input(prompt='HasÅ‚o: ', stream=stream)
        mock_input.readline.assert_called_once_with()

    def test_raises_on_empty_input(self):
        if False:
            i = 10
            return i + 15
        input = StringIO('')
        self.assertRaises(EOFError, getpass._raw_input, input=input)

    def test_trims_trailing_newline(self):
        if False:
            i = 10
            return i + 15
        input = StringIO('test\n')
        self.assertEqual('test', getpass._raw_input(input=input))

@unittest.skipUnless(termios, 'tests require system with termios')
class UnixGetpassTest(unittest.TestCase):

    def test_uses_tty_directly(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('os.open') as open, mock.patch('io.FileIO') as fileio, mock.patch('io.TextIOWrapper') as textio:
            open.return_value = None
            getpass.unix_getpass()
            open.assert_called_once_with('/dev/tty', os.O_RDWR | os.O_NOCTTY)
            fileio.assert_called_once_with(open.return_value, 'w+')
            textio.assert_called_once_with(fileio.return_value)

    def test_resets_termios(self):
        if False:
            return 10
        with mock.patch('os.open') as open, mock.patch('io.FileIO'), mock.patch('io.TextIOWrapper'), mock.patch('termios.tcgetattr') as tcgetattr, mock.patch('termios.tcsetattr') as tcsetattr:
            open.return_value = 3
            fake_attrs = [255, 255, 255, 255, 255]
            tcgetattr.return_value = list(fake_attrs)
            getpass.unix_getpass()
            tcsetattr.assert_called_with(3, mock.ANY, fake_attrs)

    def test_falls_back_to_fallback_if_termios_raises(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('os.open') as open, mock.patch('io.FileIO') as fileio, mock.patch('io.TextIOWrapper') as textio, mock.patch('termios.tcgetattr'), mock.patch('termios.tcsetattr') as tcsetattr, mock.patch('getpass.fallback_getpass') as fallback:
            open.return_value = 3
            fileio.return_value = BytesIO()
            tcsetattr.side_effect = termios.error
            getpass.unix_getpass()
            fallback.assert_called_once_with('Password: ', textio.return_value)

    def test_flushes_stream_after_input(self):
        if False:
            return 10
        with mock.patch('os.open') as open, mock.patch('io.FileIO'), mock.patch('io.TextIOWrapper'), mock.patch('termios.tcgetattr'), mock.patch('termios.tcsetattr'):
            open.return_value = 3
            mock_stream = mock.Mock(spec=StringIO)
            getpass.unix_getpass(stream=mock_stream)
            mock_stream.flush.assert_called_with()

    def test_falls_back_to_stdin(self):
        if False:
            print('Hello World!')
        with mock.patch('os.open') as os_open, mock.patch('sys.stdin', spec=StringIO) as stdin:
            os_open.side_effect = IOError
            stdin.fileno.side_effect = AttributeError
            with support.captured_stderr() as stderr:
                with self.assertWarns(getpass.GetPassWarning):
                    getpass.unix_getpass()
            stdin.readline.assert_called_once_with()
            self.assertIn('Warning', stderr.getvalue())
            self.assertIn('Password:', stderr.getvalue())
if __name__ == '__main__':
    unittest.main()