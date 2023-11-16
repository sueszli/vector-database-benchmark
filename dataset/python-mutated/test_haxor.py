from __future__ import print_function
from __future__ import division
import mock
import platform
from tests.compat import unittest
from haxor_news.haxor import Haxor

class HaxorTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.haxor = Haxor()

    def test_add_comment_pagination(self):
        if False:
            for i in range(10):
                print('nop')
        text = 'hn view 1'
        result = self.haxor._add_comment_pagination(text)
        assert result == text
        text = 'hn view 1 -c'
        result = self.haxor._add_comment_pagination(text)
        if platform.system() == 'Windows':
            assert result == text + self.haxor.PAGINATE_CMD_WIN
        else:
            assert result == text + self.haxor.PAGINATE_CMD
        text = 'hn view 1 -c -b'
        result = self.haxor._add_comment_pagination(text)
        assert result == text

    @mock.patch('haxor_news.haxor.subprocess.call')
    def test_run_command(self, mock_subprocess_call):
        if False:
            return 10
        document = mock.Mock()
        document.text = 'hn view 1 -c'
        self.haxor.run_command(document)
        mock_subprocess_call.assert_called_with('hn view 1 -c | less -r', shell=True)
        document.text = 'hn view 1'
        self.haxor.run_command(document)
        mock_subprocess_call.assert_called_with('hn view 1', shell=True)

    @mock.patch('haxor_news.haxor.sys.exit')
    def test_exit_command(self, mock_sys_exit):
        if False:
            for i in range(10):
                print('nop')
        document = mock.Mock()
        document.text = 'exit'
        self.haxor.handle_exit(document)
        mock_sys_exit.assert_called_with()