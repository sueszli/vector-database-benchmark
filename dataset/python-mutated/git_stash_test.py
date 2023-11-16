import unittest
import mock
import tempfile
import shutil
import sh
import powerline_shell.segments.git_stash as git_stash
from powerline_shell.utils import RepoStats

class GitStashTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.powerline = mock.MagicMock()
        self.dirname = tempfile.mkdtemp()
        sh.cd(self.dirname)
        sh.git('init', '.')
        self.segment = git_stash.Segment(self.powerline, {})

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.dirname)

    def _add_and_commit(self, filename):
        if False:
            while True:
                i = 10
        sh.touch(filename)
        sh.git('add', filename)
        sh.git('commit', '-m', 'add file ' + filename)

    def _overwrite_file(self, filename, content):
        if False:
            i = 10
            return i + 15
        sh.echo(content, _out=filename)

    def _stash(self):
        if False:
            return 10
        sh.git('stash')

    @mock.patch('powerline_shell.utils.get_PATH')
    def test_git_not_installed(self, get_PATH):
        if False:
            for i in range(10):
                print('nop')
        get_PATH.return_value = ''
        self.segment.start()
        self.segment.add_to_powerline()
        self.assertEqual(self.powerline.append.call_count, 0)

    def test_non_git_directory(self):
        if False:
            print('Hello World!')
        shutil.rmtree('.git')
        self.segment.start()
        self.segment.add_to_powerline()
        self.assertEqual(self.powerline.append.call_count, 0)

    def test_no_stashes(self):
        if False:
            for i in range(10):
                print('nop')
        self._add_and_commit('foo')
        self.segment.start()
        self.segment.add_to_powerline()
        self.assertEqual(self.powerline.append.call_count, 0)

    def test_one_stash(self):
        if False:
            for i in range(10):
                print('nop')
        self._add_and_commit('foo')
        self._overwrite_file('foo', 'some new content')
        self._stash()
        self.segment.start()
        self.segment.add_to_powerline()
        expected = u' {} '.format(RepoStats.symbols['stash'])
        self.assertEqual(self.powerline.append.call_args[0][0], expected)

    def test_multiple_stashes(self):
        if False:
            print('Hello World!')
        self._add_and_commit('foo')
        self._overwrite_file('foo', 'some new content')
        self._stash()
        self._overwrite_file('foo', 'some different content')
        self._stash()
        self._overwrite_file('foo', 'more different content')
        self._stash()
        self.segment.start()
        self.segment.add_to_powerline()
        expected = u' 3{} '.format(RepoStats.symbols['stash'])
        self.assertEqual(self.powerline.append.call_args[0][0], expected)