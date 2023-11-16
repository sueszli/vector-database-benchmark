import unittest
import mock
import tempfile
import shutil
import sh
import powerline_shell.segments.hg as hg
from powerline_shell.utils import RepoStats
from ..testing_utils import dict_side_effect_fn
test_cases = {'? new-file': RepoStats(new=1), 'M modified-file': RepoStats(changed=1), 'R removed-file': RepoStats(changed=1), '! missing-file': RepoStats(changed=1), 'A added-file': RepoStats(staged=1)}

class HgTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.powerline = mock.MagicMock()
        self.powerline.segment_conf.side_effect = dict_side_effect_fn({('vcs', 'show_symbol'): False})
        self.dirname = tempfile.mkdtemp()
        sh.cd(self.dirname)
        sh.hg('init', '.')
        self.segment = hg.Segment(self.powerline, {})

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.dirname)

    def _add_and_commit(self, filename):
        if False:
            print('Hello World!')
        sh.touch(filename)
        sh.hg('add', filename)
        sh.hg('commit', '-m', 'add file ' + filename)

    def _checkout_new_branch(self, branch):
        if False:
            while True:
                i = 10
        sh.hg('branch', branch)

    @mock.patch('powerline_shell.utils.get_PATH')
    def test_hg_not_installed(self, get_PATH):
        if False:
            for i in range(10):
                print('nop')
        get_PATH.return_value = ''
        self.segment.start()
        self.segment.add_to_powerline()
        self.assertEqual(self.powerline.append.call_count, 0)

    def test_non_hg_directory(self):
        if False:
            print('Hello World!')
        shutil.rmtree('.hg')
        self.segment.start()
        self.segment.add_to_powerline()
        self.assertEqual(self.powerline.append.call_count, 0)

    def test_standard(self):
        if False:
            i = 10
            return i + 15
        self._add_and_commit('foo')
        self.segment.start()
        self.segment.add_to_powerline()
        self.assertEqual(self.powerline.append.call_args[0][0], ' default ')

    def test_different_branch(self):
        if False:
            return 10
        self._add_and_commit('foo')
        self._checkout_new_branch('bar')
        self.segment.start()
        self.segment.add_to_powerline()
        self.assertEqual(self.powerline.append.call_args[0][0], ' bar ')

    @mock.patch('powerline_shell.segments.hg._get_hg_status')
    def test_all(self, check_output):
        if False:
            for i in range(10):
                print('nop')
        for (stdout, result) in test_cases.items():
            stats = hg.parse_hg_stats([stdout])
            self.assertEquals(result, stats)