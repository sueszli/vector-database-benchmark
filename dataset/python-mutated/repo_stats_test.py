import unittest
from powerline_shell.utils import RepoStats

class RepoStatsTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.repo_stats = RepoStats()
        self.repo_stats.changed = 1
        self.repo_stats.conflicted = 4

    def test_dirty(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.repo_stats.dirty)

    def test_simple(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.repo_stats.new, 0)

    def test_n_or_empty__empty(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.repo_stats.n_or_empty('changed'), u'')

    def test_n_or_empty__n(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.repo_stats.n_or_empty('conflicted'), u'4')

    def test_index(self):
        if False:
            return 10
        self.assertEqual(self.repo_stats['changed'], 1)