"""Black-box tests for 'bzr deleted', which shows newly deleted files."""
import os
from bzrlib.branch import Branch
from bzrlib.tests import TestCaseWithTransport

class TestDeleted(TestCaseWithTransport):

    def test_deleted_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test --directory option'
        tree = self.make_branch_and_tree('a')
        self.build_tree(['a/README'])
        tree.add('README')
        tree.commit('r1')
        tree.remove('README')
        (out, err) = self.run_bzr(['deleted', '--directory=a'])
        self.assertEqual('README\n', out)