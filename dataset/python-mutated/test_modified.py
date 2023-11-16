"""Black-box tests for 'bzr modified', which shows modified files."""
import os
from bzrlib.branch import Branch
from bzrlib.tests import TestCaseWithTransport

class TestModified(TestCaseWithTransport):

    def test_modified(self):
        if False:
            while True:
                i = 10
        "Test that 'modified' command reports modified files"
        self._test_modified('a', 'a')

    def test_modified_with_spaces(self):
        if False:
            i = 10
            return i + 15
        "Test that 'modified' command reports modified files with spaces in their names quoted"
        self._test_modified('a filename with spaces', '"a filename with spaces"')

    def _test_modified(self, name, output):
        if False:
            while True:
                i = 10

        def check_modified(expected, null=False):
            if False:
                return 10
            command = 'modified'
            if null:
                command += ' --null'
            (out, err) = self.run_bzr(command)
            self.assertEqual(out, expected)
            self.assertEqual(err, '')
        tree = self.make_branch_and_tree('.')
        check_modified('')
        self.build_tree_contents([(name, 'contents of %s\n' % name)])
        check_modified('')
        tree.add(name)
        check_modified('')
        tree.commit(message='add %s' % output)
        check_modified('')
        self.build_tree_contents([(name, 'changed\n')])
        check_modified(output + '\n')
        check_modified(name + '\x00', null=True)
        tree.commit(message='modified %s' % name)
        check_modified('')

    def test_modified_directory(self):
        if False:
            i = 10
            return i + 15
        'Test --directory option'
        tree = self.make_branch_and_tree('a')
        self.build_tree(['a/README'])
        tree.add('README')
        tree.commit('r1')
        self.build_tree_contents([('a/README', 'changed\n')])
        (out, err) = self.run_bzr(['modified', '--directory=a'])
        self.assertEqual('README\n', out)