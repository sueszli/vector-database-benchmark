"""Black-box tests for 'bzr added', which shows newly-added files."""
import os
from bzrlib.branch import Branch
from bzrlib.tests import TestCaseWithTransport

class TestAdded(TestCaseWithTransport):

    def test_added(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that 'added' command reports added files"
        self._test_added('a', 'a\n')

    def test_added_with_spaces(self):
        if False:
            print('Hello World!')
        "Test that 'added' command reports added files with spaces in their names quoted"
        self._test_added('a filename with spaces', '"a filename with spaces"\n')

    def test_added_null_separator(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that added uses its null operator properly'
        self._test_added('a', 'a\x00', null=True)

    def _test_added(self, name, output, null=False):
        if False:
            for i in range(10):
                print('nop')

        def check_added(expected, null=False):
            if False:
                i = 10
                return i + 15
            command = 'added'
            if null:
                command += ' --null'
            (out, err) = self.run_bzr(command)
            self.assertEqual(out, expected)
            self.assertEqual(err, '')
        tree = self.make_branch_and_tree('.')
        check_added('')
        self.build_tree_contents([(name, 'contents of %s\n' % name)])
        check_added('')
        tree.add(name)
        check_added(output, null)
        tree.commit(message='add "%s"' % name)
        check_added('')

    def test_added_directory(self):
        if False:
            print('Hello World!')
        'Test --directory option'
        tree = self.make_branch_and_tree('a')
        self.build_tree(['a/README'])
        tree.add('README')
        (out, err) = self.run_bzr(['added', '--directory=a'])
        self.assertEqual('README\n', out)