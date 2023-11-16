"""Test that all WorkingTree's implement get_file_with_stat."""
import os
from bzrlib.tests.per_tree import TestCaseWithTree

class TestGetFileWithStat(TestCaseWithTree):

    def test_get_file_with_stat_id_only(self):
        if False:
            for i in range(10):
                print('nop')
        work_tree = self.make_branch_and_tree('.')
        self.build_tree(['foo'])
        work_tree.add(['foo'], ['foo-id'])
        tree = self._convert_tree(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        (file_obj, statvalue) = tree.get_file_with_stat('foo-id')
        self.addCleanup(file_obj.close)
        if statvalue is not None:
            expected = os.lstat('foo')
            self.assertEqualStat(expected, statvalue)
        self.assertEqual(['contents of foo\n'], file_obj.readlines())

    def test_get_file_with_stat_id_and_path(self):
        if False:
            for i in range(10):
                print('nop')
        work_tree = self.make_branch_and_tree('.')
        self.build_tree(['foo'])
        work_tree.add(['foo'], ['foo-id'])
        tree = self._convert_tree(work_tree)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        (file_obj, statvalue) = tree.get_file_with_stat('foo-id', 'foo')
        self.addCleanup(file_obj.close)
        if statvalue is not None:
            expected = os.lstat('foo')
            self.assertEqualStat(expected, statvalue)
        self.assertEqual(['contents of foo\n'], file_obj.readlines())