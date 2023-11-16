"""Test that all Tree's implement get_file_mtime"""
import time
from bzrlib import errors
from bzrlib.tests.per_tree import TestCaseWithTree

class TestGetFileMTime(TestCaseWithTree):

    def get_basic_tree(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/one'])
        tree.add(['one'], ['one-id'])
        return self._convert_tree(tree)

    def test_get_file_mtime(self):
        if False:
            print('Hello World!')
        now = time.time()
        tree = self.get_basic_tree()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        mtime_file_id = tree.get_file_mtime(file_id='one-id')
        self.assertIsInstance(mtime_file_id, (float, int))
        self.assertTrue(now - 5 < mtime_file_id < now + 5, 'now: %f, mtime_file_id: %f' % (now, mtime_file_id))
        mtime_path = tree.get_file_mtime(file_id='one-id', path='one')
        self.assertEqual(mtime_file_id, mtime_path)

    def test_nonexistant(self):
        if False:
            while True:
                i = 10
        tree = self.get_basic_tree()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertRaises(errors.NoSuchId, tree.get_file_mtime, file_id='unexistant')