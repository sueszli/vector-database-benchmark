"""Tests for interface conformance of 'workingtree.put_file*'"""
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

class TestPutFileBytesNonAtomic(TestCaseWithWorkingTree):

    def test_put_new_file(self):
        if False:
            i = 10
            return i + 15
        t = self.make_branch_and_tree('t1')
        t.add(['foo'], ids=['foo-id'], kinds=['file'])
        t.put_file_bytes_non_atomic('foo-id', 'barshoom')
        self.assertEqual('barshoom', t.get_file('foo-id').read())

    def test_put_existing_file(self):
        if False:
            print('Hello World!')
        t = self.make_branch_and_tree('t1')
        t.add(['foo'], ids=['foo-id'], kinds=['file'])
        t.put_file_bytes_non_atomic('foo-id', 'first-content')
        t.put_file_bytes_non_atomic('foo-id', 'barshoom')
        self.assertEqual('barshoom', t.get_file('foo-id').read())