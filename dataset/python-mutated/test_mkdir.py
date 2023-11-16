"""Black-box tests for bzr mkdir.
"""
import os
from bzrlib import tests

class TestMkdir(tests.TestCaseWithTransport):

    def test_mkdir(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.run_bzr(['mkdir', 'somedir'])
        self.assertEqual(tree.kind(tree.path2id('somedir')), 'directory')

    def test_mkdir_multi(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('.')
        self.run_bzr(['mkdir', 'somedir', 'anotherdir'])
        self.assertEqual(tree.kind(tree.path2id('somedir')), 'directory')
        self.assertEqual(tree.kind(tree.path2id('anotherdir')), 'directory')

    def test_mkdir_parents(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.')
        self.run_bzr(['mkdir', '-p', 'somedir/foo'])
        self.assertEqual(tree.kind(tree.path2id('somedir/foo')), 'directory')

    def test_mkdir_parents_existing_versioned_dir(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        tree.mkdir('somedir')
        self.assertEqual(tree.kind(tree.path2id('somedir')), 'directory')
        self.run_bzr(['mkdir', '-p', 'somedir'])

    def test_mkdir_parents_existing_unversioned_dir(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.')
        os.mkdir('somedir')
        self.run_bzr(['mkdir', '-p', 'somedir'])
        self.assertEqual(tree.kind(tree.path2id('somedir')), 'directory')

    def test_mkdir_parents_with_unversioned_parent(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('.')
        os.mkdir('somedir')
        self.run_bzr(['mkdir', '-p', 'somedir/foo'])
        self.assertEqual(tree.kind(tree.path2id('somedir/foo')), 'directory')